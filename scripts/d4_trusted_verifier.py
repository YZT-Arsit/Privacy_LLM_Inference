"""D4 trusted-evaluation verifier (runs on H800 in a trusted-eval role, NOT the
deployed worker). Holds the plaintext checkpoint + mask secrets (re-derived from the
frozen builder seeds) to check the protected run against a plaintext reference:

  * un-permute the worker's masked logits (vocab perm) -> plaintext logits, compute the
    L1 plaintext CE (== what real TDX must return);
  * un-FOLD the worker's per-target effective masked dW to plaintext effective dW using
    the exact linear per-projection folds, apply to the plaintext HF model, and compare
    the worker's effective (un-permuted) logits to the plaintext-model+effective-LoRA
    logits: next-logit KL + top-1 agreement (the effective-equivalence metric).

This is EVALUATION only; it never runs inside the untrusted worker path.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from h800_unified_worker import TrustedVerifier, CKPT  # noqa: E402

DT = torch.float64


def fold_matrices(v: TrustedVerifier, l, proj):
    """(OUT_left, IN_right) such that W_tilde = OUT_left @ W_plain @ IN_right (out,in)."""
    Nr = v.Nr
    ga = v.sd[f"model.layers.{l}.input_layernorm.weight"].to(DT)
    gm = v.sd[f"model.layers.{l}.post_attention_layernorm.weight"].to(DT)
    nh, nkv = v.nh, v.nkv
    B = v.B[l]; S = v.S[l]; P = v.P[l]
    Bq = torch.block_diag(*([B] * nh)); Bk = torch.block_diag(*([B] * nkv))
    Sv = torch.block_diag(*([S] * nkv)); So = torch.block_diag(*([S] * nh))
    diag = lambda x: torch.diag(x)
    if proj == "q_proj":   return Bq.T, diag(ga) @ Nr
    if proj == "k_proj":   return Bk.T, diag(ga) @ Nr
    if proj == "v_proj":   return Sv.T, diag(ga) @ Nr
    if proj == "o_proj":   return Nr.T, So
    if proj == "gate_proj": return P.T, diag(gm) @ Nr
    if proj == "up_proj":   return P.T, diag(gm) @ Nr
    if proj == "down_proj": return Nr.T, P
    raise ValueError(proj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked-logits", required=True)
    ap.add_argument("--lora-state", default="", help="masked LoRA state; effective dW "
                    "is computed here (scale*B@A per target)")
    ap.add_argument("--lora-scale", type=float, default=2.0)   # alpha/rank = 16/8
    ap.add_argument("--input-ids", required=True)
    ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads((CKPT / "config.json").read_text())
    device = torch.device("cuda")
    v = TrustedVerifier(cfg, device)
    meta = json.loads(Path(args.input_ids).read_text())
    ids = torch.tensor(meta["input_ids"][:args.seq_len])

    masked = torch.load(args.masked_logits, map_location="cpu").to(DT)   # (T,V)
    perm = v.perm
    plain_from_masked = masked[:, perm]                                  # un-permute
    # L1 plaintext CE from the worker's effective (un-permuted) logits
    lg = plain_from_masked[:-1]; tg = ids[1:]
    l1_ce = float(F.cross_entropy(lg.float(), tg))

    result = {"l1_ce_from_worker_effective": l1_ce, "seq_len": int(ids.shape[0])}

    # effective equivalence: HF plaintext model + un-folded effective LoRA dW.
    # Compute effective masked dW here from the LoRA state: dW_t = scale * B_t @ A_t.
    if args.lora_state:
        from transformers import AutoModelForCausalLM
        state = torch.load(args.lora_state, map_location="cpu")
        dw = {k: args.lora_scale * (B.to(DT) @ A.to(DT)) for k, (A, B) in state.items()}
        hf = AutoModelForCausalLM.from_pretrained(str(CKPT), dtype=torch.float32).to(device).eval()
        # apply un-folded plaintext effective dW to each target weight
        with torch.no_grad():
            for l in range(cfg["num_hidden_layers"]):
                lyr = hf.model.layers[l]
                mods = {"q_proj": lyr.self_attn.q_proj, "k_proj": lyr.self_attn.k_proj,
                        "v_proj": lyr.self_attn.v_proj, "o_proj": lyr.self_attn.o_proj,
                        "gate_proj": lyr.mlp.gate_proj, "up_proj": lyr.mlp.up_proj,
                        "down_proj": lyr.mlp.down_proj}
                for proj, mod in mods.items():
                    dW_t = dw[f"{l}.{proj}"].to(DT)                      # masked effective dW
                    OUT_left, IN_right = fold_matrices(v, l, proj)
                    dW_plain = torch.linalg.solve(OUT_left, dW_t)
                    dW_plain = torch.linalg.solve(IN_right.T, dW_plain.T).T
                    mod.weight.add_(dW_plain.to(mod.weight.dtype).to(device))
            hf_logits = hf(ids.to(device).unsqueeze(0)).logits[0].float().cpu()
        wl = plain_from_masked.float()
        kl = float(F.kl_div(F.log_softmax(hf_logits, -1),
                            F.softmax(wl, -1), reduction="batchmean"))
        top1 = float((wl.argmax(-1) == hf_logits.argmax(-1)).float().mean())
        maxabs = float((wl - hf_logits).abs().max())
        hf_ce = float(F.cross_entropy(hf_logits[:-1], tg))
        result.update({
            "effective_equivalence_top1_agreement": top1,
            "effective_equivalence_next_logit_kl": kl,
            "effective_equivalence_max_abs": maxabs,
            "hf_plaintext_plus_effective_lora_ce": hf_ce,
            "l1_vs_hf_ce_abs_diff": abs(l1_ce - hf_ce)})
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
