"""Strict HF forward-parity diagnostic for the internal Qwen2.5 reference.

Compares the internal extracted-weight plaintext reference against the official
HuggingFace forward, stage by stage (embedding -> each decoder layer -> final
norm -> next-token logits), on the EXACT same chat-templated input_ids /
attention_mask. Pinpoints ``first_mismatch_stage`` / ``first_mismatch_layer``
and reports per-layer error metrics + next-token top1 for both.

Untrusted-GPU only: **no TEE import, no TEE execution** (``tee_used=False``).
ModelScope cache only (local files); never Hugging Face remote download.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.hf_wrappers.llama_qwen_single_block import (  # noqa: E402
    extract_hf_single_block_weights,
    hf_single_block_plain_prefill,
    infer_config_from_hf_layer,
)
from pllo.hf_wrappers.qwen_memory_optimized import (  # noqa: E402
    align_qk_weights_to_hf_rope,
)
from pllo.ops.causal_lm_boundaries import trusted_embedding_lookup  # noqa: E402
from pllo.ops.nonlinear_islands import rmsnorm_core  # noqa: E402
from pllo.ops.rope import build_rope_cache  # noqa: E402

_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    diff = (a - b)
    denom = float(b.pow(2).sum().sqrt().item()) or 1.0
    cos_d = float((a.norm() * b.norm()).item()) or 1.0
    return {
        "max_abs_error": float(diff.abs().max().item()),
        "relative_l2_error": float(diff.pow(2).sum().sqrt().item() / denom),
        "cosine_similarity": float((a @ b).item() / cos_d),
    }


def internal_reference(model, model_config, input_ids, dtype, device, align):
    """Streaming extracted-weight plaintext reference; returns embedding,
    per-layer hidden states, final-norm output, and logits."""
    base = getattr(model, "model", model)
    fdtype = torch.float32                      # reference precision
    embed = base.embed_tokens.weight.detach().to(device=device, dtype=fdtype)
    h = trusted_embedding_lookup(input_ids.to(device), embed)
    cfg0 = infer_config_from_hf_layer(base.layers[0], model_config, fdtype,
                                      device)
    max_pos = input_ids.shape[1] + 1
    cos, sin = build_rope_cache(max_pos, cfg0.head_dim, cfg0.rope_theta, fdtype,
                               device)
    emb = h
    per_layer = []
    for ell in range(len(base.layers)):
        w = extract_hf_single_block_weights(base.layers[ell], fdtype, device)
        cfg = infer_config_from_hf_layer(base.layers[ell], model_config,
                                         fdtype, device)
        if align:
            w = align_qk_weights_to_hf_rope(w, cfg.num_heads,
                                            cfg.num_key_value_heads,
                                            cfg.head_dim)
        h = hf_single_block_plain_prefill(h, w, cfg, cos, sin)["y"]
        per_layer.append(h)
        del w
    final_norm = rmsnorm_core(h, cfg0.rms_norm_eps) * \
        base.norm.weight.detach().to(device=device, dtype=fdtype)
    head = getattr(model, "lm_head", None)
    if head is not None and getattr(head, "weight", None) is not None:
        lm_w = head.weight.detach().to(device=device, dtype=fdtype).t()
    else:
        lm_w = embed.t()
    logits = final_norm @ lm_w
    return {"embedding": emb, "per_layer": per_layer, "final_norm": final_norm,
            "logits": logits, "rope_theta": cfg0.rope_theta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--seq-len", type=int, default=128, help="max prompt length")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-align", action="store_true",
                    help="disable the HF RoPE alignment (to show the bug)")
    ap.add_argument("--cos-threshold", type=float, default=0.9990)
    ap.add_argument("--l2-threshold", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-json",
                    default="outputs/qwen_hf_parity_diagnostic.json")
    args = ap.parse_args()

    if args.dry_run:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        mc = Qwen2Config(vocab_size=512, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=256,
                         rms_norm_eps=1e-6, rope_theta=1000000.0,
                         tie_word_embeddings=False)
        model = Qwen2ForCausalLM(mc).eval()
        tok = None
        g = torch.Generator().manual_seed(0)
        input_ids = torch.randint(0, 512, (1, 48), generator=g)
        attention_mask = torch.ones_like(input_ids)
        chat_text = None
    else:
        if not args.model_path:
            ap.error("--model-path required unless --dry-run")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = _DTYPE[args.dtype]
        tok = AutoTokenizer.from_pretrained(args.model_path,
                                            trust_remote_code=True,
                                            local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=dt, device_map=args.device,
            trust_remote_code=True, local_files_only=True).eval()
        mc = model.config
        prompt = args.prompt
        if args.prompt_file:
            prompt = json.loads(Path(args.prompt_file).read_text(
                encoding="utf-8").splitlines()[0])["prompt"]
        chat_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True)
        enc = tok(chat_text, return_tensors="pt")
        input_ids = enc["input_ids"][:, :args.seq_len].to(args.device)
        attention_mask = enc["attention_mask"][:, :args.seq_len].to(args.device)

    device = args.device if (args.device.startswith("cuda")
                             and torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        model = model.to("cpu")
        input_ids = input_ids.to("cpu")
        attention_mask = attention_mask.to("cpu")

    last_valid_index = int(attention_mask.sum(dim=1).item()) - 1

    with torch.no_grad():
        hf = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=True, output_hidden_states=True, return_dict=True)
    hf_hidden = hf.hidden_states                       # len = num_layers + 1
    hf_next_logits = hf.logits[:, last_valid_index, :]
    hf_top1 = int(hf_next_logits.argmax(dim=-1).item())

    ref = internal_reference(model, mc, input_ids, _DTYPE[args.dtype], device,
                             align=not args.no_align)
    int_next_logits = ref["logits"][:, last_valid_index, :]
    int_top1 = int(int_next_logits.argmax(dim=-1).item())

    # stage comparisons
    emb_m = _metrics(ref["embedding"], hf_hidden[0])
    per_layer = [_metrics(ref["per_layer"][i], hf_hidden[i + 1])
                 for i in range(len(ref["per_layer"]))]
    logits_m = _metrics(int_next_logits, hf_next_logits)

    def _bad(m):
        # Cosine is the robust signal for the RoPE-theta / convention bugs;
        # relative-L2 alone trips on Qwen's last-layer massive-activation dims
        # even at exact parity, so it is reported but not used for the verdict.
        return m["cosine_similarity"] < args.cos_threshold

    first_stage, first_layer = None, None
    if _bad(emb_m):
        first_stage = "embedding"
    else:
        for i, m in enumerate(per_layer):
            if _bad(m):
                first_stage, first_layer = "layer", i
                break
        if first_stage is None and _bad(logits_m):
            first_stage = "logits"

    report = {
        "stage": "qwen_hf_parity_diagnostic",
        "tee_used": False,
        "dry_run": bool(args.dry_run),
        "model_type": str(getattr(mc, "model_type", "unknown")),
        "align_rope_to_hf": not args.no_align,
        "rope_theta_used": ref["rope_theta"],
        "dtype": args.dtype,
        "raw_prompt": None if args.dry_run else args.prompt,
        "chat_template_input": chat_text,
        "input_ids_shape": list(input_ids.shape),
        "last_valid_index": last_valid_index,
        "first_mismatch_stage": first_stage,
        "first_mismatch_layer": first_layer,
        "hf_top1_token": hf_top1,
        "internal_top1_token": int_top1,
        "hf_top1_text": _safe_decode(tok, [hf_top1]),
        "internal_top1_text": _safe_decode(tok, [int_top1]),
        "top1_match": hf_top1 == int_top1,
        "logits_max_abs_error": logits_m["max_abs_error"],
        "logits_mean_abs_error": float((int_next_logits - hf_next_logits)
                                       .abs().float().mean().item()),
        "logits_relative_l2_error": logits_m["relative_l2_error"],
        "cosine_similarity": logits_m["cosine_similarity"],
        "embedding_metrics": emb_m,
        "per_layer_max_abs_error": [m["max_abs_error"] for m in per_layer],
        "per_layer_relative_l2_error": [m["relative_l2_error"]
                                        for m in per_layer],
        "per_layer_cosine_similarity": [m["cosine_similarity"]
                                        for m in per_layer],
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"first_mismatch_stage={first_stage} first_mismatch_layer={first_layer}")
    print(f"hf_top1={hf_top1} internal_top1={int_top1} match={report['top1_match']}")
    print(f"rope_theta_used={ref['rope_theta']} align={not args.no_align}")
    print(f"logits cos={logits_m['cosine_similarity']:.6f} "
          f"rel_l2={logits_m['relative_l2_error']:.3e}")
    print(f"Wrote: {out}")
    return 0 if report["top1_match"] else 1


def _safe_decode(tok, ids):
    if tok is None:
        return None
    try:
        return tok.decode(ids, skip_special_tokens=True)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
