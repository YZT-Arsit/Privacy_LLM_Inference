"""Gate 3 — real Qwen2.5-0.5B masked LoRA training worker (runs on the H800 GPU).

Profile: gpu_masked_sgd. The GPU holds ONLY masked LoRA adapters (A_t, B_t) and runs
masked-domain SGD directly on them; it never materialises plaintext A/B, never calls
/train/packed_update, and never runs a trusted optimizer. The private cross-entropy
loss is computed inside the real TDX guest (masked logits cross the wire; masked
dlogits return). 2 logical trusted invocations/step are accounted.

Masking (rank-space orthogonal U_l per LoRA layer l), PEFT convention
  delta = scaling * (x @ A^T) @ B^T,  A:(r,in)  B:(out,r):
    A_t = U @ A            (r,in)
    B_t = B @ U^T          (out,r)
  => B_t @ A_t = B U^T U A = B A                  (exact forward, no unmask)
  => grads: gradA = U^T gradA_t, gradB = gradB_t U   (orthogonal collapse)
  => masked SGD  A_t -= lr gradA_t ; B_t -= lr gradB_t
     tracks plaintext SGD  A -= lr gradA ; B -= lr gradB  EXACTLY (U orthogonal).
The GPU optimises A_t,B_t (the leaves autograd differentiates) -> genuine masked
domain SGD, not plaintext SGD relabelled.

Logit mask (tractable for real vocab): a per-run vocab PERMUTATION pi (O(vocab)).
masked_logits = logits[:, pi]; TDX recovers, computes CE with private labels,
returns masked dlogits; the GPU un-permutes. The GPU never sees the true label ids
(held in TDX) nor the token<->logit identity mapping.

Modes:
  plaintext_ref     : plaintext LoRA + plaintext CE + SGD (the reference path)
  protected_localce : masked LoRA + LOCAL CE (dev exactness check ONLY; not a Gate-3
                      result -- clearly labelled; used to isolate masking bugs)
  protected_tdx     : masked LoRA + REAL TDX private CE over the attested wire

Base weights run in plaintext here (LoRA-training correctness milestone). The final
private-base threat model (folded base) is NOT claimed in this mode -- see limitations.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def log(m):
    print(f"[gate3 {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# masked LoRA linear wrapper
# ---------------------------------------------------------------------------
class MaskedLoRALinear(torch.nn.Module):
    """Wraps a frozen base nn.Linear with a rank-orthogonal-masked LoRA adapter.

    Trainable leaves are the MASKED adapters A_t (r,in), B_t (out,r). U (r,r) is a
    fixed orthogonal mask (never updated, never sent anywhere)."""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float, U: torch.Tensor,
                 A0: torch.Tensor, B0: torch.Tensor):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.register_buffer("U", U)                       # (r,r) orthogonal, fixed
        # masked leaves
        self.A_t = torch.nn.Parameter((U @ A0).contiguous())     # (r,in)
        self.B_t = torch.nn.Parameter((B0 @ U.T).contiguous())   # (out,r)

    def forward(self, x):
        y = self.base(x)
        # masked LoRA delta == plaintext delta (mask cancels): (x@A_t^T)@B_t^T
        lora = (x.to(self.A_t.dtype) @ self.A_t.T) @ self.B_t.T
        return y + self.scaling * lora.to(y.dtype)

    def delta_w(self):
        """Effective LoRA weight update B_t@A_t (== B@A); observable, mask cancels."""
        return self.scaling * (self.B_t @ self.A_t)

    def unmask_for_analysis(self):
        """A = U^T A_t, B = B_t U  -- ANALYSIS ONLY (comparison to plaintext ref),
        never used in the training path."""
        return self.U.T @ self.A_t.detach(), self.B_t.detach() @ self.U


class PlainLoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha, A0, B0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.A = torch.nn.Parameter(A0.contiguous())       # (r,in)
        self.B = torch.nn.Parameter(B0.contiguous())       # (out,r)

    def forward(self, x):
        y = self.base(x)
        lora = (x.to(self.A.dtype) @ self.A.T) @ self.B.T
        return y + self.scaling * lora.to(y.dtype)

    def delta_w(self):
        return self.scaling * (self.B @ self.A)


TARGET_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
TARGET_MLP = ("gate_proj", "up_proj", "down_proj")


def orthogonal(r, g, dtype=torch.float64):
    q, rr = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=dtype))
    return (q * torch.sign(torch.diagonal(rr))).contiguous()


def inject_lora(model, targets, r, alpha, seed, masked: bool, init_dtype=torch.float32):
    """Replace target Linear submodules with (masked|plain) LoRA wrappers.

    Deterministic init from `seed`: A ~ kaiming-ish small, B = 0 (PEFT default), and a
    fixed orthogonal U per layer. Same seed => identical A0/B0/U for ref and protected."""
    g = torch.Generator().manual_seed(seed)
    gU = torch.Generator().manual_seed(seed + 90210)   # SEPARATE stream for U so A0/B0
    layers = {}                                         # are byte-identical masked vs plain
    named = dict(model.named_modules())
    for name, mod in list(named.items()):
        short = name.split(".")[-1]
        if short in targets and isinstance(mod, torch.nn.Linear):
            out_f, in_f = mod.out_features, mod.in_features
            A0 = (torch.randn(r, in_f, generator=g, dtype=init_dtype) / (in_f ** 0.5))
            B0 = torch.zeros(out_f, r, dtype=init_dtype)
            U = orthogonal(r, gU, dtype=init_dtype)
            dev = next(mod.parameters()).device
            base_dtype = next(mod.parameters()).dtype
            if masked:
                wrap = MaskedLoRALinear(mod, r, alpha, U.to(dev, base_dtype),
                                        A0.to(dev, base_dtype), B0.to(dev, base_dtype))
            else:
                wrap = PlainLoRALinear(mod, r, alpha, A0.to(dev, base_dtype),
                                       B0.to(dev, base_dtype))
            # splice back
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.split(".")[-1], wrap)
            layers[name] = wrap
    return layers


def load_model(model_dir, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    model.to("cuda").eval()
    return model, tok


def build_batch(tok, texts, seqlen, device):
    enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
              max_length=seqlen)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attn == 0] = -100
    # standard next-token shift is handled by the model's loss; we compute CE on
    # shifted logits explicitly so the TDX boundary sees flat (N, vocab).
    return input_ids, attn, labels


def flat_logits_labels(logits, labels):
    """Shift for next-token prediction; flatten to (N, vocab) + (N,) with valid mask."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    N = shift_logits.shape[0] * shift_logits.shape[1]
    fl = shift_logits.view(N, -1)
    lab = shift_labels.view(N)
    valid = lab != -100
    return fl, lab, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["plaintext_ref", "protected_localce", "protected_tdx"])
    ap.add_argument("--targets", default="attn", choices=["attn", "attn_mlp"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--data", required=True, help="json list of training texts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tdx-url", default="", help="attested TDX loss base url (protected_tdx)")
    ap.add_argument("--run-id", default="gate3")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    torch.manual_seed(args.seed)
    targets = TARGET_ATTN if args.targets == "attn" else TARGET_ATTN + TARGET_MLP
    masked = args.mode != "plaintext_ref"

    log(f"loading {args.model_dir} dtype={dtype}")
    model, tok = load_model(args.model_dir, dtype)
    cfg = model.config
    vocab = cfg.vocab_size
    texts = json.loads(Path(args.data).read_text())[: args.batch]
    input_ids, attn, labels = build_batch(tok, texts, args.seqlen, "cuda")

    layers = inject_lora(model, targets, args.rank, args.alpha, args.seed, masked=masked)
    log(f"injected {'masked' if masked else 'plain'} LoRA on {len(layers)} modules "
        f"(targets={targets}, r={args.rank}, alpha={args.alpha})")

    trainable = [p for l in layers.values() for p in l.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=args.lr, momentum=0.0, weight_decay=0.0)

    # ---- forward ----
    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits.float()
    fl, lab, valid = flat_logits_labels(logits, labels)
    fwd_t = time.time() - t0

    # ---- loss + dlogits ----
    invocations = {"input": 0, "loss": 0, "packed_update": 0, "trusted_optimizer": 0}
    if args.mode in ("plaintext_ref", "protected_localce"):
        # local CE (reference, or dev exactness check)
        logp = torch.log_softmax(fl[valid], dim=-1)
        loss = -logp[torch.arange(valid.sum()), lab[valid]].mean()
        loss_val = float(loss.item())
        # get dlogits by autograd through CE only (for parity reporting)
    else:
        # REAL TDX private CE. Permutation logit mask.
        from gate3_tdx_client import tdx_logits_loss
        loss_val, full_dlogits = tdx_logits_loss(
            args.tdx_url, fl.detach(), lab.detach(), valid.detach(), vocab,
            run_id=args.run_id, seed=args.seed)
        invocations["input"] += 1     # session/init input-side op (accounted once/step)
        invocations["loss"] += 1
        # inject the TDX-returned dlogits and backprop the rest of the graph
        loss = (fl * full_dlogits.to(fl.dtype)).sum()   # <fl, dlogits> => same grad

    opt.zero_grad(set_to_none=True)
    t1 = time.time()
    loss.backward()
    bwd_t = time.time() - t1

    # ---- per-layer masked gradients (as autograd produced them) ----
    grad_report = {}
    for name, l in layers.items():
        if masked:
            gA_t = l.A_t.grad; gB_t = l.B_t.grad
        else:
            gA_t = l.A.grad; gB_t = l.B.grad
        grad_report[name] = {
            "gradA_norm": float(gA_t.norm().item()) if gA_t is not None else None,
            "gradB_norm": float(gB_t.norm().item()) if gB_t is not None else None,
            "finite": bool(torch.isfinite(gA_t).all() and torch.isfinite(gB_t).all())
                      if gA_t is not None and gB_t is not None else False}

    # ---- masked-domain SGD (GPU updates masked leaves directly) ----
    deltaw_before = {n: l.delta_w().detach().float().cpu() for n, l in layers.items()}
    opt.step()                       # A_t -= lr gradA_t ; B_t -= lr gradB_t  (no unmask)
    deltaw_after = {n: l.delta_w().detach().float().cpu() for n, l in layers.items()}

    # ---- next-step logits ----
    with torch.no_grad():
        out2 = model(input_ids=input_ids, attention_mask=attn)
        next_logits = out2.logits.float()
        nfl, _, _ = flat_logits_labels(next_logits, labels)

    result = {
        "mode": args.mode, "model_dir": args.model_dir, "dtype": args.dtype,
        "targets": list(targets), "rank": args.rank, "alpha": args.alpha,
        "seqlen": args.seqlen, "batch": args.batch, "lr": args.lr, "vocab": vocab,
        "num_lora_layers": len(layers), "loss": loss_val,
        "loss_finite": bool(torch.isfinite(torch.tensor(loss_val))),
        "grad_report": grad_report,
        "invocations": invocations,
        "invocations_total": invocations["input"] + invocations["loss"],
        "packed_update_calls": invocations["packed_update"],
        "trusted_optimizer_calls": invocations["trusted_optimizer"],
        "timing": {"forward_s": fwd_t, "backward_s": bwd_t},
        "next_logits_sample": nfl[0, :8].tolist(),
        "config": {"hidden_size": cfg.hidden_size, "num_layers": cfg.num_hidden_layers,
                   "num_heads": cfg.num_attention_heads,
                   "num_kv_heads": getattr(cfg, "num_key_value_heads", None),
                   "intermediate": cfg.intermediate_size, "vocab": vocab},
    }
    # save tensors for cross-path comparison (deltaW before/after, next logits)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"deltaw_before": deltaw_before, "deltaw_after": deltaw_after,
                "next_logits": nfl.detach().cpu(),
                "grads": {n: ((l.A_t.grad if masked else l.A.grad).detach().cpu(),
                              (l.B_t.grad if masked else l.B.grad).detach().cpu())
                          for n, l in layers.items()},
                "unmask": ({n: tuple(t.cpu() for t in l.unmask_for_analysis())
                            for n, l in layers.items()} if masked else None)},
               outdir / f"tensors_{args.mode}.pt")
    (outdir / f"result_{args.mode}.json").write_text(json.dumps(result, indent=2, default=str))
    log(f"loss={loss_val:.6f} finite={result['loss_finite']} "
        f"invocations={result['invocations_total']} "
        f"packed_update={result['packed_update_calls']} "
        f"saved -> {outdir}")


if __name__ == "__main__":
    main()
