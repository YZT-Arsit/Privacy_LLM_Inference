"""Gate 3 — plaintext N-step SGD reference (H800, local CE). Matches gate3_tdx_step's
data/seed/LoRA init so the protected masked-SGD trajectory can be compared step by step.
Saves per-step loss trajectory + FINAL-step deltaw/next_logits/grads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate3_worker import (  # noqa: E402
    TARGET_ATTN, TARGET_MLP, build_batch, flat_logits_labels, inject_lora, load_model, log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--targets", default="attn", choices=["attn", "attn_mlp"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="/root/gate3/out")
    args = ap.parse_args()

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    targets = TARGET_ATTN if args.targets == "attn" else TARGET_ATTN + TARGET_MLP
    model, tok = load_model(args.model_dir, dtype)
    texts = json.loads(Path(args.data).read_text())[: args.batch]
    input_ids, attn, labels = build_batch(tok, texts, args.seqlen, "cuda")
    layers = inject_lora(model, targets, args.rank, args.alpha, args.seed, masked=False)
    trainable = [p for l in layers.values() for p in l.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=args.lr, momentum=0.0, weight_decay=0.0)

    traj = []
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps):
        out = model(input_ids=input_ids, attention_mask=attn)
        Z, lab, valid = flat_logits_labels(out.logits.float(), labels)
        logp = torch.log_softmax(Z[valid], dim=-1)
        loss = -logp[torch.arange(int(valid.sum())), lab[valid]].mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        dw_before = {n: l.delta_w().detach().float().cpu() for n, l in layers.items()}
        opt.step()
        dw_after = {n: l.delta_w().detach().float().cpu() for n, l in layers.items()}
        traj.append({"step": step, "loss": float(loss.item()),
                     "finite": bool(torch.isfinite(loss).all())})
        log(f"plain step {step}: loss={loss.item():.6f}")
        if step == args.steps - 1:
            with torch.no_grad():
                nfl, _, _ = flat_logits_labels(model(input_ids=input_ids, attention_mask=attn).logits.float(), labels)
            torch.save({"deltaw_before": dw_before, "deltaw_after": dw_after,
                        "next_logits": nfl.cpu(),
                        "grads": {n: (l.A.grad.detach().cpu(), l.B.grad.detach().cpu())
                                  for n, l in layers.items()}},
                       outdir / "tensors_plaintext_multi.pt")
    (outdir / "result_plaintext_multi.json").write_text(json.dumps(
        {"mode": "plaintext_multi", "steps": args.steps, "trajectory": traj,
         "num_lora_layers": len(layers), "lr": args.lr}, indent=2))
    log(f"plaintext_multi done: {args.steps} steps")


if __name__ == "__main__":
    main()
