"""ObfuscaTune LoRA finetuning smoke test (does the loop run + loss drop?).

This is ONLY a smoke test: it overfits a tiny GPT-2 to a small synthetic
next-token batch with LoRA adapters (rank=16, alpha=32, dropout=0.05, matching
the paper) and checks the loss decreases. It does NOT reproduce the paper's
WebQs/OBQA/PIQA/SciQ results, and it does NOT run the obfuscated forward during
training (inference-time obfuscation is validated by the correctness scripts).

Under ObfuscaTune the LoRA parameters live OUTSIDE the TEE (obfuscated with the
layer); we record that as a structural note only.

Backends:
  --backend torch  (default): a minimal pure-torch LoRA on GPT-2 Conv1D layers,
                    no extra dependency; always runnable.
  --backend peft :  use the 'peft' library (optional import; clear error if
                    absent -- we never auto-install).

Example:
    python scripts/obfuscatune/run_lora_smoke.py --steps 40
    python scripts/obfuscatune/run_lora_smoke.py --dry-run
"""

from __future__ import annotations

import argparse

from _common import (
    OUTPUT_ROOT, add_model_args, build_model, planned_config, write_json,
)


def _lora_torch(model, args):
    import torch
    import torch.nn as nn

    class LoRAConv1D(nn.Module):
        """Wrap a frozen GPT-2 Conv1D with a trainable low-rank delta."""

        def __init__(self, base, r, alpha, dropout):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            d_in, d_out = base.weight.shape  # Conv1D weight is (d_in, d_out)
            self.a = nn.Parameter(torch.zeros(d_in, r))
            self.b = nn.Parameter(torch.zeros(r, d_out))
            nn.init.normal_(self.a, std=1.0 / r)
            self.scaling = alpha / r
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.base(x) + (self.drop(x) @ self.a @ self.b) * self.scaling

    n_wrapped = 0
    for blk in model.transformer.h:
        blk.mlp.c_fc = LoRAConv1D(blk.mlp.c_fc, args.rank, args.alpha, args.dropout)
        blk.mlp.c_proj = LoRAConv1D(blk.mlp.c_proj, args.rank, args.alpha, args.dropout)
        n_wrapped += 2
    for p in [p for p in model.parameters() if p.requires_grad]:
        pass
    trainable = [p for p in model.parameters() if p.requires_grad]
    return trainable, {"lora_backend": "torch", "wrapped_modules": n_wrapped}


def _lora_peft(model, args):
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "--backend peft requires the 'peft' package, which is not installed. "
            "Install it yourself (we never auto-install) or use --backend torch."
        ) from exc
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=["c_fc", "c_proj"], bias="none")
    model = get_peft_model(model, cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    return model, trainable, {"lora_backend": "peft"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--backend", choices=["torch", "peft"], default="torch")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-examples", type=int, default=4)
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(planned_config(args, {"backend": args.backend, "rank": args.rank,
                                    "alpha": args.alpha, "dropout": args.dropout,
                                    "steps": args.steps}))
        return

    import torch

    model, minfo = build_model(args)
    model = model.to(torch.float32).train()
    if args.backend == "peft":
        model, trainable, linfo = _lora_peft(model, args)
    else:
        trainable, linfo = _lora_torch(model, args)

    g = torch.Generator().manual_seed(args.seed)
    batch = torch.randint(0, minfo["vocab_size"], (args.n_examples, args.seq_len), generator=g)
    opt = torch.optim.Adam(trainable, lr=args.lr)

    n_trainable = sum(p.numel() for p in trainable)
    losses = []
    for step in range(args.steps):
        opt.zero_grad()
        out = model(batch, labels=batch)
        loss = out.loss if hasattr(out, "loss") else out[0]
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"step {step:3d} loss={losses[-1]:.4f}")

    result = {
        "config": planned_config(args, {**linfo, "rank": args.rank, "alpha": args.alpha,
                                        "dropout": args.dropout, "steps": args.steps,
                                        "n_trainable_params": n_trainable}),
        "structural_note": "ObfuscaTune places LoRA params OUTSIDE the TEE (obfuscated with the layer); training here is plaintext smoke only.",
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "losses": losses,
    }
    out = OUTPUT_ROOT / "lora_smoke" / f"lora_smoke_{args.backend}.json"
    write_json(out, result)
    print(f"initial_loss={losses[0]:.4f} final_loss={losses[-1]:.4f} "
          f"decreased={result['loss_decreased']}  n_trainable={n_trainable}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
