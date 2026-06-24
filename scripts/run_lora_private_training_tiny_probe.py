"""E7: minimal private LoRA update prototype probe.

Updates a private LoRA adapter on a tiny synthetic task while keeping the training
inputs, labels, raw A/B, gradients, optimizer step, loss, and mask secrets on the
trusted boundary; the untrusted GPU does only the masked frozen-base matmul. The
recorded GPU trace is audited (no raw A/B / data / labels / gradients / optimizer
/ mask secrets crossed). Reports loss_before/after + adapter_delta_norm + the
security flags. NOT full fine-tuning -- see ``limitations``.

numpy only; no H800 / TDX / CUDA / Qwen checkpoint.

Example::

    python scripts/run_lora_private_training_tiny_probe.py \\
        --target-modules q_proj,v_proj --rank 8 --alpha 16 --steps 5 \\
        --output-json outputs/lora_private_training_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.training.lora_private_trainer import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    run_private_lora_training,
)


def _csv(s):
    return [p for p in str(s).replace(" ", "").split(",") if p]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--in-dim", type=int, default=16)
    ap.add_argument("--out-dim", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-json",
                    default="outputs/lora_private_training_probe.json")
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    report = run_private_lora_training(
        _csv(args.target_modules), rank=args.rank, alpha=args.alpha,
        steps=args.steps, lr=args.lr, in_dim=args.in_dim, out_dim=args.out_dim,
        batch=args.batch, seed=args.seed)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), report)

    print("=== private LoRA training probe (%s) ===" % report["task"])
    print("target_modules=%s rank=%s alpha=%s steps=%s"
          % (report["target_modules"], report["rank"], report["alpha"],
             report["training_steps"]))
    print("loss_before=%.6e loss_after=%.6e loss_decreased=%s"
          % (report["loss_before"], report["loss_after"],
             report["loss_decreased"]))
    print("adapter_delta_norm=%.6e" % report["adapter_delta_norm"])
    print("raw_lora_visible_to_gpu=%s optimizer_state_visible_to_gpu=%s "
          "training_data_visible_to_gpu=%s labels_visible_to_gpu=%s"
          % (report["raw_lora_visible_to_gpu"],
             report["optimizer_state_visible_to_gpu"],
             report["training_data_visible_to_gpu"],
             report["labels_visible_to_gpu"]))
    print("worker_has_mask_secrets=%s tee_used_on_gpu=%s "
          "gpu_visible_plaintext_fields=%s leaked_secret_fields=%s"
          % (report["worker_has_mask_secrets"], report["tee_used_on_gpu"],
             report["gpu_visible_plaintext_fields"] or "[]",
             report["leaked_secret_fields"] or "[]"))
    print("audit_passed=%s" % report["audit_passed"])

    ok = (report["loss_decreased"] and report["audit_passed"]
          and not report["raw_lora_visible_to_gpu"]
          and not report["optimizer_state_visible_to_gpu"]
          and not report["training_data_visible_to_gpu"]
          and not report["labels_visible_to_gpu"]
          and not report["worker_has_mask_secrets"]
          and not report["tee_used_on_gpu"]
          and not report["leaked_secret_fields"]
          and not report["gpu_visible_plaintext_fields"])
    print("\nPRIVATE LoRA TRAINING PROBE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def _write_md(path: Path, r: dict) -> None:
    L = ["# Private LoRA training probe (%s)" % r["task"], "",
         "- target_modules=`%s`  rank=%s  alpha=%s  steps=%s  lr=%s"
         % (r["target_modules"], r["rank"], r["alpha"], r["training_steps"],
            r["lr"]),
         "- **loss_before=%.6e  loss_after=%.6e  loss_decreased=%s**"
         % (r["loss_before"], r["loss_after"], r["loss_decreased"]),
         "- **adapter_delta_norm=%.6e**" % r["adapter_delta_norm"],
         "", "## Security (audited against the exact GPU trace)", "",
         "- raw_lora_visible_to_gpu=%s" % r["raw_lora_visible_to_gpu"],
         "- optimizer_state_visible_to_gpu=%s"
         % r["optimizer_state_visible_to_gpu"],
         "- training_data_visible_to_gpu=%s" % r["training_data_visible_to_gpu"],
         "- labels_visible_to_gpu=%s" % r["labels_visible_to_gpu"],
         "- gradients_visible_to_gpu=%s" % r["gradients_visible_to_gpu"],
         "- worker_has_mask_secrets=%s  tee_used_on_gpu=%s"
         % (r["worker_has_mask_secrets"], r["tee_used_on_gpu"]),
         "- gpu_visible_plaintext_fields=%s  leaked_secret_fields=%s"
         % (r["gpu_visible_plaintext_fields"] or "[]",
            r["leaked_secret_fields"] or "[]"),
         "- **audit_passed=%s**" % r["audit_passed"],
         "", "## Limitations", ""]
    L += ["%d. %s" % (i + 1, t) for i, t in enumerate(r["limitations"])]
    L += [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
