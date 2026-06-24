"""Run the protected LoRA *training* protection experiment suite.

Runs, for ranks 4/8/16: synthetic-linear and tiny-transformer protected-vs-
plaintext LoRA training (exact correctness), the training-stage security audit
over the GPU trace, and the three attack baselines (adapter recovery, gradient
inversion, membership). GPT-2 and Qwen2.5-7B are gated feasibility probes.

Writes outputs/lora_training_protection_summary.{json,csv,md}. numpy only for the
implemented tasks (torch is only touched by the gated probes).

Example::

    python scripts/run_lora_training_protection_experiments.py \\
        --ranks 4,8,16 --output-dir outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.lora_training_protection import run_all  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md(path: Path, summary: dict) -> None:
    L = ["# Protected LoRA Training Protection — Summary", "",
         f"- tee_used_on_gpu: **{summary['tee_used_on_gpu']}**",
         f"- ranks: {summary['ranks']}",
         f"- all_allclose (protected == plaintext): **{summary['all_allclose']}**",
         f"- all_audits_passed: **{summary['all_audits_passed']}**", "",
         "## 1. Correctness (protected vs plaintext LoRA training)", "",
         "| task | rank | allclose | max ΔW err | top1 | loss_curve_dist | "
         "final_eval_delta | task metric (plain/prot) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in summary["correctness"]:
        dW = r.get("max_delta_w_error", r.get("max_param_error", 0.0))
        L.append(
            f"| {r['task']} | {r['rank']} | {r['final_logits_allclose']} | "
            f"{dW:.2e} | {r['top1_match_rate']:.3f} | "
            f"{r['loss_curve_distance']:.2e} | {r['final_eval_delta']:.2e} | "
            f"{r['final_task_metric_plain']:.4f} / "
            f"{r['final_task_metric_protected']:.4f} |")
    L += ["", "## 2. Training-stage security audit (GPU-visible trace)", "",
          "All flags must be False; `audit_passed` True.", "",
          "| run | train ex | labels | input_ids | LoRA A | LoRA B | ΔW | "
          "grad A | grad B | optim | adapter Δ | plain hidden | leaked | "
          "passed |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, a in summary["security_audit"].items():
        L.append(
            f"| {name} | {a['gpu_visible_train_examples']} | "
            f"{a['gpu_visible_labels']} | {a['gpu_visible_input_ids']} | "
            f"{a['gpu_visible_lora_a']} | {a['gpu_visible_lora_b']} | "
            f"{a['gpu_visible_delta_w']} | {a['gpu_visible_lora_grad_a']} | "
            f"{a['gpu_visible_lora_grad_b']} | "
            f"{a['gpu_visible_optimizer_state']} | "
            f"{a['gpu_visible_adapter_update']} | "
            f"{a['gpu_visible_plain_hidden']} | "
            f"{a['leaked_secret_fields'] or 'none'} | **{a['audit_passed']}** |")
    L += ["", "## 3. Attack evaluation (synthetic linear)", "",
          "| rank | adapter recovery rel.err | ΔW recovery rel.err | "
          "grad-inv baseline | grad-inv protected | membership AUC | "
          "membership baseline AUC |", "|---|---|---|---|---|---|---|"]
    for rk, at in summary["attacks"].items():
        L.append(
            f"| {rk} | {at['adapter_recovery_relative_error']:.3f} | "
            f"{at['delta_w_recovery_relative_error']:.3f} | "
            f"{at['gradient_inversion_baseline_error']:.3f} | "
            f"{at['gradient_inversion_reconstruction_error']:.3f} | "
            f"{at['membership_attack_auc']:.3f} | "
            f"{at['membership_baseline_auc']:.3f} |")
    L += ["", "_Adapter recovery ≈ 1.0 (no information on the wire); gradient-"
          "inversion baseline ≈ 0 (plaintext grads leak the input) vs protected "
          "≈ 1+ (only masked activations); membership AUC ≈ 0.5 (random) vs "
          "baseline ≈ 1.0._", "",
          "## 4. Scale tasks / probes", "",
          f"- GPT-2: `{summary['gpt2_probe'].get('status')}` — "
          f"{summary['gpt2_probe'].get('reason', '')}",
          f"- Qwen2.5-7B: `{summary['qwen_probe'].get('status')}` — "
          f"{summary['qwen_probe'].get('reason', '')} "
          f"(probe_only={summary['qwen_probe'].get('probe_only')})", "",
          "See `docs/lora_training_protection.md` for scope and claims "
          "discipline (Qwen is a feasibility probe, not full fine-tuning)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", default="4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=16.0,
                    help="fixed LoRA alpha across ranks (main setting 16 or 32)")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW decoupled weight decay")
    ap.add_argument("--gpt2-model-path", default=None,
                    help="local GPT-2 checkpoint (skipped if absent)")
    ap.add_argument("--qwen-model-path", default=None,
                    help="local Qwen2.5-7B checkpoint for the one/few-step probe")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--no-gpt2", action="store_true")
    ap.add_argument("--no-qwen", action="store_true")
    args = ap.parse_args()

    ranks = tuple(int(x) for x in args.ranks.split(",") if x.strip())
    summary = run_all(ranks, seed=args.seed, alpha=args.alpha,
                      weight_decay=args.weight_decay,
                      gpt2_model_path=args.gpt2_model_path,
                      qwen_model_path=args.qwen_model_path,
                      include_gpt2=not args.no_gpt2,
                      include_qwen=not args.no_qwen)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lora_training_protection_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_csv(out / "lora_training_protection_summary.csv",
               summary["correctness"])
    _write_md(out / "lora_training_protection_summary.md", summary)

    print("=== LoRA training protection suite ===")
    print(f"ranks={summary['ranks']} all_allclose={summary['all_allclose']} "
          f"all_audits_passed={summary['all_audits_passed']} "
          f"tee_used_on_gpu={summary['tee_used_on_gpu']}")
    for r in summary["correctness"]:
        print(f"  {r['task']:<18} r={r['rank']:>2} allclose={r['final_logits_allclose']} "
              f"top1={r['top1_match_rate']:.3f} "
              f"task(plain/prot)={r['final_task_metric_plain']:.4f}/"
              f"{r['final_task_metric_protected']:.4f}")
    for rk, at in summary["attacks"].items():
        print(f"  attack {rk}: adapter_rec={at['adapter_recovery_relative_error']:.2f} "
              f"grad_inv(base/prot)={at['gradient_inversion_baseline_error']:.2f}/"
              f"{at['gradient_inversion_reconstruction_error']:.2f} "
              f"membership_auc={at['membership_attack_auc']:.2f}")
    print(f"  gpt2={summary['gpt2_probe'].get('status')} "
          f"qwen={summary['qwen_probe'].get('status')}")
    ok = summary["all_allclose"] and summary["all_audits_passed"]
    print(f"\nSUITE {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
