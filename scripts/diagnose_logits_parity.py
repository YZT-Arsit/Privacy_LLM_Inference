"""Stepwise logits-parity diagnostic: plaintext trusted greedy vs folded_remote.

Runs the SAME prompt(s) through (a) the plaintext trusted greedy loop and (b) the
folded_remote masked prefill+decode, capturing the top-K recovered logits/token ids
for the prefill step and the first N decode steps on both sides, then reports:

* top1 / top5 agreement per step + overall top1-agreement rate;
* ``max_abs_error`` (max |plain - ours| over the compared vocab) per step;
* ``rank_error`` (rank of the plaintext-argmax token in ours' ordering);
* the FIRST divergence step and whether it begins at **prefill** (step 0) or
  **decode** (step >= 1) -- the key signal for the IFEval id=1005 degeneration.

A JSON report is written; ``--require-top1`` (default) makes the exit code non-zero
when any step's top1 disagrees, so the diagnostic can gate paper-facing sanity
prompts. With no real model/worker, ``--self-test`` exercises the comparison core on
deterministic synthetic logits (CI-runnable, no downloads).

Example (real, on the TDX boundary client)::

    python scripts/diagnose_logits_parity.py \\
      --model-path <MODEL> --gpu-worker-url http://127.0.0.1:18082 \\
      --embedding-path <EMB> --nonlinear-backend A_rightmul \\
      --seq-len 1024 --max-steps 8 --prompt "The capital of France is" \\
      --output-json outputs/aaai/qwen/logits_parity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.logits_parity import (  # noqa: E402
    PARITY_SANITY_PROMPTS, compare_run)


def _run_real(args, prompts):
    from pllo.benchmarks.real_predictors import build_predictor
    evidence = None
    if args.attestation_evidence_json:
        evidence = json.loads(
            Path(args.attestation_evidence_json).read_text(encoding="utf-8"))
    plain = build_predictor(
        "plaintext_local", model_path=args.model_path,
        model_name=args.model_name, seq_len=args.seq_len,
        max_new_tokens=args.max_steps, dtype=args.dtype, device=args.device,
        use_chat_template=bool(args.use_chat_template),
        align_generation_config=bool(args.align_generation_config),
        repetition_penalty=args.repetition_penalty)
    ours = build_predictor(
        "folded_remote", model_path=args.model_path, model_name=args.model_name,
        gpu_worker_url=args.gpu_worker_url, embedding_path=args.embedding_path,
        attestation_evidence=evidence, expected_mr_td=args.expected_mr_td,
        seq_len=args.seq_len, max_new_tokens=args.max_steps, dtype=args.dtype,
        device=args.device, audit=False, nonlinear_backend=args.nonlinear_backend,
        use_chat_template=bool(args.use_chat_template),
        align_generation_config=bool(args.align_generation_config),
        repetition_penalty=args.repetition_penalty)
    per_prompt = []
    for p in prompts:
        _, plain_steps = plain.greedy_with_logits(p, max_steps=args.max_steps)
        ours.enable_logits_capture(max_steps=args.max_steps)
        ours.generate(p)
        ours_steps = ours.captured_logits()
        n = min(len(plain_steps), len(ours_steps))
        run = compare_run(list(zip(plain_steps[:n], ours_steps[:n])),
                          k=args.topk, max_abs_tol=args.max_abs_tol,
                          require_top1=args.require_top1)
        run["prompt_sha256"] = __import__("hashlib").sha256(
            p.encode("utf-8")).hexdigest()[:16]
        per_prompt.append(run)
    for pr in (plain, ours):
        try:
            pr.close()
        except Exception:                                    # noqa: BLE001
            pass
    return per_prompt


def _self_test(args, prompts):
    """Deterministic synthetic parity runs (no model): one matching, one that
    diverges at a decode step -- exercises the comparison + reporting."""
    runs = []
    # prompt 0: perfect parity over max_steps
    steps_ok = [([0.1 * (j == (i % 5)) for j in range(8)],
                 [0.1 * (j == (i % 5)) for j in range(8)])
                for i in range(args.max_steps)]
    runs.append(compare_run(steps_ok, k=args.topk, max_abs_tol=args.max_abs_tol,
                            require_top1=args.require_top1))
    # prompt 1: diverges at decode step 2 (ours picks a different top1)
    steps_div = []
    for i in range(args.max_steps):
        plain = [float(j == 0) for j in range(8)]
        ours = [float(j == (1 if i == 2 else 0)) for j in range(8)]
        steps_div.append((plain, ours))
    runs.append(compare_run(steps_div, k=args.topk, max_abs_tol=args.max_abs_tol,
                            require_top1=args.require_top1))
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--attestation-evidence-json", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--nonlinear-backend", default="A_rightmul")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-abs-tol", type=float, default=1e-2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-chat-template", action="store_true", default=False)
    ap.add_argument("--align-generation-config", action="store_true",
                    default=False)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--require-top1", dest="require_top1", action="store_true",
                    default=True)
    ap.add_argument("--no-require-top1", dest="require_top1",
                    action="store_false")
    ap.add_argument("--prompt", action="append", default=None,
                    help="prompt(s) to diagnose (repeatable); default = the "
                    "built-in neutral sanity prompts")
    ap.add_argument("--self-test", action="store_true", default=False,
                    help="run the synthetic comparison (no model/worker)")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    prompts = args.prompt or list(PARITY_SANITY_PROMPTS)
    if args.self_test:
        per_prompt = _self_test(args, prompts)
    else:
        if not (args.model_path and args.gpu_worker_url and args.embedding_path):
            print("ERROR: real diagnostic needs --model-path / --gpu-worker-url / "
                  "--embedding-path (or use --self-test)", file=sys.stderr)
            return 3
        per_prompt = _run_real(args, prompts)

    all_pass = all(r["passed"] for r in per_prompt) if per_prompt else False
    first_phase = next((r["divergence_phase"] for r in per_prompt
                        if r.get("divergence_phase")), None)
    report = {
        "stage": "logits_parity_diagnostic",
        "nonlinear_backend": args.nonlinear_backend,
        "num_prompts": len(per_prompt),
        "logits_parity_sanity_passed": bool(all_pass),
        "first_divergence_phase": first_phase,
        "self_test": bool(args.self_test),
        "per_prompt": per_prompt,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2),
                                      encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "num_prompts", "logits_parity_sanity_passed",
        "first_divergence_phase")}, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
