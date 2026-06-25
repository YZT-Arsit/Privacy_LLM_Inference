"""Open-ended generation preservation benchmark -- ONE backend (JSON/CSV/MD).

Runs deterministic greedy generation over a ``{id, prompt, category}`` JSONL with
a single backend and writes a per-backend generation report. Run it once for the
plaintext baseline and once for the protected folded path, then compare them with
``scripts/run_generation_preservation_pairwise.py``.

Backends (CURRENT design only -- trusted_shortcut is refused):
* ``plaintext_local`` -- real Qwen checkpoint via transformers (greedy).
* ``folded_remote``   -- trusted lite boundary + remote GPU worker over the
  folded package (masked embeddings only cross to the GPU; greedy).

Honest labeling: with no model/worker resources the deterministic stub runs and
the report is ``dry_run=True, paper_ready=False``. ``--require-real`` forbids the
stub fallback. No downloads (loads are local-files-only), no LLM judge, no
subjective quality scoring.

Example (H800, baseline then protected)::

    python scripts/run_generation_preservation_benchmark.py \\
        --dataset-jsonl data/opengen_prompts.jsonl --backend plaintext_local \\
        --model-path $MODEL --require-real --seq-len 256 --max-new-tokens 64 \\
        --output-json out/gen_plaintext.json --output-md out/gen_plaintext.md

    python scripts/run_generation_preservation_benchmark.py \\
        --dataset-jsonl data/opengen_prompts.jsonl --backend folded_remote \\
        --nonlinear-backend current --model-path $MODEL \\
        --gpu-worker-url $URL --embedding-path $ART --require-real \\
        --seq-len 256 --max-new-tokens 64 \\
        --output-json out/gen_folded.json --output-md out/gen_folded.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_preservation import (  # noqa: E402
    SUPPORTED_BACKENDS,
    assert_current_only,
    load_examples,
    render_benchmark_csv,
    render_benchmark_md,
    run_generation_benchmark,
)
from pllo.benchmarks.real_predictors import (  # noqa: E402
    RealBackendUnavailable,
    build_predictor,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-jsonl", required=True,
                    help="JSONL of {id, prompt, category}")
    ap.add_argument("--backend", required=True, choices=list(SUPPORTED_BACKENDS))
    ap.add_argument("--nonlinear-backend", default="current",
                    help="CURRENT only (trusted_shortcut is refused here)")
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=200,
                    help="cap generations stored in the report (all are scored)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--require-real", action="store_true", default=False)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    # current-only guard (refuse trusted_shortcut up front)
    try:
        nb = assert_current_only(args.nonlinear_backend)
    except ValueError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 3

    examples = load_examples(args.dataset_jsonl)
    if args.max_examples and args.max_examples > 0:
        examples = examples[:args.max_examples]
    if not examples:
        print("ERROR: no examples in %s" % args.dataset_jsonl, file=sys.stderr)
        return 3

    audit = str(args.audit).strip().lower() in {"1", "true", "yes", "y", "on"}

    # decide real vs stub
    if args.backend == "plaintext_local":
        have_real = bool(args.model_path)
    else:
        have_real = bool(args.model_path and args.gpu_worker_url
                         and args.embedding_path)

    predictor = None
    if have_real or args.require_real:
        try:
            predictor = build_predictor(
                args.backend, model_path=args.model_path,
                model_name=args.model_name, gpu_worker_url=args.gpu_worker_url,
                embedding_path=args.embedding_path,
                seq_len=args.seq_len, max_new_tokens=args.max_new_tokens,
                dtype=args.dtype, device=args.device, audit=audit,
                nonlinear_backend=nb)
        except RealBackendUnavailable as exc:
            if args.require_real:
                print("ERROR: --require-real but real backend unavailable: %s"
                      % exc, file=sys.stderr)
                return 3
            predictor = None

    try:
        report = run_generation_benchmark(
            examples, backend=args.backend, predictor=predictor,
            model_name=args.model_name, nonlinear_backend=nb,
            seq_len=args.seq_len, max_new_tokens=args.max_new_tokens,
            max_records=args.max_records)
    finally:
        if predictor is not None and hasattr(predictor, "close"):
            try:
                predictor.close()
            except Exception:                                # noqa: BLE001
                pass

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_csv:
        pc = Path(args.output_csv)
        pc.parent.mkdir(parents=True, exist_ok=True)
        pc.write_text(render_benchmark_csv(report), encoding="utf-8")
    if args.output_md:
        pm = Path(args.output_md)
        pm.parent.mkdir(parents=True, exist_ok=True)
        pm.write_text(render_benchmark_md(report), encoding="utf-8")

    print("=== generation preservation benchmark (%s) ===" % report["backend"])
    print("nonlinear_backend=%s num_examples=%d token_ids_available=%s"
          % (report["nonlinear_backend"], report["num_examples"],
             report["token_ids_available"]))
    print("dry_run=%s paper_ready=%s audit_passed=%s latency_s_mean=%s"
          % (report["dry_run"], report["paper_ready"], report["audit_passed"],
             report["latency_s_mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
