"""Generation utility benchmark -- ONE backend (JSON/CSV/MD).

Runs deterministic greedy generation over a normalized generation JSONL
({id, dataset_name, task_type, prompt, reference}) with a single backend and
computes task-aware metrics:

* generation_exact (GSM8K)   -> numeric_exact_match, extracted_number,
  output_length_tokens, latency_s.
* summarization (CNN/DM,XSum)-> rouge1/2/L (+ rouge_unavailable), length, latency.
* open_ended (custom)        -> exact_text_match + normalized_edit_similarity
  vs reference (when present), length, latency.

Backends (CURRENT design only; trusted_shortcut refused):
* ``plaintext_local`` -- real Qwen checkpoint (greedy, attention_mask passed).
* ``folded_remote``   -- trusted lite boundary + remote folded GPU worker.

Honest labeling: no model/worker -> deterministic stub, ``dry_run=True,
paper_ready=False``. ``--require-real`` forbids the stub. No downloads, no LLM
judge, no subjective quality scoring. Run once per backend, then compare with
``scripts/run_generation_pairwise_preservation.py``.

Example (H800, GSM8K-128)::

    python scripts/run_generation_utility_benchmark.py \\
        --dataset-jsonl $CONV/gsm8k_gen.jsonl --backend plaintext_local \\
        --model-path $MODEL --require-real --seq-len 1024 --max-new-tokens 128 \\
        --output-json out/gsm8k128_plaintext.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.generation_metrics import (  # noqa: E402
    SUPPORTED_BACKENDS,
    assert_current_only,
    load_examples,
    render_benchmark_csv,
    render_benchmark_md,
    run_generation_utility_benchmark,
)
from pllo.benchmarks.real_predictors import (  # noqa: E402
    RealBackendUnavailable,
    build_predictor,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-jsonl", required=True)
    ap.add_argument("--backend", required=True, choices=list(SUPPORTED_BACKENDS))
    ap.add_argument("--nonlinear-backend", default="current",
                    help="CURRENT only (trusted_shortcut refused here)")
    ap.add_argument("--task-type", default=None,
                    help="override task_type (default: read from examples)")
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=200)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--require-real", action="store_true", default=False)
    # ---- reuse the validated folded_remote boundary-client path (parity with
    # run_ifeval_generation.py): chat template + generation-config alignment +
    # EOS stop. folded_remote NEVER loads the full 7B on the trusted side. ----
    ap.add_argument("--use-chat-template", action="store_true", default=False,
                    help="apply the Qwen chat template trusted-side (same shared "
                    "formatter the folded path validated for IFEval/GSM8K)")
    ap.add_argument("--align-generation-config", action="store_true",
                    default=False, help="apply the plaintext generation-config "
                    "logit processors (repetition_penalty) TRUSTED-SIDE")
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="explicit repetition_penalty for --align-generation-config "
                    "(else read from the model generation_config.json)")
    ap.add_argument("--disable-eos-stop", action="store_true", default=False,
                    help="keep fixed-length decode (default is EOS stopping, "
                    "aligned with the plaintext model.generate stop condition)")
    # ---- TDX boundary-client provenance (folded_remote only) ----
    ap.add_argument("--tdx-boundary-client", action="store_true", default=False,
                    help="run folded_remote as a TDX boundary client: assert the "
                    "trusted side loads NO full 7B weights (only tokenizer/config "
                    "+ embedding artifact); the H800 worker does GPU compute")
    ap.add_argument("--trusted-runtime", default="process",
                    help="trusted runtime label for the report (process / "
                    "tdx_guest / real_tdx)")
    ap.add_argument("--tee-mode", default=None,
                    help="explicit TEE mode label (else derived)")
    ap.add_argument("--h800-worker-ssh-alias", default=None,
                    help="SSH alias of the untrusted H800 GPU worker (provenance)")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

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

    # TDX boundary-client mode is folded_remote only (plaintext_local loads the
    # full 7B weights, which must NEVER happen inside the trusted TDX guest).
    if args.tdx_boundary_client and args.backend != "folded_remote":
        print("ERROR: --tdx-boundary-client requires --backend folded_remote "
              "(plaintext_local loads full model weights, forbidden in the TDX "
              "guest)", file=sys.stderr)
        return 3

    audit = str(args.audit).strip().lower() in {"1", "true", "yes", "y", "on"}
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
                embedding_path=args.embedding_path, seq_len=args.seq_len,
                max_new_tokens=args.max_new_tokens, dtype=args.dtype,
                device=args.device, audit=audit, nonlinear_backend=nb,
                align_generation_config=args.align_generation_config,
                repetition_penalty=args.repetition_penalty,
                stop_on_eos=(not args.disable_eos_stop),
                use_chat_template=bool(args.use_chat_template))
        except RealBackendUnavailable as exc:
            if args.require_real:
                print("ERROR: --require-real but real backend unavailable: %s"
                      % exc, file=sys.stderr)
                return 3
            predictor = None

    # The folded_remote boundary client must hold NO full model weights on the
    # trusted side (only tokenizer/config + embedding artifact). Prove it.
    full_weights_loaded = bool(predictor is not None
                               and hasattr(predictor, "_model"))
    if args.tdx_boundary_client and full_weights_loaded:
        print("ERROR: folded_remote boundary client unexpectedly loaded full "
              "model weights on the trusted side", file=sys.stderr)
        if predictor is not None and hasattr(predictor, "close"):
            predictor.close()
        return 3

    try:
        report = run_generation_utility_benchmark(
            examples, backend=args.backend, predictor=predictor,
            model_name=args.model_name, nonlinear_backend=nb,
            seq_len=args.seq_len, max_new_tokens=args.max_new_tokens,
            dataset_name=args.dataset_name, task_type=args.task_type,
            max_records=args.max_records)
    finally:
        if predictor is not None and hasattr(predictor, "close"):
            try:
                predictor.close()
            except Exception:                                # noqa: BLE001
                pass

    # ---- boundary-client provenance (parity with run_ifeval_generation.py) ----
    tdx_client = bool(args.tdx_boundary_client)
    is_tdx = bool(tdx_client or args.trusted_runtime in ("real_tdx", "tdx_guest"))
    report["trusted_runtime"] = "tdx_guest" if is_tdx else (
        args.trusted_runtime or "process")
    report["tee_mode"] = (args.tee_mode if args.tee_mode
                          else ("real_tdx" if is_tdx else "process_boundary"))
    report["tdx_boundary_client"] = tdx_client
    report["full_model_weights_loaded_in_trusted_runtime"] = full_weights_loaded
    report["h800_worker_url"] = args.gpu_worker_url
    report["h800_worker_ssh_alias"] = args.h800_worker_ssh_alias
    report["use_chat_template"] = bool(args.use_chat_template)
    report["align_generation_config"] = bool(args.align_generation_config)

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

    print("=== generation utility benchmark (%s / %s) ==="
          % (report["dataset_name"], report["backend"]))
    print("task_type=%s metric %s=%s num_examples=%d max_new_tokens=%d"
          % (report["task_type"], report["metric_name"], report["metric_value"],
             report["num_examples"], report["max_new_tokens"]))
    print("dry_run=%s paper_ready=%s audit_passed=%s latency_s_mean=%s"
          % (report["dry_run"], report["paper_ready"], report["audit_passed"],
             report["latency_s_mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
