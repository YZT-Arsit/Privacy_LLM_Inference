"""E3 runner: scaling sweep for remote package-backed decode.

Sweeps ``--max-new-tokens-list`` (and optionally ``--seq-len-list``) by driving
the ALREADY-WORKING remote folded-package decode path
(``run_tee_gpu_protocol_demo.build_remote_folded_package_decode_report``) once per
point -- no protocol logic is duplicated here. Supports both deployment modes:

* H800 full-reference: ``--model-path`` + ``--folded-package-path`` (in-process
  folded reference);
* TDX-lite: ``--embedding-path --skip-reference`` with ``--input-ids-file`` and
  ``--expected-token-ids``/``--expected-token-ids-file`` (no full model / no 26GB
  package on the boundary).

Emits per-row JSON + CSV + a Markdown summary (pass/fail, latency, bytes,
boundary-call, and security tables). No TDX attestation is claimed.

``--dry-run`` runs a tiny model + tiny package on CPU (never a paper result).

Example (TDX-lite)::

    python scripts/run_e3_remote_decode_scaling.py \\
        --gpu-worker-url http://127.0.0.1:18082 \\
        --gpu-backend qwen7b_folded_package --model-name Qwen2.5-7B-Instruct \\
        --embedding-path /root/.../qwen7b_boundary_artifact_cuda \\
        --input-ids-file outputs/qwen7b_folded_remote_decode_reference_for_tdx.json \\
        --expected-token-ids-file outputs/qwen7b_folded_remote_decode_reference_for_tdx.json \\
        --max-new-tokens-list 1,4,8,16 --seq-len 128 --dtype bfloat16 \\
        --device cpu --audit true \\
        --output-json outputs/e3_remote_decode_scaling.json \\
        --output-csv outputs/e3_remote_decode_scaling.csv \\
        --output-md  outputs/e3_remote_decode_scaling.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.e3_remote_decode_scaling import (  # noqa: E402
    build_e3_summary,
    render_e3_csv,
    render_e3_md,
    run_e3_scaling,
)
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)


def _load_demo():
    """Load the protocol demo module (source of the decode report builder)."""
    path = REPO_ROOT / "scripts" / "run_tee_gpu_protocol_demo.py"
    spec = importlib.util.spec_from_file_location("tee_gpu_protocol_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _csv_ints(s):
    if not s:
        return []
    return [int(p) for p in str(s).replace(" ", "").split(",") if p != ""]


def _ids_from_json(path):
    """Pull token ids from a prior decode report JSON: prefer package_token_ids,
    then reference_token_ids, then expected_token_ids, then a bare input_ids."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for k in ("package_token_ids", "reference_token_ids",
                  "expected_token_ids"):
            v = data.get(k)
            if v:
                return [int(x) for x in (v[0] if v and isinstance(v[0], list)
                                         else v)]
    raise SystemExit("could not find token ids in %s "
                     "(expected package_token_ids/reference_token_ids/"
                     "expected_token_ids)" % path)


def _bool(s):
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _base_args(args):
    """The fields (besides seq_len/max_new_tokens/expected) the demo reads."""
    return dict(
        model_path=args.model_path, model_name=args.model_name,
        prompt=args.prompt, use_chat_template="true",
        folded_package_path=args.folded_package_path,
        embedding_path=args.embedding_path,
        skip_reference=args.skip_reference, boundary_lite=args.boundary_lite,
        input_ids=args.input_ids,
        input_ids_file=args.input_ids_file, tokenizer_path=args.tokenizer_path,
        seed=args.seed, dtype=args.dtype, device=args.device,
        dry_run=args.dry_run, gpu_worker_url=args.gpu_worker_url,
        boundary_backend=args.boundary_backend,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-worker-url", required=True)
    ap.add_argument("--gpu-backend", default="qwen7b_folded_package")
    ap.add_argument("--boundary-backend", default="process")
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--folded-package-path", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--skip-reference", default="false")
    ap.add_argument("--boundary-lite", default="false")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--input-ids", default=None)
    ap.add_argument("--input-ids-file", default=None)
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--expected-token-ids", default=None)
    ap.add_argument("--expected-token-ids-file", default=None,
                    help="JSON to read expected token ids from (a prior decode "
                         "report); compared against each sweep point")
    ap.add_argument("--max-new-tokens-list", default="1,4,8,16")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seq-len-list", default=None,
                    help="optional comma list of seq_len values to also sweep")
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--output-json", default="outputs/e3_remote_decode_scaling.json")
    ap.add_argument("--output-csv", default="outputs/e3_remote_decode_scaling.csv")
    ap.add_argument("--output-md", default="outputs/e3_remote_decode_scaling.md")
    ap.add_argument("--nonlinear-backend", default="current",
                    help="nonlinear design (current|trusted_shortcut, aliases ok)")
    args = ap.parse_args()
    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)

    if args.gpu_backend != "qwen7b_folded_package":
        ap.error("E3 currently sweeps --gpu-backend qwen7b_folded_package")

    max_new_tokens_list = _csv_ints(args.max_new_tokens_list)
    if not max_new_tokens_list:
        ap.error("--max-new-tokens-list must contain at least one value")
    seq_lens = (_csv_ints(args.seq_len_list) if args.seq_len_list
                else [int(args.seq_len)])

    expected_ids = _csv_ints(args.expected_token_ids) if args.expected_token_ids \
        else (_ids_from_json(args.expected_token_ids_file)
              if args.expected_token_ids_file else None)

    demo = _load_demo()
    base = _base_args(args)
    run_audit = _bool(args.audit)

    def decode_fn(*, seq_len, max_new_tokens):
        # greedy decode is a deterministic prefix: an n-token run must match the
        # FIRST n expected tokens from a (>=n) reference run.
        expected_csv = (",".join(str(t) for t in expected_ids[:max_new_tokens])
                        if expected_ids else None)
        ns = SimpleNamespace(seq_len=int(seq_len),
                             max_new_tokens=int(max_new_tokens),
                             expected_token_ids=expected_csv, **base)
        return demo.build_remote_folded_package_decode_report(ns, run_audit)

    rows = run_e3_scaling(decode_fn, seq_lens=seq_lens,
                          max_new_tokens_list=max_new_tokens_list)
    summary = build_e3_summary(rows)
    meta = {
        "experiment": "E3", "stage": "remote_package_decode_scaling",
        "gpu_backend": args.gpu_backend, "gpu_worker_url": args.gpu_worker_url,
        "boundary_mode": rows[0]["boundary_mode"] if rows else None,
        "model_name": args.model_name, "dtype": args.dtype, "device": args.device,
        "seq_lens": seq_lens, "max_new_tokens_list": max_new_tokens_list,
        "dry_run": bool(args.dry_run),
    }
    report = {"experiment": "E3", "stage": "remote_package_decode_scaling",
              "meta": meta, "rows": rows, "summary": summary}
    report.update(nonlinear_design_report_fields(args.nonlinear_backend))

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_csv:
        p = Path(args.output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_e3_csv(rows), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_e3_md(rows, summary, meta), encoding="utf-8")

    print("=== E3 remote package-backed decode scaling ===")
    print("boundary_mode=%s gpu_backend=%s rows=%d"
          % (meta["boundary_mode"], args.gpu_backend, len(rows)))
    for p in summary["passfail"]:
        print("  seq_len=%s max_new_tokens=%s pass=%s tokens_exact_match=%s "
              "token_match_rate=%s" % (p["seq_len"], p["max_new_tokens"],
                                       p["pass"], p["tokens_exact_match"],
                                       p["token_match_rate"]))
    print("all_pass=%s all_security_ok=%s (%d/%d)"
          % (summary["all_pass"], summary["all_security_ok"],
             summary["num_pass"], summary["num_rows"]))
    print("\nE3 %s" % ("PASSED" if summary["all_pass"] else "FAILED"))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
