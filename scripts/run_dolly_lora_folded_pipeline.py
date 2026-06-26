"""Wrapper: fold a Dolly-trained raw HF LoRA adapter + run the E6 remote probe.

Does NOT reimplement the protocol. It checks the raw HF adapter exists, then calls
the existing ``scripts/run_e6_lora_real_h800_pipeline.py`` with ``--lora-mode hf``
(build folded LoRA package from the adapter -> verify no raw A/B / optimizer /
training-data / mask secrets -> local probe -> remote folded LoRA decode), and
summarizes the E6 JSON into ``outputs/lora_dolly/``.

Example::

    python scripts/run_dolly_lora_folded_pipeline.py \\
        --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \\
        --raw-lora-adapter-path /root/autodl-tmp/privacy_llm_packages/qwen7b_lora_dolly_r16 \\
        --lora-rank 16 --start-worker true --seq-len 1024 --max-new-tokens 16 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_BASE_PKG = "/root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_current_seq1024"
_EMBED = "/root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current"


def _summarize(e6, *, adapter, output_pkg, e6_json):
    """Project the E6 report into a compact folded-pipeline summary (security +
    correctness + cost), with safe defaults if a field is missing."""
    def g(k, d=None):
        return e6.get(k, d) if isinstance(e6, dict) else d
    return {
        "stage": "dolly_lora_folded_pipeline",
        "raw_lora_adapter_path": adapter,
        "output_lora_package": output_pkg,
        "e6_report_json": e6_json,
        "lora_package_built": bool(g("lora_package_built", False)),
        "lora_package_valid": bool(g("lora_package_valid", False)),
        # security (must all be safe)
        "contains_raw_lora": bool(g("contains_raw_lora", True)),
        "contains_optimizer_state": bool(g("contains_optimizer_state", True)),
        "contains_training_data": bool(g("contains_training_data", True)),
        "contains_mask_secrets": bool(g("contains_mask_secrets", True)),
        "worker_has_raw_lora": bool(g("worker_has_raw_lora", True)),
        "worker_has_mask_secrets": bool(g("worker_has_mask_secrets", True)),
        "tee_used_on_gpu": bool(g("tee_used_on_gpu", False)),
        "gpu_visible_plaintext_fields": g("gpu_visible_plaintext_fields", []),
        "leaked_secret_fields": g("leaked_secret_fields", []),
        "audit_passed": g("audit_passed"),
        # correctness
        "local_allclose": g("local_allclose"),
        "local_max_abs_error": g("local_max_abs_error"),
        "tokens_exact_match": g("tokens_exact_match"),
        "token_match_rate": g("token_match_rate"),
        "lora_differs_from_no_lora": g("lora_differs_from_no_lora"),
        # cost
        "latency_s": g("latency_s"),
        "trusted_bytes": g("trusted_bytes"),
        "gpu_bytes": g("gpu_bytes"),
        "boundary_calls": g("boundary_calls"),
        "peak_gpu_memory_mb": g("peak_gpu_memory_mb"),
        "paper_ready": bool(g("lora_package_valid", False)
                            and not g("contains_raw_lora", True)
                            and not g("worker_has_raw_lora", True)),
    }


def _md(s):
    keys = ["lora_package_valid", "contains_raw_lora", "contains_optimizer_state",
            "contains_training_data", "contains_mask_secrets", "worker_has_raw_lora",
            "worker_has_mask_secrets", "tee_used_on_gpu", "leaked_secret_fields",
            "audit_passed", "local_allclose", "local_max_abs_error",
            "tokens_exact_match", "token_match_rate", "latency_s", "trusted_bytes",
            "gpu_bytes", "boundary_calls", "peak_gpu_memory_mb", "paper_ready"]
    L = ["# Dolly LoRA folded pipeline summary", "",
         "- raw_lora_adapter_path: `%s`" % s["raw_lora_adapter_path"],
         "- output_lora_package: `%s`" % s["output_lora_package"], "",
         "| field | value |", "|---|---|"]
    L += ["| %s | %s |" % (k, s.get(k)) for k in keys]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--raw-lora-adapter-path", required=True)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--target-modules",
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--base-folded-package-path", default=_BASE_PKG)
    ap.add_argument("--embedding-artifact-path", default=_EMBED)
    ap.add_argument("--output-lora-package", default=None)
    ap.add_argument("--gpu-worker-url", default="http://127.0.0.1:18083")
    ap.add_argument("--start-worker", default="true")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nonlinear-backend", default="current")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="outputs/lora_dolly")
    args = ap.parse_args()

    adapter = Path(args.raw_lora_adapter_path)
    if not adapter.exists():
        print("ERROR: raw HF adapter not found: %s" % adapter, file=sys.stderr)
        return 3

    out_pkg = (args.output_lora_package
               or "/root/autodl-tmp/privacy_llm_packages/qwen7b_lora_folded_dolly_r%d"
               % int(args.lora_rank))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    e6_json = out_dir / "dolly_lora_e6.json"
    e6_md = out_dir / "dolly_lora_e6.md"

    argv = [sys.executable,
            str(REPO_ROOT / "scripts/run_e6_lora_real_h800_pipeline.py"),
            "--lora-mode", "hf",
            "--raw-lora-adapter-path", str(adapter),
            "--adapter-format", "hf_peft",
            "--base-folded-package-path", args.base_folded_package_path,
            "--embedding-artifact-path", args.embedding_artifact_path,
            "--output-lora-package", out_pkg,
            "--lora-rank", str(args.lora_rank),
            "--lora-alpha", str(args.lora_alpha),
            "--target-modules", args.target_modules,
            "--gpu-worker-url", args.gpu_worker_url,
            "--start-worker", str(args.start_worker),
            "--seq-len", str(args.seq_len),
            "--max-new-tokens", str(args.max_new_tokens),
            "--dtype", args.dtype, "--device", args.device,
            "--nonlinear-backend", args.nonlinear_backend,
            "--output-json", str(e6_json), "--output-md", str(e6_md)]
    if args.model_path:
        argv += ["--model-path", args.model_path]
    if args.dry_run:
        argv += ["--dry-run"]

    print("=== invoking E6 LoRA pipeline ===\n%s" % " ".join(argv))
    rc = subprocess.call(argv)
    e6 = {}
    if e6_json.exists():
        try:
            e6 = json.loads(e6_json.read_text())
        except Exception:                                    # noqa: BLE001
            e6 = {}
    summary = _summarize(e6, adapter=str(adapter), output_pkg=out_pkg,
                         e6_json=str(e6_json))
    summary["e6_return_code"] = rc

    sj = out_dir / "dolly_lora_folded_pipeline_summary.json"
    sm = out_dir / "dolly_lora_folded_pipeline_summary.md"
    sj.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    sm.write_text(_md(summary), encoding="utf-8")
    print("=== folded pipeline summary ===")
    print("lora_package_valid=%s contains_raw_lora=%s worker_has_raw_lora=%s "
          "audit_passed=%s" % (summary["lora_package_valid"],
                               summary["contains_raw_lora"],
                               summary["worker_has_raw_lora"],
                               summary["audit_passed"]))
    print("summary -> %s" % sj)
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
