"""Remote HTTP private folded-LoRA decode probe.

Drives the validated remote folded-package decode path (reused, not duplicated)
against a worker that loaded BOTH the base folded package and a private
folded-LoRA package. The TDX/boundary side runs in lite mode (boundary embedding
artifact only -- no full Qwen checkpoint, no 26GB package, no raw LoRA);
correctness is checked against expected token ids from a prior trusted local
reference (``--expected-token-ids`` / ``--expected-token-ids-file`` /
``--input-ids-file``). The worker merges the folded LoRA over masked activations
and never holds raw A/B or masks.

Start the worker (separately) with both packages::

    python scripts/run_tee_gpu_protocol_demo.py --mode gpu_worker_server \\
        --gpu-backend qwen7b_folded_package \\
        --folded-package-path /root/.../qwen7b_folded_full \\
        --folded-lora-package-path /root/.../qwen7b_lora_folded \\
        --device cuda --dtype bfloat16 --audit true

Then this probe (TDX-lite boundary client)::

    python scripts/run_qwen7b_lora_folded_remote_decode_probe.py \\
        --gpu-worker-url http://127.0.0.1:18082 \\
        --embedding-path /root/.../qwen7b_boundary_artifact_cuda \\
        --input-ids-file outputs/qwen7b_lora_folded_local_probe.json \\
        --expected-token-ids-file outputs/qwen7b_lora_folded_local_probe.json \\
        --max-new-tokens 4 --seq-len 128 --dtype bfloat16 --device cpu \\
        --audit true --output-json outputs/qwen7b_lora_folded_remote_decode_probe.json
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


def _load_demo():
    path = REPO_ROOT / "scripts" / "run_tee_gpu_protocol_demo.py"
    spec = importlib.util.spec_from_file_location("tee_gpu_protocol_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ids_from_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("package_token_ids", "reference_token_ids", "expected_token_ids"):
        v = data.get(k) if isinstance(data, dict) else None
        if v:
            return [int(x) for x in (v[0] if isinstance(v[0], list) else v)]
    raise SystemExit("no token ids in %s" % path)


def _bool(s):
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-worker-url", required=True)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--boundary-backend", default="process")
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--model-path", default=None,
                    help="trusted-reference mode (full model on boundary); "
                         "omit for TDX-lite")
    ap.add_argument("--folded-package-path", default=None,
                    help="only needed for non-lite full-reference mode")
    ap.add_argument("--adapter-path", default=None,
                    help="raw LoRA for trusted-reference mode (NOT lite)")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--input-ids", default=None)
    ap.add_argument("--input-ids-file", default=None)
    ap.add_argument("--tokenizer-path", default=None)
    ap.add_argument("--expected-token-ids", default=None)
    ap.add_argument("--expected-token-ids-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--attestation-evidence", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--write-runtime-manifest", default=None,
                    help="write the trusted-boundary manifest JSON to this path")
    ap.add_argument("--output-json",
                    default="outputs/qwen7b_lora_folded_remote_decode_probe.json")
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    lite = bool(args.embedding_path and not args.model_path)
    expected_csv = args.expected_token_ids
    if expected_csv is None and args.expected_token_ids_file:
        expected_csv = ",".join(str(t) for t in
                                _ids_from_json(args.expected_token_ids_file))

    demo = _load_demo()
    ns = SimpleNamespace(
        model_path=args.model_path, model_name=args.model_name,
        prompt=args.prompt, use_chat_template="true",
        folded_package_path=args.folded_package_path,
        embedding_path=args.embedding_path,
        skip_reference="true" if lite else "false",
        boundary_lite="true" if lite else "false",
        expected_token_ids=expected_csv, input_ids=args.input_ids,
        input_ids_file=args.input_ids_file, tokenizer_path=args.tokenizer_path,
        seed=args.seed, dtype=args.dtype, device=args.device,
        dry_run=args.dry_run, gpu_worker_url=args.gpu_worker_url,
        boundary_backend=args.boundary_backend,
        seq_len=args.seq_len, max_new_tokens=args.max_new_tokens)

    report = demo.build_remote_folded_package_decode_report(ns, _bool(args.audit))
    report["stage"] = "qwen7b_lora_folded_remote_decode_probe"
    report["probe"] = "lora_remote_decode"

    # TDX attestation: only attach when evidence is supplied, so a non-attested
    # run makes NO attestation claim. Mirrors the demo's folded attested branch.
    attested_requested = bool(args.attestation_evidence)
    if attested_requested:
        if args.write_runtime_manifest:
            md = demo.boundary_manifest_metadata(
                report.get("boundary_backend", "process"),
                report.get("gpu_backend", "qwen7b_folded_package"),
                args.expected_mr_td)
            demo.write_runtime_manifest(args.write_runtime_manifest, metadata=md)
        demo.attach_attestation(
            report, evidence=args.attestation_evidence,
            expected_mr_td=args.expected_mr_td,
            manifest_path=args.write_runtime_manifest)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        demo._write_remote_folded_md(Path(args.output_md), report)

    print("=== remote folded-LoRA decode probe ===")
    print("lora_enabled=%s folded_lora_loaded=%s folded_lora_valid=%s "
          "worker_has_raw_lora=%s" % (report.get("lora_enabled"),
                                      report.get("folded_lora_loaded"),
                                      report.get("folded_lora_valid"),
                                      report.get("worker_has_raw_lora")))
    print("package_backed_prefill=%s package_backed_decode=%s max_new_tokens=%s"
          % (report["package_backed_prefill"], report["package_backed_decode"],
             report["max_new_tokens"]))
    print("reference_token_ids=%s" % report["reference_token_ids"])
    print("package_token_ids  =%s" % report["package_token_ids"])
    print("tokens_exact_match=%s token_match_rate=%s"
          % (report["tokens_exact_match"], report["token_match_rate"]))
    print("worker_has_mask_secrets=%s tee_used_on_gpu=%s "
          "gpu_visible_plaintext_fields=%s leaked_secret_fields=%s"
          % (report["worker_has_mask_secrets"], report["tee_used_on_gpu"],
             report["gpu_visible_plaintext_fields"] or "[]",
             report["leaked_secret_fields"] or "[]"))
    if attested_requested:
        print("attestation: boundary_tee_type=%s boundary_attested=%s "
              "runtime_hash_bound=%s mr_td=%s"
              % (report.get("boundary_tee_type"), report.get("boundary_attested"),
                 report.get("runtime_hash_bound"), report.get("mr_td")))
        if report.get("binding_mismatch_reason"):
            print("binding_mismatch_reason: %s"
                  % report.get("binding_mismatch_reason"))

    ok = (report.get("lora_enabled") and report.get("folded_lora_loaded")
          and (report["tokens_exact_match"] is not False)
          and not report.get("worker_has_raw_lora")
          and not report["worker_has_mask_secrets"]
          and not report["tee_used_on_gpu"]
          and not report["leaked_secret_fields"]
          and not report["gpu_visible_plaintext_fields"]
          and report["audit_passed"] is not False)
    # when attestation evidence is supplied, the binding MUST verify
    if attested_requested:
        ok = bool(ok and report.get("boundary_attested") is True
                  and report.get("runtime_hash_bound") is True)
    print("\nREMOTE FOLDED-LoRA DECODE %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
