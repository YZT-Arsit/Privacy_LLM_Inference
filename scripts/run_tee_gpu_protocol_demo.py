"""TEE-boundary <-> untrusted-GPU-worker protocol demo (Stage 8.5).

Runs a full masked decode through the message protocol: a trusted boundary
(Stage 8.3 runtime, optionally process-isolated) drives an *untrusted* GPU
worker (separate process) that sees only masked embeddings + public metadata +
the folded LM head and returns only masked logits. The trusted side tokenizes,
masks, recovers logits, samples greedily, and remasks each new token. The model
is NEVER placed in the TEE (``tee_used_on_gpu=False``).

The security audit then proves, against the *exact* recorded GPU traffic, that
no raw prompt / input_ids / generated tokens / recovered logits / mask secret
crossed to the GPU, and that a wrong mask cannot recover the plaintext.

numpy only for ``--gpu-backend mock`` (the default). ``--gpu-backend qwen7b``
plugs the same protocol into the real masked pipeline on the GPU server.

Example::

    python scripts/run_tee_gpu_protocol_demo.py --mode local_two_process \\
        --boundary-backend process --gpu-backend mock --max-new-tokens 8 \\
        --audit true --output-json outputs/tee_gpu_protocol.json \\
        --output-md outputs/tee_gpu_protocol.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataclasses import asdict  # noqa: E402

from pllo.protocol.attestation import attest_boundary  # noqa: E402
from pllo.protocol.gpu_worker import LocalGpuWorker  # noqa: E402
from pllo.protocol.orchestrator import run_protocol  # noqa: E402
from pllo.protocol.security_audit import (  # noqa: E402
    assert_no_gpu_visible_plaintext,
    assert_no_mask_secret_leak,
    assert_wrong_mask_recovery_fails,
)
from pllo.protocol.tee_gpu_messages import (  # noqa: E402
    BoundaryInitRequest,
    ProtocolTrace,
)


def _bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def boundary_runtime_components(boundary_backend: str, gpu_backend: str,
                                hidden_size: int, vocab_size: int) -> dict:
    """Public identity bound into the TDX quote's report_data (no secrets)."""
    return {
        "component": "pllo-tee-boundary",
        "stage": "8.5",
        "version": "1",
        "boundary_backend": boundary_backend,
        "gpu_backend": gpu_backend,
        "mask_mode": "signed_permutation",
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
    }


def attach_attestation(report: dict, *, hidden_size: int, vocab_size: int,
                       evidence: str | None, expected_mr_td: str | None) -> None:
    """Attest the trusted boundary and fold the result into ``report``."""
    components = boundary_runtime_components(
        report["boundary_backend"], report["gpu_backend"], hidden_size,
        vocab_size)
    ev = attest_boundary(components, evidence=evidence,
                         expected_mr_td=expected_mr_td)
    report["attestation"] = asdict(ev)
    report["boundary_tee_type"] = ev.tee_type
    report["boundary_attested"] = ev.verified
    report["runtime_hash"] = ev.runtime_hash_hex
    report["runtime_hash_bound"] = ev.runtime_hash_bound
    report["mr_td"] = ev.mr_td


def build_report(prompt: str, boundary_backend: str, gpu_backend: str,
                 max_new_tokens: int, run_audit: bool, **run_kwargs) -> dict:
    out = run_protocol(prompt, boundary_backend=boundary_backend,
                       gpu_backend=gpu_backend, max_new_tokens=max_new_tokens,
                       **run_kwargs)
    trace = out["trace"]

    plaintext_fields: list[str] = []
    secret_fields: list[str] = []
    wrong_mask: dict = {}
    audit_passed = None
    if run_audit:
        plaintext_fields = assert_no_gpu_visible_plaintext(
            trace, raw_prompt=prompt, input_ids=out["input_ids"],
            generated_token_ids=out["generated_token_ids"],
            recovered_logits=None, raise_on_fail=False)
        secret_fields = assert_no_mask_secret_leak(
            trace, out["handles"], raise_on_fail=False)
        wrong_mask = assert_wrong_mask_recovery_fails(
            out["masked_logits_first"], out["handles"], out["wrong_handles"],
            out["plaintext_logits_first"], raise_on_fail=False)
        audit_passed = bool(
            not plaintext_fields and not secret_fields
            and not wrong_mask.get("findings")
            and not trace.tee_used_on_gpu)

    report = {
        "stage": "tee_gpu_protocol_demo",
        "mode": "local_two_process",
        "tee_used": False,                     # nothing of ours runs in a TEE here
        "tee_used_on_gpu": trace.tee_used_on_gpu,
        "boundary_backend": boundary_backend,
        "gpu_backend": gpu_backend,
        "max_new_tokens": max_new_tokens,
        "boundary_calls": trace.boundary_calls,
        "gpu_calls": trace.gpu_calls,
        "trusted_bytes": trace.trusted_bytes,
        "gpu_bytes": trace.gpu_bytes,
        "gpu_inbound_message_count": len(trace.gpu_inbound_messages),
        "gpu_outbound_message_count": len(trace.gpu_outbound_messages),
        "gpu_visible_plaintext_fields": plaintext_fields,
        "leaked_secret_fields": secret_fields,
        "wrong_mask_control": wrong_mask,
        "recovered_tokens": out["recovered_tokens"],
        "tokens_match_plaintext_reference": out["tokens_match_reference"],
        "audit_performed": run_audit,
        "audit_passed": audit_passed,
    }
    return report


def build_qwen7b_probe_report(prompt: str, boundary_backend: str,
                              max_new_tokens: int, run_audit: bool,
                              model_path: str | None, device: str,
                              dtype: str) -> dict:
    """Init-only probe for the qwen7b GPU backend.

    The real masked prefill/decode runs on the GPU server (CUDA + checkpoint);
    locally we still exercise the protocol's init handshake in a separate
    process, confirm ``tee_used_on_gpu=False``, and audit the init traffic (it
    carries only public metadata -- no folded head, no plaintext, no secret)."""
    trace = ProtocolTrace(boundary_backend=boundary_backend,
                          gpu_backend="qwen7b", max_new_tokens=max_new_tokens,
                          tee_used_on_gpu=False)

    def _record(direction, method, msg):
        (trace.record_gpu_inbound if direction == "inbound"
         else trace.record_gpu_outbound)(msg)

    worker = LocalGpuWorker(
        "qwen7b", {"model_path": model_path, "device": device, "dtype": dtype},
        recorder=_record)
    try:
        init_req = BoundaryInitRequest(
            session_id="sess-0", hidden_size=3584, vocab_size=152064,
            num_layers=28, dtype=dtype, gpu_backend="qwen7b",
            folded_lm_head=None,
            public_metadata={"model": "qwen2.5-7b", "note": "decode on server"})
        init_resp = worker.init(init_req)
        trace.tee_used_on_gpu = bool(init_resp.tee_used_on_gpu)
        # describe info is built locally (it is control metadata, not a boundary
        # message, and must not ride the audited GPU channel).
        describe = {"backend": "qwen7b", "tee_used": False,
                    "model_path": model_path, "device": device, "dtype": dtype}
    finally:
        worker.close()

    plaintext_fields, secret_fields, audit_passed = [], [], None
    if run_audit:
        plaintext_fields = assert_no_gpu_visible_plaintext(
            trace, raw_prompt=prompt, raise_on_fail=False)
        secret_fields = assert_no_mask_secret_leak(trace, None,
                                                   raise_on_fail=False)
        audit_passed = bool(not plaintext_fields and not secret_fields
                            and not trace.tee_used_on_gpu)

    return {
        "stage": "tee_gpu_protocol_demo",
        "mode": "local_two_process",
        "tee_used": False,
        "tee_used_on_gpu": trace.tee_used_on_gpu,
        "boundary_backend": boundary_backend,
        "gpu_backend": "qwen7b",
        "qwen7b_probe_only": True,
        "qwen7b_describe": describe,
        "max_new_tokens": max_new_tokens,
        "boundary_calls": trace.boundary_calls,
        "gpu_calls": trace.gpu_calls,
        "trusted_bytes": trace.trusted_bytes,
        "gpu_bytes": trace.gpu_bytes,
        "gpu_inbound_message_count": len(trace.gpu_inbound_messages),
        "gpu_outbound_message_count": len(trace.gpu_outbound_messages),
        "gpu_visible_plaintext_fields": plaintext_fields,
        "leaked_secret_fields": secret_fields,
        "wrong_mask_control": {},
        "recovered_tokens": [],
        "tokens_match_plaintext_reference": None,
        "audit_performed": run_audit,
        "audit_passed": audit_passed,
        "note": "qwen7b masked prefill/decode runs on the GPU server; this is "
                "an init-only protocol/audit probe.",
    }


def _write_md(path: Path, r: dict) -> None:
    lines = [
        f"# TEE ↔ GPU protocol demo ({r['mode']})", "",
        f"- boundary_backend: `{r['boundary_backend']}`",
        f"- gpu_backend: `{r['gpu_backend']}`",
        f"- max_new_tokens: {r['max_new_tokens']}",
        f"- **tee_used_on_gpu: {r['tee_used_on_gpu']}**",
        f"- boundary_calls: `{r['boundary_calls']}`",
        f"- gpu_calls: `{r['gpu_calls']}`",
        f"- trusted_bytes: {r['trusted_bytes']:,}",
        f"- gpu_bytes: {r['gpu_bytes']:,}",
        f"- tokens_match_plaintext_reference: {r['tokens_match_plaintext_reference']}",
        "", "## Security audit", "",
        f"- audit_performed: {r['audit_performed']}",
        f"- **audit_passed: {r['audit_passed']}**",
        f"- gpu_visible_plaintext_fields: {r['gpu_visible_plaintext_fields'] or 'none'}",
        f"- leaked_secret_fields: {r['leaked_secret_fields'] or 'none'}",
        f"- wrong_mask_control: `{r['wrong_mask_control']}`",
    ]
    att = r.get("attestation")
    if att is not None:
        lines += [
            "", "## Boundary attestation (TDX)", "",
            f"- boundary_tee_type: `{r['boundary_tee_type']}`",
            f"- **boundary_attested: {r['boundary_attested']}**",
            f"- quote_status: `{att['quote_status']}`",
            f"- runtime_hash: `{r['runtime_hash']}`",
            f"- runtime_hash_bound: {r['runtime_hash_bound']}",
            f"- mr_td: `{r['mr_td']}`",
            f"- debug: {att['debug']}",
            f"- jwt_present: {att['jwt_present']} (parts={att['jwt_parts']})",
            f"- mr_td_match: {att['mr_td_match']}",
        ]
    lines += [
        "",
        "_The GPU worker received only masked embeddings + public metadata + "
        "the folded LM head, and returned only masked logits. The model is "
        "never placed in the TEE._",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="local_two_process",
                    choices=["local_two_process"])
    ap.add_argument("--boundary-backend", default="process",
                    choices=["process", "simulated"])
    ap.add_argument("--gpu-backend", default="mock",
                    choices=["mock", "qwen7b"])
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--model-path", default=None,
                    help="qwen7b checkpoint (GPU server); probe-only locally")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--attestation-evidence", default=None,
                    help="path to TDX attestation evidence JSON from the VM "
                         "(tee/mr_td/report_data/jwt); verifies the binding")
    ap.add_argument("--expected-mr-td", default=None,
                    help="expected mr_td to match against the evidence")
    ap.add_argument("--output-json", default="outputs/tee_gpu_protocol.json")
    ap.add_argument("--output-md", default="outputs/tee_gpu_protocol.md")
    args = ap.parse_args()

    if args.gpu_backend == "qwen7b":
        print("NOTE: --gpu-backend qwen7b runs masked prefill/decode on the GPU "
              "server (CUDA + checkpoint). Locally this is an init-only "
              "protocol/audit probe; --gpu-backend mock runs end-to-end.")
        report = build_qwen7b_probe_report(
            args.prompt, args.boundary_backend, args.max_new_tokens,
            _bool(args.audit), args.model_path, args.device, args.dtype)
    else:
        report = build_report(
            args.prompt, args.boundary_backend, args.gpu_backend,
            args.max_new_tokens, _bool(args.audit),
            hidden_size=args.hidden_size, vocab_size=args.vocab_size,
            seq_len=args.seq_len, seed=args.seed)

    attach_attestation(report, hidden_size=args.hidden_size,
                       vocab_size=args.vocab_size,
                       evidence=args.attestation_evidence,
                       expected_mr_td=args.expected_mr_td)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_md(p, report)

    print("=== TEE ↔ GPU protocol demo ===")
    print(f"boundary_backend={report['boundary_backend']} "
          f"gpu_backend={report['gpu_backend']} "
          f"max_new_tokens={report['max_new_tokens']}")
    print(f"tee_used_on_gpu={report['tee_used_on_gpu']}")
    print(f"boundary_calls={report['boundary_calls']}")
    print(f"gpu_calls={report['gpu_calls']}")
    print(f"trusted_bytes={report['trusted_bytes']} "
          f"gpu_bytes={report['gpu_bytes']}")
    print(f"tokens_match_plaintext_reference="
          f"{report['tokens_match_plaintext_reference']}")
    print(f"gpu_visible_plaintext_fields="
          f"{report['gpu_visible_plaintext_fields'] or 'none'}")
    print(f"leaked_secret_fields={report['leaked_secret_fields'] or 'none'}")
    print(f"wrong_mask_control={report['wrong_mask_control']}")
    print(f"audit_passed={report['audit_passed']}")
    att = report["attestation"]
    print(f"boundary_tee_type={report['boundary_tee_type']} "
          f"boundary_attested={report['boundary_attested']} "
          f"quote_status={att['quote_status']}")
    print(f"runtime_hash={report['runtime_hash'][:32]}... "
          f"runtime_hash_bound={report['runtime_hash_bound']} "
          f"mr_td={report['mr_td']}")

    if report["audit_performed"] and not report["audit_passed"]:
        print("\nDEMO FAILED (audit)")
        return 1
    # qwen7b is an init-only probe (tokens_match is None) -> success on a clean
    # audit; mock must also match the plaintext reference.
    ok = (report["tokens_match_plaintext_reference"]
          if report["tokens_match_plaintext_reference"] is not None
          else (report["audit_passed"] is not False))
    print(f"\nDEMO {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
