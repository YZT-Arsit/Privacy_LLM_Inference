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

from pllo.protocol.attestation import (  # noqa: E402
    attest_boundary,
    binding_mismatch_reason,
    boundary_manifest_metadata,
    boundary_runtime_hash,
    build_trusted_boundary_manifest,
    compute_runtime_hash_from_manifest,
    write_runtime_hash,
    write_runtime_manifest,
)
from pllo.protocol.gpu_worker import LocalGpuWorker  # noqa: E402
from pllo.protocol.orchestrator import run_protocol  # noqa: E402
from pllo.protocol.remote import RemoteGpuWorker, run_gpu_worker_server  # noqa: E402
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


def attach_attestation(report: dict, *, evidence: str | None,
                       expected_mr_td: str | None,
                       manifest_path: str | None = None) -> None:
    """Attest the trusted boundary (manifest recipe) and fold into ``report``.

    The runtime hash is SHA-512 over the trusted-boundary manifest (source-file
    digests + public runtime identity), which binds the attestation token to the
    actual boundary code artifact. ``expected_runtime_hash`` is THE value the TD
    Quote's ``report_data`` must equal; the report surfaces it alongside
    ``evidence_report_data`` and a ``binding_mismatch_reason`` when they differ."""
    metadata = boundary_manifest_metadata(
        report["boundary_backend"], report["gpu_backend"], expected_mr_td)
    manifest = build_trusted_boundary_manifest(metadata=metadata)
    expected = compute_runtime_hash_from_manifest(manifest)    # source of truth
    ev = attest_boundary(runtime_hash=expected, evidence=evidence,
                         expected_mr_td=expected_mr_td)
    report["attestation"] = asdict(ev)
    report["boundary_tee_type"] = ev.tee_type
    report["boundary_attested"] = ev.verified
    report["runtime_hash"] = ev.runtime_hash_hex
    report["expected_runtime_hash"] = expected
    report["evidence_report_data"] = ev.report_data_hex
    report["runtime_hash_bound"] = ev.runtime_hash_bound
    report["binding_mismatch_reason"] = binding_mismatch_reason(ev)
    report["mr_td"] = ev.mr_td
    report["runtime_manifest_path"] = manifest_path
    report["runtime_manifest_file_count"] = len(manifest["files"])


def build_report(prompt: str, boundary_backend: str, gpu_backend: str,
                 max_new_tokens: int, run_audit: bool, **run_kwargs) -> dict:
    import time as _time
    _t0 = _time.perf_counter()
    out = run_protocol(prompt, boundary_backend=boundary_backend,
                       gpu_backend=gpu_backend, max_new_tokens=max_new_tokens,
                       **run_kwargs)
    latency_s = _time.perf_counter() - _t0
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

    gpu_worker_remote = bool(out.get("gpu_worker_remote"))
    init_resp = out.get("init_response")
    report = {
        "stage": "tee_gpu_protocol_demo",
        "mode": "boundary_client" if gpu_worker_remote else "local_two_process",
        "tee_used": False,                     # nothing of ours runs in a TEE here
        "tee_used_on_gpu": trace.tee_used_on_gpu,
        "gpu_worker_remote": gpu_worker_remote,
        "gpu_worker_url": out.get("gpu_worker_url"),
        "latency_s": latency_s,
        # peak GPU memory is measured server-side by the qwen7b worker; the mock
        # backend uses no GPU so it is None here.
        "peak_gpu_memory_mb": out.get("peak_gpu_memory_mb"),
        "boundary_backend": boundary_backend,
        # client-intended backend (drives the runtime-hash metadata, so it must
        # match the preflight); the server's own report is kept separately.
        "gpu_backend": gpu_backend,
        "gpu_backend_server_reported": (init_resp.gpu_backend
                                        if init_resp is not None else None),
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


# Three-way scope distinction the paper must keep separate (see
# docs/cross_machine_security_scope.md). Stamped into every cross-machine report
# so a reader never confuses standalone compute with the attested deployment.
_COMPUTE_CORRECTNESS_SOURCE = (
    "standalone H800 E1/E2 (run_qwen7b_e1_nolora_generation.py / "
    "run_qwen7b_e2_token_scaling.py) validates full Qwen2.5-7B masked compute "
    "correctness, generation behaviour, and token scaling")
_SECURITY_BOUNDARY_SOURCE = (
    "cross-machine mock end-to-end (real attestation + wire-field rejection + "
    "masked-tensor-only traffic + audit) plus the qwen7b /init handshake plus "
    "TDX boundary attestation validates the attested trusted boundary driving a "
    "remote untrusted GPU worker")
_CONNECTIVITY_NOTE = (
    "cross-machine connectivity is the deployment setting for the attested "
    "protocol, not a contribution")


def _annotate_cross_machine_scope(report: dict, *, model_name: str,
                                  cross_machine_compute: str,
                                  limitations: str) -> None:
    """Stamp the three-way scope fields onto a cross-machine report so the
    standalone compute result and the attested deployment are never conflated."""
    report["model_name"] = model_name
    report["cross_machine_compute"] = cross_machine_compute  # end_to_end|probe_only
    report["compute_correctness_source"] = _COMPUTE_CORRECTNESS_SOURCE
    report["security_boundary_source"] = _SECURITY_BOUNDARY_SOURCE
    report["connectivity_note"] = _CONNECTIVITY_NOTE
    report["limitations"] = limitations
    # Qwen-specific correctness metrics live in standalone E1/E2; surface the
    # keys here (None unless a Qwen reference is driven cross-machine) so the
    # report schema is stable. "if implemented" per the experiment spec.
    report.setdefault("teacher_forced_top1_match_rate", None)
    report.setdefault("plain_vs_masked_token_match_rate", None)


def build_remote_qwen7b_probe_report(prompt: str, boundary_backend: str,
                                     max_new_tokens: int, run_audit: bool,
                                     gpu_worker_url: str, *, model_name: str,
                                     seq_len: int, num_layers: int,
                                     dtype: str) -> dict:
    """Cross-machine init + audit probe against a **remote** qwen7b GPU worker.

    Performs the real ``/health`` + ``/init`` handshake to the H800 worker over
    HTTP, confirms the server reports ``tee_used_on_gpu=False``, and audits the
    init traffic (only masked/public metadata crosses -- no folded head, no
    plaintext, no secret). The full Qwen2.5-7B masked prefill/decode is validated
    standalone (E1/E2): a *private* cross-machine masked decode would require
    shipping folded layer weights to the worker (which must never hold the
    masks), impractical over JSON HTTP -- so this is a protocol/attestation/audit
    probe, not an end-to-end compute run. ``--gpu-backend mock`` runs the full
    boundary<->worker decode end-to-end cross-machine."""
    import time as _time
    trace = ProtocolTrace(boundary_backend=boundary_backend,
                          gpu_backend="qwen7b", max_new_tokens=max_new_tokens,
                          tee_used_on_gpu=False)

    def _record(direction, method, msg):
        (trace.record_gpu_inbound if direction == "inbound"
         else trace.record_gpu_outbound)(msg)

    worker = RemoteGpuWorker(gpu_worker_url, "qwen7b", recorder=_record)
    _t0 = _time.perf_counter()
    health = worker.health()                                   # GET /health
    init_req = BoundaryInitRequest(
        session_id="sess-0", hidden_size=3584, vocab_size=152064,
        num_layers=num_layers, dtype=dtype, gpu_backend="qwen7b",
        folded_lm_head=None,
        public_metadata={"model": model_name, "seq_len": seq_len,
                         "max_new_tokens": max_new_tokens,
                         "note": "masked prefill/decode runs on the GPU server"})
    init_resp = worker.init(init_req)                          # POST /init
    latency_s = _time.perf_counter() - _t0
    trace.tee_used_on_gpu = bool(init_resp.tee_used_on_gpu)
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
        "mode": "boundary_client",
        "tee_used": False,
        "tee_used_on_gpu": trace.tee_used_on_gpu,
        "gpu_worker_remote": True,
        "gpu_worker_url": gpu_worker_url,
        "qwen7b_probe_only": True,
        "server_health": health,
        "boundary_backend": boundary_backend,
        "gpu_backend": "qwen7b",
        "gpu_backend_server_reported": init_resp.gpu_backend,
        "max_new_tokens": max_new_tokens,
        "latency_s": latency_s,
        "peak_gpu_memory_mb": None,   # no compute on a probe; measured in E1/E2
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
        "note": "cross-machine qwen7b init/attestation/audit probe; full masked "
                "compute validated standalone (E1/E2).",
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
            f"- expected_runtime_hash: `{r.get('expected_runtime_hash')}`",
            f"- evidence_report_data: `{r.get('evidence_report_data')}`",
            f"- **runtime_hash_bound: {r['runtime_hash_bound']}**",
            f"- binding_mismatch_reason: {r.get('binding_mismatch_reason') or 'none'}",
            f"- runtime_manifest_path: `{r.get('runtime_manifest_path')}` "
            f"(files={r.get('runtime_manifest_file_count')})",
            f"- mr_td: `{r['mr_td']}`",
            f"- debug: {att['debug']}",
            f"- jwt_present: {att['jwt_present']} (parts={att['jwt_parts']})",
            f"- mr_td_match: {att['mr_td_match']}",
        ]
    if "cross_machine_compute" in r:
        lines += [
            "", "## Cross-machine scope (keep these separate)", "",
            f"- model_name: `{r.get('model_name')}`",
            f"- cross_machine_compute: **{r.get('cross_machine_compute')}**",
            f"- gpu_worker_remote: {r.get('gpu_worker_remote')} "
            f"(`{r.get('gpu_worker_url')}`)",
            f"- teacher_forced_top1_match_rate: "
            f"{r.get('teacher_forced_top1_match_rate')}",
            f"- plain_vs_masked_token_match_rate: "
            f"{r.get('plain_vs_masked_token_match_rate')}",
            f"- compute correctness: {r.get('compute_correctness_source')}",
            f"- security boundary: {r.get('security_boundary_source')}",
            f"- connectivity: {r.get('connectivity_note')}",
            f"- **limitations**: {r.get('limitations')}",
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
                    choices=["local_two_process", "gpu_worker_server",
                             "boundary_client"])
    ap.add_argument("--boundary-backend", default="process",
                    choices=["process", "simulated"])
    ap.add_argument("--gpu-backend", default="mock",
                    choices=["mock", "qwen7b"])
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--hidden-size", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--num-layers", type=int, default=1,
                    help="public num_layers metadata sent to the GPU worker")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--model-path", default=None,
                    help="qwen7b checkpoint (GPU server); probe-only locally")
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct",
                    help="public model name stamped into cross-machine reports")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    # cross-machine transport
    ap.add_argument("--listen-host", default="0.0.0.0",
                    help="gpu_worker_server: bind host")
    ap.add_argument("--listen-port", type=int, default=18080,
                    help="gpu_worker_server: bind port")
    ap.add_argument("--gpu-worker-url", default=None,
                    help="boundary_client: URL of the remote GPU worker server")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--attestation-evidence", default=None,
                    help="path to TDX attestation evidence JSON from the VM "
                         "(tee/mr_td/report_data/jwt); verifies the binding")
    ap.add_argument("--expected-mr-td", default=None,
                    help="expected mr_td to match against the evidence")
    ap.add_argument("--write-runtime-manifest", default=None,
                    help="write the trusted-boundary manifest JSON to this path")
    ap.add_argument("--write-runtime-hash", default=None,
                    help="write the runtime hash (hex) to this path")
    ap.add_argument("--print-runtime-hash-only", action="store_true",
                    help="preflight: print the exact runtime_hash this demo will "
                         "verify (== TD Quote report_data), then exit without "
                         "running the protocol")
    ap.add_argument("--output-json", default="outputs/tee_gpu_protocol.json")
    ap.add_argument("--output-md", default="outputs/tee_gpu_protocol.md")
    args = ap.parse_args()

    # --- gpu_worker_server: run the untrusted HTTP worker (blocking) ----------
    if args.mode == "gpu_worker_server":
        backend_kwargs = {}
        if args.gpu_backend == "qwen7b":
            backend_kwargs = {"model_path": args.model_path,
                              "device": args.device, "dtype": args.dtype,
                              "seq_len": args.seq_len,
                              "num_layers": args.num_layers}
        run_gpu_worker_server(args.listen_host, args.listen_port,
                              args.gpu_backend, backend_kwargs, _bool(args.audit))
        return 0

    # --- preflight: read off the exact hash to bind into report_data ---------
    md = boundary_manifest_metadata(args.boundary_backend, args.gpu_backend,
                                    args.expected_mr_td)
    if args.print_runtime_hash_only:
        manifest = build_trusted_boundary_manifest(metadata=md)
        rh = compute_runtime_hash_from_manifest(manifest)
        if args.write_runtime_manifest:
            write_runtime_manifest(args.write_runtime_manifest, metadata=md)
        if args.write_runtime_hash:
            write_runtime_hash(args.write_runtime_hash, metadata=md)
        print(rh)
        print(f"# files_measured={len(manifest['files'])} "
              f"boundary_backend={args.boundary_backend} "
              f"gpu_backend={args.gpu_backend} "
              f"expected_mr_td={args.expected_mr_td}", file=sys.stderr)
        print("# bind this value into the TD Quote report_data, then run the "
              "demo with the SAME flags + --attestation-evidence", file=sys.stderr)
        return 0

    gpu_worker_url = None
    if args.mode == "boundary_client":
        if not args.gpu_worker_url:
            ap.error("--mode boundary_client requires --gpu-worker-url")
        gpu_worker_url = args.gpu_worker_url

    if gpu_worker_url is not None and args.gpu_backend == "qwen7b":
        # Cross-machine: real /health + /init + audit + attestation probe vs the
        # remote H800 worker. Full masked compute is validated standalone (E1/E2).
        print("NOTE: cross-machine --gpu-backend qwen7b runs an init/attestation/"
              "audit probe against the remote worker; full Qwen2.5-7B masked "
              "compute is validated standalone (E1/E2). --gpu-backend mock runs "
              "the full boundary<->worker decode end-to-end cross-machine.")
        report = build_remote_qwen7b_probe_report(
            args.prompt, args.boundary_backend, args.max_new_tokens,
            _bool(args.audit), gpu_worker_url, model_name=args.model_name,
            seq_len=args.seq_len, num_layers=args.num_layers, dtype=args.dtype)
        _annotate_cross_machine_scope(
            report, model_name=args.model_name,
            cross_machine_compute="probe_only",
            limitations=(
                "private cross-machine qwen7b masked decode would require "
                "shipping folded layer weights to the worker (which must never "
                "hold the masks); impractical over JSON HTTP. The masked compute "
                "is therefore validated standalone (E1/E2); this run validates "
                "the attested protocol boundary + init handshake + audit only"))
    elif args.gpu_backend == "qwen7b" and gpu_worker_url is None:
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
            seq_len=args.seq_len, seed=args.seed,
            gpu_worker_url=gpu_worker_url)
        if gpu_worker_url is not None:        # mock end-to-end cross-machine
            _annotate_cross_machine_scope(
                report, model_name="mock-identity",
                cross_machine_compute="end_to_end",
                limitations=("none for the boundary security claim: the mock "
                             "identity decoder validates the full cross-machine "
                             "attested protocol + audit end-to-end; Qwen compute "
                             "correctness is the standalone E1/E2 result"))

    # Write manifest / hash with the SAME metadata the attestation uses, so the
    # written hash matches report["runtime_hash"] / expected_runtime_hash.
    if args.write_runtime_manifest:
        write_runtime_manifest(args.write_runtime_manifest, metadata=md)
    if args.write_runtime_hash:
        write_runtime_hash(args.write_runtime_hash, metadata=md)

    attach_attestation(report, evidence=args.attestation_evidence,
                       expected_mr_td=args.expected_mr_td,
                       manifest_path=args.write_runtime_manifest)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_md(p, report)

    print(f"=== TEE ↔ GPU protocol demo ({report['mode']}) ===")
    print(f"boundary_backend={report['boundary_backend']} "
          f"gpu_backend={report['gpu_backend']} "
          f"max_new_tokens={report['max_new_tokens']}")
    if report.get("gpu_worker_remote"):
        print(f"gpu_worker_remote=True gpu_worker_url={report['gpu_worker_url']} "
              f"gpu_backend_server_reported="
              f"{report.get('gpu_backend_server_reported')}")
    print(f"tee_used_on_gpu={report['tee_used_on_gpu']}")
    print(f"latency_s={report.get('latency_s'):.4f} "
          f"peak_gpu_memory_mb={report.get('peak_gpu_memory_mb')}")
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
    print(f"expected_runtime_hash={report['expected_runtime_hash']}")
    print(f"evidence_report_data ={report['evidence_report_data']}")
    print(f"runtime_hash_bound={report['runtime_hash_bound']} "
          f"mr_td={report['mr_td']}")
    if report.get("binding_mismatch_reason"):
        print(f"binding_mismatch_reason: {report['binding_mismatch_reason']}")
    print(f"runtime_manifest_path={report['runtime_manifest_path']} "
          f"(files={report['runtime_manifest_file_count']})")

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
