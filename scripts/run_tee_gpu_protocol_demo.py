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
                       manifest_path: str | None = None,
                       nonlinear_backend: str | None = None) -> None:
    """Attest the trusted boundary (manifest recipe) and fold into ``report``.

    The runtime hash is SHA-512 over the trusted-boundary manifest (source-file
    digests + public runtime identity), which binds the attestation token to the
    actual boundary code artifact. ``expected_runtime_hash`` is THE value the TD
    Quote's ``report_data`` must equal; the report surfaces it alongside
    ``evidence_report_data`` and a ``binding_mismatch_reason`` when they differ."""
    if nonlinear_backend is None:
        nonlinear_backend = report.get("nonlinear_backend")
    metadata = boundary_manifest_metadata(
        report["boundary_backend"], report["gpu_backend"], expected_mr_td,
        nonlinear_backend=nonlinear_backend)
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
        # H. folded-weight provisioning fields (strict cross-machine deployment)
        "folded_weight_setup_required": True,
        "folded_weight_source": "trusted_setup",
        "folded_weight_transfer_required": True,
        "worker_has_mask_secrets": False,
        "note": "cross-machine qwen7b init/attestation/audit probe; full masked "
                "compute validated standalone (E1/E2). Strict private cross-"
                "machine decode uses a provisioned folded-weight package "
                "(scripts/build_qwen7b_folded_package.py, gpu-backend "
                "qwen7b_folded_package).",
    }


def _greedy_token(rec) -> int:
    return int(rec.argmax(-1).item())


def _trusted_ref_tokens(session, h_tilde, n_new: int, seq_len: int):
    """Trusted in-process folded greedy decode (the reference the remote
    package-backed path must reproduce). Uses the session's worker_prefill /
    worker_decode + the trusted recover/sample -- masks never leave the boundary."""
    import torch
    toks = []
    out = session.worker_prefill(h_tilde)
    tok = _greedy_token(session.recover(out["logits_tilde"][:, -1, :]))
    toks.append(tok)
    kv, position = out["kv"], seq_len
    for _ in range(n_new - 1):
        x = session.mask_token_embedding(torch.tensor([tok]))
        out = session.worker_decode(x, kv, position)
        kv = out["kv"]
        tok = _greedy_token(session.recover(out["logits_tilde"][:, -1, :]))
        toks.append(tok)
        position += 1
    return toks


def build_remote_folded_package_decode_report(args, run_audit: bool,
                                              transcript_recorder=None) -> dict:
    """Cross-machine package-backed prefill+decode against a **remote** folded-
    package GPU worker.

    The trusted boundary owns the model + masks: it tokenizes, masks the prompt
    embeddings, computes the trusted in-process folded reference, then sends ONLY
    masked embeddings + public model/RoPE metadata to the remote worker. The
    worker loads the folded shards locally, executes package-backed prefill/decode
    over the masked tensors (NO mask secrets, not a TEE), and returns masked
    logits; the boundary recovers + samples. Generated tokens are compared to the
    trusted reference. ``--dry-run`` uses a tiny model + tiny package on CPU."""
    import time as _time

    import numpy as np
    import torch

    from pllo.experiments.folded_probe_common import (
        LiteBoundary, folded_exec_metadata, load_model_and_ids,
        seed_from_manifest)
    from pllo.protocol.remote import RemoteGpuWorker
    from pllo.protocol.security_audit import (
        assert_no_gpu_visible_plaintext, assert_no_mask_secret_leak)
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest,
        ProtocolTrace)

    n_new = int(args.max_new_tokens)
    expected = _parse_int_csv(getattr(args, "expected_token_ids", None))
    lite = bool(_bool(getattr(args, "skip_reference", "false"))
                or _bool(getattr(args, "boundary_lite", "false"))
                or (getattr(args, "embedding_path", None) and not args.model_path))
    pkg_dir = Path(args.folded_package_path) if args.folded_package_path else None
    ref_tokens = None
    art_size_gb = None

    if lite:
        # TDX-friendly: NO full Qwen model, NO local folded package. Load only the
        # small trusted boundary artifact (embedding table + N_0 + vocab mask) and
        # the trusted input ids; the worker holds + verifies the folded package.
        if not args.embedding_path:
            raise SystemExit("lite / --skip-reference mode requires "
                             "--embedding-path (the trusted boundary embedding "
                             "artifact built by build_qwen7b_embedding_artifact.py)")
        from pllo.deployment.embedding_artifact import embedding_artifact_size_gb
        boundary = LiteBoundary.from_artifact(args.embedding_path,
                                              device=args.device)
        meta = boundary.meta
        n = int(meta["num_layers"])
        vocab_size = int(meta["vocab_size"])
        seed = int(meta["seed"]) if meta.get("seed") is not None else None
        ids = _resolve_input_ids(args, vocab_size)     # trusted; never sent to GPU
        device, dtype = args.device, args.dtype
        seq_len = int(ids.shape[1])
        exec_meta = boundary.exec_metadata(seq_len=seq_len, max_new_tokens=n_new)
        h_tilde = boundary.mask_embeddings(ids)
        dry_run = bool(getattr(args, "dry_run", False))
        art_size_gb = round(embedding_artifact_size_gb(args.embedding_path), 6)
    else:
        # H800 reference mode: full session owns model + masks + in-process folded
        # reference (kept intact). Requires --folded-package-path for the manifest.
        if not args.folded_package_path:
            raise SystemExit("non-lite mode requires --folded-package-path (the "
                             "in-process folded reference reads its manifest); "
                             "use --skip-reference for the TDX-lite path")
        from pllo.deployment import load_manifest
        from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession
        from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig
        dry_run = bool(getattr(args, "dry_run", False) or not args.model_path)
        model, mc, ids, device, dtype = load_model_and_ids(args, dry_run)
        manifest = load_manifest(pkg_dir)
        n = int(manifest.num_layers)
        seed = seed_from_manifest(pkg_dir, args.seed)
        if manifest.model_path_or_id and args.model_path and \
                manifest.model_path_or_id != args.model_path:
            print("WARNING: --model-path != package model_path_or_id (%s)."
                  % manifest.model_path_or_id)
        seq_len = int(ids.shape[1])
        cfg = MemoryOptimizedConfig(
            num_layers=n, batch_size=1, seq_len=seq_len, max_new_tokens=n_new,
            device=device, dtype=dtype, folding_dtype="float32",
            folded_weight_device=device, seed=seed)
        boundary = MaskedQwenSession(model, mc, cfg)
        vocab_size = int(getattr(mc, "vocab_size"))
        h_tilde = boundary.mask_embeddings(ids)
        ref_tokens = _trusted_ref_tokens(boundary, h_tilde, n_new, seq_len)
        exec_meta = folded_exec_metadata(
            boundary, model_name=args.model_name, num_layers=n, seq_len=seq_len,
            max_new_tokens=n_new, vocab_size=vocab_size)

    # --- remote package-backed decode (shared loop) --------------------------
    trace = ProtocolTrace(boundary_backend=args.boundary_backend,
                          gpu_backend="qwen7b_folded_package",
                          max_new_tokens=n_new, tee_used_on_gpu=False)
    trace.trusted_bytes += int(ids.detach().to("cpu").numpy().nbytes)

    def _record(direction, method, msg):
        (trace.record_gpu_inbound if direction == "inbound"
         else trace.record_gpu_outbound)(msg)
        # Optional metadata-only security transcript (Task D). Backward
        # compatible: when no recorder is supplied, behaviour is unchanged.
        if transcript_recorder is not None:
            from pllo.security import record_message
            record_message(transcript_recorder, direction, msg)

    def _to_np(t):
        return np.asarray(t.detach().to("cpu").float().numpy())

    def _recover_token(masked_logits):
        rec = boundary.recover(torch.as_tensor(np.asarray(masked_logits)).to(
            boundary.compute_device, boundary.fdtype))
        trace.trusted_bytes += int(np.asarray(rec.detach().to("cpu")).nbytes)
        return _greedy_token(rec)

    worker = RemoteGpuWorker(args.gpu_worker_url, "qwen7b_folded_package",
                             recorder=_record)
    _t0 = _time.perf_counter()
    health = worker.health()
    init_resp = worker.init(BoundaryInitRequest(
        session_id="folded-0", hidden_size=int(exec_meta["hidden_size"]),
        vocab_size=vocab_size, num_layers=n, dtype=dtype,
        gpu_backend="qwen7b_folded_package", folded_lm_head=None,
        public_metadata=exec_meta))
    trace.tee_used_on_gpu = bool(init_resp.tee_used_on_gpu)
    trace.bump_boundary("init")

    pkg_tokens: list[int] = []
    pre = worker.prefill(MaskedPrefillRequest(
        session_id="folded-0", masked_embeddings=_to_np(h_tilde),
        positions=list(range(seq_len)), batch_size=1, seq_len=seq_len))
    trace.bump_boundary("prefill")
    tok = _recover_token(pre.masked_logits)
    pkg_tokens.append(tok)
    position = seq_len
    for step in range(n_new - 1):
        x = boundary.mask_token_embedding(torch.tensor([tok]))
        trace.bump_boundary("mask_token_embedding")
        dec = worker.decode(MaskedDecodeRequest(
            session_id="folded-0", masked_embedding=_to_np(x),
            position=position, step=step + 1))
        trace.bump_boundary("decode")
        tok = _recover_token(dec.masked_logits)
        pkg_tokens.append(tok)
        position += 1
    latency_s = _time.perf_counter() - _t0
    # peak GPU memory is measured server-side; read it back over /health.
    peak_mb = None
    try:
        peak_mb = worker.health().get("peak_gpu_memory_mb")
    except Exception:                                        # noqa: BLE001
        pass
    worker.close()

    # --- server's own view of the package (from the init notes) --------------
    snotes: dict = {}
    try:
        snotes = json.loads(init_resp.notes)
    except Exception:                                        # noqa: BLE001
        pass

    # --- audit the EXACT recorded GPU traffic --------------------------------
    plaintext_fields, secret_fields, audit_passed = [], [], None
    if run_audit:
        plaintext_fields = assert_no_gpu_visible_plaintext(
            trace, raw_prompt=args.prompt,
            input_ids=ids.detach().to("cpu").numpy(),
            generated_token_ids=np.asarray([pkg_tokens], dtype=np.int64)
            if pkg_tokens else None, raise_on_fail=False)
        secret_fields = assert_no_mask_secret_leak(trace, None,
                                                   raise_on_fail=False)
        audit_passed = bool(not plaintext_fields and not secret_fields
                            and not trace.tee_used_on_gpu)

    # --- correctness: expected ids (if provided) else in-process reference ----
    if expected is not None:
        reference_for_report = expected
        reference_basis = "expected_token_ids"
        denom = expected
    elif ref_tokens is not None:
        reference_for_report = ref_tokens
        reference_basis = "in_process_folded_reference"
        denom = ref_tokens
    else:
        reference_for_report = None
        reference_basis = "none"
        denom = None
    if denom is not None:
        matches = sum(1 for a, b in zip(pkg_tokens, denom) if a == b)
        token_match_rate = matches / max(1, len(denom))
        tokens_exact_match = bool(pkg_tokens == list(denom))
    else:
        token_match_rate = None
        tokens_exact_match = None
    # cross-check both when available (full reference run that also got expected)
    ref_vs_expected_match = (None if (expected is None or ref_tokens is None)
                             else bool(list(ref_tokens) == list(expected)))

    pkg_size_gb = snotes.get("folded_package_size_gb")
    if pkg_size_gb is None and pkg_dir is not None and pkg_dir.exists():
        from pllo.deployment import package_size_gb
        pkg_size_gb = round(package_size_gb(pkg_dir), 6)
    pkg_path_report = (str(pkg_dir) if pkg_dir is not None
                       else snotes.get("folded_package_path"))

    report = {
        "stage": "qwen7b_folded_remote_package_decode",
        "mode": "boundary_client",
        "boundary_mode": "lite" if lite else "full_reference",
        "dry_run": dry_run,
        "model_name": args.model_name,
        "gpu_worker_remote": True,
        "gpu_worker_url": args.gpu_worker_url,
        "server_health": health,
        "gpu_backend": "qwen7b_folded_package",
        "gpu_backend_server_reported": init_resp.gpu_backend,
        "boundary_backend": args.boundary_backend,
        "tee_used": False,
        "tee_used_on_gpu": trace.tee_used_on_gpu,
        "num_exec_layers": n, "num_package_layers": n,
        "seq_len": seq_len, "dtype": dtype, "seed": seed,
        "max_new_tokens": n_new,
        "folded_package_path": pkg_path_report,
        "embedding_artifact_path": args.embedding_path if lite else None,
        "embedding_artifact_size_gb": art_size_gb,
        "folded_package_loaded": bool(snotes.get("folded_package_loaded", False)),
        "folded_package_valid": bool(snotes.get("folded_package_valid"))
        if snotes.get("folded_package_valid") is not None else None,
        "package_size_gb": pkg_size_gb,
        "num_shards": snotes.get("num_shards"),
        "manifest_hash": snotes.get("manifest_hash"),
        "package_backed_prefill": True,
        "package_backed_decode": True,
        "reference_basis": reference_basis,
        "reference_token_ids": reference_for_report,
        "expected_token_ids": expected,
        "ref_vs_expected_match": ref_vs_expected_match,
        "package_token_ids": pkg_tokens,
        "tokens_exact_match": tokens_exact_match,
        "token_match_rate": token_match_rate,
        # trusted-only echo of the input ids so a later run can REPLAY them via
        # --input-ids-file (stays in the report file; never on the GPU channel).
        "input_ids": ids.detach().to("cpu").reshape(-1).tolist(),
        "worker_has_mask_secrets": bool(
            snotes.get("worker_has_mask_secrets", False)),
        # private folded-LoRA metadata (from the worker init notes; all default
        # to no-LoRA values so the no-LoRA report schema is unchanged)
        "lora_enabled": bool(snotes.get("lora_enabled", False)),
        "folded_lora_loaded": bool(snotes.get("folded_lora_loaded", False)),
        "folded_lora_valid": snotes.get("folded_lora_valid"),
        "lora_rank": snotes.get("lora_rank"),
        "lora_alpha": snotes.get("lora_alpha"),
        "lora_target_modules": snotes.get("lora_target_modules"),
        "lora_adapter_hash": snotes.get("lora_adapter_hash"),
        "worker_has_raw_lora": bool(snotes.get("worker_has_raw_lora", False)),
        "gpu_visible_plaintext_fields": plaintext_fields,
        "leaked_secret_fields": secret_fields,
        "audit_performed": run_audit,
        "audit_passed": audit_passed,
        "boundary_calls": trace.boundary_calls,
        "gpu_calls": trace.gpu_calls,
        "trusted_bytes": trace.trusted_bytes,
        "gpu_bytes": trace.gpu_bytes,
        "gpu_inbound_message_count": len(trace.gpu_inbound_messages),
        "gpu_outbound_message_count": len(trace.gpu_outbound_messages),
        "latency_s": latency_s,
        "peak_gpu_memory_mb": peak_mb,
        # Strict-deployment provisioning fields (folded-weight package).
        "folded_weight_setup_required": True,
        "folded_weight_source": "trusted_setup",
        "worker_has_mask_secrets_claim": False,
        "note": "cross-machine package-backed masked prefill+decode: the trusted "
                "boundary sent only masked embeddings + public model/RoPE "
                "metadata; the untrusted worker executed the folded shards (no "
                "mask secrets, tee_used_on_gpu=False). In lite mode the boundary "
                "holds only the small embedding artifact (no full model, no "
                "26GB package). TDX attestation is NOT claimed by this run -- "
                "rerun the attested demo on TDX if the measured trusted files "
                "changed.",
    }
    return report


def _parse_int_csv(s):
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    parts = [p for p in str(s).replace(" ", "").split(",") if p != ""]
    return [int(p) for p in parts] if parts else None


def _resolve_input_ids(args, vocab_size: int):
    """Trusted input ids for the lite boundary (NEVER sent to the GPU).

    Source order: explicit ``--input-ids`` > ``--input-ids-file`` (JSON with an
    ``input_ids`` field; e.g. a prior reference run's output) > ``--tokenizer-path``
    (+ ``--prompt``) > deterministic dry-run ids. Errors if none is available."""
    import torch
    seq_len = int(args.seq_len)
    if getattr(args, "input_ids", None):
        ids = _parse_int_csv(args.input_ids) or [1]
        return torch.tensor([ids[:seq_len]], dtype=torch.long)
    if getattr(args, "input_ids_file", None):
        data = json.loads(Path(args.input_ids_file).read_text(encoding="utf-8"))
        raw = data.get("input_ids") if isinstance(data, dict) else data
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        return torch.tensor([[int(x) for x in raw][:seq_len]], dtype=torch.long)
    if getattr(args, "tokenizer_path", None):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True, local_files_only=True)
        text = args.prompt
        if _bool(getattr(args, "use_chat_template", "true")):
            text = tok.apply_chat_template(
                [{"role": "user", "content": args.prompt}], tokenize=False,
                add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")["input_ids"][:, :seq_len]
        return ids.to(torch.long)
    if getattr(args, "dry_run", False):
        g = torch.Generator().manual_seed(int(args.seed))
        return torch.randint(0, vocab_size, (1, min(seq_len, 8)), generator=g)
    raise SystemExit(
        "lite mode needs a trusted input source: --input-ids, --input-ids-file "
        "(e.g. a prior reference run's output JSON), or --tokenizer-path "
        "(+ --prompt). These stay trusted-side; the GPU never sees input ids.")


def _write_remote_folded_md(path: Path, r: dict) -> None:
    L = ["# Cross-machine package-backed decode (%s)" % r["stage"], "",
         "- model_name=`%s`  seq_len=%s  dtype=%s  seed=%s  max_new_tokens=%s"
         % (r["model_name"], r["seq_len"], r["dtype"], r["seed"],
            r["max_new_tokens"]),
         "- gpu_worker_remote=%s  url=`%s`  gpu_backend=`%s`"
         % (r["gpu_worker_remote"], r["gpu_worker_url"], r["gpu_backend"]),
         "- folded_package_path=`%s`  num_shards=%s  package_size_gb=%s"
         % (r["folded_package_path"], r["num_shards"], r["package_size_gb"]),
         "- manifest_hash=`%s`" % r["manifest_hash"],
         "- **folded_package_loaded=%s** **folded_package_valid=%s**"
         % (r["folded_package_loaded"], r["folded_package_valid"]),
         "- **package_backed_prefill=%s** **package_backed_decode=%s**"
         % (r["package_backed_prefill"], r["package_backed_decode"]),
         "- **worker_has_mask_secrets=%s** **tee_used_on_gpu=%s**"
         % (r["worker_has_mask_secrets"], r["tee_used_on_gpu"]),
         "- boundary_mode=**%s**  embedding_artifact_path=`%s` (%s GB)"
         % (r["boundary_mode"], r.get("embedding_artifact_path"),
            r.get("embedding_artifact_size_gb")),
         "- lora_enabled=%s folded_lora_loaded=%s folded_lora_valid=%s "
         "worker_has_raw_lora=%s (rank=%s alpha=%s modules=%s)"
         % (r.get("lora_enabled"), r.get("folded_lora_loaded"),
            r.get("folded_lora_valid"), r.get("worker_has_raw_lora"),
            r.get("lora_rank"), r.get("lora_alpha"),
            r.get("lora_target_modules")),
         "", "## Generated tokens (remote package vs %s)" % r["reference_basis"],
         "",
         "- reference_basis=%s  reference_token_ids=%s"
         % (r["reference_basis"], r["reference_token_ids"]),
         "- expected_token_ids=%s  ref_vs_expected_match=%s"
         % (r.get("expected_token_ids"), r.get("ref_vs_expected_match")),
         "- package_token_ids=%s" % r["package_token_ids"],
         "- **tokens_exact_match=%s**  token_match_rate=%s"
         % (r["tokens_exact_match"],
            "n/a" if r["token_match_rate"] is None
            else ("%.4f" % r["token_match_rate"])),
         "", "## Security audit + accounting", "",
         "- audit_performed=%s  **audit_passed=%s**"
         % (r["audit_performed"], r["audit_passed"]),
         "- gpu_visible_plaintext_fields=%s  leaked_secret_fields=%s"
         % (r["gpu_visible_plaintext_fields"] or "[]",
            r["leaked_secret_fields"] or "[]"),
         "- boundary_calls=`%s`  gpu_calls=`%s`"
         % (r["boundary_calls"], r["gpu_calls"]),
         "- trusted_bytes=%s  gpu_bytes=%s  latency_s=%s  peak_gpu_memory_mb=%s"
         % (r["trusted_bytes"], r["gpu_bytes"], r["latency_s"],
            r["peak_gpu_memory_mb"]),
         "", "_%s_" % r["note"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


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
                    choices=["mock", "qwen7b", "qwen7b_folded_package"])
    ap.add_argument("--folded-package-path", default=None,
                    help="qwen7b_folded_package: local folded-weight package dir")
    ap.add_argument("--folded-lora-package-path", default=None,
                    help="qwen7b_folded_package: private folded-LoRA package dir "
                         "(worker merges it; never holds raw A/B or masks)")
    # TDX-friendly lite boundary (no full model / no local 26GB package)
    ap.add_argument("--embedding-path", default=None,
                    help="qwen7b_folded_package lite: trusted boundary embedding "
                         "artifact dir (embed table + N_0 + vocab mask)")
    ap.add_argument("--skip-reference", default="false",
                    help="qwen7b_folded_package: skip the in-process folded "
                         "reference (TDX-lite). Requires --embedding-path; "
                         "correctness checked vs --expected-token-ids if given")
    ap.add_argument("--boundary-lite", default="false",
                    help="alias for --skip-reference (TDX-lite boundary)")
    ap.add_argument("--expected-token-ids", default=None,
                    help="comma-separated token ids from a prior reference run; "
                         "package tokens are compared against them (lite mode)")
    ap.add_argument("--input-ids", default=None,
                    help="lite: comma-separated trusted input ids (never sent to "
                         "the GPU)")
    ap.add_argument("--input-ids-file", default=None,
                    help="lite: JSON file with an 'input_ids' field (e.g. a prior "
                         "reference run's output JSON) to replay")
    ap.add_argument("--tokenizer-path", default=None,
                    help="lite: tokenizer dir to tokenize --prompt locally "
                         "(small; no full model needed)")
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
    ap.add_argument("--dry-run", action="store_true",
                    help="qwen7b_folded_package boundary_client: tiny model + "
                         "tiny package on CPU (never a paper result)")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--attestation-evidence", default=None,
                    help="path to TDX attestation evidence JSON from the VM "
                         "(tee/mr_td/report_data/jwt); verifies the binding")
    ap.add_argument("--expected-mr-td", default=None,
                    help="expected mr_td to match against the evidence")
    ap.add_argument("--nonlinear-backend", default=None,
                    help="bind a nonlinear design (current|trusted_shortcut, "
                         "aliases ok) into the runtime hash / attestation; omit "
                         "to preserve the legacy no-nonlinear binding")
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
    ap.add_argument("--record-transcript", default=None,
                    help="optional path: write a metadata-only GPU-channel "
                         "security transcript JSONL for the boundary_client "
                         "folded path (default None = unchanged behaviour). "
                         "Scan it with scripts/scan_security_transcript.py")
    args = ap.parse_args()

    # --- gpu_worker_server: run the untrusted HTTP worker (blocking) ----------
    if args.mode == "gpu_worker_server":
        backend_kwargs = {}
        if args.gpu_backend == "qwen7b":
            backend_kwargs = {"model_path": args.model_path,
                              "device": args.device, "dtype": args.dtype,
                              "seq_len": args.seq_len,
                              "num_layers": args.num_layers}
        elif args.gpu_backend == "qwen7b_folded_package":
            # forward the selected nonlinear DESIGN so the untrusted worker
            # genuinely EXECUTES it (design B lifts the activation onto this GPU).
            srv_nb = "current"
            if getattr(args, "nonlinear_backend", None) is not None:
                from pllo.experiments.nonlinear_designs import (
                    normalize_nonlinear_backend)
                srv_nb = normalize_nonlinear_backend(args.nonlinear_backend)
            backend_kwargs = {"folded_package_path": args.folded_package_path,
                              "device": args.device, "dtype": args.dtype,
                              "nonlinear_backend": srv_nb,
                              "folded_lora_package_path":
                                  args.folded_lora_package_path}
        run_gpu_worker_server(args.listen_host, args.listen_port,
                              args.gpu_backend, backend_kwargs, _bool(args.audit))
        return 0

    # --- preflight: read off the exact hash to bind into report_data ---------
    nonlinear_backend = None
    if getattr(args, "nonlinear_backend", None) is not None:
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)
    md = boundary_manifest_metadata(args.boundary_backend, args.gpu_backend,
                                    args.expected_mr_td,
                                    nonlinear_backend=nonlinear_backend)
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

    # --- cross-machine package-backed prefill+decode (the executable folded
    #     path over HTTP); self-contained, no TDX-attestation overclaim --------
    if (args.mode == "boundary_client"
            and args.gpu_backend == "qwen7b_folded_package"):
        transcript_recorder = None
        if getattr(args, "record_transcript", None):
            from pllo.security import TranscriptRecorder
            transcript_recorder = TranscriptRecorder()
        report = build_remote_folded_package_decode_report(
            args, _bool(args.audit), transcript_recorder=transcript_recorder)
        if nonlinear_backend is not None:
            from pllo.experiments.nonlinear_designs import (
                nonlinear_design_report_fields)
            report.update(nonlinear_design_report_fields(nonlinear_backend))
            # OVERRIDE the capability stamp with the worker's MEASURED execution
            # evidence (post-run health), so a wired trusted_shortcut run carries
            # genuine lift counters (amulet_lift_executed / lifted_*).
            try:
                from pllo.protocol.remote import RemoteGpuWorker
                _h = RemoteGpuWorker(
                    args.gpu_worker_url, "qwen7b_folded_package").health()
                _ev = (_h or {}).get("nonlinear_execution_evidence") or {}
                if _ev:
                    report.update(_ev)
                    report["nonlinear_execution_evidence_source"] = "worker_health"
            except Exception:                                # noqa: BLE001
                pass
        if transcript_recorder is not None:
            tpath = transcript_recorder.to_jsonl(args.record_transcript)
            print("security transcript written: %s (%d entries)"
                  % (tpath, len(transcript_recorder.entries)))
        # Optional TDX attestation: only attached when evidence is supplied, so
        # the default folded run still makes NO attestation claim. The folded-LoRA
        # metadata already in `report` is thereby covered by the attested run.
        if args.attestation_evidence:
            if args.write_runtime_manifest:
                write_runtime_manifest(args.write_runtime_manifest, metadata=md)
            if args.write_runtime_hash:
                write_runtime_hash(args.write_runtime_hash, metadata=md)
            attach_attestation(report, evidence=args.attestation_evidence,
                               expected_mr_td=args.expected_mr_td,
                               manifest_path=args.write_runtime_manifest,
                               nonlinear_backend=nonlinear_backend)
            print("attestation: boundary_attested=%s runtime_hash_bound=%s "
                  "mr_td=%s" % (report.get("boundary_attested"),
                                report.get("runtime_hash_bound"),
                                report.get("mr_td")))
        if args.output_json:
            p = Path(args.output_json)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=2, default=str),
                         encoding="utf-8")
        if args.output_md:
            _write_remote_folded_md(Path(args.output_md), report)
        print("=== cross-machine package-backed decode (%s) ===" % report["stage"])
        print("gpu_worker_remote=%s url=%s gpu_backend_server_reported=%s"
              % (report["gpu_worker_remote"], report["gpu_worker_url"],
                 report["gpu_backend_server_reported"]))
        print("folded_package_loaded=%s folded_package_valid=%s "
              "package_backed_prefill=%s package_backed_decode=%s"
              % (report["folded_package_loaded"], report["folded_package_valid"],
                 report["package_backed_prefill"], report["package_backed_decode"]))
        print("boundary_mode=%s worker_has_mask_secrets=%s tee_used_on_gpu=%s"
              % (report["boundary_mode"], report["worker_has_mask_secrets"],
                 report["tee_used_on_gpu"]))
        if report.get("lora_enabled"):
            print("lora_enabled=True folded_lora_loaded=%s folded_lora_valid=%s "
                  "worker_has_raw_lora=%s rank=%s alpha=%s target_modules=%s"
                  % (report["folded_lora_loaded"], report["folded_lora_valid"],
                     report["worker_has_raw_lora"], report["lora_rank"],
                     report["lora_alpha"], report["lora_target_modules"]))
        print("reference_basis=%s reference_token_ids=%s expected_token_ids=%s"
              % (report["reference_basis"], report["reference_token_ids"],
                 report["expected_token_ids"]))
        print("package_token_ids  =%s" % report["package_token_ids"])
        _tmr = ("n/a" if report["token_match_rate"] is None
                else ("%.4f" % report["token_match_rate"]))
        print("tokens_exact_match=%s token_match_rate=%s latency_s=%.3f "
              "peak_gpu_memory_mb=%s"
              % (report["tokens_exact_match"], _tmr,
                 report["latency_s"], report["peak_gpu_memory_mb"]))
        print("boundary_calls=%s gpu_calls=%s trusted_bytes=%s gpu_bytes=%s"
              % (report["boundary_calls"], report["gpu_calls"],
                 report["trusted_bytes"], report["gpu_bytes"]))
        print("gpu_visible_plaintext_fields=%s leaked_secret_fields=%s"
              % (report["gpu_visible_plaintext_fields"] or "none",
                 report["leaked_secret_fields"] or "none"))
        print("audit_passed=%s" % report["audit_passed"])
        # tokens_exact_match must be True when a reference/expected basis exists;
        # if neither was provided (tokens_exact_match is None) the security +
        # protocol invariants still gate success (correctness reported, not gated).
        toks_ok = (report["tokens_exact_match"] is not False)
        if report["tokens_exact_match"] is None:
            print("NOTE: no --expected-token-ids / reference; correctness not "
                  "gated (security + protocol invariants still checked).")
        ok = (report["folded_package_loaded"]
              and toks_ok
              and not report["worker_has_mask_secrets"]
              and not report["tee_used_on_gpu"]
              and not report["leaked_secret_fields"]
              and not report["gpu_visible_plaintext_fields"]
              and report["audit_passed"] is not False)
        print("\nREMOTE PACKAGE-BACKED DECODE %s"
              % ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1

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
