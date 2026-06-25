"""TDX attestation + runtime-hash binding for the trusted boundary.

The trusted boundary process is meant to run inside an Intel TDX guest on
Alibaba Cloud. Remote attestation binds a *runtime hash* (a 64-byte SHA-512 over
the boundary's public identity/config) into the TD Quote's ``report_data`` field;
a verifier then checks ``expected_runtime_hash == report_data`` from the signed
attestation JWT, alongside ``tee == tdx`` and ``td_attributes.debug == false``.

On the validated Alibaba Cloud TDX VM this whole chain succeeds (TD Quote
generated, Attestation API returns a 3-part signed JWT, ``mr_td`` reported,
runtime-hash binding verified, debug mode off). This module:

* computes the runtime hash deterministically (so the same recipe runs on the VM
  and in CI);
* **verifies** real attestation evidence (a JSON blob produced on the VM) without
  fabricating a quote off-TDX;
* degrades gracefully off-TDX (reports ``simulated`` + the runtime hash the
  boundary *would* bind), so deployment can be prepared and unit-tested with
  numpy/stdlib only.

No quote is generated here -- quote generation is the deployment's attestation
client (configfs-tsm / Alibaba SDK). This module ingests + verifies its output.
numpy + standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pllo.tee.runtime_api import TDX_GUEST_DEVICE

__all__ = [
    "AttestationEvidence",
    "DEFAULT_TRUSTED_BOUNDARY_PATHS",
    "MANIFEST_EXCLUDES",
    "binding_mismatch_reason",
    "boundary_manifest_metadata",
    "boundary_runtime_hash",
    "build_trusted_boundary_manifest",
    "compute_runtime_hash",
    "compute_runtime_hash_from_manifest",
    "runtime_report_data_hex",
    "write_runtime_manifest",
    "write_runtime_hash",
    "attest_boundary",
    "verify_evidence",
]

# Report data in TDX is 64 bytes; SHA-512 is a natural exact fit.
REPORT_DATA_BYTES = 64

# Repo root: .../src/pllo/protocol/attestation.py -> parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]

# The trusted-boundary / protocol code measured into the runtime hash. Globs are
# expanded relative to the manifest base (repo root by default); __pycache__ is
# never matched by the *.py glob. Override via ``build_trusted_boundary_manifest``.
#
# This MUST cover every trusted-side source file the boundary process loads at
# runtime -- including the TDX-lite boundary used by the (no-LoRA and LoRA)
# package-backed decode path. ``scripts/check_tdx_measurement_coverage.py``
# recomputes the boundary's first-party import closure and fails if any
# boundary-imported module here is unmeasured. Worker-only / trusted-setup files
# (gpu_worker, folded_worker, lora_folded_package, the folded-package writer, full
# hf_wrappers) are intentionally excluded: they run on the untrusted GPU host or
# the offline trusted-setup box, NOT inside the TD. The private folded LoRA adds
# NO new boundary file -- folding/merging happens on the worker -- so the LoRA
# attested path measures exactly this same set.
DEFAULT_TRUSTED_BOUNDARY_PATHS: tuple[str, ...] = (
    "src/pllo/protocol/attestation.py",
    "src/pllo/protocol/tee_gpu_messages.py",
    "src/pllo/protocol/security_audit.py",
    "src/pllo/protocol/wire.py",
    "src/pllo/protocol/remote.py",
    "src/pllo/tee/*.py",
    # TDX-lite boundary runtime surface (embed+mask / recover / RPC drive):
    "src/pllo/experiments/folded_probe_common.py",
    "src/pllo/deployment/embedding_artifact.py",
    "src/pllo/ops/causal_lm_boundaries.py",
    "src/pllo/ops/nonlinear_islands.py",
    "src/pllo/ops/mitigation_bundles.py",
    # optional metadata-only security transcript recorder/scanner (loaded by the
    # boundary when --record-transcript is enabled; trusted-side instrumentation).
    # Listed explicitly (not a glob) so non-runtime security tooling added to the
    # package later does not silently change the attestation hash.
    "src/pllo/security/__init__.py",
    "src/pllo/security/transcript_recorder.py",
    "src/pllo/security/transcript_scanner.py",
    # nonlinear-design registry: the selected design is folded into the runtime
    # identity (boundary_manifest_metadata), so the design contract source is
    # part of the measured boundary. This binds design A vs design B to distinct
    # runtime hashes -- evidence for one design cannot be replayed for the other.
    "src/pllo/experiments/nonlinear_designs.py",
    "scripts/run_tee_gpu_protocol_demo.py",
)

# Explicitly NOT part of the manifest: per-request, secret, or volatile data.
# Recorded in the manifest itself for auditability (and the paper).
MANIFEST_EXCLUDES: tuple[str, ...] = (
    "raw_prompt",
    "input_ids",
    "generated_token_ids",
    "recovered_logits",
    "mask_secrets",
    "tokenizer_output",
    "temporary_outputs",
    "logs",
    "model_weights",
)

# Package versions folded into runtime identity (best effort; "if available").
_MANIFEST_PACKAGES = ("pllo", "numpy")


@dataclass
class AttestationEvidence:
    """Attestation status for the trusted boundary (verified, not generated)."""
    tee_type: str                       # "tdx" | "simulated"
    available: bool
    verified: bool
    runtime_hash_hex: str               # 128 hex chars (64 bytes)
    report_data_hex: str | None         # report_data carried in the quote
    runtime_hash_bound: bool | None     # report_data == runtime_hash
    debug: bool | None                  # td_attributes.debug (must be False)
    mr_td: str | None
    mr_td_match: bool | None            # vs an expected_mr_td (if provided)
    jwt_present: bool
    jwt_parts: int
    quote_available: bool
    quote_status: str
    tdx_guest_device_present: bool
    notes: str = ""
    claims: dict[str, Any] = field(default_factory=dict)


def compute_runtime_hash(components: dict[str, Any]) -> bytes:
    """Deterministic 64-byte runtime hash over the boundary's public identity.

    ``components`` is canonicalised (sorted-key JSON) before hashing so the same
    inputs always yield the same bytes on the TDX VM and in CI. Include only
    public, reproducible identity here (component name/version + boundary config)
    -- never mask secrets or the raw prompt."""
    canon = json.dumps(components, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")
    return hashlib.sha512(canon).digest()        # 64 bytes


def runtime_report_data_hex(runtime_hash: bytes) -> str:
    """Hex string the boundary binds into ``report_data`` (zero-padded to 64B)."""
    if len(runtime_hash) > REPORT_DATA_BYTES:
        runtime_hash = runtime_hash[:REPORT_DATA_BYTES]
    elif len(runtime_hash) < REPORT_DATA_BYTES:
        runtime_hash = runtime_hash + b"\x00" * (
            REPORT_DATA_BYTES - len(runtime_hash))
    return runtime_hash.hex()


# ---------------------------------------------------------------------------
# Trusted-boundary manifest (the runtime-hash recipe)
# ---------------------------------------------------------------------------


def _rel_path(path: Path, base: Path) -> str:
    """Path relative to ``base`` (POSIX) so the manifest is location-independent.

    Falls back to the file name if ``path`` is not under ``base`` (e.g. a temp
    file in a test), keeping the manifest deterministic within a run."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def _expand_paths(paths: Iterable[str], base: Path) -> list[Path]:
    """Resolve each entry against ``base``; expand glob entries; drop pycache."""
    out: list[Path] = []
    for entry in paths:
        if any(c in entry for c in "*?[]"):
            for m in sorted(base.glob(entry)):
                if "__pycache__" in m.parts:
                    continue
                out.append(m)
        else:
            out.append(base / entry)
    # de-dup while preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    try:
        from importlib import metadata as _md
    except Exception:                                   # pragma: no cover
        return {name: None for name in _MANIFEST_PACKAGES}
    for name in _MANIFEST_PACKAGES:
        try:
            versions[name] = _md.version(name)
        except Exception:
            versions[name] = None
    return versions


def build_trusted_boundary_manifest(
    paths: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    base: Path | str | None = None,
) -> dict[str, Any]:
    """Build the deterministic trusted-boundary manifest (a canonical dict).

    Contents:

    * ``files`` -- for each measured trusted-boundary source file: repo-relative
      path + SHA-256 digest + size (sorted by path). This binds the attestation
      to the actual code artifact, not just a config string.
    * ``runtime_identity`` -- ``protocol_version``, ``boundary_backend``,
      ``allowed_gpu_backend``, ``expected_mr_td`` (if supplied), the Python
      major.minor, and best-effort package versions.
    * ``excludes`` -- the per-request / secret / volatile data explicitly NOT
      measured (raw prompt, input_ids, generated tokens, mask secrets, recovered
      logits, tokenizer output, temp outputs, logs, model weights).

    The manifest never contains prompt text, token ids, mask secrets, or model
    weights -- only file digests + public runtime identity. Hash it with
    :func:`compute_runtime_hash_from_manifest`."""
    base_path = Path(base) if base is not None else REPO_ROOT
    selected = list(paths) if paths is not None \
        else list(DEFAULT_TRUSTED_BOUNDARY_PATHS)

    files: list[dict[str, Any]] = []
    for p in _expand_paths(selected, base_path):
        rel = _rel_path(p, base_path)
        if p.is_file():
            data = p.read_bytes()
            files.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(),
                          "size": len(data)})
        else:
            files.append({"path": rel, "sha256": None, "size": 0,
                          "missing": True})
    files.sort(key=lambda e: e["path"])

    md = dict(metadata or {})
    runtime_identity = {
        "protocol_version": md.get("protocol_version", "8.5"),
        "boundary_backend": md.get("boundary_backend"),
        "allowed_gpu_backend": md.get("allowed_gpu_backend"),
        "expected_mr_td": (md.get("expected_mr_td").lower()
                           if md.get("expected_mr_td") else None),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "packages": _package_versions(),
    }
    # carry through any extra public identity keys the caller supplied
    for k, v in md.items():
        runtime_identity.setdefault(k, v)

    return {
        "kind": "trusted_boundary_manifest",
        "manifest_version": "1",
        "files": files,
        "runtime_identity": runtime_identity,
        "excludes": list(MANIFEST_EXCLUDES),
    }


def compute_runtime_hash_from_manifest(manifest: dict[str, Any]) -> str:
    """64-byte SHA-512 hex over the canonicalised manifest (== report_data hex).

    Canonicalisation is sorted-key, separator-tight JSON so the same manifest
    always yields the same 128-hex-char digest on the TDX VM and in CI."""
    canon = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")
    return hashlib.sha512(canon).hexdigest()            # 64 bytes -> 128 hex


def write_runtime_manifest(
    path: str | Path,
    paths: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    base: Path | str | None = None,
) -> dict[str, Any]:
    """Build + write the manifest as indented JSON; return the manifest dict."""
    manifest = build_trusted_boundary_manifest(paths, metadata, base=base)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def write_runtime_hash(
    path: str | Path,
    paths: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    base: Path | str | None = None,
) -> str:
    """Build the manifest, compute the runtime hash, write the hex; return it."""
    manifest = build_trusted_boundary_manifest(paths, metadata, base=base)
    rh = compute_runtime_hash_from_manifest(manifest)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rh + "\n", encoding="utf-8")
    return rh


def boundary_manifest_metadata(
    boundary_backend: str,
    gpu_backend: str,
    expected_mr_td: str | None = None,
    *,
    protocol_version: str = "8.5",
    nonlinear_backend: str | None = None,
    nonlinear_design_metadata_hash: str | None = None,
) -> dict[str, Any]:
    """Canonical runtime-identity metadata shared by the demo + preflight tool.

    Both the quote-binding step (``write_tee_boundary_runtime_hash.py`` /
    ``--print-runtime-hash-only``) and the verification step (the demo) MUST use
    this exact metadata so they compute the identical runtime hash. Changing any
    field (notably ``expected_mr_td`` or the selected ``nonlinear_backend``)
    changes the binding.

    The nonlinear design is part of the runtime identity: design A and design B
    deliberately produce different runtime hashes, so attestation evidence bound
    for one nonlinear design cannot be reused for the other. When
    ``nonlinear_backend`` is given the canonical name + its design metadata hash
    are folded into ``runtime_identity``. ``None`` (the default) preserves the
    legacy hash for backward compatibility with already-bound no-nonlinear
    runs."""
    md: dict[str, Any] = {
        "protocol_version": protocol_version,
        "boundary_backend": boundary_backend,
        "allowed_gpu_backend": gpu_backend,
        "expected_mr_td": expected_mr_td,
    }
    if nonlinear_backend is not None:
        # Resolve canonically so an alias cannot dodge the binding; compute the
        # design hash here if the caller did not supply it.
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend,
            nonlinear_design_metadata_hash as _ndmh,
        )
        canon = normalize_nonlinear_backend(nonlinear_backend)
        md["nonlinear_backend"] = canon
        md["nonlinear_design_metadata_hash"] = (
            nonlinear_design_metadata_hash or _ndmh(canon))
    return md


def boundary_runtime_hash(
    metadata: dict[str, Any] | None = None,
    paths: Iterable[str] | None = None,
    *,
    base: Path | str | None = None,
) -> str:
    """The single source of truth: the runtime hash the boundary binds + verifies.

    Equals ``report_data`` of the TD Quote. The preflight tool prints this; the
    demo recomputes it identically and checks it against the evidence."""
    return compute_runtime_hash_from_manifest(
        build_trusted_boundary_manifest(paths, metadata, base=base))


def binding_mismatch_reason(ev: "AttestationEvidence") -> str | None:
    """Explain why ``runtime_hash_bound`` is not True (or ``None`` if it is)."""
    if ev.runtime_hash_bound is True:
        return None
    if ev.report_data_hex is None:
        return ("no report_data in evidence (no TD Quote was bound, or this is a "
                "simulated/off-TDX run)")
    if ev.report_data_hex != ev.runtime_hash_hex:
        return (
            f"evidence.report_data ({ev.report_data_hex[:16]}...) != "
            f"expected_runtime_hash ({ev.runtime_hash_hex[:16]}...): the TD Quote "
            "was bound to a stale/different runtime hash. The boundary code or "
            "metadata (e.g. expected_mr_td) changed since binding. Recompute with "
            "scripts/write_tee_boundary_runtime_hash.py (identical flags), bind "
            "the quote's report_data to that value, and re-run.")
    return None


def _jwt_parts(evidence: dict[str, Any]) -> tuple[bool, int]:
    jwt = evidence.get("jwt")
    if isinstance(jwt, str) and jwt:
        return True, len([p for p in jwt.split(".") if p != ""])
    parts = evidence.get("jwt_parts")
    if isinstance(parts, int) and parts > 0:
        return True, parts
    return False, 0


def _get_debug(evidence: dict[str, Any]) -> bool | None:
    td = evidence.get("tdx") or {}
    attrs = td.get("td_attributes") if isinstance(td, dict) else None
    if isinstance(attrs, dict) and "debug" in attrs:
        return bool(attrs["debug"])
    if "debug" in evidence:
        return bool(evidence["debug"])
    if "td_attributes.debug" in evidence:
        return bool(evidence["td_attributes.debug"])
    return None


def verify_evidence(
    evidence: dict[str, Any],
    runtime_hash: bytes,
    *,
    expected_mr_td: str | None = None,
) -> AttestationEvidence:
    """Verify real TDX attestation evidence against the expected runtime hash.

    Checks: ``tee == tdx``; ``td_attributes.debug == false``; a signed JWT is
    present (3 parts: header.payload.signature); ``report_data`` equals the
    runtime hash (binding); and, if given, ``mr_td`` matches ``expected_mr_td``.
    ``verified`` is the conjunction of all required checks. This does NOT verify
    the JWT signature / certificate chain -- that is the remote verifier's job;
    we record that a signed token was returned."""
    rd_hex = runtime_report_data_hex(runtime_hash)
    tee = str(evidence.get("tee", "")).lower()
    debug = _get_debug(evidence)
    mr_td = evidence.get("mr_td")
    report_data = evidence.get("report_data")
    if isinstance(report_data, str):
        report_data = report_data.lower()
        if report_data.startswith("0x"):       # py3.6-safe (no str.removeprefix)
            report_data = report_data[2:]
    jwt_present, jwt_parts = _jwt_parts(evidence)

    bound = (report_data == rd_hex) if isinstance(report_data, str) else None
    mr_td_match = (str(mr_td).lower() == str(expected_mr_td).lower()
                   if (expected_mr_td and mr_td) else None)

    checks = [
        tee == "tdx",
        debug is False,
        jwt_present and jwt_parts == 3,
        bound is True,
    ]
    if expected_mr_td:
        checks.append(mr_td_match is True)
    verified = all(checks)

    return AttestationEvidence(
        tee_type=tee or "unknown", available=True, verified=verified,
        runtime_hash_hex=rd_hex, report_data_hex=report_data,
        runtime_hash_bound=bound, debug=debug, mr_td=mr_td,
        mr_td_match=mr_td_match, jwt_present=jwt_present, jwt_parts=jwt_parts,
        quote_available=True,
        quote_status="verified" if verified else "evidence_check_failed",
        tdx_guest_device_present=os.path.exists(TDX_GUEST_DEVICE),
        notes="JWT signature/cert-chain verified by the remote attestation "
              "service; this module verifies tee/debug/binding/mr_td claims.",
        claims={"tee": tee, "debug": debug, "mr_td": mr_td,
                "jwt_parts": jwt_parts})


def attest_boundary(
    runtime_components: dict[str, Any] | None = None,
    *,
    runtime_hash: bytes | str | None = None,
    evidence: dict[str, Any] | str | None = None,
    expected_mr_td: str | None = None,
    tdx_guest_device: str = TDX_GUEST_DEVICE,
) -> AttestationEvidence:
    """Attest the trusted boundary, binding the runtime hash into the quote.

    Provide the runtime hash one of two ways:

    * ``runtime_hash`` -- a precomputed 64-byte digest (bytes or 128-char hex),
      normally :func:`compute_runtime_hash_from_manifest` over the trusted-
      boundary manifest (the preferred, code-binding recipe); or
    * ``runtime_components`` -- a legacy config dict hashed via
      :func:`compute_runtime_hash` (weaker; does not bind the code artifact).

    Then:

    * ``evidence`` given (dict or path to JSON produced on the TDX VM) -> verify
      it against the runtime hash (real, evidence-backed path).
    * no evidence, TDX guest device present -> report ``tdx`` available but mark
      that evidence was not supplied (quote not generated here).
    * otherwise -> ``simulated``: still report the runtime hash the boundary
      would bind, so deployment can be prepared/tested off-TDX.

    The runtime hash never depends on secrets; it is a public identity binding."""
    if runtime_hash is not None:
        rh = (bytes.fromhex(runtime_hash) if isinstance(runtime_hash, str)
              else bytes(runtime_hash))
    elif runtime_components is not None:
        rh = compute_runtime_hash(runtime_components)
    else:
        raise ValueError("provide runtime_hash or runtime_components")
    rd_hex = runtime_report_data_hex(rh)

    if evidence is not None:
        if isinstance(evidence, str):
            evidence = json.loads(
                Path(evidence).read_text(encoding="utf-8"))
        return verify_evidence(evidence, rh, expected_mr_td=expected_mr_td)

    device_present = False
    try:
        device_present = os.path.exists(tdx_guest_device)
    except OSError:
        device_present = False

    if device_present:
        return AttestationEvidence(
            tee_type="tdx", available=True, verified=False,
            runtime_hash_hex=rd_hex, report_data_hex=None,
            runtime_hash_bound=None, debug=None, mr_td=None, mr_td_match=None,
            jwt_present=False, jwt_parts=0, quote_available=False,
            quote_status="tdx_guest_present_evidence_not_provided",
            tdx_guest_device_present=True,
            notes="TDX guest device present; provide --attestation-evidence "
                  "(from the VM's attestation client) to verify the binding.")
    return AttestationEvidence(
        tee_type="simulated", available=False, verified=False,
        runtime_hash_hex=rd_hex, report_data_hex=None, runtime_hash_bound=None,
        debug=None, mr_td=None, mr_td_match=None, jwt_present=False,
        jwt_parts=0, quote_available=False, quote_status="simulated_no_tdx",
        tdx_guest_device_present=False,
        notes="No TDX guest device; runtime hash shown is what the boundary "
              "would bind into report_data when deployed in TDX.")
