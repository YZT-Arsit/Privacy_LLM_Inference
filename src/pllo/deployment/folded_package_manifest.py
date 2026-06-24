"""Manifest for a folded-weight package (trusted-setup output).

The manifest describes a folded weight package without containing any secret:
public model identity, fold configuration, attestation provenance, the per-shard
index (name / relative path / sha256 / byte size / tensor names), and explicit
``contains_*`` security flags that must all be ``False``. ``compute_manifest_hash``
binds the whole manifest (including every shard's sha256) into one digest, so a
single tampered shard is detectable.

stdlib only (json / hashlib / dataclasses). No torch import here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "PACKAGE_FORMAT_VERSION",
    "VALID_PACKAGE_TYPES",
    "VALID_CREATED_BY",
    "SECURITY_CLAIM",
    "FoldedPackageManifest",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "compute_manifest_hash",
    "validate_manifest",
]

PACKAGE_FORMAT_VERSION = "1.0"
VALID_PACKAGE_TYPES = ("base_model", "lora_adapter")
VALID_CREATED_BY = ("tdx_trusted_setup", "trusted_setup", "test")
SECURITY_CLAIM = "gpu_receives_folded_weights_without_mask_secrets"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class FoldedPackageManifest:
    """Describes a folded weight package. No secret material is stored here."""

    package_format_version: str
    package_type: str                       # base_model | lora_adapter
    model_name: str | None
    model_path_or_id: str | None
    num_layers: int
    dtype: str
    nonlinear_backend: str
    created_by: str                         # tdx_trusted_setup|trusted_setup|test
    model_hash: str | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None
    mask_schedule_id: str | None = None
    folding_runtime_hash: str | None = None
    tee_type: str | None = None
    mr_td: str | None = None
    report_data: str | None = None
    security_claim: str = SECURITY_CLAIM
    contains_mask_secrets: bool = False
    contains_plaintext_inputs: bool = False
    contains_raw_lora: bool = False
    contains_optimizer_state: bool = False
    created_at: str | None = None
    # per-shard index: {name, path, sha256, nbytes, tensors:[...], shard_index}
    shard_index: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FoldedPackageManifest":
        fields = {f for f in cls.__dataclass_fields__}      # ignore extras (hash)
        return cls(**{k: v for k, v in d.items() if k in fields})

    @property
    def num_shards(self) -> int:
        return len(self.shard_index)


def build_manifest(*, package_type: str, model_name: str | None,
                   model_path_or_id: str | None, num_layers: int, dtype: str,
                   nonlinear_backend: str, created_by: str,
                   shard_index: list[dict[str, Any]],
                   model_hash: str | None = None, hidden_size: int | None = None,
                   vocab_size: int | None = None,
                   mask_schedule_id: str | None = None,
                   folding_runtime_hash: str | None = None,
                   tee_type: str | None = None, mr_td: str | None = None,
                   report_data: str | None = None,
                   created_at: str | None = None) -> FoldedPackageManifest:
    """Assemble a :class:`FoldedPackageManifest`. The ``contains_*`` flags are
    forced ``False`` (a folded package never carries secrets); the shard index is
    taken verbatim from the writer."""
    return FoldedPackageManifest(
        package_format_version=PACKAGE_FORMAT_VERSION, package_type=package_type,
        model_name=model_name, model_path_or_id=model_path_or_id,
        num_layers=int(num_layers), dtype=dtype,
        nonlinear_backend=nonlinear_backend, created_by=created_by,
        model_hash=model_hash, hidden_size=hidden_size, vocab_size=vocab_size,
        mask_schedule_id=mask_schedule_id,
        folding_runtime_hash=folding_runtime_hash, tee_type=tee_type, mr_td=mr_td,
        report_data=report_data, created_at=created_at,
        shard_index=list(shard_index))


def _canonical(manifest: FoldedPackageManifest | dict[str, Any]) -> bytes:
    """Canonical JSON bytes for hashing (sorted keys, manifest_hash excluded)."""
    d = manifest.to_dict() if isinstance(manifest, FoldedPackageManifest) \
        else dict(manifest)
    d.pop("manifest_hash", None)
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_manifest_hash(manifest: FoldedPackageManifest | dict[str, Any]
                          ) -> str:
    """SHA-256 over the canonical manifest (binds every shard's sha256)."""
    return hashlib.sha256(_canonical(manifest)).hexdigest()


def write_manifest(manifest: FoldedPackageManifest, package_dir: str | Path
                   ) -> Path:
    """Write ``manifest.json`` (with an embedded ``manifest_hash``) into the
    package directory; returns its path."""
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    d = manifest.to_dict()
    d["manifest_hash"] = compute_manifest_hash(manifest)
    path = package_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_manifest(package_dir: str | Path) -> FoldedPackageManifest:
    """Load ``manifest.json`` from a package directory (or a direct file path)."""
    p = Path(package_dir)
    if p.is_dir():
        p = p / MANIFEST_FILENAME
    d = json.loads(p.read_text(encoding="utf-8"))
    return FoldedPackageManifest.from_dict(d)


def validate_manifest(manifest: FoldedPackageManifest) -> tuple[bool, list[str]]:
    """Structural + security validation of a manifest (no disk access).

    Returns ``(ok, problems)``. Checks the format version / package type /
    created_by are recognised, the security claim is the expected one, every
    ``contains_*`` flag is ``False``, and the shard index is non-empty and
    well-formed."""
    problems: list[str] = []
    if manifest.package_format_version != PACKAGE_FORMAT_VERSION:
        problems.append(
            f"package_format_version {manifest.package_format_version!r} != "
            f"{PACKAGE_FORMAT_VERSION!r}")
    if manifest.package_type not in VALID_PACKAGE_TYPES:
        problems.append(f"invalid package_type {manifest.package_type!r}")
    if manifest.created_by not in VALID_CREATED_BY:
        problems.append(f"invalid created_by {manifest.created_by!r}")
    if manifest.security_claim != SECURITY_CLAIM:
        problems.append(f"security_claim {manifest.security_claim!r} != "
                        f"{SECURITY_CLAIM!r}")
    for flag in ("contains_mask_secrets", "contains_plaintext_inputs",
                 "contains_raw_lora", "contains_optimizer_state"):
        if getattr(manifest, flag):
            problems.append(f"{flag} must be False")
    if not manifest.shard_index:
        problems.append("shard_index is empty")
    for i, sh in enumerate(manifest.shard_index):
        for key in ("name", "path", "sha256", "nbytes"):
            if key not in sh:
                problems.append(f"shard[{i}] missing {key!r}")
    return (not problems), problems
