"""Trusted private-base checkpoint packager + plaintext-absence scanner + guard.

The plaintext checkpoint and the mask secrets may exist ONLY in the trusted offline
packager (or a TDX provisioning env). The packager emits a GPU package that contains
**transformed artifacts only** — every base-weight/embedding/LM-head tensor is
`*_tilde` (masked); no plaintext tensor, no mask secret (`N`, `N_inv`, `R`, `S`, `U`,
`pi`, `perm`, `D_vocab`) is ever written into the GPU package.

Three pieces:
* :class:`PrivateBasePackager` — build a package from transformed tensors; REFUSES
  (fail-closed) to write any artifact that is not `*_tilde` or that matches a
  mask-secret / plaintext-checkpoint pattern.
* :func:`scan_package_for_plaintext` — independently scan a package directory and
  assert the absence of plaintext safetensors / original shards / plaintext embedding
  table / recoverable temp files / plaintext model cache / serialized masks / debug
  dumps.
* :func:`assert_private_base_or_abort` — hard runtime guard: if a plaintext
  checkpoint/model is present in the GPU process or the experiment's filesystem path,
  abort.

stdlib + torch (torch only for the artifact writer). The scanner/guard are stdlib so
they can run anywhere, including inside the untrusted worker's own self-check.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "MASK_SECRET_PATTERNS",
    "PLAINTEXT_CHECKPOINT_PATTERNS",
    "PLAINTEXT_TENSOR_NAME_HINTS",
    "PrivateBasePackager",
    "PrivateBasePackageError",
    "scan_package_for_plaintext",
    "assert_private_base_or_abort",
]

# names that must NEVER appear as an artifact in the GPU package
MASK_SECRET_PATTERNS = (
    r"(^|[_./])n_res(_inv)?([_./]|$)", r"(^|[_./])n_in([_./]|$)",
    r"(^|[_./])n_out([_./]|$)", r"(^|[_./])r_mask([_./]|$)",
    r"(^|[_./])s_mask([_./]|$)", r"(^|[_./])u_rank([_./]|$)",
    r"(^|[_./])pi(_inv)?([_./]|$)", r"(^|[_./])perm([_./]|$)",
    r"(^|[_./])d_vocab([_./]|$)", r"(^|[_./])mask_secret",
    r"(^|[_./])inverse", r"_inv([_./]|$)",
)
# filenames that indicate a plaintext checkpoint / cache leaked into the package
PLAINTEXT_CHECKPOINT_PATTERNS = (
    r"model-\d+-of-\d+\.safetensors$", r"pytorch_model.*\.bin$",
    r"consolidated.*\.pth$", r"\.ckpt$", r"model\.safetensors$",
    r"models--", r"modelscope_cache", r"blobs/", r"snapshots/",
    r"debug.*dump", r"\.tmp$", r"~$",
)
# tensor-name hints for a PLAINTEXT (un-transformed) model tensor
PLAINTEXT_TENSOR_NAME_HINTS = (
    "embed_tokens.weight", "lm_head.weight", "q_proj.weight", "k_proj.weight",
    "v_proj.weight", "o_proj.weight", "gate_proj.weight", "up_proj.weight",
    "down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
)


class PrivateBasePackageError(RuntimeError):
    """A packaging / scan / guard violation (fail-closed)."""


_TENSOR_EXTS = (".pt", ".safetensors", ".bin", ".pth")


def _strip_tensor_ext(name: str) -> str:
    for ext in _TENSOR_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _is_transformed_name(name: str) -> bool:
    # dots are namespace separators in tensor names, NOT extensions; only strip a
    # real tensor file extension before checking the *_tilde suffix.
    return _strip_tensor_ext(name).endswith("_tilde")


def _matches_any(name: str, patterns: Tuple[str, ...]) -> Optional[str]:
    low = name.lower()
    for p in patterns:
        if re.search(p, low):
            return p
    return None


@dataclass
class PrivateBasePackager:
    """Builds a GPU package of transformed artifacts only (fail-closed on plaintext)."""
    out_dir: Path
    model_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    _artifacts: Dict[str, Path] = field(default_factory=dict)
    _hashes: Dict[str, str] = field(default_factory=dict)
    _sizes: Dict[str, int] = field(default_factory=dict)
    _mask_domains: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add_transformed_tensor(self, name: str, tensor) -> None:
        """Write one transformed artifact. REFUSES non-`*_tilde` names and any name
        matching a mask-secret / plaintext pattern."""
        if not _is_transformed_name(name):
            raise PrivateBasePackageError(
                "refusing artifact %r: not a transformed (*_tilde) tensor" % name)
        hit = _matches_any(name, MASK_SECRET_PATTERNS)
        if hit:
            raise PrivateBasePackageError(
                "refusing artifact %r: matches mask-secret pattern %r" % (name, hit))
        if any(h in name for h in PLAINTEXT_TENSOR_NAME_HINTS):
            raise PrivateBasePackageError(
                "refusing artifact %r: looks like a plaintext model tensor" % name)
        import torch  # local import: scanner/guard stay stdlib
        path = self.out_dir / f"{name}.pt"
        blob = tensor.detach().cpu().contiguous()
        torch.save({name: blob}, path)
        data = path.read_bytes()
        self._artifacts[name] = path
        self._hashes[name] = hashlib.sha256(data).hexdigest()
        self._sizes[name] = len(data)

    def record_mask_domain(self, domain: str, members: List[str]) -> None:
        """Record (names only, NO secrets) that a mask domain was used."""
        self._mask_domains[domain] = list(members)

    def finalize(self) -> Dict[str, Any]:
        manifest = {
            "schema": "private_base_gpu_package",
            "version": "1.0",
            "model_id": self.model_id,
            "private_base_required": True,
            "contains_plaintext_base_weights": False,
            "contains_plaintext_embeddings": False,
            "contains_plaintext_lm_head": False,
            "contains_mask_secrets": False,
            "artifacts": sorted(self._artifacts),
            "artifact_count": len(self._artifacts),
            "mask_domains_recorded": self._mask_domains,
            "config": self.config,
        }
        (self.out_dir / "package_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True))
        (self.out_dir / "artifact_hashes.json").write_text(
            json.dumps(self._hashes, indent=2, sort_keys=True))
        (self.out_dir / "package_size.json").write_text(json.dumps(
            {"per_artifact_bytes": self._sizes,
             "total_bytes": sum(self._sizes.values()),
             "artifact_count": len(self._sizes)}, indent=2, sort_keys=True))
        return manifest


def scan_package_for_plaintext(package_dir: str | Path) -> Dict[str, Any]:
    """Independently scan a package dir. Returns a report with ``clean`` bool + the
    findings. Flags plaintext safetensors/shards, plaintext embedding/head/proj
    tensors, serialized masks, temp/debug dumps, and model caches."""
    package_dir = Path(package_dir)
    findings: List[Dict[str, str]] = []
    scanned = 0
    for p in sorted(package_dir.rglob("*")):
        if p.is_dir():
            # directory names can themselves signal a leaked cache
            hit = _matches_any(p.name + "/", PLAINTEXT_CHECKPOINT_PATTERNS)
            if hit:
                findings.append({"path": str(p), "kind": "cache_dir",
                                 "pattern": hit})
            continue
        scanned += 1
        rel = p.name
        allowed_meta = rel in ("package_manifest.json", "artifact_hashes.json",
                               "package_size.json", "plaintext_absence_scan.json",
                               "config.json", "summary.md") or rel.endswith(
                                   "tokenizer.json") or rel.endswith(".txt")
        ck = _matches_any(rel, PLAINTEXT_CHECKPOINT_PATTERNS)
        if ck:
            findings.append({"path": str(p), "kind": "plaintext_checkpoint",
                             "pattern": ck})
        ms = _matches_any(rel, MASK_SECRET_PATTERNS)
        if ms:
            findings.append({"path": str(p), "kind": "mask_secret", "pattern": ms})
        if p.suffix in (".pt", ".bin", ".safetensors", ".pth"):
            base = rel.rsplit(".", 1)[0]
            if not base.endswith("_tilde"):
                findings.append({"path": str(p), "kind": "untransformed_tensor",
                                 "pattern": "name!=*_tilde"})
            if any(h in rel for h in PLAINTEXT_TENSOR_NAME_HINTS):
                findings.append({"path": str(p), "kind": "plaintext_model_tensor",
                                 "pattern": "plaintext_tensor_name"})
        if not allowed_meta and p.suffix not in (".pt", ".bin", ".safetensors",
                                                 ".pth", ".json", ".md", ".txt"):
            findings.append({"path": str(p), "kind": "unexpected_file",
                             "pattern": p.suffix})
    report = {
        "package_dir": str(package_dir),
        "files_scanned": scanned,
        "clean": len(findings) == 0,
        "finding_count": len(findings),
        "findings": findings,
    }
    (package_dir / "plaintext_absence_scan.json").write_text(
        json.dumps(report, indent=2, sort_keys=True))
    return report


def assert_private_base_or_abort(*, gpu_process_paths: List[str],
                                 require: bool = True) -> None:
    """Hard runtime guard. If ``require`` and any path used by the GPU process looks
    like a plaintext checkpoint/model, abort (raise). Call this at worker startup with
    the paths the worker will read (model dir, cache dir, package dir)."""
    if not require:
        return
    offenders = []
    for raw in gpu_process_paths:
        hit = _matches_any(str(raw), PLAINTEXT_CHECKPOINT_PATTERNS)
        if hit:
            offenders.append({"path": str(raw), "pattern": hit})
        # a directory that contains original shards / plaintext model tensors
        pth = Path(raw)
        if pth.exists() and pth.is_dir():
            for f in pth.rglob("*"):
                if f.is_file() and f.suffix in (".safetensors", ".bin", ".pth"):
                    base = f.name.rsplit(".", 1)[0]
                    if not base.endswith("_tilde") or any(
                            h in f.name for h in PLAINTEXT_TENSOR_NAME_HINTS):
                        offenders.append({"path": str(f),
                                          "pattern": "plaintext_model_tensor_in_gpu_path"})
    if offenders:
        raise PrivateBasePackageError(
            "private_base_required=True but plaintext checkpoint/model present in the "
            "GPU path(s): %s" % json.dumps(offenders[:8]))
