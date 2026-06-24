"""Folded-weight package: fold operators, shard IO, writer, verification.

A *folded operator* is ``W_tilde = N_in^{-1} W N_out`` (the codebase convention
is ``y = x @ W``, so masked ``x_tilde = x @ N_in`` gives
``x_tilde @ W_tilde = (x @ W) @ N_out = y @ N_out`` -- the worker computes the
masked output with no knowledge of ``N_in`` / ``N_out``). Trusted setup writes
the folded operators to disk shards (safetensors preferred, ``torch.save``
fallback) plus a manifest; the untrusted worker loads the shards and never sees a
mask.

Every artifact is screened by tensor name: any name resembling a mask matrix,
plaintext input, raw LoRA state, gradient, or optimizer state is rejected before
it can be written. The folded operator names (``*_tilde``) are allowed.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "FORBIDDEN_PACKAGE_SUBSTRINGS",
    "forbidden_tensor_names",
    "fold_linear",
    "save_shard",
    "load_shard",
    "compute_file_sha256",
    "list_package_shards",
    "package_size_bytes",
    "package_size_gb",
    "FoldedPackageWriter",
    "verify_package",
]

# Substrings that must NOT appear in any folded-package tensor name. Curated so
# the allowed folded-operator names (wq_tilde, wo_tilde, wgate_tilde, wup_tilde,
# wdown_tilde, w_lm_tilde, b*_tilde, lora_a_tilde, lora_b_tilde) never match.
FORBIDDEN_PACKAGE_SUBSTRINGS = frozenset({
    "mask", "perm", "_sign", "signs", "n_res", "n_in", "n_out", "n0",
    "raw_prompt", "prompt", "input_id", "label", "recovered", "plain_hidden",
    "tokenizer", "vocab_inv", "vocab_perm", "vocab_scale", "out_scale", "scale",
    "secret", "grad", "optim", "adam", "delta_w",
    # raw (un-folded) LoRA state -- only *_tilde folded LoRA may be packaged
    "lora_a_raw", "lora_b_raw", "raw_lora",
})

_SHARD_EXTS = (".safetensors", ".pt")


def forbidden_tensor_names(names: Any) -> list[str]:
    """Return the subset of ``names`` containing a forbidden substring."""
    bad: list[str] = []
    for n in names:
        low = str(n).lower()
        if any(sub in low for sub in FORBIDDEN_PACKAGE_SUBSTRINGS):
            bad.append(str(n))
    return bad


def fold_linear(weight: torch.Tensor, n_in_inv: torch.Tensor,
                n_out: torch.Tensor) -> torch.Tensor:
    """``W_tilde = N_in^{-1} W N_out`` for the ``y = x @ W`` convention
    (``weight`` is ``[in, out]``; ``n_in_inv`` ``[in, in]``; ``n_out``
    ``[out, out]``)."""
    return n_in_inv @ weight @ n_out


# ---------------------------------------------------------------------------
# Shard IO (safetensors preferred, torch.save fallback)
# ---------------------------------------------------------------------------


def _have_safetensors() -> bool:
    try:
        import safetensors.torch  # noqa: F401
        return True
    except Exception:
        return False


def save_shard(path: str | Path, tensors: dict[str, torch.Tensor], *,
               prefer_safetensors: bool = True,
               allow_torch_save_fallback: bool = True) -> dict[str, Any]:
    """Write a shard of folded tensors. Rejects forbidden tensor names first.

    Uses safetensors when available (``.safetensors``); otherwise falls back to
    ``torch.save`` (``.pt``) with an explicit warning. Returns shard info
    ``{path, sha256, nbytes, tensors:[names], format}``."""
    bad = forbidden_tensor_names(tensors.keys())
    if bad:
        raise ValueError(f"refusing to write forbidden tensor names: {bad}")
    path = Path(path)
    cpu = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
    use_st = prefer_safetensors and _have_safetensors()
    if use_st:
        from safetensors.torch import save_file
        path = path.with_suffix(".safetensors")
        save_file(cpu, str(path))
        fmt = "safetensors"
    else:
        if prefer_safetensors and not allow_torch_save_fallback:
            raise RuntimeError("safetensors unavailable and torch.save fallback "
                               "disabled")
        warnings.warn("safetensors unavailable; using torch.save (.pt) fallback "
                      "-- install safetensors for the preferred format",
                      RuntimeWarning, stacklevel=2)
        path = path.with_suffix(".pt")
        torch.save(cpu, str(path))
        fmt = "torch_save"
    return {"path": path.name, "sha256": compute_file_sha256(path),
            "nbytes": int(path.stat().st_size),
            "tensors": sorted(cpu.keys()), "format": fmt}


def load_shard(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a shard (``.safetensors`` or ``.pt``) into a name->tensor dict."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(load_file(str(path)))
    if path.suffix == ".pt":
        return dict(torch.load(str(path), map_location="cpu"))
    raise ValueError(f"unknown shard format {path.suffix!r}")


def compute_file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_package_shards(package_dir: str | Path) -> list[Path]:
    """Sorted shard files in a package directory (excludes the manifest)."""
    package_dir = Path(package_dir)
    out: list[Path] = []
    for p in sorted(package_dir.iterdir()):
        if p.is_file() and p.suffix in _SHARD_EXTS:
            out.append(p)
    return out


def package_size_bytes(package_dir: str | Path) -> int:
    """Total bytes of all shard files in a package directory."""
    return int(sum(p.stat().st_size for p in list_package_shards(package_dir)))


def package_size_gb(package_dir: str | Path) -> float:
    return package_size_bytes(package_dir) / (1024 ** 3)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class FoldedPackageWriter:
    """Streams folded shards to disk and accumulates the manifest shard index.

    Shards are written one at a time so the full folded model never needs to be
    resident; each ``add_shard`` records the shard's sha256 + size + tensor names
    for the manifest."""

    def __init__(self, output_dir: str | Path, *,
                 prefer_safetensors: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefer_safetensors = prefer_safetensors
        self._entries: list[dict[str, Any]] = []

    def add_shard(self, name: str, tensors: dict[str, torch.Tensor]
                  ) -> dict[str, Any]:
        info = save_shard(self.output_dir / name, tensors,
                          prefer_safetensors=self.prefer_safetensors)
        entry = {"name": name, "shard_index": len(self._entries), **info}
        self._entries.append(entry)
        return entry

    @property
    def shard_index(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def total_bytes(self) -> int:
        return int(sum(e["nbytes"] for e in self._entries))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_package(package_dir: str | Path, *, check_manifest: bool = True,
                   check_hashes: bool = True,
                   check_no_secret_fields: bool = True) -> dict[str, Any]:
    """Verify a folded package on disk.

    Returns a report with ``package_valid`` plus diagnostics: manifest hash,
    package size, shard count, missing shards, per-shard hash mismatches, and any
    forbidden field/tensor names found in the manifest or shard contents."""
    from pllo.deployment.folded_package_manifest import (
        compute_manifest_hash,
        load_manifest,
        validate_manifest,
    )

    package_dir = Path(package_dir)
    report: dict[str, Any] = {
        "package_dir": str(package_dir),
        "package_valid": False,
        "manifest_hash": None,
        "manifest_problems": [],
        "package_size_gb": round(package_size_gb(package_dir), 6),
        "num_shards": 0,
        "missing_shards": [],
        "hash_mismatches": [],
        "forbidden_fields_found": [],
        "contains_mask_secrets": None,
        "contains_plaintext_inputs": None,
        "contains_raw_lora": None,
        "contains_optimizer_state": None,
    }

    manifest = load_manifest(package_dir)
    report["manifest_hash"] = compute_manifest_hash(manifest)
    report["num_shards"] = manifest.num_shards
    report["contains_mask_secrets"] = manifest.contains_mask_secrets
    report["contains_plaintext_inputs"] = manifest.contains_plaintext_inputs
    report["contains_raw_lora"] = manifest.contains_raw_lora
    report["contains_optimizer_state"] = manifest.contains_optimizer_state

    manifest_ok = True
    if check_manifest:
        manifest_ok, problems = validate_manifest(manifest)
        report["manifest_problems"] = problems

    # NOTE: manifest *field names* (e.g. mask_schedule_id, a public id) are not
    # screened -- only the actual stored tensor names are, since that is where
    # secret material would have to live.

    # per-shard existence + hash + tensor-name screen
    for sh in manifest.shard_index:
        shard_path = package_dir / sh["path"]
        if not shard_path.exists():
            report["missing_shards"].append(sh["path"])
            continue
        if check_hashes:
            actual = compute_file_sha256(shard_path)
            if actual != sh.get("sha256"):
                report["hash_mismatches"].append(
                    {"path": sh["path"], "expected": sh.get("sha256"),
                     "actual": actual})
        if check_no_secret_fields:
            bad = forbidden_tensor_names(sh.get("tensors", []))
            if bad:
                report["forbidden_fields_found"].extend(
                    [f"{sh['path']}:{b}" for b in bad])

    report["package_valid"] = bool(
        manifest_ok and not report["missing_shards"]
        and not report["hash_mismatches"] and not report["forbidden_fields_found"]
        and not manifest.contains_mask_secrets
        and not manifest.contains_plaintext_inputs
        and not manifest.contains_raw_lora
        and not manifest.contains_optimizer_state)
    return report
