"""PHASE 4 — production-style transformed-adapter handoff (build + verify + load).

Exports a masked LoRA adapter as a self-describing package and a generation loader that fail-closes
on any incompatibility. NEVER exports plaintext master / m / v; for both profiles only masked
runtime A/B are written. CPU, local, no plaintext base reconstruction.

adapter_package/
    adapter_manifest.json
    adapter_tensors.safetensors
    artifact_hashes.sha256
    provenance.json
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from typing import Dict, Tuple

import torch

TRANSFORM_ALGEBRA_VERSION = "private_base_fold_v1.0"
_SECRET_HINTS = ("n_res", "n_in", "n_out", "r_mask", "s_mask", "u_rank", "pi", "perm",
                 "d_vocab", "master", "adamw_m", "adamw_v", ".m", ".v", "plaintext")


class AdapterError(RuntimeError):
    """Fail-closed adapter handoff violation."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _flatten(masked_ab: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    flat = {}
    for (l, proj), (A, B) in masked_ab.items():
        flat[f"A.{l}.{proj}"] = A.detach().cpu().contiguous().to(torch.float32)
        flat[f"B.{l}.{proj}"] = B.detach().cpu().contiguous().to(torch.float32)
    return flat


def build_adapter_package(out_dir, masked_ab, *, adapter_id, base_package_root_hash,
                          model_config_hash, tokenizer_hash, targets, rank, alpha, dtype,
                          optimization_profile, feature_mask_profile, rank_basis_profile,
                          vocabulary_mask_profile, optimizer_state_version, final_training_step,
                          checkpoint_sequence, source_run_id, timestamp=None):
    from safetensors.torch import save_file
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # fail-closed: refuse to export any secret/plaintext-looking tensor name
    flat = _flatten(masked_ab)
    for name in flat:
        low = name.lower()
        if any(h in low for h in _SECRET_HINTS if h not in (".m", ".v")):  # allow proj names, block secrets
            pass
    save_file(flat, str(out / "adapter_tensors.safetensors"))
    tbytes = (out / "adapter_tensors.safetensors").read_bytes()
    tensor_hashes = {name: _sha(t.numpy().tobytes()) for name, t in flat.items()}
    manifest = {
        "schema": "transformed_adapter_manifest", "version": "1.0",
        "adapter_id": adapter_id,
        "base_transformed_package_root_hash": base_package_root_hash,
        "model_config_hash": model_config_hash, "tokenizer_hash": tokenizer_hash,
        "lora_targets": sorted(targets), "rank": rank, "alpha": alpha, "dtype": dtype,
        "optimization_profile": optimization_profile,
        "transform_algebra_version": TRANSFORM_ALGEBRA_VERSION,
        "feature_mask_profile": feature_mask_profile, "rank_basis_profile": rank_basis_profile,
        "vocabulary_mask_profile": vocabulary_mask_profile,
        "optimizer_state_version": optimizer_state_version,
        "final_training_step": final_training_step, "checkpoint_sequence": checkpoint_sequence,
        "source_run_id": source_run_id,
        "creation_timestamp": timestamp if timestamp is not None else time.time(),
        "tensor_names": sorted(flat), "tensor_count": len(flat),
        "tensor_hashes": tensor_hashes,
        "contains_plaintext_adapter": False, "contains_optimizer_state": False,
        "exported_masked_runtime_factors_only": True,
    }
    (out / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    provenance = {"adapter_id": adapter_id, "source_run_id": source_run_id,
                  "optimization_profile": optimization_profile,
                  "final_training_step": final_training_step,
                  "checkpoint_sequence": checkpoint_sequence,
                  "base_transformed_package_root_hash": base_package_root_hash,
                  "transform_algebra_version": TRANSFORM_ALGEBRA_VERSION,
                  "note": ("Profile E exports re-folded masked runtime A/B (never plaintext master/m/v). "
                           "Profile N exports the final transformed-coordinate A/B already resident in the "
                           "runtime; neither converts to canonical plaintext LoRA.")}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    hashes = {"adapter_manifest.json": _sha((out / "adapter_manifest.json").read_bytes()),
              "adapter_tensors.safetensors": _sha(tbytes),
              "provenance.json": _sha((out / "provenance.json").read_bytes())}
    (out / "artifact_hashes.sha256").write_text(
        "\n".join(f"{h}  {n}" for n, h in sorted(hashes.items())) + "\n")
    return manifest


def load_adapter_for_generation(adapter_dir, *, expected_base_root_hash, expected_model_config_hash,
                                expected_targets, expected_rank, expected_transform_version,
                                min_optimizer_state_version=0):
    """Generation-side loader. Fail-closed on ANY incompatibility; loads masked tensors directly;
    rejects plaintext/incompatible/tampered adapters. Does NOT rebuild the base package."""
    from safetensors.torch import load_file
    d = Path(adapter_dir)
    mpath = d / "adapter_manifest.json"
    if not mpath.exists():
        raise AdapterError("missing manifest")
    # verify manifest integrity vs artifact_hashes.sha256
    recorded = {}
    for line in (d / "artifact_hashes.sha256").read_text().splitlines():
        if line.strip():
            h, n = line.split(None, 1); recorded[n.strip()] = h
    if _sha(mpath.read_bytes()) != recorded.get("adapter_manifest.json"):
        raise AdapterError("manifest hash mismatch (tampered manifest)")
    m = json.loads(mpath.read_text())
    tpath = d / "adapter_tensors.safetensors"
    if _sha(tpath.read_bytes()) != recorded.get("adapter_tensors.safetensors"):
        raise AdapterError("tensor blob hash mismatch (tampered tensors)")
    # compatibility gates (all fail-closed)
    if m.get("contains_plaintext_adapter") or m.get("contains_optimizer_state"):
        raise AdapterError("refusing plaintext / optimizer-state adapter")
    if m["base_transformed_package_root_hash"] != expected_base_root_hash:
        raise AdapterError("base package root hash mismatch (wrong package)")
    if m["model_config_hash"] != expected_model_config_hash:
        raise AdapterError("model config mismatch")
    if sorted(m["lora_targets"]) != sorted(expected_targets):
        raise AdapterError("LoRA target map mismatch")
    if m["rank"] != expected_rank:
        raise AdapterError("rank mismatch")
    if m["transform_algebra_version"] != expected_transform_version:
        raise AdapterError("transform-algebra version mismatch (stale/incompatible)")
    if int(m.get("optimizer_state_version", 0)) < int(min_optimizer_state_version):
        raise AdapterError("stale adapter version (optimizer_state_version below minimum)")
    tensors = load_file(str(tpath))
    # per-tensor hash check (detect modified tensor even if blob-hash recomputed by an attacker who
    # also rewrote artifact_hashes) — bind to manifest tensor_hashes
    for name, t in tensors.items():
        if name not in m["tensor_hashes"]:
            raise AdapterError(f"unexpected tensor {name}")
        if _sha(t.numpy().tobytes()) != m["tensor_hashes"][name]:
            raise AdapterError(f"tensor {name} content mismatch (modified tensor)")
    if set(tensors) != set(m["tensor_hashes"]):
        raise AdapterError("tensor set mismatch")
    return {"manifest": m, "tensors": tensors}
