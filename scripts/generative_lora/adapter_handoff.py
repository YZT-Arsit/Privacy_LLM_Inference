"""Production transformed-adapter package builder and fail-closed generation loader."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

TRANSFORM_ALGEBRA_VERSION = "private_base_fold_v1.0"


class AdapterError(RuntimeError):
    """Fail-closed adapter handoff violation."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _flatten(masked_ab: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]):
    flat = {}
    for (layer, projection), (a_factor, b_factor) in masked_ab.items():
        flat[f"A.{layer}.{projection}"] = a_factor.detach().cpu().contiguous().float()
        flat[f"B.{layer}.{projection}"] = b_factor.detach().cpu().contiguous().float()
    return flat


def build_adapter_package(out_dir, masked_ab, *, adapter_id, base_package_root_hash,
                          model_config_hash, tokenizer_hash, targets, rank, alpha, dtype,
                          optimization_profile, feature_mask_profile, rank_basis_profile,
                          vocabulary_mask_profile, optimizer_state_version, final_training_step,
                          checkpoint_sequence, source_run_id, timestamp=None):
    """Export only transformed runtime A/B factors and bind every file and tensor by hash."""
    from safetensors.torch import save_file

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise AdapterError(f"refusing to overwrite non-empty adapter package: {out}")
    out.mkdir(parents=True, exist_ok=True)
    flat = _flatten(masked_ab)
    if not flat:
        raise AdapterError("refusing empty adapter package")
    forbidden = ("master", "optimizer", "adamw", "plaintext", "delta", "moment", "mask_secret")
    if any(any(hint in name.lower() for hint in forbidden) for name in flat):
        raise AdapterError("forbidden plaintext/optimizer/secret tensor name")

    tensor_path = out / "adapter_tensors.safetensors"
    save_file(flat, str(tensor_path))
    tensor_hashes = {name: _sha(tensor.numpy().tobytes()) for name, tensor in flat.items()}
    manifest = {
        "schema": "transformed_adapter_manifest",
        "version": "1.0",
        "adapter_id": adapter_id,
        "base_transformed_package_root_hash": base_package_root_hash,
        "model_config_hash": model_config_hash,
        "tokenizer_hash": tokenizer_hash,
        "lora_targets": sorted(targets),
        "rank": rank,
        "alpha": alpha,
        "dtype": dtype,
        "optimization_profile": optimization_profile,
        "transform_algebra_version": TRANSFORM_ALGEBRA_VERSION,
        "feature_mask_profile": feature_mask_profile,
        "rank_basis_profile": rank_basis_profile,
        "vocabulary_mask_profile": vocabulary_mask_profile,
        "optimizer_state_version": optimizer_state_version,
        "final_training_step": final_training_step,
        "checkpoint_sequence": checkpoint_sequence,
        "source_run_id": source_run_id,
        "creation_timestamp": timestamp if timestamp is not None else time.time(),
        "tensor_names": sorted(flat),
        "tensor_count": len(flat),
        "tensor_hashes": tensor_hashes,
        "contains_plaintext_adapter": False,
        "contains_optimizer_state": False,
        "exported_masked_runtime_factors_only": True,
    }
    manifest_path = out / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    provenance = {
        "adapter_id": adapter_id,
        "source_run_id": source_run_id,
        "optimization_profile": optimization_profile,
        "final_training_step": final_training_step,
        "checkpoint_sequence": checkpoint_sequence,
        "base_transformed_package_root_hash": base_package_root_hash,
        "transform_algebra_version": TRANSFORM_ALGEBRA_VERSION,
        "note": "Exports re-folded masked runtime A/B only; never plaintext master, moments, or DeltaW.",
    }
    provenance_path = out / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    hashes = {
        "adapter_manifest.json": _sha(manifest_path.read_bytes()),
        "adapter_tensors.safetensors": _sha(tensor_path.read_bytes()),
        "provenance.json": _sha(provenance_path.read_bytes()),
    }
    (out / "artifact_hashes.sha256").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n")
    return manifest


def load_adapter_for_generation(adapter_dir, *, expected_base_root_hash, expected_model_config_hash,
                                expected_targets, expected_rank, expected_transform_version,
                                min_optimizer_state_version=0):
    """Verify package integrity/compatibility and return transformed tensors without reconstruction."""
    from safetensors.torch import load_file

    directory = Path(adapter_dir)
    required = {"adapter_manifest.json", "adapter_tensors.safetensors",
                "artifact_hashes.sha256", "provenance.json"}
    missing_files = sorted(name for name in required if not (directory / name).is_file())
    if missing_files:
        raise AdapterError(f"missing package files: {missing_files}")

    recorded = {}
    for line in (directory / "artifact_hashes.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            recorded[name.strip()] = digest
    manifest_path = directory / "adapter_manifest.json"
    tensor_path = directory / "adapter_tensors.safetensors"
    if _sha(manifest_path.read_bytes()) != recorded.get("adapter_manifest.json"):
        raise AdapterError("manifest hash mismatch (tampered manifest)")
    if _sha(tensor_path.read_bytes()) != recorded.get("adapter_tensors.safetensors"):
        raise AdapterError("tensor blob hash mismatch (tampered tensors)")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contains_plaintext_adapter") or manifest.get("contains_optimizer_state"):
        raise AdapterError("refusing plaintext / optimizer-state adapter")
    if manifest["base_transformed_package_root_hash"] != expected_base_root_hash:
        raise AdapterError("base package root hash mismatch (wrong package)")
    if manifest["model_config_hash"] != expected_model_config_hash:
        raise AdapterError("model config mismatch")
    if sorted(manifest["lora_targets"]) != sorted(expected_targets):
        raise AdapterError("LoRA target map mismatch")
    if manifest["rank"] != expected_rank:
        raise AdapterError("rank mismatch")
    if manifest["transform_algebra_version"] != expected_transform_version:
        raise AdapterError("transform-algebra version mismatch (stale/incompatible)")
    if int(manifest.get("optimizer_state_version", 0)) < int(min_optimizer_state_version):
        raise AdapterError("stale adapter version (optimizer_state_version below minimum)")

    tensors = load_file(str(tensor_path))
    forbidden = ("master", "optimizer", "adamw", "plaintext", "delta", "moment", "mask_secret")
    for name, tensor in tensors.items():
        if any(hint in name.lower() for hint in forbidden):
            raise AdapterError(f"forbidden tensor {name}")
        if name not in manifest["tensor_hashes"]:
            raise AdapterError(f"unexpected tensor {name}")
        if _sha(tensor.numpy().tobytes()) != manifest["tensor_hashes"][name]:
            raise AdapterError(f"tensor {name} content mismatch (modified tensor)")
    if set(tensors) != set(manifest["tensor_hashes"]):
        raise AdapterError("tensor set mismatch")
    if sorted(tensors) != sorted(manifest.get("tensor_names", [])):
        raise AdapterError("tensor name registry mismatch")
    return {"manifest": manifest, "tensors": tensors}


__all__ = ["AdapterError", "TRANSFORM_ALGEBRA_VERSION",
           "build_adapter_package", "load_adapter_for_generation"]
