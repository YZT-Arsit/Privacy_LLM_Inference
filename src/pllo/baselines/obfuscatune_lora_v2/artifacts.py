"""Fail-closed checkpoints and transformed-adapter export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .optimizer import TransformedAdamW

SCHEMA = "obfuscatune_style_adapted_lora_v2_artifact_v1"
CONTRACT = "OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1"


def _tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _inventory(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _tensor_hash(value)}
            for name, value in sorted(tensors.items())}


def _validate_names(tensors: dict[str, torch.Tensor]) -> None:
    if not tensors or any(not name.endswith((".a_star", ".b_star")) for name in tensors):
        raise ValueError("adapter must contain only transformed a_star/b_star tensors")
    forbidden = ("plaintext", "canonical", "rotation", "mask", "inverse", "gamma", "optimizer")
    if any(any(token in name.lower() for token in forbidden) for name in tensors):
        raise ValueError("forbidden adapter tensor name")


def save_training_checkpoint(
    path: str | Path,
    *,
    parameters: dict[str, torch.Tensor],
    optimizer: TransformedAdamW,
    binding: dict[str, Any],
    step: int,
    counters: dict[str, int],
) -> str:
    _validate_names(parameters)
    if int(step) != optimizer.step_index:
        raise ValueError("checkpoint/optimizer step mismatch")
    payload = {
        "schema": SCHEMA,
        "kind": "training_checkpoint",
        "contract_id": CONTRACT,
        "binding": dict(binding),
        "step": int(step),
        "inventory": _inventory(parameters),
        "optimizer": optimizer.state_dict(),
        "counters": dict(counters),
        "contains_transform_secret": False,
        "contains_canonical_factor": False,
    }
    path = Path(path)
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_checkpoint(
    path: str | Path,
    *,
    parameters: dict[str, torch.Tensor],
    optimizer: TransformedAdamW,
    expected_binding: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("kind") != "training_checkpoint" or payload.get("contract_id") != CONTRACT:
        raise ValueError("checkpoint schema/contract mismatch")
    if payload.get("binding") != expected_binding:
        raise ValueError("checkpoint binding mismatch")
    if payload.get("contains_transform_secret") or payload.get("contains_canonical_factor"):
        raise ValueError("forbidden checkpoint material")
    state = payload.get("optimizer", {})
    stored = state.get("parameters", {})
    if set(stored) != set(parameters) or payload.get("inventory") != _inventory(stored):
        raise ValueError("checkpoint inventory/hash mismatch")
    optimizer.load_state_dict(state)
    return payload


def export_adapter(
    directory: str | Path,
    *,
    parameters: dict[str, torch.Tensor],
    binding: dict[str, Any],
    final_step: int,
) -> dict[str, Any]:
    _validate_names(parameters)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tensor_path = directory / "adapter_tensors.safetensors"
    cpu = {name: value.detach().cpu().contiguous() for name, value in parameters.items()}
    save_file(cpu, tensor_path)
    manifest = {
        "schema": SCHEMA,
        "kind": "transformed_adapter",
        "paper_facing_name": "OBFUSCATUNE_STYLE_ADAPTED_LORA_BASELINE",
        "contract_id": CONTRACT,
        "binding": dict(binding),
        "final_step": int(final_step),
        "inventory": _inventory(cpu),
        "tensor_file_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "contains_transform_secret": False,
        "contains_canonical_factor": False,
        "optimizer_state_included": False,
    }
    manifest_path = directory / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_adapter(directory: str | Path, *, expected_binding: dict[str, Any]) -> dict[str, torch.Tensor]:
    directory = Path(directory)
    manifest_path = directory / "adapter_manifest.json"
    tensor_path = directory / "adapter_tensors.safetensors"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("kind") != "transformed_adapter" or manifest.get("contract_id") != CONTRACT:
        raise ValueError("adapter schema/contract mismatch")
    if manifest.get("binding") != expected_binding:
        raise ValueError("adapter binding mismatch")
    if manifest.get("contains_transform_secret") or manifest.get("contains_canonical_factor") or manifest.get("optimizer_state_included"):
        raise ValueError("forbidden adapter material")
    if hashlib.sha256(tensor_path.read_bytes()).hexdigest() != manifest.get("tensor_file_sha256"):
        raise ValueError("adapter tensor-file hash mismatch")
    tensors = load_file(tensor_path)
    _validate_names(tensors)
    if manifest.get("inventory") != _inventory(tensors):
        raise ValueError("adapter tensor inventory/hash mismatch")
    return tensors


def apply_adapter(parameters: dict[str, torch.Tensor], tensors: dict[str, torch.Tensor]) -> None:
    if set(parameters) != set(tensors):
        raise ValueError("adapter target inventory mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != tensors[name].shape or parameter.dtype != tensors[name].dtype:
                raise ValueError(f"adapter metadata mismatch: {name}")
            parameter.copy_(tensors[name].to(parameter.device))
