"""Validation and minimal checkpoint helpers for the frozen v2 contract."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import torch

from .optimizer import TransformedAdamW

CONTRACT_ID = "OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1"


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def build_checkpoint(
    optimizer: TransformedAdamW,
    *,
    direction: str,
    rank: int,
    alpha: float,
    base_hash: str,
    config_hash: str,
    transform_generation: str,
) -> dict[str, Any]:
    if direction not in {"input", "output"}:
        raise ValueError("invalid direction")
    state = optimizer.state_dict()
    inventory = {
        group: {
            name: {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": tensor_sha256(t)}
            for name, t in state[group].items()
        }
        for group in ("parameters", "m", "v")
    }
    return {
        "schema": "obfuscatune_lora_v2_minimal_checkpoint_v1",
        "contract_id": CONTRACT_ID,
        "binding": {"direction": direction, "rank": rank, "alpha": alpha, "base_hash": base_hash, "config_hash": config_hash, "transform_generation": transform_generation},
        "optimizer": state,
        "inventory": inventory,
        "contains_transform_secret": False,
        "contains_canonical_factors": False,
    }


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    Path(path).write_bytes(buffer.getvalue())


def load_checkpoint(path: str | Path, *, expected_binding: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "obfuscatune_lora_v2_minimal_checkpoint_v1" or checkpoint.get("contract_id") != CONTRACT_ID:
        raise ValueError("checkpoint schema or contract mismatch")
    if checkpoint.get("binding") != expected_binding:
        raise ValueError("checkpoint binding mismatch")
    if checkpoint.get("contains_transform_secret") or checkpoint.get("contains_canonical_factors"):
        raise ValueError("forbidden plaintext/transform material")
    state = checkpoint.get("optimizer", {})
    inventory = checkpoint.get("inventory", {})
    if set(inventory) != {"parameters", "m", "v"}:
        raise ValueError("checkpoint inventory groups mismatch")
    for group in ("parameters", "m", "v"):
        if set(inventory[group]) != set(state.get(group, {})):
            raise ValueError("checkpoint tensor inventory mismatch")
        for name, tensor in state[group].items():
            meta = inventory[group][name]
            if list(tensor.shape) != meta["shape"] or str(tensor.dtype) != meta["dtype"] or tensor_sha256(tensor) != meta["sha256"]:
                raise ValueError(f"checkpoint tensor validation failed: {group}.{name}")
    return checkpoint


def adamw_rotation_counterexample(dtype=torch.float64) -> dict[str, Any]:
    root2 = 2.0**0.5
    rotation = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=dtype) / root2
    theta = torch.tensor([1.0, 2.0], dtype=dtype)
    gradient = torch.tensor([0.3, -0.7], dtype=dtype)
    h = {"lr": 0.1, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.01}

    def one_step(value: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        m = (1 - h["beta1"]) * grad
        v = (1 - h["beta2"]) * grad.square()
        m_hat = m / (1 - h["beta1"])
        v_hat = v / (1 - h["beta2"])
        return value * (1 - h["lr"] * h["weight_decay"]) - h["lr"] * m_hat / (v_hat.sqrt() + h["eps"])

    canonical = one_step(theta, gradient)
    mapped_back = rotation.transpose(0, 1) @ one_step(rotation @ theta, rotation @ gradient)
    return {"canonical": canonical, "mapped_back": mapped_back, "max_abs_difference": float((canonical - mapped_back).abs().max())}
