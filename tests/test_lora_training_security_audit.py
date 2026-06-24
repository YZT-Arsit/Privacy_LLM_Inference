"""Training-stage security audit tests for protected LoRA training.

Verifies that a clean protected-training GPU trace exposes none of the sensitive
artifacts, and that the audit catches each kind of injected leak (training data,
labels, LoRA A/B, gradients, optimizer state, delta_W, plaintext hidden states,
mask secrets, forbidden field names). numpy only.

Run: python -m pytest tests/test_lora_training_security_audit.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pllo.experiments.lora_training_protection import run_synthetic_linear
from pllo.protocol.lora_training_audit import (
    LoRAMaskedInitRequest,
    LoRAMaskedMatmulRequest,
    audit_lora_training_trace,
)


@pytest.fixture(scope="module")
def run():
    return run_synthetic_linear(8)


def _audit(run):
    return audit_lora_training_trace(
        run["trace"], plaintext=run["plaintext"],
        secrets=run["secrets"].secret_dict())


def _leak_msg(arr) -> LoRAMaskedMatmulRequest:
    arr = np.asarray(arr, dtype=np.float64)
    arr2d = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr.reshape(1, -1)
    return LoRAMaskedMatmulRequest(
        session_id="x", layer="proj", step=99, phase="forward",
        masked_input=arr2d, batch_size=arr2d.shape[0],
        in_features=arr2d.shape[1], out_features=arr2d.shape[1])


def test_clean_trace_audit_passes(run) -> None:
    rep = _audit(run)
    assert rep.audit_passed is True
    assert rep.gpu_visible_train_examples is False
    assert rep.gpu_visible_labels is False
    assert rep.gpu_visible_input_ids is False
    assert rep.gpu_visible_lora_a is False
    assert rep.gpu_visible_lora_b is False
    assert rep.gpu_visible_delta_w is False
    assert rep.gpu_visible_lora_grad_a is False
    assert rep.gpu_visible_lora_grad_b is False
    assert rep.gpu_visible_optimizer_state is False
    assert rep.gpu_visible_adapter_update is False
    assert rep.gpu_visible_plain_hidden is False
    assert rep.gpu_visible_recovered_logits is False
    assert rep.leaked_secret_fields == []
    assert rep.forbidden_field_names == []
    assert rep.tee_used_on_gpu is False


def test_audit_detects_leaked_train_examples(run) -> None:
    run["trace"].inbound.append(_leak_msg(run["plaintext"]["train_examples"]))
    rep = _audit(run)
    assert rep.gpu_visible_train_examples is True
    assert rep.audit_passed is False
    run["trace"].inbound.pop()


def test_audit_detects_leaked_labels(run) -> None:
    run["trace"].inbound.append(_leak_msg(run["plaintext"]["labels"]))
    rep = _audit(run)
    assert rep.gpu_visible_labels is True
    assert rep.audit_passed is False
    run["trace"].inbound.pop()


def test_audit_detects_leaked_lora_params(run) -> None:
    run["trace"].inbound.append(_leak_msg(run["plaintext"]["lora_a"][-1]))
    run["trace"].inbound.append(_leak_msg(run["plaintext"]["lora_b"][-1]))
    rep = _audit(run)
    assert rep.gpu_visible_lora_a is True
    assert rep.gpu_visible_lora_b is True
    assert rep.audit_passed is False
    run["trace"].inbound.pop()
    run["trace"].inbound.pop()


def test_audit_detects_leaked_gradients_and_optimizer(run) -> None:
    run["trace"].inbound.append(_leak_msg(run["plaintext"]["lora_grad_a"][-1]))
    run["trace"].inbound.append(_leak_msg(run["plaintext"]["optimizer_state"][-1]))
    rep = _audit(run)
    assert rep.gpu_visible_lora_grad_a is True
    assert rep.gpu_visible_optimizer_state is True
    assert rep.audit_passed is False
    run["trace"].inbound.pop()
    run["trace"].inbound.pop()


def test_audit_detects_leaked_delta_w_and_hidden(run) -> None:
    run["trace"].inbound.append(_leak_msg(run["plaintext"]["delta_w"][-1]))
    run["trace"].inbound.append(_leak_msg(run["plaintext"]["plain_hidden"][-1]))
    rep = _audit(run)
    assert rep.gpu_visible_delta_w is True
    assert rep.gpu_visible_plain_hidden is True
    assert rep.audit_passed is False
    run["trace"].inbound.pop()
    run["trace"].inbound.pop()


def test_audit_detects_secret_leak(run) -> None:
    secret = run["secrets"].in_perm
    run["trace"].inbound.append(LoRAMaskedInitRequest(
        session_id="x", folded_base_weights={"leak": np.asarray(secret)},
        public_metadata={}))
    rep = _audit(run)
    assert "in_perm" in rep.leaked_secret_fields
    assert rep.audit_passed is False
    run["trace"].inbound.pop()


def test_audit_detects_forbidden_field_name(run) -> None:
    run["trace"].inbound.append(LoRAMaskedInitRequest(
        session_id="x", folded_base_weights={},
        public_metadata={"input_ids": [1, 2, 3]}))   # forbidden key name
    rep = _audit(run)
    assert "input_ids" in rep.forbidden_field_names
    assert rep.audit_passed is False
    run["trace"].inbound.pop()
