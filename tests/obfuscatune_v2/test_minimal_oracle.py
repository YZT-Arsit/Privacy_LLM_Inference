from __future__ import annotations

import copy

import pytest
import torch

from pllo.baselines.obfuscatune_lora_v2.backward import transformed_lora_backward
from pllo.baselines.obfuscatune_lora_v2.lora_layers import (
    plaintext_lora_forward,
    transformed_lora_forward,
)
from pllo.baselines.obfuscatune_lora_v2.optimizer import (
    AdamWHyperparameters,
    TransformedAdamW,
)
from pllo.baselines.obfuscatune_lora_v2.transforms import (
    orthogonal_matrix,
    restore_factors,
    restore_input_gradient,
    restore_output,
    transform_base,
    transform_factors,
    transform_input,
    transform_output_gradient,
)
from pllo.baselines.obfuscatune_lora_v2.validation import (
    adamw_rotation_counterexample,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


DTYPE = torch.float64


def fixture(seed=11, d_in=5, d_out=4, rank=2, n=3):
    generator = torch.Generator().manual_seed(seed)
    values = {
        "x": torch.randn(n, d_in, generator=generator, dtype=DTYPE),
        "w": torch.randn(d_in, d_out, generator=generator, dtype=DTYPE),
        "a": torch.randn(d_in, rank, generator=generator, dtype=DTYPE),
        "b": torch.randn(rank, d_out, generator=generator, dtype=DTYPE),
        "g": torch.randn(n, d_out, generator=generator, dtype=DTYPE),
    }
    values["ri"] = orthogonal_matrix(d_in, seed + 1, dtype=DTYPE)
    values["ro"] = orthogonal_matrix(d_out, seed + 2, dtype=DTYPE)
    return values


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_dense_linear_forward(seed):
    f = fixture(seed)
    x_star = transform_input(f["x"], f["ri"])
    input_result = x_star @ transform_base(f["w"], f["ri"], "input")
    torch.testing.assert_close(input_result, f["x"] @ f["w"], atol=1e-11, rtol=1e-11)
    output_star = f["x"] @ transform_base(f["w"], f["ro"], "output")
    torch.testing.assert_close(restore_output(output_star, f["ro"]), f["x"] @ f["w"], atol=1e-11, rtol=1e-11)


def test_dense_linear_backward():
    f = fixture()
    w_star = transform_base(f["w"], f["ri"], "input")
    dx_star = f["g"] @ w_star.T
    torch.testing.assert_close(restore_input_gradient(dx_star, f["ri"]), f["g"] @ f["w"].T, atol=1e-11, rtol=1e-11)
    w_out = transform_base(f["w"], f["ro"], "output")
    g_star = transform_output_gradient(f["g"], f["ro"])
    torch.testing.assert_close(g_star @ w_out.T, f["g"] @ f["w"].T, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("direction", ["input", "output"])
def test_lora_forward(direction):
    f = fixture()
    rotation = f["ri"] if direction == "input" else f["ro"]
    w_star = transform_base(f["w"], rotation, direction)
    a_star, b_star = transform_factors(f["a"], f["b"], rotation, direction)
    scale = 1.75
    if direction == "input":
        result = transformed_lora_forward(transform_input(f["x"], rotation), w_star, a_star, b_star, scale)
    else:
        result = restore_output(transformed_lora_forward(f["x"], w_star, a_star, b_star, scale), rotation)
    oracle = plaintext_lora_forward(f["x"], f["w"], f["a"], f["b"], scale)
    torch.testing.assert_close(result, oracle, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("direction", ["input", "output"])
def test_lora_da_db(direction):
    f = fixture()
    rotation = f["ri"] if direction == "input" else f["ro"]
    x_stored = transform_input(f["x"], rotation) if direction == "input" else f["x"]
    g_stored = f["g"] if direction == "input" else transform_output_gradient(f["g"], rotation)
    w_star = transform_base(f["w"], rotation, direction)
    a_star, b_star = transform_factors(f["a"], f["b"], rotation, direction)
    x_var = x_stored.clone().requires_grad_()
    a_var = a_star.clone().requires_grad_()
    b_var = b_star.clone().requires_grad_()
    y = transformed_lora_forward(x_var, w_star, a_var, b_var, 1.75)
    (y * g_stored).sum().backward()
    manual = transformed_lora_backward(x_stored, g_stored, w_star, a_star, b_star, 1.75)
    torch.testing.assert_close(manual.da, a_var.grad, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(manual.db, b_var.grad, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("direction", ["input", "output"])
def test_input_gradient(direction):
    f = fixture()
    rotation = f["ri"] if direction == "input" else f["ro"]
    x_stored = transform_input(f["x"], rotation) if direction == "input" else f["x"]
    g_stored = f["g"] if direction == "input" else transform_output_gradient(f["g"], rotation)
    w_star = transform_base(f["w"], rotation, direction)
    a_star, b_star = transform_factors(f["a"], f["b"], rotation, direction)
    manual = transformed_lora_backward(x_stored, g_stored, w_star, a_star, b_star, 1.75)
    canonical = f["g"] @ (f["w"] + 1.75 * f["a"] @ f["b"]).T
    actual = restore_input_gradient(manual.dx, rotation) if direction == "input" else manual.dx
    torch.testing.assert_close(actual, canonical, atol=1e-11, rtol=1e-11)


def _torch_adamw_oracle(initial, gradients, hparams):
    value = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([value], lr=hparams.lr, betas=(hparams.beta1, hparams.beta2), eps=hparams.eps, weight_decay=hparams.weight_decay)
    for gradient in gradients:
        value.grad = gradient.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return value.detach()


def test_one_optimizer_step_matches_stored_coordinate_oracle():
    initial = torch.tensor([[0.2, -0.4], [0.8, 0.1]], dtype=DTYPE)
    gradient = torch.tensor([[0.3, -0.7], [0.2, 0.5]], dtype=DTYPE)
    h = AdamWHyperparameters(lr=0.01, weight_decay=0.02)
    parameter = initial.clone()
    optimizer = TransformedAdamW({"A_star": parameter}, h)
    optimizer.step({"A_star": gradient})
    torch.testing.assert_close(parameter, _torch_adamw_oracle(initial, [gradient], h), atol=1e-13, rtol=1e-13)


def test_ten_optimizer_steps_match_stored_coordinate_oracle():
    generator = torch.Generator().manual_seed(99)
    initial = torch.randn(3, 2, generator=generator, dtype=DTYPE)
    gradients = [torch.randn(3, 2, generator=generator, dtype=DTYPE) for _ in range(10)]
    h = AdamWHyperparameters(lr=0.003, weight_decay=0.01)
    parameter = initial.clone()
    optimizer = TransformedAdamW({"B_star": parameter}, h)
    for gradient in gradients:
        optimizer.step({"B_star": gradient})
    torch.testing.assert_close(parameter, _torch_adamw_oracle(initial, gradients, h), atol=2e-13, rtol=2e-13)


def test_zero_gradient_weight_decay():
    initial = torch.tensor([[1.0, -2.0]], dtype=DTYPE)
    h = AdamWHyperparameters(lr=0.1, weight_decay=0.01)
    parameter = initial.clone()
    optimizer = TransformedAdamW({"A_star": parameter}, h)
    optimizer.step({"A_star": torch.zeros_like(parameter)})
    torch.testing.assert_close(parameter, initial * (1 - h.lr * h.weight_decay), atol=1e-14, rtol=1e-14)


def test_checkpoint_round_trip_and_binding_rejection(tmp_path):
    f = fixture()
    a_star, b_star = transform_factors(f["a"], f["b"], f["ri"], "input")
    optimizer = TransformedAdamW({"A_star": a_star.clone(), "B_star": b_star.clone()})
    optimizer.step({"A_star": torch.ones_like(a_star), "B_star": torch.ones_like(b_star)})
    binding = {"direction": "input", "rank": 2, "alpha": 4.0, "base_hash": "base", "config_hash": "config", "transform_generation": "g1"}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, build_checkpoint(optimizer, **binding))
    restored = load_checkpoint(path, expected_binding=binding)
    assert restored["optimizer"]["step"] == 1
    assert not restored["contains_transform_secret"] and not restored["contains_canonical_factors"]
    wrong = copy.deepcopy(binding)
    wrong["transform_generation"] = "stale"
    with pytest.raises(ValueError, match="binding mismatch"):
        load_checkpoint(path, expected_binding=wrong)


def test_checkpoint_corrupted_tensor_rejected(tmp_path):
    f = fixture()
    a_star, b_star = transform_factors(f["a"], f["b"], f["ri"], "input")
    optimizer = TransformedAdamW({"A_star": a_star.clone(), "B_star": b_star.clone()})
    binding = {"direction": "input", "rank": 2, "alpha": 4.0, "base_hash": "base", "config_hash": "config", "transform_generation": "g1"}
    checkpoint = build_checkpoint(optimizer, **binding)
    checkpoint["optimizer"]["parameters"]["A_star"][0, 0] += 1
    path = tmp_path / "corrupted.pt"
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match="tensor validation failed"):
        load_checkpoint(path, expected_binding=binding)


def test_adamw_rotation_equivalence_claim_is_rejected():
    result = adamw_rotation_counterexample()
    assert result["max_abs_difference"] > 0.09
    with pytest.raises(AssertionError):
        torch.testing.assert_close(result["canonical"], result["mapped_back"], atol=1e-8, rtol=1e-8)
