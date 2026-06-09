"""Stage 7.6 tests — masked-gradient LoRA training."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import torch

from pllo.experiments.masked_gradient_lora_security_proxy import (
    MaskedGradientLoRASecurityProxyConfig,
    run_masked_gradient_lora_security_proxy,
)
from pllo.experiments.masked_gradient_lora_training import (
    render_markdown,
    run_masked_gradient_lora_training,
    write_reports,
)
from pllo.ops.masked_gradient_lora import (
    DenseMaskedAdamWUnsupported,
    MaskedGradientLoRAConfig,
    create_cancellation_padded_lora,
    create_masked_lora_state,
    create_orthogonal_matrix,
    dummy_contribution_norm,
    masked_adamw_step_unsupported,
    masked_lora_forward,
    masked_momentum_sgd_step,
    masked_sgd_step,
    recover_lora_from_masked,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


_REQUIRED_PHRASES = (
    "The GPU never receives plaintext LoRA adapters or plaintext LoRA "
    "gradients in this experiment.",
    "Masked SGD is algebraically equivalent under orthogonal masks.",
    "Dense masked AdamW is not claimed because coordinate-wise second "
    "moments are not invariant under dense orthogonal mixing.",
    "This is a CPU-only algebraic and proxy-leakage experiment, not a "
    "real TEE/GPU training benchmark.",
    "The user side does not require a GPU; the simulated cloud "
    "accelerator performs masked forward, backward, and optimizer "
    "updates.",
    "No formal, cryptographic, or semantic security is claimed.",
)


@pytest.fixture(scope="module")
def small_config() -> MaskedGradientLoRAConfig:
    return MaskedGradientLoRAConfig(
        d_in=8, d_out=4, true_rank=2, padded_rank=4,
        batch_size=4, lr=0.01, momentum=0.9,
        use_rank_padding=True, dummy_strategy="paired_cancellation",
        seed=0, dtype="float64",
    )


@pytest.fixture(scope="module")
def training_report(small_config) -> dict:
    return run_masked_gradient_lora_training(small_config, num_steps=4)


# 1. Small config runs end-to-end.
def test_small_synthetic_runs_end_to_end(training_report: dict) -> None:
    assert training_report["status"] == "ok"
    assert training_report["stage"] == "7.6"
    assert len(training_report["per_step"]) == 4


# 2. Forward masked output matches plaintext output after recovery.
def test_forward_recovery_equals_plain(training_report: dict) -> None:
    for ps in training_report["per_step"]:
        assert ps["forward_max_abs_err"] < 1e-10


# 3. Plain loss and masked orthogonal loss match.
def test_masked_loss_equals_plain_loss(training_report: dict) -> None:
    for ps in training_report["per_step"]:
        assert ps["loss_abs_err"] < 1e-12


# 4. grad_A_tilde relation holds.
def test_grad_A_tilde_relation(training_report: dict) -> None:
    for ps in training_report["per_step"]:
        assert ps["grad_A_tilde_relation_max_abs_err"] < 1e-10


# 5. grad_B_tilde relation holds.
def test_grad_B_tilde_relation(training_report: dict) -> None:
    for ps in training_report["per_step"]:
        assert ps["grad_B_tilde_relation_max_abs_err"] < 1e-10


# 6. Masked SGD update recovers to plaintext SGD update.
def test_masked_sgd_update_recovers_to_plain(training_report: dict) -> None:
    for ps in training_report["per_step"]:
        assert ps["masked_sgd_update_A_max_abs_err_after_recovery"] < 1e-10
        assert ps["masked_sgd_update_B_max_abs_err_after_recovery"] < 1e-10


# 7. Masked momentum SGD update recovers to plaintext momentum update.
def test_masked_momentum_sgd_update_recovers_to_plain(
    training_report: dict,
) -> None:
    for ps in training_report["per_step"]:
        assert (
            ps["masked_momentum_sgd_update_A_max_abs_err_after_recovery"]
            < 1e-10
        )
        assert (
            ps["masked_momentum_sgd_update_B_max_abs_err_after_recovery"]
            < 1e-10
        )


# 8. AdamW dense-mask exactness is explicitly unsupported / raises.
def test_adamw_dense_mask_explicitly_unsupported(
    training_report: dict,
) -> None:
    record = training_report["adamw_dense_mask_unsupported"]
    assert record["status"] == "explicitly_raised_as_designed"
    assert record["exception_type"] == "DenseMaskedAdamWUnsupported"
    # Direct call must raise.
    with pytest.raises(DenseMaskedAdamWUnsupported):
        masked_adamw_step_unsupported()


# 9. Rank padding cancellation preserves A B (at init).
def test_rank_padding_cancellation_preserves_product_at_init() -> None:
    g = torch.Generator(device="cpu").manual_seed(7)
    A = torch.randn(6, 2, dtype=torch.float64, generator=g)
    B = torch.randn(2, 4, dtype=torch.float64, generator=g)
    A_pad, B_pad, meta = create_cancellation_padded_lora(
        A, B, padded_rank=6, strategy="paired_cancellation", generator=g,
    )
    assert A_pad.shape == (6, 6)
    assert B_pad.shape == (6, 4)
    dummy_norm = dummy_contribution_norm(A_pad, B_pad, true_rank=2)
    assert dummy_norm < 1e-12
    assert meta["strategy"] == "paired_cancellation"


# 10. JSON / CSV / Markdown contain no raw tensors, no `tensor(`,
#     no overlong numeric arrays.
def test_outputs_have_no_raw_tensors(
    training_report: dict, tmp_path: Path,
) -> None:
    j, c, m = write_reports(training_report, outputs_dir=str(tmp_path))
    for path in (j, c, m):
        text = Path(path).read_text()
        assert "tensor(" not in text, f"{path} contains tensor() repr"
        long_arr = re.search(r"\[(\s*-?\d+(\.\d+)?\s*,\s*){50,}", text)
        assert long_arr is None, f"{path} has long numeric array"


# 11. Markdown contains all required honesty phrases.
def test_markdown_contains_required_honesty_phrases(
    training_report: dict,
) -> None:
    md = render_markdown(training_report)
    for phrase in _REQUIRED_PHRASES:
        assert phrase in md, f"missing honesty phrase: {phrase!r}"


# 12. Runner exits successfully.
def test_runner_exits_successfully() -> None:
    script = REPO_ROOT / "scripts" / "run_masked_gradient_lora_training.py"
    result = subprocess.run(
        ["python", str(script)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "status=ok" in result.stdout
    assert "adamw_status=explicitly_raised_as_designed" in result.stdout


# Bonus: direct module-level checks.
def test_module_level_forward_and_sgd_algebra() -> None:
    g = torch.Generator(device="cpu").manual_seed(42)
    d_in, d_out, r = 6, 4, 3
    X = torch.randn(8, d_in, dtype=torch.float64, generator=g)
    A = torch.randn(d_in, r, dtype=torch.float64, generator=g)
    B = torch.randn(r, d_out, dtype=torch.float64, generator=g)
    N_x = create_orthogonal_matrix(d_in, generator=g, dtype=torch.float64)
    N_y = create_orthogonal_matrix(d_out, generator=g, dtype=torch.float64)
    M = create_orthogonal_matrix(r, generator=g, dtype=torch.float64)
    state = create_masked_lora_state(
        A, B, N_x=N_x, N_y=N_y, M=M, padded_rank=r, true_rank=r,
    )
    X_tilde = X @ N_x
    Y_tilde = masked_lora_forward(X_tilde, state.A_tilde, state.B_tilde)
    Y_plain = X @ A @ B
    Y_recovered = Y_tilde @ N_y.transpose(-2, -1)
    assert (Y_recovered - Y_plain).abs().max().item() < 1e-12
    # Recovery of A, B from masked state.
    A_rec, B_rec = recover_lora_from_masked(
        state.A_tilde, state.B_tilde, N_x=N_x, N_y=N_y, M=M,
    )
    assert (A_rec - A).abs().max().item() < 1e-12
    assert (B_rec - B).abs().max().item() < 1e-12
    # SGD step.
    grad_A_tilde = torch.randn_like(state.A_tilde)
    grad_B_tilde = torch.randn_like(state.B_tilde)
    A_next, B_next = masked_sgd_step(
        state.A_tilde, state.B_tilde, grad_A_tilde, grad_B_tilde, lr=0.1,
    )
    # Recovery of update.
    grad_A_plain = N_x @ grad_A_tilde @ M.transpose(-2, -1)
    grad_B_plain = M @ grad_B_tilde @ N_y.transpose(-2, -1)
    A_next_rec = N_x @ A_next @ M.transpose(-2, -1)
    A_next_plain = A - 0.1 * grad_A_plain
    assert (A_next_rec - A_next_plain).abs().max().item() < 1e-12


def test_security_proxy_runs_and_publishes_labels(
    small_config: MaskedGradientLoRAConfig,
) -> None:
    proxy_cfg = MaskedGradientLoRASecurityProxyConfig(
        base=small_config, num_trials=2, num_steps=3,
        fresh_masks_per_step=True, fixed_masks_baseline=True,
    )
    rep = run_masked_gradient_lora_security_proxy(proxy_cfg)
    assert rep["status"] == "ok"
    assert rep["formal_security_claim"] is False
    assert len(rep["trials_fresh_masks"]) == 2
    assert len(rep["trials_fixed_masks_baseline"]) == 2
    # Labels must be from the conservative set.
    allowed = {
        "low_proxy_risk", "medium_proxy_risk", "high_proxy_risk",
        "needs_more_evaluation",
    }
    for t in rep["trials_fresh_masks"] + rep["trials_fixed_masks_baseline"]:
        assert t["rank_proxy_label"] in allowed
        assert t["linkability_label"] in allowed


def test_formal_security_claim_is_false(training_report: dict) -> None:
    assert training_report["formal_security_claim"] is False
    md = render_markdown(training_report)
    assert "`formal_security_claim`: `False`" in md


def test_dummy_strategy_recorded_in_metadata(training_report: dict) -> None:
    drp = training_report["dummy_rank_padding"]
    assert drp["strategy_used"] in ("paired_cancellation", "none")
    # At initialisation, cancellation must be at machine precision.
    assert drp["initial_dummy_contribution_norm"] < 1e-12


def test_gpu_visibility_table_marks_plaintext_invisible(
    training_report: dict,
) -> None:
    vis = training_report["gpu_visibility"]
    by_var = {entry["variable"]: entry for entry in vis}
    for var in (
        "plaintext_A", "plaintext_B",
        "plaintext_grad_A", "plaintext_grad_B",
        "plaintext_optimizer_state", "N_x / N_y / M",
    ):
        assert by_var[var]["visible_to_gpu"] is False, var
    for var in ("X_tilde", "A_tilde", "B_tilde",
                "grad_A_tilde", "grad_B_tilde"):
        assert by_var[var]["visible_to_gpu"] is True, var
