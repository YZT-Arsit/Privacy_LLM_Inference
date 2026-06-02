"""Stage 7.3 — tests for the multi-layer LoRA end-to-end training probe."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pllo.experiments.multilayer_lora_training import (
    MultiLayerLoRATrainingConfig,
    VALID_LORA_TARGETS,
    VALID_OPTIMIZERS,
    multilayer_lora_training_csv_rows,
    normalize_optimizer,
    run_multilayer_lora_training,
)
from pllo.model_zoo.tiny_lora_transformer import (
    TinyLoRATransformerConfig,
    init_base_weights,
    init_lora_adapters,
    model_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_multilayer_lora_training_experiments.py"


def _cfg(**overrides) -> MultiLayerLoRATrainingConfig:
    base = dict(
        seed=2026,
        num_layers=2,
        hidden_size=16,
        intermediate_size=24,
        vocab_size=32,
        seq_len=4,
        batch_size=2,
        true_rank=2,
        padded_rank=4,
        num_steps=2,
        lr=1e-2,
        optimizer="sgd",
        use_pad=True,
        fresh_u_per_step=True,
        dummy_strategy="paired_cancellation_dummy",
        dtype="float64",
    )
    base.update(overrides)
    return MultiLayerLoRATrainingConfig(**base)


# ---------------------------------------------------------------------------
# 1. Tiny multi-layer LoRA model forward shape correct
# ---------------------------------------------------------------------------


def test_tiny_lora_model_spec_shape() -> None:
    cfg = TinyLoRATransformerConfig(
        num_layers=2, hidden_size=16, intermediate_size=24,
        vocab_size=32, seq_len=4, batch_size=2,
        true_rank=2, padded_rank=4,
        dtype="float64",
    )
    spec = model_spec(cfg)
    assert spec["num_layers"] == 2
    assert spec["modules_per_layer"] == len(VALID_LORA_TARGETS)
    assert spec["total_lora_modules"] == 2 * len(VALID_LORA_TARGETS)
    # Every per-layer module has the right shape based on its name.
    for entry in spec["per_layer_modules"]:
        if entry["module_name"] in (
            "q_proj", "k_proj", "v_proj", "o_proj",
        ):
            assert entry["d_in"] == 16 and entry["d_out"] == 16
        elif entry["module_name"] in ("gate_proj", "up_proj"):
            assert entry["d_in"] == 16 and entry["d_out"] == 24
        elif entry["module_name"] == "down_proj":
            assert entry["d_in"] == 24 and entry["d_out"] == 16


def test_init_base_weights_and_adapters_shapes() -> None:
    cfg = TinyLoRATransformerConfig(
        num_layers=2, hidden_size=16, intermediate_size=24,
        vocab_size=32, seq_len=4, batch_size=2,
        true_rank=2, padded_rank=4,
        dtype="float64",
    )
    import torch as _torch
    gen = _torch.Generator(device="cpu").manual_seed(2026)
    base = init_base_weights(cfg, generator=gen)
    adapters = init_lora_adapters(cfg, generator=gen)
    # 2 layers × 7 modules + head
    assert len(base) == 2 * 7 + 1
    assert "head.W" in base
    assert base["head.W"].shape == (16, 32)
    assert len(adapters) == 2 * 7
    for k, v in adapters.items():
        assert v["a"].shape[1] == 2  # true_rank
        assert v["b"].shape[0] == 2


# ---------------------------------------------------------------------------
# 2. run_multilayer_lora_training small config runs
# ---------------------------------------------------------------------------


def test_run_multilayer_lora_training_runs() -> None:
    report = run_multilayer_lora_training(_cfg())
    assert report["lora_multilayer_training_status"] == "prototype"
    assert (
        report["security_profile"] == "proxy-evaluated, not formal"
    )


# ---------------------------------------------------------------------------
# 3. loss plain vs masked close
# ---------------------------------------------------------------------------


def test_loss_plain_vs_masked_allclose() -> None:
    report = run_multilayer_lora_training(_cfg())
    tc = report["training_correctness"]
    assert tc["allclose"] is True
    assert tc["max_loss_diff"] < 1e-9
    assert tc["max_dummy_contribution_norm"] < 1e-9


# ---------------------------------------------------------------------------
# 4. per-layer grad_A / grad_B close
# ---------------------------------------------------------------------------


def test_per_layer_grad_close() -> None:
    report = run_multilayer_lora_training(_cfg())
    tc = report["training_correctness"]
    assert tc["max_grad_a_real_err"] < 1e-7
    assert tc["max_grad_b_real_err"] < 1e-7
    for entry in report["per_layer_metrics"]:
        assert entry["grad_a_real_max_abs_err"] < 1e-7
        assert entry["grad_b_real_max_abs_err"] < 1e-7


# ---------------------------------------------------------------------------
# 5. per-layer adapter update close
# ---------------------------------------------------------------------------


def test_per_layer_adapter_update_close() -> None:
    report = run_multilayer_lora_training(_cfg())
    tc = report["training_correctness"]
    assert tc["max_update_a_err"] < 1e-7
    assert tc["max_update_b_err"] < 1e-7
    for entry in report["per_layer_metrics"]:
        assert entry["update_a_max_abs_err"] < 1e-7
        assert entry["update_b_max_abs_err"] < 1e-7


# ---------------------------------------------------------------------------
# 6. use_pad=True works (default) and use_pad=False also runs allclose
# ---------------------------------------------------------------------------


def test_use_pad_toggle() -> None:
    for use_pad in (True, False):
        report = run_multilayer_lora_training(_cfg(use_pad=use_pad))
        tc = report["training_correctness"]
        assert tc["allclose"] is True
        assert tc["max_loss_diff"] < 1e-9


# ---------------------------------------------------------------------------
# 7. rank padding enabled for every LoRA module
# ---------------------------------------------------------------------------


def test_rank_padding_enabled_per_module() -> None:
    report = run_multilayer_lora_training(_cfg())
    for entry in report["per_layer_metrics"]:
        assert entry["padded_rank"] == 4
        assert entry["true_rank"] == 2
        assert entry["visible_rank_from_a_shape"] == 4
        assert entry["visible_rank_from_b_shape"] == 4


# ---------------------------------------------------------------------------
# 8. dummy_update_applied=False for all modules
# ---------------------------------------------------------------------------


def test_dummy_update_not_applied() -> None:
    report = run_multilayer_lora_training(_cfg())
    for entry in report["per_layer_metrics"]:
        assert entry["dummy_update_applied"] is False
    assert report["optimizer_summary"]["any_dummy_update_applied"] is False


# ---------------------------------------------------------------------------
# 9. optimizer_state_contains_dummy=False per module + global
# ---------------------------------------------------------------------------


def test_optimizer_state_no_dummy() -> None:
    for opt in ("sgd", "adamw"):
        report = run_multilayer_lora_training(_cfg(optimizer=opt, lr=1e-3))
        assert (
            report["optimizer_summary"]["any_optimizer_state_contains_dummy"]
            is False
        )
        for entry in report["per_layer_metrics"]:
            assert entry["optimizer_state_contains_dummy"] is False
            assert entry["trainable_adapter_shape_a"][1] == 2  # true_rank
            assert entry["trainable_adapter_shape_b"][0] == 2


# ---------------------------------------------------------------------------
# 10. visible_rank_from_shape == padded_rank
# ---------------------------------------------------------------------------


def test_visible_rank_equals_padded_rank() -> None:
    report = run_multilayer_lora_training(_cfg(padded_rank=6))
    rp = report["rank_padding_summary"]
    assert rp["padded_rank"] == 6
    for entry in report["per_layer_metrics"]:
        assert entry["visible_rank_from_a_shape"] == 6
        assert entry["visible_rank_from_b_shape"] == 6


# ---------------------------------------------------------------------------
# 11. true_rank_hidden_from_shape=True for every module
# ---------------------------------------------------------------------------


def test_true_rank_hidden_from_shape() -> None:
    report = run_multilayer_lora_training(_cfg())
    rp = report["rank_padding_summary"]
    assert rp["true_rank_hidden_from_shape"] is True
    assert rp["padded_rank_visible"] is True
    for entry in report["per_layer_metrics"]:
        assert entry["true_rank_hidden_from_shape"] is True


# ---------------------------------------------------------------------------
# 12. Markdown / JSON / CSV generated
# ---------------------------------------------------------------------------


def test_runner_script_emits_required_artifacts(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--num-layers", "2",
        "--hidden-size", "16",
        "--intermediate-size", "24",
        "--vocab-size", "32",
        "--seq-len", "4",
        "--batch-size", "2",
        "--true-rank", "2",
        "--padded-rank", "4",
        "--num-steps", "2",
        "--dtype", "float64",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_path = tmp_path / "multilayer_lora_training_experiments.json"
    csv_path = tmp_path / "multilayer_lora_training_experiments.csv"
    md_path = tmp_path / "multilayer_lora_training_experiments.md"
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()
    report = json.loads(json_path.read_text())
    assert report["training_correctness"]["allclose"] is True
    md_text = md_path.read_text().lower()
    for needle in (
        "multi-layer lora",
        "experiment scope",
        "forward correctness",
        "backward correctness",
        "rank padding across layers",
        "optimizer handling",
        "per-layer metrics",
        "limitations",
        "next stage plan",
    ):
        assert needle in md_text, f"missing markdown section: {needle!r}"


# ---------------------------------------------------------------------------
# 13. No raw tensor in JSON/MD (tensor_free output)
# ---------------------------------------------------------------------------


def test_outputs_have_no_raw_tensors(tmp_path: Path) -> None:
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-dir", str(tmp_path),
        "--num-layers", "2",
        "--hidden-size", "16",
        "--intermediate-size", "24",
        "--vocab-size", "32",
        "--seq-len", "4",
        "--batch-size", "2",
        "--true-rank", "2",
        "--padded-rank", "4",
        "--num-steps", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    json_text = (
        tmp_path / "multilayer_lora_training_experiments.json"
    ).read_text()
    md_text = (
        tmp_path / "multilayer_lora_training_experiments.md"
    ).read_text()
    csv_text = (
        tmp_path / "multilayer_lora_training_experiments.csv"
    ).read_text()
    for text in (json_text, md_text, csv_text):
        assert "tensor(" not in text


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_normalize_optimizer() -> None:
    assert normalize_optimizer(None) == "sgd"
    assert normalize_optimizer("sgd") == "sgd"
    assert normalize_optimizer("adamw") == "adamw"
    with pytest.raises(ValueError):
        normalize_optimizer("rmsprop")


def test_csv_rows_no_tensor_text() -> None:
    report = run_multilayer_lora_training(_cfg())
    rows = multilayer_lora_training_csv_rows(report)
    for row in rows:
        assert "tensor(" not in str(row["value"])


def test_subset_lora_targets_runs() -> None:
    report = run_multilayer_lora_training(
        _cfg(lora_targets=("q_proj", "v_proj", "down_proj"))
    )
    tc = report["training_correctness"]
    assert tc["allclose"] is True
    rp = report["rank_padding_summary"]
    assert rp["num_lora_modules"] == 2 * 3
