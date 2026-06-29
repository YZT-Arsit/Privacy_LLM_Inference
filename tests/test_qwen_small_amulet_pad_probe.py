"""F: small-model Amulet + Linear-pad MLP probe over REAL Qwen2 weights.

Builds a tiny (randomly-initialised) Qwen2 model with the real Qwen2 architecture,
saves it, and runs ``scripts/run_qwen_small_amulet_pad_mlp_probe.py``'s ``run()``
against it. Skips cleanly if transformers / Qwen2 are unavailable.

    PYTHONPATH=$PWD/src pytest tests/test_qwen_small_amulet_pad_probe.py -q
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "qwen_small_amulet_probe",
        REPO_ROOT / "scripts" / "run_qwen_small_amulet_pad_mlp_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_qwen2(path: Path) -> None:
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(path)


def _args(model_path, out_dir, **kw):
    base = dict(
        model_path=str(model_path), model_name="tiny-qwen2",
        num_layers=2, seq_len=8, dtype="float64", device="cpu",
        kronecker_size=3, pad_scale=0.1, seed=0, linear_boundary_pad=True,
        output_json=out_dir / "probe.json", output_md=out_dir / "probe.md")
    base.update(kw)
    return argparse.Namespace(**base)


def test_small_qwen_amulet_pad_probe_real_weights(tmp_path) -> None:
    mod = _load_probe()
    model_dir = tmp_path / "tiny_qwen2"
    _tiny_qwen2(model_dir)
    rep = mod.run(_args(model_dir, tmp_path))

    assert rep.get("skipped") is False
    assert rep["uses_real_qwen_weights"] is True
    assert rep["num_layers_checked"] == 2
    assert rep["linear_boundary_pad_enabled"] is True
    assert rep["amulet_right_mask_swiglu_enabled"] is True
    assert rep["gate_up_down_pad_enabled"] is True
    assert rep["pad_enters_nonlinear_island"] is False
    assert rep["production_qwen7b_integration"] is False
    assert rep["formal_security_claim"] is False
    assert rep["paper_scope"] == "small_model_real_weight_nonlinear_island_probe"
    # fp64 strict correctness against plaintext MLP
    assert rep["tokens_or_mlp_output_match"] is True
    assert rep["paper_ready_small_model_amulet_probe"] is True
    assert rep["max_abs_error"] <= 1e-6


def test_small_qwen_probe_clean_skip_when_model_missing(tmp_path) -> None:
    mod = _load_probe()
    rep = mod.run(_args(tmp_path / "does_not_exist", tmp_path))
    assert rep["skipped"] is True
    assert rep["uses_real_qwen_weights"] is False
    assert rep["paper_ready_small_model_amulet_probe"] is False
