"""Task 2: exercise the REAL HF/PEFT LoRA adapter path with a tiny, locally
generated fixture (no external download). Validates the loader (PEFT layout ->
codebase convention), config extraction, the no-raw-LoRA package screen, and that
a folded package built from the HF adapter matches the trusted raw-LoRA reference.

Tiny CPU only (no H800/TDX/CUDA/full Qwen).

Run: python -m pytest tests/test_lora_hf_adapter_fixture.py -q
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.deployment.lora_folded_package import (  # noqa: E402
    apply_lora_to_model,
    build_lora_folded_package,
    fold_lora_for_layer,
    load_hf_lora_adapter,
    lora_scaling,
    merge_folded_lora,
    read_hf_adapter_config,
    verify_lora_folded_package,
)
from pllo.experiments.folded_probe_common import tiny_model  # noqa: E402
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402

SEED = 2035
N = 4
RANK = 4
ALPHA = 8.0
TM = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _fixture_fn():
    spec = importlib.util.spec_from_file_location(
        "ctf", REPO_ROOT / "scripts" / "create_tiny_hf_lora_fixture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.create_tiny_hf_lora_fixture


def _cfg():
    return MemoryOptimizedConfig(
        num_layers=N, batch_size=1, seq_len=8, max_new_tokens=1, device="cpu",
        dtype="float32", folding_dtype="float32", folded_weight_device="cpu",
        seed=SEED)


@pytest.fixture()
def fixture(tmp_path):
    _model, mc = tiny_model()
    path, raw = _fixture_fn()(
        tmp_path / "tiny_hf_lora", mc, num_layers=N, target_modules=TM,
        rank=RANK, alpha=ALPHA, seed=1)
    return path, raw, mc


def test_adapter_config_extracted(fixture) -> None:
    path, _raw, _mc = fixture
    acfg = read_hf_adapter_config(path)
    assert acfg["rank"] == RANK
    assert acfg["alpha"] == ALPHA
    assert acfg["scaling"] == pytest.approx(lora_scaling(ALPHA, RANK))
    # all foldable target modules recognised, in canonical order
    assert acfg["target_modules"] == TM


def test_loader_round_trips_peft_layout(fixture) -> None:
    """PEFT lora_A [r,in] / lora_B [out,r] -> codebase A [in,r] / B [r,out]."""
    path, raw, mc = fixture
    loaded = load_hf_lora_adapter(path, mc, N, TM)
    assert set(loaded) == set(raw)
    for ell in raw:
        assert set(loaded[ell]) == set(raw[ell])
        for m in TM:
            a_l, b_l = loaded[ell][m]
            a_r, b_r = raw[ell][m]
            assert a_l.shape == a_r.shape       # [in, r]
            assert b_l.shape == b_r.shape       # [r, out]
            assert torch.allclose(a_l, a_r, atol=1e-6)
            assert torch.allclose(b_l, b_r, atol=1e-6)


def test_folded_package_from_hf_adapter_has_no_raw_lora(fixture, tmp_path) -> None:
    path, _raw, mc = fixture
    model, _mc2 = tiny_model()
    sess = MaskedQwenSession(model, mc, _cfg())
    lora = load_hf_lora_adapter(path, mc, N, TM)
    out_dir = tmp_path / "hf_folded"
    build_lora_folded_package(
        out_dir, session=sess, lora=lora, target_modules=TM, rank=RANK,
        alpha=ALPHA, rank_seed=SEED, base_manifest_hash=None,
        model_name="tiny", created_by="test")
    rep = verify_lora_folded_package(str(out_dir))
    assert rep["lora_package_valid"] is True
    assert rep["forbidden_fields_found"] == []
    assert rep["raw_lora_tensor_names_found"] == []
    assert rep["contains_raw_lora"] is False
    assert rep["target_modules_missing_coverage"] == []
    assert rep["rank"] == RANK


def test_folded_hf_adapter_matches_reference(fixture) -> None:
    """folded base + merged folded-LoRA == folded(base + raw HF LoRA)."""
    path, _raw, mc = fixture
    model, _mc2 = tiny_model()
    model_lora = copy.deepcopy(model)
    scaling = lora_scaling(ALPHA, RANK)
    lora = load_hf_lora_adapter(path, mc, N, TM)
    sess = MaskedQwenSession(model, mc, _cfg())
    apply_lora_to_model(model_lora, lora, TM, scaling)
    sess_lora = MaskedQwenSession(model_lora, mc, _cfg())

    max_err = 0.0
    for ell in range(N):
        base = {k: v.clone()
                for k, v in sess.export_folded_layer_tensors(ell).items()}
        fl = fold_lora_for_layer(sess, ell, lora[ell], scaling=scaling,
                                 rank=RANK, rank_seed=SEED, target_modules=TM)
        merged = merge_folded_lora(base, fl, TM)
        ref = sess_lora.export_folded_layer_tensors(ell)
        for k in ref:
            max_err = max(max_err, (merged[k] - ref[k]).abs().max().item())
    assert max_err < 1e-4
