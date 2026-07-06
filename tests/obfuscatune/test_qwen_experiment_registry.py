"""ObfuscaTune-Qwen method registration + non-interference with existing methods."""

from __future__ import annotations

from pllo.experiments import experiment_registry as reg


def test_obfuscatune_qwen_methods_registered():
    for name in ("obfuscatune_qwen_orthogonal", "obfuscatune_qwen_random",
                 "obfuscatune_qwen_cond_8", "obfuscatune_qwen_cond_32",
                 "obfuscatune_qwen_cond_128"):
        assert name in reg.OBFUSCATUNE_QWEN_METHODS
        spec = reg.get_baseline_method(name)
        assert spec.implemented is True
        assert spec.family == "baseline"
        assert spec.model_family == "qwen"
        assert spec.uses_tee == "simulated"
        assert spec.requires_private_base_model is True
        assert spec.comparable_to_amulet_style is True


def test_registry_schema_completeness():
    spec = reg.get_baseline_method("obfuscatune_qwen_orthogonal")
    for field in ("protects_base_model", "protects_user_input", "protects_lora",
                  "protects_kv_cache", "uses_obfuscation", "uses_orthogonal_random_matrix",
                  "notes"):
        assert hasattr(spec, field)
    # honest labels: base model protected, LoRA not implemented, KV not protected
    assert spec.protects_base_model is True
    assert spec.protects_lora == "not_implemented"
    assert spec.protects_kv_cache == "false"
    assert spec.uses_orthogonal_random_matrix is True
    assert reg.get_baseline_method("obfuscatune_qwen_random").uses_orthogonal_random_matrix is False


def test_does_not_disturb_existing_workload_methods():
    # amulet-style / trusted-shortcut workload registry is untouched.
    assert "amulet_style_reference" in reg.METHOD_BY_NAME
    assert "ours_current" in reg.METHOD_BY_NAME
    assert "plain_hf_gpu" in reg.METHOD_BY_NAME
    # baseline specs are a SEPARATE dict, not mixed into the workload methods
    assert "obfuscatune_qwen_orthogonal" not in reg.METHOD_BY_NAME
    assert len(reg.WORKLOAD_METHODS) == 6


def test_notes_flag_threat_model_difference():
    spec = reg.get_baseline_method("obfuscatune_qwen_orthogonal")
    assert "different threat model" in spec.notes.lower() or "differ" in spec.notes.lower()
