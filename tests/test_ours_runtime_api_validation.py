"""Stage 7.5c tests for the deployable runtime API validation runner."""

from __future__ import annotations

from pathlib import Path

from pllo.experiments.ours_runtime_api_validation import (
    OursRuntimeAPIValidationConfig,
    run_ours_runtime_api_validation,
)


def _cfg(tmp: Path, **overrides) -> OursRuntimeAPIValidationConfig:
    base = dict(
        output_dir=str(tmp),
        seed=2026,
        batch_size=2, seq_len=4, hidden_size=8,
        true_rank=2, padded_rank=4, num_decode_steps=2,
    )
    base.update(overrides)
    return OursRuntimeAPIValidationConfig(**base)


def test_runner_runs(tmp_path: Path) -> None:
    report = run_ours_runtime_api_validation(_cfg(tmp_path))
    assert report["ours_runtime_api_validation_status"] == "implemented"
    assert report["backend"] == "local_cpu"
    assert report["tee_gpu_ready_interface"] is True
    # The runtime API claim is interface-readiness only -- not real-hardware.
    assert report["real_tee_implemented"] is False
    assert report["real_gpu_backend_implemented"] is False


def test_covers_forward_decode_lora(tmp_path: Path) -> None:
    report = run_ours_runtime_api_validation(_cfg(tmp_path))
    components = {r["component"] for r in report["rows"]}
    for required in (
        "linear_pad_compensation",
        "modern_decoder_full_forward",
        "modern_decoder_prefill",
        "modern_decoder_decode_step",
        "modern_decoder_greedy_generation",
        "lora_forward",
        "lora_backward",
        "rank_padding",
        "multilayer_lora_training_step",
    ):
        assert required in components, f"missing {required}"


def test_transcript_sanitized_and_no_secret_leak(tmp_path: Path) -> None:
    report = run_ours_runtime_api_validation(_cfg(tmp_path))
    for row in report["rows"]:
        assert row["transcript_sanitized"] is True, row["component"]
        assert row["raw_secret_leaked"] is False, row["component"]


def test_local_cpu_only_disclaimer_in_md(tmp_path: Path) -> None:
    run_ours_runtime_api_validation(_cfg(tmp_path))
    md = (tmp_path / "ours_runtime_api_validation.md").read_text(encoding="utf-8").lower()
    assert "local cpu" in md
    assert "real tee" in md or "real tee/gpu" in md
    assert "not implemented" in md or "not deployed" in md


def test_outputs_written(tmp_path: Path) -> None:
    run_ours_runtime_api_validation(_cfg(tmp_path))
    for name in (
        "ours_runtime_api_validation.json",
        "ours_runtime_api_validation.csv",
        "ours_runtime_api_validation.md",
    ):
        assert (tmp_path / name).exists()


def test_no_raw_tensors_in_json(tmp_path: Path) -> None:
    run_ours_runtime_api_validation(_cfg(tmp_path))
    blob = (tmp_path / "ours_runtime_api_validation.json").read_text(encoding="utf-8")
    assert "tensor(" not in blob
