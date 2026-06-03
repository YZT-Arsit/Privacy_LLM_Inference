"""Stage 7.5c tests for the deployable runtime boundary."""

from __future__ import annotations

from pathlib import Path

import torch

from pllo.runtime import (
    AcceleratorBackend,
    LocalCPUBackend,
    RuntimeTranscript,
    TrustedController,
    TrustedControllerConfig,
    get_backend,
    list_backends,
    register_backend,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_trusted_controller_samples_mask_and_pad() -> None:
    ctrl = TrustedController(config=TrustedControllerConfig(dtype="float64"))
    n, n_inv = ctrl.sample_mask(8)
    assert n.shape == (8, 8)
    assert torch.allclose(
        n @ n_inv, torch.eye(8, dtype=torch.float64), atol=1e-9,
    )
    pad = ctrl.sample_pad((4, 8))
    assert pad.shape == (4, 8)


def test_local_cpu_backend_satisfies_protocol() -> None:
    backend = LocalCPUBackend(dtype=torch.float64)
    assert isinstance(backend, AcceleratorBackend)
    assert backend.name == "local_cpu"


def test_transcript_redacts_mask_seeds() -> None:
    tr = RuntimeTranscript()
    tr.record_mask_id(2026)
    tr.record_mask_id("hidden_seed")
    assert tr.mask_ids_redacted, "redacted ids should be present"
    for digest in tr.mask_ids_redacted:
        assert digest != "2026"
        assert digest != "hidden_seed"
    summary = tr.to_summary()
    assert summary["contains_raw_secret"] is False


def test_transcript_summary_does_not_carry_raw_secret() -> None:
    backend = LocalCPUBackend(dtype=torch.float64)
    x = torch.randn(4, 8, dtype=torch.float64)
    w = torch.randn(8, 16, dtype=torch.float64)
    backend.linear(x, w, None)
    backend.matmul(x, w)
    summary = backend.collect_transcript_summary()
    assert summary["contains_raw_secret"] is False
    assert summary["boundary_calls"] == 2
    # No raw tensors should appear anywhere in the summary string.
    assert "tensor(" not in str(summary)


def test_backend_registry_returns_local_cpu() -> None:
    backends = list_backends()
    assert "local_cpu" in backends
    inst = get_backend("local_cpu")
    assert isinstance(inst, AcceleratorBackend)


def test_register_backend_allows_future_tee_or_gpu_drop_in() -> None:
    class FakeBackend:
        name = "fake_tee"

        # Minimum protocol surface for the test: not exercised here.
        def __init__(self) -> None:
            pass

    register_backend("fake_tee", FakeBackend)
    assert "fake_tee" in list_backends()


def test_docs_runtime_boundary_exists() -> None:
    doc = PROJECT_ROOT / "docs" / "runtime_boundary.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "TrustedController" in text
    assert "AcceleratorBackend" in text
    assert "LocalCPUBackend" in text
    lowered = text.lower()
    assert "not a real tee" in lowered or "not real tee" in lowered


def test_sanitize_summary_strips_forbidden_fields() -> None:
    ctrl = TrustedController()
    cleaned = ctrl.sanitize_summary({
        "ok_field": 1,
        "raw_input": "secret-prompt",
        "private_data": [1, 2, 3],
        "optimizer_state": {"momentum": 0.9},
        "labels": [0, 1, 0],
    })
    assert "raw_input" not in cleaned
    assert "private_data" not in cleaned
    assert "optimizer_state" not in cleaned
    assert "labels" not in cleaned
    assert cleaned["ok_field"] == 1
    assert cleaned["contains_raw_secret"] is False
    assert cleaned["sanitized_by_trusted_controller"] is True
