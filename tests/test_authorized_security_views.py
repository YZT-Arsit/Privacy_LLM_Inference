from __future__ import annotations

import pytest

from scripts.authorized_security.view_contracts import (
    ALLOWLISTS,
    View,
    ViewAccessError,
    ViewRecord,
)


@pytest.mark.parametrize("view", list(View))
def test_each_view_requires_explicit_valid_name(view: View) -> None:
    record = ViewRecord.build(view.value, {"sample_id": "s1"})
    assert record.get("sample_id") == "s1"
    with pytest.raises(ValueError):
        ViewRecord.build("AUTO", {"sample_id": "s1"})


@pytest.mark.parametrize("view", [View.V0, View.V1, View.V2])
@pytest.mark.parametrize(
    "field",
    ["plaintext_weights", "plaintext_adapter", "reference", "labels", "model", "oracle"],
)
def test_untrusted_views_reject_evaluator_fields(view: View, field: str) -> None:
    with pytest.raises(ViewAccessError):
        ViewRecord.build(view, {"sample_id": "s1", field: "secret"})


def test_v0_rejects_every_v2_only_field() -> None:
    for field in ALLOWLISTS[View.V2] - ALLOWLISTS[View.V0]:
        with pytest.raises(ViewAccessError):
            ViewRecord.build(View.V0, {"sample_id": "s1", field: []})


def test_access_is_checked_again() -> None:
    record = ViewRecord.build(View.V0, {"sample_id": "s1", "output_text": "ok"})
    with pytest.raises(ViewAccessError):
        record.get("logits")


def test_live_model_like_object_is_rejected() -> None:
    DummyModel = type("DummyModel", (), {})
    with pytest.raises(ViewAccessError):
        ViewRecord.build(View.V3, {"sample_id": "s1", "plaintext_weights": DummyModel()})
