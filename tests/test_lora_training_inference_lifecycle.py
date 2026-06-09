"""Stage 7.6 tests -- LoRA training-to-inference lifecycle report
and Stage 7.6 claims consistency audit."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pllo.experiments.lora_training_inference_lifecycle import (
    build_lifecycle_report,
    render_markdown as render_lifecycle_markdown,
    write_reports as write_lifecycle_reports,
)
from pllo.experiments.stage_7_6_claims_consistency import (
    build_claims_consistency_report,
    render_markdown as render_claims_markdown,
    scan_paths,
    write_reports as write_claims_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


_REQUIRED_LIFECYCLE_PHASES: tuple[str, ...] = (
    "lora_initialization",
    "masked_forward",
    "masked_backward",
    "masked_sgd",
    "masked_momentum_sgd",
    "final_adapter_recovery_or_audit",
    "trained_lora_inference",
)


_REQUIRED_TRACKED_PHRASES: tuple[str, ...] = (
    "formal security",
    "cryptographically secure",
    "semantic security",
    "AdamW supported",
    "plaintext gradients hidden by proof",
    "optimizer fully outsourced",
    "LoRA rank is hidden",
)


# ---------------------------------------------------------------------------
# Lifecycle report tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lifecycle_report() -> dict:
    return build_lifecycle_report()


def test_lifecycle_report_status_and_stage(lifecycle_report: dict) -> None:
    assert lifecycle_report["status"] == "ok"
    assert lifecycle_report["stage"] == "7.6"
    assert (
        lifecycle_report["report"] == "lora_training_inference_lifecycle"
    )


def test_lifecycle_required_phases_present(lifecycle_report: dict) -> None:
    phases = set(lifecycle_report["phases_present"])
    for phase in _REQUIRED_LIFECYCLE_PHASES:
        assert phase in phases, f"missing lifecycle phase {phase!r}"


def test_lifecycle_rows_have_required_fields(lifecycle_report: dict) -> None:
    for r in lifecycle_report["lifecycle_rows"]:
        for field in ("phase", "object", "visibility", "exposed_form", "note"):
            assert field in r, f"row missing field {field!r}: {r}"
        assert r["visibility"] in (
            "plaintext_trusted_only",
            "gpu_visible_masked",
            "public",
        )


def test_lifecycle_plaintext_lora_factors_never_gpu_visible(
    lifecycle_report: dict,
) -> None:
    for r in lifecycle_report["lifecycle_rows"]:
        if r["object"] in ("A_real, B_real", "A_pad, B_pad"):
            assert r["visibility"] == "plaintext_trusted_only", r


def test_lifecycle_masks_never_gpu_visible(lifecycle_report: dict) -> None:
    found = False
    for r in lifecycle_report["lifecycle_rows"]:
        if r["object"] == "N_x, N_y, M":
            assert r["visibility"] == "plaintext_trusted_only", r
            found = True
    assert found, "N_x, N_y, M row must be present"


def test_lifecycle_gpu_sees_only_masked(lifecycle_report: dict) -> None:
    expected_masked = (
        ("masked_forward", "X_tilde = X N_x"),
        ("masked_forward", "A_tilde, B_tilde"),
        ("masked_backward", "grad_A_tilde"),
        ("masked_backward", "grad_B_tilde"),
        ("masked_momentum_sgd", "V_A_tilde, V_B_tilde (masked momentum buffers)"),
        ("trained_lora_inference", "X_infer_tilde = X_infer N_x"),
    )
    rows_by_key = {
        (r["phase"], r["object"]): r
        for r in lifecycle_report["lifecycle_rows"]
    }
    for phase, obj in expected_masked:
        assert (phase, obj) in rows_by_key, f"missing ({phase}, {obj})"
        assert rows_by_key[(phase, obj)]["visibility"] == "gpu_visible_masked"


def test_lifecycle_base_model_is_public(lifecycle_report: dict) -> None:
    rows = [
        r for r in lifecycle_report["lifecycle_rows"]
        if r["object"] == "base_model_W"
    ]
    assert rows, "base_model_W row must be present"
    for r in rows:
        assert r["visibility"] == "public", r
        assert (
            "transformed_to_preserve_hidden_state_masks"
            in r["exposed_form"]
        )


def test_lifecycle_adamw_unsupported_present(lifecycle_report: dict) -> None:
    md = render_lifecycle_markdown(lifecycle_report)
    assert "DenseMaskedAdamWUnsupported" in md
    assert "AdamW under dense masks is unsupported" in md
    inv = lifecycle_report["invariants"]
    assert inv["adamw_under_dense_masks_supported"] is False


def test_lifecycle_formal_security_claim_false(
    lifecycle_report: dict,
) -> None:
    assert lifecycle_report["formal_security_claim"] is False
    md = render_lifecycle_markdown(lifecycle_report)
    assert "`formal_security_claim`: `False`" in md


def test_lifecycle_invariants_match_threat_model(
    lifecycle_report: dict,
) -> None:
    inv = lifecycle_report["invariants"]
    for k in (
        "plaintext_A_or_B_visible_to_gpu",
        "plaintext_grad_A_or_B_visible_to_gpu",
        "masks_visible_to_gpu",
        "plaintext_optimizer_state_visible_to_gpu",
        "user_input_visible_to_gpu",
    ):
        assert inv[k] is False, k
    for k in (
        "base_model_W_public",
        "base_model_W_transformed_to_preserve_hidden_state_masks",
        "gpu_sees_masked_hidden_states",
        "gpu_sees_masked_lora_adapters",
        "gpu_sees_masked_lora_gradients",
        "gpu_sees_masked_momentum_buffers",
    ):
        assert inv[k] is True, k


def test_lifecycle_outputs_no_raw_tensors(
    lifecycle_report: dict, tmp_path: Path,
) -> None:
    j, c, m = write_lifecycle_reports(
        lifecycle_report, outputs_dir=str(tmp_path),
    )
    for path in (j, c, m):
        text = Path(path).read_text()
        assert "tensor(" not in text, f"{path} contains tensor(...)"
        long_arr = re.search(r"\[(\s*-?\d+(\.\d+)?\s*,\s*){50,}", text)
        assert long_arr is None, f"{path} has long numeric array"


def test_lifecycle_outputs_written_to_outputs_dir(tmp_path: Path) -> None:
    rep = build_lifecycle_report()
    j, c, m = write_lifecycle_reports(rep, outputs_dir=str(tmp_path))
    assert Path(j).is_file()
    assert Path(c).is_file()
    assert Path(m).is_file()
    payload = json.loads(Path(j).read_text())
    assert payload["status"] == "ok"
    assert payload["stage"] == "7.6"


# ---------------------------------------------------------------------------
# Claims consistency tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def claims_report() -> dict:
    return build_claims_consistency_report(repo_root=REPO_ROOT)


def test_claims_status_and_stage(claims_report: dict) -> None:
    assert claims_report["status"] == "ok"
    assert claims_report["stage"] == "7.6"
    assert claims_report["report"] == "stage_7_6_claims_consistency"


def test_claims_tracks_required_phrases(claims_report: dict) -> None:
    for phrase in _REQUIRED_TRACKED_PHRASES:
        assert phrase in claims_report["tracked_phrases"]


def test_claims_no_unsafe_wording_present_in_repo(
    claims_report: dict,
) -> None:
    unsafe_rows = [
        o for o in claims_report["occurrences"]
        if o["classification"] == "unsafe_wording_present"
    ]
    assert not unsafe_rows, (
        "Unsafe wording present in project markdown / LaTeX:\n"
        + "\n".join(
            f"  {o['file']}:{o['line']} [{o['phrase']}] -> {o['snippet']}"
            for o in unsafe_rows
        )
    )
    assert claims_report["passes_consistency_check"] is True


def test_claims_listed_safe_occurrences_present(
    claims_report: dict,
) -> None:
    # We expect the repo to enumerate these forbidden phrases in
    # at least one disclaimer / audit document (the existing
    # paper_draft/unsafe_wording_review.md and the new
    # masked_gradient_lora_training.md / lifecycle / claims docs).
    safe_rows = [
        o for o in claims_report["occurrences"]
        if o["classification"] == "listed_as_unsafe_wording_to_avoid"
    ]
    assert len(safe_rows) > 0


def test_claims_formal_security_claim_false(claims_report: dict) -> None:
    assert claims_report["formal_security_claim"] is False
    md = render_claims_markdown(claims_report)
    assert "`formal_security_claim`: `False`" in md


def test_claims_outputs_no_raw_tensors(
    claims_report: dict, tmp_path: Path,
) -> None:
    j, c, m = write_claims_reports(
        claims_report, outputs_dir=str(tmp_path),
    )
    for path in (j, c, m):
        text = Path(path).read_text()
        assert "tensor(" not in text, f"{path} contains tensor()"
        long_arr = re.search(r"\[(\s*-?\d+(\.\d+)?\s*,\s*){50,}", text)
        assert long_arr is None


def test_claims_writes_three_artifacts(tmp_path: Path) -> None:
    rep = build_claims_consistency_report(repo_root=REPO_ROOT)
    j, c, m = write_claims_reports(rep, outputs_dir=str(tmp_path))
    assert Path(j).is_file()
    assert Path(c).is_file()
    assert Path(m).is_file()


def test_claims_scan_handles_synthetic_unsafe_file(tmp_path: Path) -> None:
    """A handcrafted unsafe markdown file must be flagged as
    `unsafe_wording_present`; a handcrafted safe disclaimer file
    must be flagged as `listed_as_unsafe_wording_to_avoid`."""
    unsafe_md = tmp_path / "synthetic_unsafe.md"
    unsafe_md.write_text(
        "# A synthetic claim\n\n"
        "Our system achieves formal security in the standard "
        "model.\n"
    )
    safe_md = tmp_path / "synthetic_unsafe_wording_review.md"
    safe_md.write_text(
        "# Unsafe Wording Review\n\n"
        "We do not claim formal security. We do not claim "
        "cryptographically secure execution.\n"
    )
    occurrences = scan_paths(
        [unsafe_md, safe_md], repo_root=tmp_path,
    )
    by_file = {}
    for o in occurrences:
        by_file.setdefault(o["file"], []).append(o)
    unsafe_occ = by_file.get("synthetic_unsafe.md", [])
    safe_occ = by_file.get("synthetic_unsafe_wording_review.md", [])
    assert any(
        o["phrase"] == "formal security"
        and o["classification"] == "unsafe_wording_present"
        for o in unsafe_occ
    ), unsafe_occ
    for o in safe_occ:
        assert o["classification"] == "listed_as_unsafe_wording_to_avoid"


def test_claims_markdown_lists_tracked_phrases(claims_report: dict) -> None:
    md = render_claims_markdown(claims_report)
    for phrase in _REQUIRED_TRACKED_PHRASES:
        assert phrase in md


def test_claims_paper_safe_wording_present(claims_report: dict) -> None:
    md = render_claims_markdown(claims_report)
    assert (
        "masked-gradient LoRA provides algebraic equivalence for "
        "SGD/Momentum under orthogonal masks and proxy-evaluated "
        "leakage mitigation; it does not provide formal, "
        "cryptographic, or semantic security."
    ) in md


# ---------------------------------------------------------------------------
# Cross-report invariants
# ---------------------------------------------------------------------------


def test_both_reports_emit_outputs_outputs_dir(tmp_path: Path) -> None:
    lc = build_lifecycle_report()
    cc = build_claims_consistency_report(repo_root=REPO_ROOT)
    lj, _lc, _lm = write_lifecycle_reports(lc, outputs_dir=str(tmp_path))
    cj, _cc, _cm = write_claims_reports(cc, outputs_dir=str(tmp_path))
    lifecycle = json.loads(Path(lj).read_text())
    claims = json.loads(Path(cj).read_text())
    assert lifecycle["formal_security_claim"] is False
    assert claims["formal_security_claim"] is False


def test_lifecycle_required_phases_match_spec(
    lifecycle_report: dict,
) -> None:
    assert (
        tuple(lifecycle_report["required_phases"])
        == _REQUIRED_LIFECYCLE_PHASES
    )
