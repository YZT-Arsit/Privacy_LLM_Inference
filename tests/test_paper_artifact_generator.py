"""Stage 7.1 tests -- paper artifact generator (CPU, no HF/network)."""

from __future__ import annotations

from pllo.experiments.paper_artifact_generator import (
    UNSAFE_BEGIN,
    UNSAFE_END,
    UNSAFE_PHRASES,
    PaperArtifactConfig,
    generate_paper_artifacts,
    write_paper_artifacts,
)


def _strip_unsafe(text: str) -> str:
    out: list[str] = []
    skip = False
    for ln in text.splitlines():
        if UNSAFE_BEGIN in ln:
            skip = True
            continue
        if UNSAFE_END in ln:
            skip = False
            continue
        if not skip:
            out.append(ln)
    return "\n".join(out).lower()


# 1.
def test_config_defaults() -> None:
    cfg = PaperArtifactConfig()
    assert cfg.output_dir == "outputs/paper_artifacts"
    assert cfg.include_math and cfg.include_tables and cfg.include_claim_audit
    assert cfg.strict_no_unsafe_claims is True


# 2.
def test_generator_runs_with_missing_reports() -> None:
    cfg = PaperArtifactConfig(
        cost_report_path="/nope/cost.json",
        skeleton_report_path="/nope/skel.json",
        boundary_report_path="/nope/bnd.json",
        rope_report_path="/nope/rope.json")
    r = generate_paper_artifacts(cfg)
    src = r["metadata"]["source_reports"]
    assert all(v == "missing_source_report" for v in src.values())
    assert set(r["metadata"]["missing_inputs"]) == {
        "cost_report", "skeleton_report", "boundary_report", "rope_report"}
    # tables degrade gracefully, not crash
    assert "missing_source_report" in r["artifacts"]["complexity_table.md"]
    assert "missing_source_report" in r["artifacts"]["leakage_boundary_table.md"]


# 3.
def test_method_summary_contains_boundaries() -> None:
    ms = generate_paper_artifacts(PaperArtifactConfig())[
        "artifacts"]["method_summary.md"].lower()
    for token in ("trusted tee", "untrusted gpu", "masked embeddings",
                  "masked logits", "samples the next token"):
        assert token in ms


# 4.
def test_correctness_theorems_include_required_lemmas() -> None:
    th = generate_paper_artifacts(PaperArtifactConfig())[
        "artifacts"]["correctness_theorems.md"]
    for n in range(1, 8):
        assert f"Lemma {n}" in th
    for n in range(1, 4):
        assert f"Theorem {n}" in th


# 5.
def test_complexity_table_schema() -> None:
    art = generate_paper_artifacts(PaperArtifactConfig())["artifacts"]
    md = art["complexity_table.md"]
    csv_text = art["complexity_table.csv"]
    for col in ("gpu_flops_prefill", "handoff_gemm_flops",
                "lm_head_gpu_flops", "lm_head_tee_flops", "boundary_calls"):
        assert col in csv_text
    assert "plain_synthetic" in md
    assert "masked_per_layer_with_vocab_scaling" in md


# 6.
def test_leakage_table_schema() -> None:
    art = generate_paper_artifacts(PaperArtifactConfig())["artifacts"]
    csv_text = art["leakage_boundary_table.csv"]
    for col in ("input_ids_visible_to_gpu", "masked_logits_visible_to_gpu",
                "plaintext_logits_visible_to_gpu", "security_status"):
        assert col in csv_text


# 7.
def test_ablation_summary_contains_required_variants() -> None:
    ab = generate_paper_artifacts(PaperArtifactConfig())[
        "artifacts"]["ablation_summary.md"].lower()
    for token in ("rotation", "complex-scaling", "shared residual mask",
                  "per-layer residual mask", "masked lm head",
                  "output hidden to tee", "permutation"):
        assert token in ab


# 8.
def test_claim_audit_safe_and_unsafe_claims() -> None:
    r = generate_paper_artifacts(PaperArtifactConfig())
    ca = r["claim_audit"]
    assert ca["safe_claims"] and ca["unsafe_claims"]
    md = r["artifacts"]["claim_audit.md"]
    assert "## Safe claims" in md
    assert UNSAFE_BEGIN in md and UNSAFE_END in md


# 9.
def test_unsafe_phrases_only_in_unsafe_section() -> None:
    arts = generate_paper_artifacts(PaperArtifactConfig())["artifacts"]
    violations = []
    for name, content in arts.items():
        if not name.endswith(".md"):
            continue
        body = _strip_unsafe(content)
        for ph in UNSAFE_PHRASES:
            if ph in body:
                violations.append((name, ph))
    assert violations == [], violations


# 10.
def test_limitations_include_handoff_gemm_caveat() -> None:
    lim = generate_paper_artifacts(PaperArtifactConfig())[
        "artifacts"]["limitations.md"].lower()
    assert "handoff" in lim and "gemm" in lim
    assert "single decoder layer" in lim
    assert "greedy decode only" in lim


# 11.
def test_combined_report_required_statement() -> None:
    r = generate_paper_artifacts(PaperArtifactConfig())
    combined = r["artifacts"]["stage_7_1_paper_artifacts.md"]
    assert r["required_statement"] in combined
    assert "do not constitute" in combined.lower()


# 12.
def test_cli_writes_expected_files(tmp_path) -> None:
    cfg = PaperArtifactConfig(output_dir=str(tmp_path / "art"))
    report = write_paper_artifacts(cfg)
    expected = {
        "method_summary.md", "correctness_theorems.md", "complexity_table.md",
        "complexity_table.csv", "leakage_boundary_table.md",
        "leakage_boundary_table.csv", "ablation_summary.md", "claim_audit.md",
        "claim_audit.json", "limitations.md", "stage_7_1_paper_artifacts.md",
    }
    written = {p.split("/")[-1] for p in report["written_files"]}
    assert expected <= written
    for name in expected:
        assert (tmp_path / "art" / name).is_file()
