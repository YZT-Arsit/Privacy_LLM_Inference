"""Stage 5.8 tests -- Lookup Nonlinear Cost Proxy."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from pllo.experiments.lookup_nonlinear_cost_proxy import (
    LookupNonlinearCostProxyConfig,
    compatible_island_costs,
    lookup_table_costs,
    render_markdown,
    run_lookup_nonlinear_cost_proxy,
    write_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


_REQUIRED_PHRASES: tuple[str, ...] = (
    "This is a lookup cost proxy, not a secure lookup "
    "implementation.",
    "No garbled circuit, MPC, FHE, Tabula, FLUTE, or cryptographic "
    "lookup protocol is implemented.",
    "Lookup-style nonlinear protection may improve value hiding, but "
    "this stage evaluates only table-size and memory-access costs.",
    "The current compatible island is faster and lower-memory but "
    "preserves permutation-invariant activation statistics.",
    "No formal, cryptographic, or semantic security is claimed.",
    "No real TEE or GPU wall-time is measured.",
)


@pytest.fixture(scope="module")
def small_config() -> LookupNonlinearCostProxyConfig:
    # Synthetic small config so tests run fast.
    return LookupNonlinearCostProxyConfig(
        batch_size=1,
        seq_len=16,
        intermediate_size=64,
        bit_widths=(4, 6, 8),
        entry_bytes=2,
        num_layers=1,
        num_tables_policy="per_layer_shared",
        run_microbench=True,
        microbench_intermediate_size=64,
        microbench_seq_len=16,
        repeats=3,
        seed=0,
    )


@pytest.fixture(scope="module")
def report(small_config) -> dict:
    return run_lookup_nonlinear_cost_proxy(small_config)


# 1. End-to-end run.
def test_small_config_runs_end_to_end(report: dict) -> None:
    assert report["status"] == "ok"
    assert report["stage"] == "5.8"
    assert report["experiment"] == "lookup_nonlinear_cost_proxy"
    methods = {m["method"] for m in report["methods"]}
    expected = {
        "compatible_swiglu_island_current",
        "compatible_swiglu_full_bundle",
        "lookup_swiglu_proxy_4bit",
        "lookup_swiglu_proxy_6bit",
        "lookup_swiglu_proxy_8bit",
    }
    assert expected.issubset(methods)


# 2. JSON / CSV / MD all emitted.
def test_reports_written(report: dict, tmp_path: Path) -> None:
    j, c, m = write_reports(report, outputs_dir=str(tmp_path))
    assert Path(j).is_file()
    assert Path(c).is_file()
    assert Path(m).is_file()
    payload = json.loads(Path(j).read_text())
    assert payload["status"] == "ok"


# 3. Table size formulas are correct.
def test_table_size_formulas_4_6_8_bits() -> None:
    for b in (4, 6, 8):
        c = lookup_table_costs(
            bit_width=b, entry_bytes=2, num_layers=1,
            policy="per_layer_shared", batch_size=1, seq_len=4,
            intermediate_size=4, per_channel_proxy=True,
        )
        assert c["table_entries"] == 2 ** (2 * b)
        assert c["table_bytes"] == (2 ** (2 * b)) * 2
        assert c["num_tables"] == 1
        assert c["preprocessing_bytes"] == c["table_bytes"]
        assert c["online_lookup_bytes"] == 1 * 4 * 4 * 2
        assert c["per_channel_table_bytes"] == c["table_bytes"] * 4
        assert c["per_channel_status"] == "impractical_proxy_only"


# 4. formal_security_claim is False.
def test_formal_security_claim_is_false(report: dict) -> None:
    assert report["formal_security_claim"] is False
    md = render_markdown(report)
    assert "`formal_security_claim`: `False`" in md


# 5. cryptographic_lookup_implemented is False.
def test_cryptographic_lookup_implemented_is_false(report: dict) -> None:
    assert report["cryptographic_lookup_implemented"] is False
    md = render_markdown(report)
    assert "`cryptographic_lookup_implemented`: `False`" in md


# 6. Markdown contains all required honesty phrases.
def test_markdown_contains_required_honesty_phrases(report: dict) -> None:
    md = render_markdown(report)
    for phrase in _REQUIRED_PHRASES:
        assert phrase in md, f"missing honesty phrase: {phrase!r}"


# 7. No raw tensors / no long numeric arrays in outputs.
def test_outputs_have_no_raw_tensors(report: dict, tmp_path: Path) -> None:
    j, c, m = write_reports(report, outputs_dir=str(tmp_path))
    for path in (j, c, m):
        text = Path(path).read_text()
        assert "tensor(" not in text, f"{path} contains tensor(...)"
        long_arr = re.search(r"\[(\s*-?\d+(\.\d+)?\s*,\s*){50,}", text)
        assert long_arr is None, f"{path} has long numeric array"


# 8. Microbench can be enabled and disabled.
def test_microbench_can_be_disabled() -> None:
    cfg = LookupNonlinearCostProxyConfig(
        batch_size=1, seq_len=8, intermediate_size=16,
        bit_widths=(4,), entry_bytes=2, num_layers=1,
        num_tables_policy="per_layer_shared",
        run_microbench=False, repeats=2, seed=0,
    )
    rep = run_lookup_nonlinear_cost_proxy(cfg)
    assert rep["microbench"]["enabled"] is False
    # No mean/median rows expected.
    for v in rep["microbench"].values():
        if isinstance(v, dict):
            assert "mean_ms" not in v


def test_microbench_can_be_enabled() -> None:
    cfg = LookupNonlinearCostProxyConfig(
        batch_size=1, seq_len=8, intermediate_size=16,
        bit_widths=(4,), entry_bytes=2, num_layers=1,
        num_tables_policy="per_layer_shared",
        run_microbench=True,
        microbench_intermediate_size=16, microbench_seq_len=8,
        repeats=2, seed=0,
    )
    rep = run_lookup_nonlinear_cost_proxy(cfg)
    mb = rep["microbench"]
    assert mb["enabled"] is True
    assert "compatible_swiglu_island_current" in mb
    assert "lookup_swiglu_proxy_4bit" in mb
    cur = mb["compatible_swiglu_island_current"]
    for k in ("mean_ms", "median_ms", "std_ms"):
        assert k in cur
        assert isinstance(cur[k], float)


# 9. Current compatible island has zero table preprocessing bytes.
def test_current_island_zero_preprocessing(report: dict) -> None:
    current = next(
        m for m in report["methods"]
        if m["method"] == "compatible_swiglu_island_current"
    )
    bundle = next(
        m for m in report["methods"]
        if m["method"] == "compatible_swiglu_full_bundle"
    )
    for m in (current, bundle):
        assert m["cost"]["preprocessing_bytes"] == 0
        assert m["cost"]["table_bytes"] == 0
        assert m["cost"]["table_entries"] == 0


# 10. Lookup proxy table_bytes monotonically increase with bit width.
def test_lookup_table_bytes_monotonic_in_bit_width(report: dict) -> None:
    lookup_methods = sorted(
        (m for m in report["methods"] if m["kind"] == "lookup_proxy"),
        key=lambda x: x["cost"]["bit_width"],
    )
    bit_widths = [m["cost"]["bit_width"] for m in lookup_methods]
    table_bytes = [m["cost"]["table_bytes"] for m in lookup_methods]
    assert bit_widths == sorted(bit_widths)
    assert table_bytes == sorted(table_bytes)
    for i in range(1, len(table_bytes)):
        assert table_bytes[i] > table_bytes[i - 1]


# Bonus: compatible island costs helper.
def test_compatible_island_costs_helper() -> None:
    c = compatible_island_costs(
        batch_size=2, seq_len=4, intermediate_size=8, dtype="float32",
    )
    n = 2 * 4 * 8
    assert c["silu_ops"] == n
    assert c["multiply_ops"] == n
    assert c["read_bytes_G_plus_U"] == 2 * n * 4
    assert c["write_bytes_A"] == n * 4
    assert c["online_memory_bytes"] == 3 * n * 4
    assert c["preprocessing_bytes"] == 0
    assert c["known_leakage"] == "permutation_invariant_statistics_preserved"


# Bonus: security profile fields on every method.
def test_security_profile_fields_present(report: dict) -> None:
    for m in report["methods"]:
        for k in (
            "security_profile",
            "implemented_security",
            "security_potential",
        ):
            assert k in m and isinstance(m[k], str)
        if m["kind"] == "lookup_proxy":
            assert m["implemented_security"] == "none_cost_proxy_only"
            assert (
                m["security_potential"]
                == "stronger_value_hiding_if_combined_with_secure_lookup"
                   "_protocol"
            )


# Bonus: recommended_use and policy validation.
def test_recommended_use_field(report: dict) -> None:
    assert (
        report["recommended_use"]
        == "cost-baseline-and-future-work-motivation"
    )


def test_invalid_policy_raises() -> None:
    with pytest.raises(ValueError):
        lookup_table_costs(
            bit_width=4, entry_bytes=2, num_layers=1,
            policy="bogus_policy", batch_size=1, seq_len=4,
            intermediate_size=4, per_channel_proxy=False,
        )


# Bonus: runner script exits zero.
def test_runner_exits_successfully() -> None:
    script = REPO_ROOT / "scripts" / "run_lookup_nonlinear_cost_proxy.py"
    result = subprocess.run(
        ["python", str(script)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "status=ok" in result.stdout
    assert "formal_security_claim=False" in result.stdout
    assert "cryptographic_lookup_implemented=False" in result.stdout
