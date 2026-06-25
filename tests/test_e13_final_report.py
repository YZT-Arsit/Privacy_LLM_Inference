"""Task H: E13 final consolidation report. stdlib only.

Run: python -m pytest tests/test_e13_final_report.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.e13_final_report import (  # noqa: E402
    LIMITATIONS,
    build_e13_report,
    render_e13_md,
)

_SEC = {"gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
        "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
        "audit_passed": True}


def _decode(dry=False, lora=False, attested=False):
    r = dict(_SEC, stage=("qwen7b_lora_folded_remote_decode_probe" if lora
                          else "qwen7b_folded_remote_package_decode"),
             gpu_backend="qwen7b_folded_package", gpu_worker_remote=True,
             dry_run=dry, boundary_mode="lite", folded_package_loaded=True,
             folded_package_valid=True, package_backed_prefill=True,
             package_backed_decode=True, tokens_exact_match=True,
             token_match_rate=1.0, lora_enabled=lora, worker_has_raw_lora=False)
    if attested:
        r.update(boundary_attested=True, runtime_hash_bound=True,
                 attestation={"tee_type": "tdx", "verified": True,
                              "available": True, "mr_td_match": True})
    return r


def _inputs():
    return {
        "correctness": [_decode(attested=True)],
        "e9": [{"stage": "e9_task_utility_benchmark", "dataset": "mmlu",
                "backend": "tdx_attested_remote", "task_type": "multiple_choice",
                "metric_name": "accuracy", "metric_value": 0.62,
                "num_examples": 200, "dry_run": False, "paper_ready": True}],
        "e10": {"stage": "e10_lora_utility_benchmark", "dataset_name": "sst2",
                "task_type": "classification", "metric_name": "accuracy",
                "base_metric": 0.7, "plaintext_lora_metric": 0.9,
                "folded_lora_metric": 0.895, "tdx_attested_folded_lora_metric":
                None, "lora_gain_plaintext": 0.2, "lora_gain_folded": 0.195,
                "lora_gain_preserved_ratio": 0.975,
                "folded_lora_preserves_gain": True, "utility_preserved": True,
                "security_ok": True, "dry_run": False, "paper_ready": True},
        "security_negative": {"stage": "security_negative_tests",
                              "num_cases": 14, "num_pass": 14,
                              "all_passed": True,
                              "cases": [{"negative_test_name": "x",
                                         "expected_failure": True,
                                         "actually_failed": True, "pass": True}]},
        "latency": {"rows": [{"backend": "plaintext_h800", "total_latency_s": 1.0,
                              "latency_per_token_s": 0.25, "tokens_per_s": 4.0,
                              "overhead_vs_plaintext_h800": 1.0,
                              "peak_gpu_memory_mb": 2000},
                             {"backend": "folded_h800_remote",
                              "total_latency_s": 1.1,
                              "overhead_vs_plaintext_h800": 1.1}]},
        "e4": {"folded_package_size_gb": 26.0, "setup_time_s": 120.0},
        "e8": {"cost": {"folded_lora_package_size_gb": 0.05,
                        "decode_latency_overhead_s": 0.1}},
        "results": [],
    }


def test_e13_build_has_all_tables() -> None:
    rep = build_e13_report(_inputs())
    assert rep["stage"] == "e13_final_evaluation_report"
    assert len(rep["correctness"]) == 1
    assert len(rep["public_task_utility"]) == 1
    assert rep["lora_utility"]["utility_preserved"] is True
    assert len(rep["security_audit_matrix"]) >= 1
    assert rep["security_negative_tests"]["all_passed"] is True
    assert len(rep["deployment_truth"]) >= 1
    assert isinstance(rep["latency_overhead"], list)
    assert rep["setup_cost"]["folded_package_size_gb"] == 26.0
    assert rep["limitations"] == LIMITATIONS
    # claims computed inline; attested decode supports the attested claim
    sup = rep["paper_claims"]["supported_claims"]
    assert "no_lora_tdx_attested_remote_package_decode" in sup


def test_e13_md_has_ten_sections() -> None:
    md = render_e13_md(build_e13_report(_inputs()))
    for h in ("## 1. Correctness", "## 2. Public task utility preservation",
              "## 3. LoRA utility preservation", "## 4. Security audit matrix",
              "## 5. Security negative tests", "## 6. Deployment truth",
              "## 7. Latency / overhead", "## 8. Setup / provisioning cost",
              "## 9. Supported paper claims", "## 10. Limitations"):
        assert h in md
    # limitations text present
    assert "research prototype, not production transport" in md


def test_e13_empty_inputs_safe() -> None:
    rep = build_e13_report({})
    md = render_e13_md(rep)
    assert "## 10. Limitations" in md
    assert rep["lora_utility"] is None


def test_e13_script(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "e13", REPO_ROOT / "scripts" / "run_e13_final_evaluation_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    corr = tmp_path / "c.json"
    corr.write_text(json.dumps(_decode(attested=True)))
    oj = tmp_path / "e13.json"
    om = tmp_path / "e13.md"
    old = sys.argv
    try:
        sys.argv = ["x", "--correctness-json", str(corr),
                    "--output-json", str(oj), "--output-md", str(om)]
        rc = mod.main()
    finally:
        sys.argv = old
    assert rc == 0
    assert "## 10. Limitations" in om.read_text()
