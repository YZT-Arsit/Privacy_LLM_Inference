"""Dry-run tests for E3 (scaling), E4 (setup cost), E5 (final comparison).

The aggregation/reporting logic is exercised with tiny in-memory dicts + JSON
fixtures -- NO H800, TDX, CUDA, or Qwen checkpoint required. A single torch-gated
integration test drives the E3 runner against a live tiny CPU worker.

Run: python -m pytest tests/test_e3_e4_e5_reports.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.e3_remote_decode_scaling import (  # noqa: E402
    E3_ROW_FIELDS,
    build_e3_summary,
    render_e3_csv,
    render_e3_md,
    run_e3_scaling,
)
from pllo.experiments.e4_setup_cost import (  # noqa: E402
    build_e4_report,
    gather_facts,
    render_e4_csv,
    render_e4_md,
)
from pllo.experiments.e5_final_comparison import (  # noqa: E402
    build_e5_report,
    render_e5_md,
    render_e5_tex,
)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _main(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


def _fake_report(*, seq_len, max_new_tokens):
    """A demo-shaped decode report (the fields E3 extracts)."""
    return {
        "stage": "qwen7b_folded_remote_package_decode",
        "boundary_mode": "lite", "gpu_worker_remote": True,
        "gpu_backend": "qwen7b_folded_package", "seq_len": seq_len,
        "tokens_exact_match": True, "token_match_rate": 1.0,
        "reference_basis": "expected_token_ids",
        "package_backed_prefill": True, "package_backed_decode": True,
        "folded_package_loaded": True, "folded_package_valid": True,
        "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
        "gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
        "audit_passed": True,
        "latency_s": 0.5 * max_new_tokens,
        "trusted_bytes": 1000 * max_new_tokens, "gpu_bytes": 2000 * max_new_tokens,
        "boundary_calls": {"init": 1, "prefill": 1,
                           "decode": max(0, max_new_tokens - 1)},
        "gpu_calls": {"BoundaryInitRequest": 1, "MaskedPrefillRequest": 1,
                      "MaskedDecodeRequest": max(0, max_new_tokens - 1)},
        "peak_gpu_memory_mb": 2140.5,
    }


# ---------------------------------------------------------------------------
# E3 (library, torch-free via injected decode_fn)
# ---------------------------------------------------------------------------


def test_e3_scaling_rows_and_summary() -> None:
    rows = run_e3_scaling(_fake_report, seq_lens=[128],
                          max_new_tokens_list=[1, 4, 8, 16])
    assert len(rows) == 4
    for r, n in zip(rows, [1, 4, 8, 16]):
        assert r["experiment"] == "E3"
        assert r["stage"] == "remote_package_decode_scaling"
        assert r["max_new_tokens"] == n
        assert r["seq_len"] == 128
        assert r["latency_per_token_s"] == pytest.approx(0.5)
        for f in E3_ROW_FIELDS:
            assert f in r
    summary = build_e3_summary(rows)
    assert summary["num_rows"] == 4
    assert summary["all_pass"] is True
    assert summary["all_security_ok"] is True
    assert len(summary["latency_table"]) == 4
    assert summary["boundary_call_table"][1]["boundary_calls_total"] == 1 + 1 + 3

    csv_txt = render_e3_csv(rows)
    assert "max_new_tokens" in csv_txt.splitlines()[0]
    md = render_e3_md(rows, summary, {"gpu_backend": "qwen7b_folded_package",
                                      "boundary_mode": "lite",
                                      "dry_run": True})
    assert "Latency scaling" in md and "Security audit" in md


def test_e3_security_failure_flips_pass() -> None:
    def bad(*, seq_len, max_new_tokens):
        r = _fake_report(seq_len=seq_len, max_new_tokens=max_new_tokens)
        r["leaked_secret_fields"] = ["seed"]
        return r
    rows = run_e3_scaling(bad, seq_lens=[8], max_new_tokens_list=[1])
    summary = build_e3_summary(rows)
    assert summary["all_pass"] is False
    assert summary["all_security_ok"] is False


def test_e3_no_reference_does_not_fail() -> None:
    def noref(*, seq_len, max_new_tokens):
        r = _fake_report(seq_len=seq_len, max_new_tokens=max_new_tokens)
        r["tokens_exact_match"] = None
        r["token_match_rate"] = None
        return r
    rows = run_e3_scaling(noref, seq_lens=[8], max_new_tokens_list=[1, 4])
    summary = build_e3_summary(rows)
    assert summary["all_pass"] is True          # security holds; correctness ungated


# ---------------------------------------------------------------------------
# E4 (setup cost, torch-free JSON fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture()
def e4_fixtures(tmp_path):
    build = {"size_gb": 26.339369, "generation_time_s": 59.6, "num_layers": 28,
             "num_shards": 29, "manifest_hash": "90184a6b", "store_dtypes": ["F32"]}
    verify = {"package_valid": True}
    inspection = {"store_dtypes": ["F32"], "total_size_gb": 26.339369}
    load_probe = {"load_time_s": 12.3, "num_layers": 28}
    art = tmp_path / "boundary_art"
    art.mkdir()
    (art / "boundary_meta.json").write_text(json.dumps({
        "size_gb": 1.066, "tensors_sha256": "deadbeef",
        "contains_mask_secrets": True, "trusted_only": True}), encoding="utf-8")
    (art / "boundary.safetensors").write_bytes(b"\x00" * 2048)
    return build, verify, inspection, load_probe, art


def test_e4_gather_and_report(e4_fixtures) -> None:
    build, verify, inspection, load_probe, art = e4_fixtures
    facts = gather_facts(
        folded_package_path=None, embedding_artifact_path=str(art),
        build_json=build, verify_json=verify, inspection_json=inspection,
        load_probe_json=load_probe)
    assert facts["folded_package_size_gb"] == pytest.approx(26.339369)
    assert facts["folded_package_store_dtype"] == "F32"
    assert facts["folded_package_size_if_bf16_gb"] == pytest.approx(13.1696845)
    assert facts["num_layers"] == 28
    assert facts["num_shards"] == 29
    assert facts["package_verify_passed"] is True
    assert facts["package_load_time_s"] == 12.3
    assert facts["boundary_embedding_artifact_hash"] == "deadbeef"
    assert facts["boundary_artifact_contains_mask_secrets"] is True
    assert facts["boundary_artifact_trusted_only"] is True
    assert facts["boundary_embedding_artifact_size_gb"] > 0

    report = build_e4_report(facts, bandwidth_mbps_list=[100, 1000],
                             sessions_list=[1, 10, 100])
    assert report["experiment"] == "E4"
    assert report["one_time_setup_s"] == pytest.approx(59.6 + 12.3)
    tt = report["transfer_estimates"]["folded_package"]
    assert len(tt) == 2
    # 26.339369 GB * 8 bits / 100 Mbps ~ 2262s
    assert tt[0]["transfer_time_s"] == pytest.approx(
        26.339369 * (1024 ** 3) * 8 / (100 * 1e6), rel=1e-6)
    am = {a["sessions"]: a["amortized_setup_cost_s"]
          for a in report["amortized_setup_cost"]}
    assert am[1] == pytest.approx(71.9)
    assert am[10] == pytest.approx(7.19)
    assert "float32" in report["size_explanation_note"]
    assert "26.34GB" in report["size_explanation_note"]
    md = render_e4_md(report)
    assert "Amortized setup cost" in md and "Transfer time" in md
    assert "section,key,value,source" in render_e4_csv(report).splitlines()[0]


def test_e4_missing_inputs_stay_none() -> None:
    facts = gather_facts()
    rep = build_e4_report(facts, bandwidth_mbps_list=[100], sessions_list=[1])
    assert rep["folded_package_size_gb"] is None
    assert rep["one_time_setup_s"] is None
    assert rep["transfer_estimates"]["folded_package"][0]["transfer_time_s"] is None
    assert rep["amortized_setup_cost"][0]["amortized_setup_cost_s"] is None


def test_e4_script_end_to_end(e4_fixtures, tmp_path) -> None:
    build, verify, inspection, load_probe, art = e4_fixtures
    bj = tmp_path / "b.json"; bj.write_text(json.dumps(build))
    vj = tmp_path / "v.json"; vj.write_text(json.dumps(verify))
    ij = tmp_path / "i.json"; ij.write_text(json.dumps(inspection))
    lj = tmp_path / "l.json"; lj.write_text(json.dumps(load_probe))
    mod = _load("e4", "scripts/run_e4_setup_cost_report.py")
    oj = tmp_path / "e4.json"
    rc = _main(mod, ["prog", "--embedding-artifact-path", str(art),
                     "--build-json", str(bj), "--verify-json", str(vj),
                     "--inspection-json", str(ij), "--load-probe-json", str(lj),
                     "--bandwidth-mbps-list", "100,1000",
                     "--amortize-sessions-list", "1,100",
                     "--output-json", str(oj),
                     "--output-md", str(tmp_path / "e4.md"),
                     "--output-csv", str(tmp_path / "e4.csv")])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["experiment"] == "E4"
    assert r["folded_package_size_gb"] == pytest.approx(26.339369)


# ---------------------------------------------------------------------------
# E5 (final comparison, torch-free JSON fixtures)
# ---------------------------------------------------------------------------


def _e5_inputs():
    decode_common = {
        "latency_s": 1.0, "trusted_bytes": 10, "gpu_bytes": 20,
        "peak_gpu_memory_mb": 2140.5, "package_size_gb": 26.339369,
        "gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
        "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
        "audit_passed": True, "boundary_calls": {"init": 1}}
    return {
        "e1": {"token_match_rate": 1.0},
        "e2": {"token_match_rate": 1.0},
        "local_prefill": {"num_exec_layers": 28, "allclose": True,
                          "max_abs_error": 0.002197,
                          "relative_l2_error": 3.152e-06},
        "local_logits": {"top1_match": True, "next_token_match": True,
                         "topk_overlap": 1.0},
        "local_decode": dict(decode_common, tokens_exact_match=True,
                             token_match_rate=1.0, stage="local_decode"),
        "remote_scaling": {"summary": {"all_pass": True, "num_rows": 4,
                                       "latency_table": [
                                           {"seq_len": 128, "max_new_tokens": 1,
                                            "latency_s": 0.5,
                                            "latency_per_token_s": 0.5}]},
                           "meta": {"max_new_tokens_list": [1, 4, 8, 16]}},
        "tdx_lite": dict(decode_common, tokens_exact_match=True,
                         token_match_rate=1.0, boundary_mode="lite",
                         stage="qwen7b_folded_remote_package_decode"),
        "tdx_attested": dict(decode_common, tokens_exact_match=True,
                             max_new_tokens=4, boundary_attested=True,
                             runtime_hash_bound=True,
                             attestation={"mr_td_match": True},
                             stage="qwen7b_folded_remote_package_decode"),
        "setup_cost": {"folded_package_size_gb": 26.339369,
                       "folded_package_size_if_bf16_gb": 13.1696845,
                       "generation_time_s": 59.6, "package_load_time_s": 12.3,
                       "one_time_setup_s": 71.9,
                       "boundary_embedding_artifact_size_gb": 1.066},
    }


def test_e5_build_report_sections() -> None:
    report = build_e5_report(_e5_inputs())
    assert report["experiment"] == "E5"
    names = [r["name"] for r in report["correctness"]]
    assert any("k=28" in n for n in names)
    assert any("TDX-attested" in n for n in names)
    # all provided correctness rows pass
    for r in report["correctness"]:
        if r["provided"]:
            assert r["pass"] in (True, None) or r["pass"] is True
    assert len(report["security_matrix"]["matrix"]) == 11
    assert report["security_matrix"]["audit_cross_check_ok"] is True
    assert len(report["deployment"]) == 4
    assert len(report["limitations"]) >= 4
    # deployment: TDX scenarios do not need full checkpoint / 26GB package
    tdx = [d for d in report["deployment"] if "TDX" in d["scenario"]]
    assert all(d["boundary_needs_full_checkpoint"] is False for d in tdx)
    assert all(d["boundary_needs_full_26gb_package"] is False for d in tdx)


def test_e5_render_md_and_tex() -> None:
    report = build_e5_report(_e5_inputs())
    md = render_e5_md(report)
    for h in ("1. Correctness", "2. Deployment", "3. Security matrix",
              "4. Cost", "5. Limitations"):
        assert h in md
    tex = render_e5_tex(report)
    assert "\\begin{tabular}" in tex and "Security matrix" in tex


def test_e5_missing_inputs_not_assumed() -> None:
    report = build_e5_report({})           # nothing provided
    assert all(v is False for v in report["inputs_provided"].values())
    for r in report["correctness"]:
        assert r["provided"] is False
        assert r["pass"] is None
    # no decode reports -> cross-check is None (not a false pass)
    assert report["security_matrix"]["audit_cross_check_ok"] is None


def test_e5_script_end_to_end(tmp_path) -> None:
    inp = _e5_inputs()
    paths = {}
    for key, fname in [("e1", "e1"), ("local_prefill", "lp"),
                       ("local_decode", "ld"), ("remote_scaling", "rs"),
                       ("tdx_lite", "tl"), ("tdx_attested", "ta"),
                       ("setup_cost", "sc")]:
        p = tmp_path / (fname + ".json")
        p.write_text(json.dumps(inp[key]))
        paths[key] = p
    mod = _load("e5", "scripts/run_e5_final_comparison_report.py")
    oj = tmp_path / "e5.json"
    rc = _main(mod, [
        "prog", "--e1-json", str(paths["e1"]),
        "--local-prefill-json", str(paths["local_prefill"]),
        "--local-decode-json", str(paths["local_decode"]),
        "--remote-scaling-json", str(paths["remote_scaling"]),
        "--tdx-lite-json", str(paths["tdx_lite"]),
        "--tdx-attested-json", str(paths["tdx_attested"]),
        "--setup-cost-json", str(paths["setup_cost"]),
        "--output-json", str(oj), "--output-md", str(tmp_path / "e5.md"),
        "--output-tex", str(tmp_path / "e5.tex"),
        "--paper-ready-md", str(tmp_path / "paper.md")])
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["experiment"] == "E5"
    assert r["security_matrix"]["audit_cross_check_ok"] is True
    assert (tmp_path / "e5.tex").read_text().count("tabular") >= 2
    assert "paper-ready final evaluation" in (
        tmp_path / "paper.md").read_text().lower()


# ---------------------------------------------------------------------------
# E3 runner integration against a live tiny CPU worker (torch-gated)
# ---------------------------------------------------------------------------


def test_e3_runner_live_tiny_worker(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch

    from pllo.experiments.folded_probe_common import LiteBoundary, tiny_model
    from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession
    from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig
    from pllo.protocol.remote import GpuWorkerServer

    seed, n_layers, seq_len = 2035, 4, 8
    pkg = tmp_path / "pkg4"
    builder = _load("buildpkg_e3", "scripts/build_qwen7b_folded_package.py")
    assert _main(builder, ["prog", "--dry-run", "--output-dir", str(pkg),
                           "--num-layers", str(n_layers), "--seed", str(seed),
                           "--write-manifest", "true"]) == 0
    art = tmp_path / "art"
    embuild = _load("buildemb_e3", "scripts/build_qwen7b_embedding_artifact.py")
    assert _main(embuild, ["prog", "--dry-run", "--output-dir", str(art),
                           "--seed", str(seed)]) == 0

    # expected tokens from a full in-process reference (same tiny model + ids)
    torch.manual_seed(11)
    ids = torch.randint(0, 256, (1, seq_len))
    model, mc = tiny_model()
    cfg = MemoryOptimizedConfig(num_layers=n_layers, batch_size=1,
                                seq_len=seq_len, max_new_tokens=4, device="cpu",
                                dtype="float32", folding_dtype="float32",
                                folded_weight_device="cpu", seed=seed)
    session = MaskedQwenSession(model, mc, cfg)
    h = session.mask_embeddings(ids)
    out = session.worker_prefill(h)
    tok = int(session.recover(out["logits_tilde"][:, -1, :]).argmax(-1).item())
    expected, kv, pos = [tok], out["kv"], seq_len
    for _ in range(3):
        x = session.mask_token_embedding(torch.tensor([tok]))
        out = session.worker_decode(x, kv, pos); kv = out["kv"]
        tok = int(session.recover(out["logits_tilde"][:, -1, :]).argmax(-1).item())
        expected.append(tok); pos += 1
    ids_csv = ",".join(str(int(x)) for x in ids.reshape(-1).tolist())

    server = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg), "device": "cpu",
                        "dtype": "float32"}, audit=True)
    server.start_background()
    url = "http://127.0.0.1:%d" % server.port
    e3 = _load("e3_live", "scripts/run_e3_remote_decode_scaling.py")
    oj = tmp_path / "e3.json"
    try:
        rc = _main(e3, [
            "prog", "--gpu-worker-url", url, "--gpu-backend",
            "qwen7b_folded_package", "--embedding-path", str(art),
            "--skip-reference", "true", "--input-ids", ids_csv,
            "--expected-token-ids", ",".join(str(t) for t in expected),
            "--max-new-tokens-list", "1,2", "--seq-len", str(seq_len),
            "--dtype", "float32", "--device", "cpu", "--audit", "true",
            "--output-json", str(oj), "--output-csv", str(tmp_path / "e3.csv"),
            "--output-md", str(tmp_path / "e3.md")])
    finally:
        server.shutdown()
    assert rc == 0
    r = json.loads(oj.read_text())
    assert r["experiment"] == "E3"
    assert r["summary"]["all_pass"] is True
    assert len(r["rows"]) == 2
    for row in r["rows"]:
        assert row["tokens_exact_match"] is True
        assert row["worker_has_mask_secrets"] is False
        assert row["tee_used_on_gpu"] is False
        assert row["gpu_visible_plaintext_fields"] == []
        assert row["leaked_secret_fields"] == []
