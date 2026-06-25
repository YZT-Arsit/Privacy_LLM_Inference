"""Local (CPU, dry-run) tests for the folded-worker forward microbench.

Drives scripts/run_folded_worker_forward_profile.py over a tiny dry-run folded
package on CPU -- no Qwen weights, no GPU, no server paths. Verifies the profile
emits the required fields, correctly detects the incremental-KV decode path and
the per-step weight reload (the suspected bottleneck), and passes the security
audit.

Run: python -m pytest tests/test_folded_worker_forward_profile.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


def _profile(tmp_path, extra=None):
    mod = _load("fwd_prof", "scripts/run_folded_worker_forward_profile.py")
    out = tmp_path / "profile.json"
    argv = ["x", "--dry-run", "--num-layers", "4", "--device", "cpu",
            "--dtype", "float32", "--seq-len", "8", "--decode-steps", "4",
            "--profile-layer-timings", "--output-json", str(out)] + (extra or [])
    rc = _run(mod, argv)
    return rc, json.loads(out.read_text())


def test_profile_emits_required_fields(tmp_path) -> None:
    rc, r = _profile(tmp_path)
    assert rc == 0
    for k in ("total_forward_s_per_token", "layer_forward_s_per_token",
              "attention_s_per_token", "mlp_s_per_token", "nonlinear_s_per_token",
              "lm_head_s_per_token", "known_substage_total_s_per_token",
              "unattributed_forward_s_per_token", "per_layer_timing_summary",
              "worker_bottleneck_stage", "worker_timing_method",
              "worker_timing_is_cuda_synchronized", "kv_cache_reuse_enabled",
              "decode_uses_incremental_kv", "full_prefix_recomputed_each_step",
              "input_shape_per_step", "cache_shape_per_step",
              "weight_shard_loads_per_decode_step", "weight_reloaded_each_step"):
        assert k in r, "missing %s" % k
    s = r["per_layer_timing_summary"]
    assert set(("mean_s", "max_s", "min_s")) <= set(s)


def test_profile_detects_incremental_kv_not_full_prefill(tmp_path) -> None:
    _, r = _profile(tmp_path)
    # decode feeds ONE token + threads the KV; it is NOT a full prefill recompute
    assert r["decode_uses_incremental_kv"] is True
    assert r["kv_grows_by_one_per_step"] is True
    assert r["full_prefix_recomputed_each_step"] is False
    assert r["input_shape_per_step"][1] == 1            # [B, 1, H]


def test_profile_detects_per_step_weight_reload(tmp_path) -> None:
    # the real bottleneck: every decode step reloads ALL shards from disk
    _, r = _profile(tmp_path)
    assert r["weight_reloaded_each_step"] is True
    assert r["weight_shard_loads_per_decode_step"] == r["num_layers"]
    assert r["folded_layer_dict_builds_per_decode_step"] == r["num_layers"]


def test_profile_bottleneck_not_a_substage(tmp_path) -> None:
    # substages are tiny vs forward -> bottleneck must be layer_total / unattributed
    _, r = _profile(tmp_path)
    assert r["worker_bottleneck_stage"] in (
        "worker_layer_total_s", "worker_backend_forward_unattributed")
    assert r["worker_bottleneck_stage"] not in (
        "worker_attention_total_s", "worker_mlp_total_s")
    assert r["worker_timing_method"] == "wall_clock"     # CPU (synchronous)


def test_resident_removes_per_step_weight_movement(tmp_path) -> None:
    rc, r = _profile(tmp_path, ["--resident-folded-weights"])
    assert rc == 0
    assert r["resident_folded_weights"] is True
    assert r["resident_cache_active"] is True
    assert r["resident_cache_oom"] is False
    assert r["resident_cache_fallback_used"] is False
    assert r["resident_cache_num_layers"] == r["num_layers"]
    assert r["resident_weight_init_latency_s"] is not None
    assert r["resident_weight_memory_gb"] is not None
    # the whole point: no per-token reload / build / H2D copy
    assert r["weight_reloaded_each_step"] is False
    assert r["weight_shard_loads_per_decode_step"] == 0
    assert r["folded_layer_dict_builds_per_decode_step"] == 0
    assert r["cpu_to_gpu_weight_copies_per_decode_step"] == 0


def test_resident_correctness_matches_non_resident(tmp_path) -> None:
    _, r = _profile(tmp_path, ["--resident-folded-weights"])
    c = r["correctness"]
    assert c["output_shapes_match"] is True
    # identical synthetic inputs -> bit-identical (or dtype-error) logits
    assert c["max_abs_err_resident_vs_nonresident"] == 0.0
    # KV behaviour is unchanged by residency
    assert r["decode_uses_incremental_kv"] is True
    assert r["kv_grows_by_one_per_step"] is True
    assert r["full_prefix_recomputed_each_step"] is False


def test_resident_reports_comparison_and_speedup(tmp_path) -> None:
    _, r = _profile(tmp_path, ["--resident-folded-weights"])
    cmp = r["comparison_resident_vs_non_resident"]
    assert cmp is not None
    assert cmp["non_resident_weight_reloaded_each_step"] is True
    assert cmp["resident_weight_reloaded_each_step"] is False
    assert cmp["forward_speedup_x"] is not None


def test_resident_audit_unchanged_and_perf_not_security(tmp_path) -> None:
    _, r = _profile(tmp_path, ["--resident-folded-weights"])
    assert r["audit_passed"] is True
    assert r["tee_used_on_gpu"] is False
    assert r["worker_has_mask_secrets"] is False
    assert r["worker_has_raw_lora"] is False
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []
    assert r["schedule_secret_leaked_to_gpu"] is False
    assert r["gpu_request_contains_schedule_secret"] is False
    assert r["kv_cache_plaintext_visible_to_gpu"] is False
    assert r["mask_domain_reused_across_steps"] is False
    # residency must never be reported as a security improvement
    assert r["optimization_is_performance_only"] is True
    assert r["resident_weight_security_audit_passed"] is True


def test_resident_visible_via_health_and_worker_timing(tmp_path) -> None:
    """resident_folded_weights must be observable by a trusted client over /health
    and (when timing requested) in the worker_timing metadata -- no secret cross."""
    import numpy as np
    from pllo.experiments.folded_probe_common import LiteBoundary
    from pllo.protocol.remote import GpuWorkerServer, RemoteGpuWorker
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)
    from pllo.protocol.worker_timing import audit_worker_timing_no_secrets

    pkg, art = _build_pkg_art(tmp_path)
    boundary = LiteBoundary.from_artifact(art, device="cpu")
    meta = boundary.exec_metadata(seq_len=6, max_new_tokens=2)
    srv = GpuWorkerServer(
        host="127.0.0.1", port=0, backend_name="qwen7b_folded_package",
        backend_kwargs={"folded_package_path": str(pkg), "device": "cpu",
                        "dtype": "float32", "resident_folded_weights": True},
        audit=True)
    srv.start_background()
    url = "http://127.0.0.1:%d" % srv.port

    def _np(t):
        return np.asarray(t.detach().to("cpu").float().numpy())

    try:
        c = RemoteGpuWorker(url, "qwen7b_folded_package",
                            request_worker_timing=True)
        c.init(BoundaryInitRequest(
            session_id="s", hidden_size=int(meta["hidden_size"]),
            vocab_size=int(meta["vocab_size"]), num_layers=int(meta["num_layers"]),
            dtype="float32", gpu_backend="qwen7b_folded_package",
            folded_lm_head=None, public_metadata=meta))
        ids = __import__("torch").randint(0, 256, (1, 6))
        h = boundary.mask_embeddings(ids)
        c.prefill(MaskedPrefillRequest(
            session_id="s", masked_embeddings=_np(h),
            positions=list(range(6)), batch_size=1, seq_len=6))
        x = boundary.mask_token_embedding(__import__("torch").tensor([1]))
        dec = c.decode(MaskedDecodeRequest(
            session_id="s", masked_embedding=_np(x), position=6, step=1))
        health = c.health()
        c.close()
    finally:
        srv.shutdown()

    # /health exposes the public resident status + per-decode counters (the exact
    # worker metadata that real_predictors.stats() forwards into the end-to-end
    # ifeval report -- so they are no longer null when resident is on)
    rs = health.get("resident_status") or {}
    assert rs.get("resident_folded_weights") is True
    assert rs.get("resident_cache_active") is True
    assert rs.get("weight_reloaded_each_step") is False
    assert rs.get("weight_shard_loads_per_decode_step") == 0
    assert rs.get("folded_layer_dict_builds_per_decode_step") == 0
    assert rs.get("cpu_to_gpu_weight_copies_per_decode_step") == 0
    assert rs.get("resident_cache_device") == "cpu"
    assert rs.get("resident_cache_dtype") is not None
    # worker_timing carries the resident flag (and no secret)
    wt = dec.worker_timing
    assert isinstance(wt, dict)
    assert wt.get("resident_folded_weights") is True
    assert audit_worker_timing_no_secrets(wt)["ok"] is True


def _build_pkg_art(tmp_path):
    mod = _load("fwd_prof2", "scripts/run_folded_worker_forward_profile.py")
    return mod._build_dry_run(tmp_path / "_pkg", 4, 2035)


def _run_backend(pkg, art, *, resident, device="cpu", dtype="float32",
                 seq_len=8, steps=3):
    import numpy as np
    import torch
    from pllo.experiments.folded_probe_common import LiteBoundary
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)
    boundary = LiteBoundary.from_artifact(art, device=device)
    meta = boundary.exec_metadata(seq_len=seq_len, max_new_tokens=steps)
    b = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg), device=device, dtype=dtype,
        resident_folded_weights=resident)

    def _np(t):
        return np.asarray(t.detach().to("cpu").float().numpy())

    torch.manual_seed(7)
    ids = torch.randint(0, 256, (1, seq_len))
    h = boundary.mask_embeddings(ids)
    b.init(BoundaryInitRequest(
        session_id="s", hidden_size=int(meta["hidden_size"]),
        vocab_size=int(meta["vocab_size"]), num_layers=int(meta["num_layers"]),
        dtype=dtype, gpu_backend="qwen7b_folded_package", folded_lm_head=None,
        public_metadata=meta))
    outs = [np.asarray(b.prefill(MaskedPrefillRequest(
        session_id="s", masked_embeddings=_np(h),
        positions=list(range(seq_len)), batch_size=1, seq_len=seq_len
    )).masked_logits)]
    pos = seq_len
    for step in range(steps):
        x = boundary.mask_token_embedding(torch.tensor([step % 256]))
        outs.append(np.asarray(b.decode(MaskedDecodeRequest(
            session_id="s", masked_embedding=_np(x), position=pos, step=step + 1
        )).masked_logits))
        pos += 1
    return b, outs


def test_backend_resident_equivalence(tmp_path) -> None:
    """The resident backend produces the SAME masked logits as the per-step path
    for prefill AND every decode step (correctness at the backend boundary)."""
    import numpy as np
    pkg, art = _build_pkg_art(tmp_path)
    _, base = _run_backend(pkg, art, resident=False)
    b_res, res = _run_backend(pkg, art, resident=True)
    assert len(base) == len(res)
    for a, c in zip(base, res):
        assert a.shape == c.shape
        assert float(np.max(np.abs(a.astype("float64") - c.astype("float64")))) \
            == 0.0
    # resident cache is active and holds only public folded operators
    st = b_res.resident_status()
    assert st["resident_cache_active"] is True
    assert b_res.worker_has_mask_secrets is False
    # describe() surfaces the resident status without any secret
    d = b_res.describe()
    assert d["resident_folded_weights"] is True
    assert d["worker_has_mask_secrets"] is False


def test_backend_resident_decode_counters(tmp_path) -> None:
    """resident_status() reports the per-decode weight-movement counters measured
    from the actual path: resident -> 0; per-step path -> num_layers."""
    pkg, art = _build_pkg_art(tmp_path)
    b_res, _ = _run_backend(pkg, art, resident=True, steps=3)
    rs = b_res.resident_status()
    assert rs["weight_shard_loads_per_decode_step"] == 0
    assert rs["folded_layer_dict_builds_per_decode_step"] == 0
    assert rs["cpu_to_gpu_weight_copies_per_decode_step"] == 0
    assert rs["weight_reloaded_each_step"] is False
    assert rs["resident_cache_device"] == "cpu"
    assert rs["resident_cache_dtype"] is not None

    b_base, _ = _run_backend(pkg, art, resident=False, steps=3)
    rb = b_base.resident_status()
    nl = b_base._num_layers
    assert rb["weight_shard_loads_per_decode_step"] == nl
    assert rb["folded_layer_dict_builds_per_decode_step"] == nl
    assert rb["cpu_to_gpu_weight_copies_per_decode_step"] == nl
    assert rb["weight_reloaded_each_step"] is True


def test_profile_security_audit_clean(tmp_path) -> None:
    _, r = _profile(tmp_path)
    assert r["audit_passed"] is True
    assert r["tee_used_on_gpu"] is False
    assert r["worker_has_mask_secrets"] is False
    assert r["worker_has_raw_lora"] is False
    assert r["gpu_visible_plaintext_fields"] == []
    assert r["leaked_secret_fields"] == []
    assert r["schedule_secret_leaked_to_gpu"] is False
    assert r["gpu_request_contains_schedule_secret"] is False
    assert r["worker_timing_contains_secret"] is False
    assert r["kv_cache_plaintext_visible_to_gpu"] is False
    assert r["mask_domain_reused_across_steps"] is False
