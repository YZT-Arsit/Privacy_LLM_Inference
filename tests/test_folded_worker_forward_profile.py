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
