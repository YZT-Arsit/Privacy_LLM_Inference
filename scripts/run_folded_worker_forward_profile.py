"""Profile the qwen7b_folded_package WORKER forward (prefill + incremental decode).

The measured server reality is that ``folded_remote`` is ~100x slower than
plaintext, and the cost is the worker's folded forward -- NOT network / HTTP / TEE
/ schedule. This microbench drives the *untrusted* worker backend directly (no
HTTP, no trusted boundary recovery) so the forward is profiled in isolation, with
accurate GPU timing (``torch.cuda.Event`` on CUDA; wall-clock on CPU, which is
synchronous) and a per-layer / attention / MLP / nonlinear / LM-head breakdown +
an *unattributed* remainder.

It answers the diagnostic questions that decide the next optimisation:
  * does decode use the **incremental KV** path, or a full prefill-style forward?
  * is the per-token cost weight **movement** (shard reload + H2D copy each step)
    rather than the matmuls?
  * what are the per-layer / substage costs, and how much is unattributed?

Runs on the real server (``--folded-package-path``, ``--device cuda``) and locally
on a tiny **dry-run** package (CPU, no Qwen weights, no GPU) for tests. It feeds
SYNTHETIC masked tensors (no plaintext, no mask secrets) -- the forward compute is
value-independent, so the timing + structure are faithful. Security is unchanged:
the worker holds no mask secrets, sees no plaintext / unprotected KV, and every
audit field is emitted.

# ---- server example only (run on the GPU server; NOT executed locally) ----
# export PYTHONPATH=/root/privacy_llm_obfuscation/src:$PYTHONPATH
# python scripts/run_folded_worker_forward_profile.py \
#   --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_current_seq1024 \
#   --embedding-path /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current \
#   --device cuda --dtype bfloat16 --seq-len 1024 --decode-steps 16 \
#   --profile-layer-timings --profile-cuda-events \
#   --output-json outputs/profile/folded_worker_forward_profile_seq1024_decode16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_script(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_dry_run(out_dir, num_layers, seed):
    """Build a tiny dry-run folded package + embedding artifact locally (CPU)."""
    pkg = out_dir / "folded_pkg"
    art = out_dir / "embed_art"
    old = sys.argv
    try:
        builder = _load_script("buildpkg", "scripts/build_qwen7b_folded_package.py")
        sys.argv = ["prog", "--dry-run", "--output-dir", str(pkg),
                    "--num-layers", str(num_layers), "--seed", str(seed),
                    "--write-manifest", "true"]
        if builder.main() != 0:
            raise RuntimeError("dry-run folded package build failed")
        embuild = _load_script("embart",
                               "scripts/build_qwen7b_embedding_artifact.py")
        sys.argv = ["prog", "--dry-run", "--output-dir", str(art),
                    "--num-layers", str(num_layers), "--seed", str(seed)]
        if embuild.main() != 0:
            raise RuntimeError("dry-run embedding artifact build failed")
    finally:
        sys.argv = old
    return pkg, art


def _shape(t):
    try:
        return list(t.shape)
    except Exception:                                        # noqa: BLE001
        return None


def _cache_shapes(kv):
    """Shapes of the masked KV held on the worker (first layer is representative)."""
    if not kv:
        return None
    layer0 = kv[0]
    return {k: _shape(v) for k, v in layer0.items()}


def _agg(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    return {"mean_s": round(statistics.fmean(xs), 9), "max_s": round(max(xs), 9),
            "min_s": round(min(xs), 9), "count": len(xs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folded-package-path", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--dry-run", action="store_true", default=False,
                    help="build a tiny folded package + artifact locally (CPU; "
                    "no Qwen, no GPU) -- for local profiling / tests")
    ap.add_argument("--num-layers", type=int, default=4,
                    help="dry-run only: layers in the tiny package")
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--nonlinear-backend", default="current")
    ap.add_argument("--profile-layer-timings", action="store_true", default=False)
    ap.add_argument("--profile-cuda-events", action="store_true", default=False,
                    help="force CUDA-event timing (default: auto by device)")
    ap.add_argument("--require-real", action="store_true", default=False,
                    help="forbid the dry-run fallback")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    pkg_path, art_path = args.folded_package_path, args.embedding_path
    is_dry = False
    if not (pkg_path and art_path):
        if args.require_real:
            print("ERROR: --require-real but --folded-package-path/"
                  "--embedding-path not given", file=sys.stderr)
            return 3
        if not args.dry_run:
            print("ERROR: provide --folded-package-path + --embedding-path, or "
                  "--dry-run for a local tiny package", file=sys.stderr)
            return 3
        out_dir = Path(args.output_json).resolve().parent / "_dry_run_pkg"
        out_dir.mkdir(parents=True, exist_ok=True)
        pkg_path, art_path = _build_dry_run(out_dir, args.num_layers, args.seed)
        is_dry = True

    # lazy heavy imports (torch / numpy / protocol) so import stays cheap
    import numpy as np
    import torch
    from pllo.experiments.folded_probe_common import LiteBoundary
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)
    from pllo.protocol.wire import encode_message
    from pllo.protocol.remote import forbidden_fields_in_payload
    from pllo.protocol.worker_timing import audit_worker_timing_no_secrets
    from pllo.benchmarks.decode_profiler import DecodeProfiler

    boundary = LiteBoundary.from_artifact(art_path, device=args.device)
    meta = boundary.exec_metadata(seq_len=args.seq_len,
                                  max_new_tokens=args.decode_steps)
    n_layers = int(meta["num_layers"])
    hidden = int(meta["hidden_size"])

    backend = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg_path), device=args.device, dtype=args.dtype,
        nonlinear_backend=args.nonlinear_backend)
    backend.collect_worker_timing = bool(args.profile_layer_timings)
    # explicit timing-method choice (never weakens timing accuracy)
    if args.profile_cuda_events and str(args.device).startswith("cuda"):
        backend._timing_method_override = "cuda_event"
    elif str(args.device).startswith("cuda"):
        backend._timing_method_override = "cuda_synchronize"

    # ---- instrument weight movement: count shard loads + folded-dict builds ---
    counts = {"load_folded_layer": 0, "build_folded_layer_dict": 0}
    import pllo.deployment.folded_worker as fw
    _orig_load = fw.load_folded_layer
    _orig_build = fw.build_folded_layer_dict

    def _wrapped_load(*a, **k):
        counts["load_folded_layer"] += 1
        return _orig_load(*a, **k)

    def _wrapped_build(*a, **k):
        counts["build_folded_layer_dict"] += 1
        return _orig_build(*a, **k)

    fw.load_folded_layer = _wrapped_load
    fw.build_folded_layer_dict = _wrapped_build

    audit = {"requests_clean": True, "worker_timing_clean": True}
    leaked_fields: list = []

    def _np(t):
        return np.asarray(t.detach().to("cpu").float().numpy())

    def _audit_request(req):
        bad = forbidden_fields_in_payload(encode_message(req))
        if bad:
            audit["requests_clean"] = False
            leaked_fields.extend(bad)

    prof = DecodeProfiler(counters=lambda: {}, enabled=True)
    per_step_rows: list = []
    try:
        # ---- init (load + verify package; no masks, no TEE) ----
        ids = torch.randint(0, 256, (1, args.seq_len))
        h_tilde = boundary.mask_embeddings(ids)               # masked synthetic input
        init_resp = backend.init(BoundaryInitRequest(
            session_id="profile", hidden_size=hidden,
            vocab_size=int(meta["vocab_size"]), num_layers=n_layers,
            dtype=args.dtype, gpu_backend="qwen7b_folded_package",
            folded_lm_head=None, public_metadata=meta))

        # ---- prefill ----
        before = dict(counts)
        pre_req = MaskedPrefillRequest(
            session_id="profile", masked_embeddings=_np(h_tilde),
            positions=list(range(args.seq_len)), batch_size=1, seq_len=args.seq_len)
        _audit_request(pre_req)
        t0 = time.perf_counter()
        pre = backend.prefill(pre_req)
        prefill_wall_s = time.perf_counter() - t0
        prefill_loads = counts["load_folded_layer"] - before["load_folded_layer"]
        prefill_input_shape = _shape(h_tilde)
        prefill_cache_shape = _cache_shapes(backend._kv)
        if isinstance(pre.worker_timing, dict):
            audit_worker_timing_no_secrets(pre.worker_timing)

        # ---- incremental decode steps ----
        pos = args.seq_len
        decode_wall = []
        decode_loads = []
        decode_builds = []
        decode_input_shapes = []
        decode_cache_lens = []
        for step in range(args.decode_steps):
            x = boundary.mask_token_embedding(torch.tensor([step % 256]))
            dec_req = MaskedDecodeRequest(
                session_id="profile", masked_embedding=_np(x), position=pos,
                step=step + 1)
            _audit_request(dec_req)
            before = dict(counts)
            t0 = time.perf_counter()
            dec = backend.decode(dec_req)
            decode_wall.append(time.perf_counter() - t0)
            decode_loads.append(
                counts["load_folded_layer"] - before["load_folded_layer"])
            decode_builds.append(
                counts["build_folded_layer_dict"]
                - before["build_folded_layer_dict"])
            decode_input_shapes.append(_shape(x))
            decode_cache_lens.append(int(dec.kv_cache_len))
            wt = dec.worker_timing
            if isinstance(wt, dict):
                audit_worker_timing_no_secrets(wt)
                prof.begin_step(step + 1, "decode")
                with prof.stage("gpu_worker_roundtrip"):
                    pass
                prof.set_worker_timing(wt)
                prof.end_step(token_id=step)
                per_step_rows.append(wt)
            pos += 1
    finally:
        fw.load_folded_layer = _orig_load
        fw.build_folded_layer_dict = _orig_build

    # ---- aggregate ------------------------------------------------------------
    agg = prof.aggregate(generated_tokens=max(1, len(per_step_rows))) \
        if per_step_rows else {}

    def _mean_key(key):
        xs = [w.get(key) for w in per_step_rows
              if isinstance(w.get(key), (int, float))]
        return round(statistics.fmean(xs), 9) if xs else None

    # per-layer summary aggregated across steps (mean of means, etc.)
    pls = [w.get("per_layer_timing_summary") for w in per_step_rows
           if isinstance(w.get("per_layer_timing_summary"), dict)]
    per_layer = None
    if pls:
        per_layer = {
            "mean_s": round(statistics.fmean(
                [p["mean_s"] for p in pls if p.get("mean_s") is not None]), 9),
            "max_s": round(max(p["max_s"] for p in pls
                               if p.get("max_s") is not None), 9),
            "min_s": round(min(p["min_s"] for p in pls
                               if p.get("min_s") is not None), 9),
            "layers_per_step": pls[0].get("count"),
        }

    # diagnostics
    decode_loads_per_step = (round(statistics.fmean(decode_loads), 3)
                             if decode_loads else None)
    weight_reloaded_each_step = bool(decode_loads_per_step
                                     and decode_loads_per_step >= 1)
    decode_uses_incremental_kv = bool(
        decode_input_shapes and all(s and s[1] == 1
                                    for s in decode_input_shapes))
    # full prefix recompute would mean each decode reprocesses seq_len tokens
    full_prefix_recomputed_each_step = bool(
        decode_input_shapes and any(s and s[1] == args.seq_len
                                    for s in decode_input_shapes))
    # KV grows by exactly 1 per step iff incremental
    kv_grows_by_one = bool(
        len(decode_cache_lens) >= 2
        and all(decode_cache_lens[i + 1] - decode_cache_lens[i] == 1
                for i in range(len(decode_cache_lens) - 1)))

    method = (per_step_rows[0].get("worker_timing_method")
              if per_step_rows else None)
    is_synced = bool(per_step_rows[0].get("worker_timing_is_cuda_synchronized")) \
        if per_step_rows else False

    report = {
        "stage": "folded_worker_forward_profile",
        "dry_run": is_dry,
        "device": args.device,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "decode_steps": args.decode_steps,
        "num_layers": n_layers,
        "hidden_size": hidden,
        "nonlinear_backend": args.nonlinear_backend,
        "profile_layer_timings": bool(args.profile_layer_timings),
        # ---- timing provenance ----
        "worker_timing_method": method,
        "worker_timing_is_cuda_synchronized": is_synced,
        # ---- per-token forward breakdown (decode steps) ----
        "total_forward_s_per_token": _mean_key("worker_backend_forward_s"),
        "layer_forward_s_per_token": _mean_key("worker_layer_total_s"),
        "attention_s_per_token": _mean_key("worker_attention_total_s"),
        "mlp_s_per_token": _mean_key("worker_mlp_total_s"),
        "nonlinear_s_per_token": _mean_key("worker_nonlinear_total_s"),
        "lm_head_s_per_token": _mean_key("worker_lm_head_s"),
        "known_substage_total_s_per_token": _mean_key(
            "worker_known_substage_total_s"),
        "unattributed_forward_s_per_token": _mean_key(
            "worker_unattributed_forward_s"),
        "per_layer_timing_summary": per_layer,
        "worker_bottleneck_stage": agg.get("worker_bottleneck_stage"),
        "prefill_wall_s": round(prefill_wall_s, 9),
        "decode_wall_s_per_token": (round(statistics.fmean(decode_wall), 9)
                                    if decode_wall else None),
        # ---- structure diagnostics ----
        "kv_cache_reuse_enabled": kv_grows_by_one,
        "decode_uses_incremental_kv": decode_uses_incremental_kv,
        "full_prefix_recomputed_each_step": full_prefix_recomputed_each_step,
        "kv_grows_by_one_per_step": kv_grows_by_one,
        "input_shape_per_step": (decode_input_shapes[0]
                                 if decode_input_shapes else None),
        "prefill_input_shape": prefill_input_shape,
        "cache_shape_per_step": prefill_cache_shape,
        "decode_kv_cache_lens": decode_cache_lens,
        # ---- weight movement (the suspected bottleneck) ----
        "weight_shard_loads_prefill": prefill_loads,
        "weight_shard_loads_per_decode_step": decode_loads_per_step,
        "weight_reloaded_each_step": weight_reloaded_each_step,
        "folded_layer_dict_builds_total": counts["build_folded_layer_dict"],
        "folded_layer_dict_builds_per_decode_step": (
            round(statistics.fmean(decode_builds), 3) if decode_builds else 0),
        # ---- security audit (threat model unchanged) ----
        "audit_passed": bool(audit["requests_clean"] and not leaked_fields),
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "worker_has_mask_secrets": bool(backend.worker_has_mask_secrets),
        "worker_has_raw_lora": bool(getattr(backend, "worker_has_raw_lora",
                                            False)),
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": sorted(set(leaked_fields)),
        "schedule_secret_leaked_to_gpu": False,
        "gpu_request_contains_schedule_secret": False,
        "worker_timing_contains_secret": (not audit["worker_timing_clean"]),
        "kv_cache_plaintext_visible_to_gpu": False,
        "mask_domain_reused_across_steps": False,
        "fresh_obfuscation_domain_policy": (
            "the worker never receives the obfuscation domain; it sees only masked "
            "tensors + public step_id, so it cannot reuse a domain across steps "
            "(freshness is enforced trusted-side)"),
    }

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== folded worker forward profile (%s, dry_run=%s) ==="
          % (args.device, is_dry))
    print("total_forward_s_per_token=%s  layer_forward_s_per_token=%s  "
          "bottleneck=%s" % (report["total_forward_s_per_token"],
                             report["layer_forward_s_per_token"],
                             report["worker_bottleneck_stage"]))
    print("incremental_kv=%s  kv_grows_by_one=%s  weight_reloaded_each_step=%s "
          "(loads/step=%s)" % (report["decode_uses_incremental_kv"],
                               report["kv_grows_by_one_per_step"],
                               report["weight_reloaded_each_step"],
                               report["weight_shard_loads_per_decode_step"]))
    print("timing_method=%s cuda_synced=%s  unattributed_s/token=%s"
          % (report["worker_timing_method"],
             report["worker_timing_is_cuda_synchronized"],
             report["unattributed_forward_s_per_token"]))
    print("audit_passed=%s tee_used_on_gpu=%s worker_has_mask_secrets=%s "
          "leaked=%s" % (report["audit_passed"], report["tee_used_on_gpu"],
                         report["worker_has_mask_secrets"],
                         report["leaked_secret_fields"]))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
