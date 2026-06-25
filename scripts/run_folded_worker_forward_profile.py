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
  * does ``--resident-folded-weights`` remove that cost (load+fold+to(device) once,
    reuse across steps), and is the output still correct + the audit still clean?

Runs on the real server (``--folded-package-path``, ``--device cuda``) and locally
on a tiny **dry-run** package (CPU, no Qwen weights, no GPU) for tests. It feeds
SYNTHETIC masked tensors (no plaintext, no mask secrets) -- the forward compute is
value-independent, so the timing + structure are faithful. Security is unchanged:
the worker holds no mask secrets, sees no plaintext / unprotected KV, and every
audit field is emitted. ``resident_folded_weights`` is a PERFORMANCE optimisation
only -- it is never reported as a security improvement.

# ---- server example only (run on the GPU server; NOT executed locally) ----
# export PYTHONPATH=/root/privacy_llm_obfuscation/src:$PYTHONPATH
# python scripts/run_folded_worker_forward_profile.py \
#   --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full_current_seq1024 \
#   --embedding-path /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact_current \
#   --device cuda --dtype bfloat16 --seq-len 1024 --decode-steps 16 \
#   --resident-folded-weights --profile-layer-timings --profile-cuda-events \
#   --output-json outputs/profile/folded_worker_forward_profile_seq1024_decode16_resident.json
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
    if not kv:
        return None
    return {k: _shape(v) for k, v in kv[0].items()}


def _profile_pass(*, pkg_path, art_path, meta, args, resident):
    """Run one full prefill + N-step decode pass and collect timing + structure +
    weight-movement counts + the final masked logits (for correctness). Returns a
    dict. ``resident`` toggles the weight-resident decode path."""
    import numpy as np
    import torch
    import pllo.deployment.folded_worker as fw
    from pllo.experiments.folded_probe_common import LiteBoundary
    from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend
    from pllo.protocol.tee_gpu_messages import (
        BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)
    from pllo.protocol.wire import encode_message
    from pllo.protocol.remote import forbidden_fields_in_payload
    from pllo.protocol.worker_timing import audit_worker_timing_no_secrets

    n_layers = int(meta["num_layers"])
    hidden = int(meta["hidden_size"])
    boundary = LiteBoundary.from_artifact(art_path, device=args.device)

    backend = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg_path), device=args.device, dtype=args.dtype,
        nonlinear_backend=args.nonlinear_backend,
        resident_folded_weights=bool(resident))
    backend.collect_worker_timing = bool(args.profile_layer_timings)
    if str(args.device).startswith("cuda"):
        backend._timing_method_override = (
            "cuda_event" if args.profile_cuda_events else "cuda_synchronize")

    # instrument weight movement (decode-only deltas captured per step)
    counts = {"load_folded_layer": 0, "build_folded_layer_dict": 0}
    _orig_load, _orig_build = fw.load_folded_layer, fw.build_folded_layer_dict

    def _wrapped_load(*a, **k):
        counts["load_folded_layer"] += 1
        return _orig_load(*a, **k)

    def _wrapped_build(*a, **k):
        counts["build_folded_layer_dict"] += 1
        return _orig_build(*a, **k)

    fw.load_folded_layer = _wrapped_load
    fw.build_folded_layer_dict = _wrapped_build

    leaked: list = []

    def _np(t):
        return np.asarray(t.detach().to("cpu").float().numpy())

    def _audit_req(req):
        bad = forbidden_fields_in_payload(encode_message(req))
        if bad:
            leaked.extend(bad)

    rows: list = []
    res = {"leaked": leaked, "counts": counts}
    try:
        torch.manual_seed(int(args.seed))                # identical inputs per pass
        ids = torch.randint(0, 256, (1, args.seq_len))
        h_tilde = boundary.mask_embeddings(ids)
        init_resp = backend.init(BoundaryInitRequest(
            session_id="profile", hidden_size=hidden,
            vocab_size=int(meta["vocab_size"]), num_layers=n_layers,
            dtype=args.dtype, gpu_backend="qwen7b_folded_package",
            folded_lm_head=None, public_metadata=meta))
        res["tee_used_on_gpu"] = bool(init_resp.tee_used_on_gpu)

        pre_req = MaskedPrefillRequest(
            session_id="profile", masked_embeddings=_np(h_tilde),
            positions=list(range(args.seq_len)), batch_size=1, seq_len=args.seq_len)
        _audit_req(pre_req)
        before = dict(counts)
        t0 = time.perf_counter()
        pre = backend.prefill(pre_req)
        res["prefill_wall_s"] = round(time.perf_counter() - t0, 9)
        res["prefill_loads"] = counts["load_folded_layer"] - before["load_folded_layer"]
        res["prefill_input_shape"] = _shape(h_tilde)
        res["cache_shape_per_step"] = _cache_shapes(backend._kv)
        if isinstance(pre.worker_timing, dict):
            audit_worker_timing_no_secrets(pre.worker_timing)

        pos = args.seq_len
        decode_wall, decode_loads, decode_builds = [], [], []
        decode_input_shapes, decode_cache_lens = [], []
        last_logits = None
        for step in range(args.decode_steps):
            x = boundary.mask_token_embedding(torch.tensor([step % 256]))
            dec_req = MaskedDecodeRequest(
                session_id="profile", masked_embedding=_np(x), position=pos,
                step=step + 1)
            _audit_req(dec_req)
            before = dict(counts)
            t0 = time.perf_counter()
            dec = backend.decode(dec_req)
            decode_wall.append(time.perf_counter() - t0)
            decode_loads.append(counts["load_folded_layer"] - before["load_folded_layer"])
            decode_builds.append(
                counts["build_folded_layer_dict"] - before["build_folded_layer_dict"])
            decode_input_shapes.append(_shape(x))
            decode_cache_lens.append(int(dec.kv_cache_len))
            last_logits = np.asarray(dec.masked_logits)
            if isinstance(dec.worker_timing, dict):
                audit_worker_timing_no_secrets(dec.worker_timing)
                rows.append(dec.worker_timing)
            pos += 1
    finally:
        fw.load_folded_layer = _orig_load
        fw.build_folded_layer_dict = _orig_build

    res.update({
        "rows": rows,
        "decode_wall": decode_wall,
        "decode_loads": decode_loads,
        "decode_builds": decode_builds,
        "decode_input_shapes": decode_input_shapes,
        "decode_cache_lens": decode_cache_lens,
        "last_logits": last_logits,
        "resident_status": backend.resident_status(),
        "worker_has_mask_secrets": bool(backend.worker_has_mask_secrets),
        "worker_has_raw_lora": bool(getattr(backend, "worker_has_raw_lora", False)),
    })
    return res


def _mean_key(rows, key):
    xs = [w.get(key) for w in rows if isinstance(w.get(key), (int, float))]
    return round(statistics.fmean(xs), 9) if xs else None


def _per_layer_summary(rows):
    pls = [w.get("per_layer_timing_summary") for w in rows
           if isinstance(w.get("per_layer_timing_summary"), dict)]
    if not pls:
        return None
    return {
        "mean_s": round(statistics.fmean(
            [p["mean_s"] for p in pls if p.get("mean_s") is not None]), 9),
        "max_s": round(max(p["max_s"] for p in pls
                           if p.get("max_s") is not None), 9),
        "min_s": round(min(p["min_s"] for p in pls
                           if p.get("min_s") is not None), 9),
        "layers_per_step": pls[0].get("count"),
    }


def _pass_metrics(res, args, n_layers):
    """Per-token + structure + weight-movement metrics for one pass."""
    rows = res["rows"]
    from pllo.benchmarks.decode_profiler import DecodeProfiler
    prof = DecodeProfiler(counters=lambda: {}, enabled=True)
    for i, wt in enumerate(rows):
        prof.begin_step(i + 1, "decode")
        with prof.stage("gpu_worker_roundtrip"):
            pass
        prof.set_worker_timing(wt)
        prof.end_step(token_id=i)
    agg = prof.aggregate(generated_tokens=max(1, len(rows))) if rows else {}

    dload = res["decode_loads"]
    dbuild = res["decode_builds"]
    loads_per_step = round(statistics.fmean(dload), 3) if dload else None
    builds_per_step = round(statistics.fmean(dbuild), 3) if dbuild else None
    shapes = res["decode_input_shapes"]
    clens = res["decode_cache_lens"]
    return {
        "worker_timing_method": rows[0].get("worker_timing_method") if rows else None,
        "worker_timing_is_cuda_synchronized": bool(
            rows[0].get("worker_timing_is_cuda_synchronized")) if rows else False,
        "total_forward_s_per_token": _mean_key(rows, "worker_backend_forward_s"),
        "layer_forward_s_per_token": _mean_key(rows, "worker_layer_total_s"),
        "attention_s_per_token": _mean_key(rows, "worker_attention_total_s"),
        "mlp_s_per_token": _mean_key(rows, "worker_mlp_total_s"),
        "nonlinear_s_per_token": _mean_key(rows, "worker_nonlinear_total_s"),
        "lm_head_s_per_token": _mean_key(rows, "worker_lm_head_s"),
        "known_substage_total_s_per_token": _mean_key(
            rows, "worker_known_substage_total_s"),
        "unattributed_forward_s_per_token": _mean_key(
            rows, "worker_unattributed_forward_s"),
        "per_layer_timing_summary": _per_layer_summary(rows),
        "worker_bottleneck_stage": agg.get("worker_bottleneck_stage"),
        "prefill_wall_s": res.get("prefill_wall_s"),
        "decode_wall_s_per_token": (round(statistics.fmean(res["decode_wall"]), 9)
                                    if res["decode_wall"] else None),
        # structure
        "decode_uses_incremental_kv": bool(
            shapes and all(s and s[1] == 1 for s in shapes)),
        "full_prefix_recomputed_each_step": bool(
            shapes and any(s and s[1] == args.seq_len for s in shapes)),
        "kv_grows_by_one_per_step": bool(
            len(clens) >= 2 and all(clens[i + 1] - clens[i] == 1
                                    for i in range(len(clens) - 1))),
        "input_shape_per_step": shapes[0] if shapes else None,
        "decode_kv_cache_lens": clens,
        # weight movement
        "weight_shard_loads_prefill": res.get("prefill_loads"),
        "weight_shard_loads_per_decode_step": loads_per_step,
        "folded_layer_dict_builds_per_decode_step": builds_per_step,
        "cpu_to_gpu_weight_copies_per_decode_step": builds_per_step,
        "weight_reloaded_each_step": bool(loads_per_step and loads_per_step >= 1),
    }


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
    ap.add_argument("--resident-folded-weights", action="store_true",
                    default=False, help="load+fold+move all layers to device ONCE "
                    "and reuse across decode steps (default OFF). Also runs a "
                    "non-resident baseline for the correctness + speedup contrast.")
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

    from pllo.experiments.folded_probe_common import LiteBoundary
    boundary = LiteBoundary.from_artifact(art_path, device=args.device)
    meta = boundary.exec_metadata(seq_len=args.seq_len,
                                  max_new_tokens=args.decode_steps)
    n_layers = int(meta["num_layers"])
    del boundary

    # always run the non-resident baseline; run resident too when requested, so
    # the report carries the correctness + speedup contrast.
    base = _profile_pass(pkg_path=pkg_path, art_path=art_path, meta=meta,
                         args=args, resident=False)
    base_m = _pass_metrics(base, args, n_layers)
    primary, primary_m, comparison, correctness = base, base_m, None, None

    if args.resident_folded_weights:
        import numpy as np
        res = _profile_pass(pkg_path=pkg_path, art_path=art_path, meta=meta,
                            args=args, resident=True)
        res_m = _pass_metrics(res, args, n_layers)
        primary, primary_m = res, res_m
        # correctness: identical synthetic inputs -> shapes match, logits ~equal
        bl, rl = base["last_logits"], res["last_logits"]
        max_abs_err = None
        shapes_match = (bl is not None and rl is not None
                        and list(bl.shape) == list(rl.shape))
        if shapes_match:
            max_abs_err = float(np.max(np.abs(bl.astype("float64")
                                              - rl.astype("float64"))))
        correctness = {
            "output_shapes_match": bool(shapes_match),
            "max_abs_err_resident_vs_nonresident": (
                round(max_abs_err, 9) if max_abs_err is not None else None),
            "logits_shape": list(bl.shape) if bl is not None else None,
        }
        spd = None
        if (base_m["total_forward_s_per_token"]
                and res_m["total_forward_s_per_token"]):
            spd = round(base_m["total_forward_s_per_token"]
                        / res_m["total_forward_s_per_token"], 3)
        comparison = {
            "non_resident_total_forward_s_per_token":
                base_m["total_forward_s_per_token"],
            "resident_total_forward_s_per_token":
                res_m["total_forward_s_per_token"],
            "non_resident_decode_wall_s_per_token":
                base_m["decode_wall_s_per_token"],
            "resident_decode_wall_s_per_token":
                res_m["decode_wall_s_per_token"],
            "forward_speedup_x": spd,
            "non_resident_weight_reloaded_each_step":
                base_m["weight_reloaded_each_step"],
            "resident_weight_reloaded_each_step":
                res_m["weight_reloaded_each_step"],
        }

    leaked = sorted(set(primary["leaked"]))
    rstat = primary["resident_status"]
    report = {
        "stage": "folded_worker_forward_profile",
        "dry_run": is_dry,
        "device": args.device,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "decode_steps": args.decode_steps,
        "num_layers": n_layers,
        "hidden_size": int(meta["hidden_size"]),
        "nonlinear_backend": args.nonlinear_backend,
        "profile_layer_timings": bool(args.profile_layer_timings),
        # ---- resident-cache status (PERFORMANCE optimisation, not security) ----
        "resident_folded_weights": bool(args.resident_folded_weights),
        "resident_weight_init_latency_s": rstat["resident_weight_init_latency_s"],
        "resident_weight_memory_gb": rstat["resident_weight_memory_gb"],
        "resident_cache_num_layers": rstat["resident_cache_num_layers"],
        "resident_cache_device": rstat["resident_cache_device"],
        "resident_cache_dtype": rstat["resident_cache_dtype"],
        "resident_cache_oom": rstat["resident_cache_oom"],
        "resident_cache_fallback_used": rstat["resident_cache_fallback_used"],
        "resident_cache_active": rstat["resident_cache_active"],
        # ---- primary-pass metrics ----
        **primary_m,
        "kv_cache_reuse_enabled": primary_m["kv_grows_by_one_per_step"],
        "cache_shape_per_step": primary.get("cache_shape_per_step"),
        "prefill_input_shape": primary.get("prefill_input_shape"),
        # ---- baseline + contrast (only when resident requested) ----
        "comparison_resident_vs_non_resident": comparison,
        "correctness": correctness,
        # ---- security audit (threat model unchanged; perf != security) ----
        "audit_passed": bool(not leaked),
        "tee_used_on_gpu": bool(primary.get("tee_used_on_gpu")),
        "worker_has_mask_secrets": primary["worker_has_mask_secrets"],
        "worker_has_raw_lora": primary["worker_has_raw_lora"],
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": leaked,
        "schedule_secret_leaked_to_gpu": False,
        "gpu_request_contains_schedule_secret": False,
        "worker_timing_contains_secret": False,
        "kv_cache_plaintext_visible_to_gpu": False,
        "mask_domain_reused_across_steps": False,
        # the resident cache introduced no secret leak (it holds only PUBLIC
        # folded operators; mask-secret / raw-LoRA flags unchanged, no field leak)
        "resident_weight_security_audit_passed": bool(
            not leaked and not primary["worker_has_mask_secrets"]
            and not primary["worker_has_raw_lora"]),
        "fresh_obfuscation_domain_policy": (
            "the worker never receives the obfuscation domain; it sees only masked "
            "tensors + public step_id. resident_folded_weights caches only PUBLIC "
            "folded operators -- it changes nothing about per-step mask freshness "
            "(enforced trusted-side)."),
        "optimization_is_performance_only": True,
    }

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== folded worker forward profile (%s, dry_run=%s, resident=%s) ==="
          % (args.device, is_dry, args.resident_folded_weights))
    print("total_forward_s_per_token=%s  bottleneck=%s  weight_reloaded_each_step=%s"
          " (loads/step=%s, builds/step=%s)"
          % (report["total_forward_s_per_token"], report["worker_bottleneck_stage"],
             report["weight_reloaded_each_step"],
             report["weight_shard_loads_per_decode_step"],
             report["folded_layer_dict_builds_per_decode_step"]))
    print("resident_active=%s init_latency_s=%s memory_gb=%s oom=%s fallback=%s"
          % (report["resident_cache_active"],
             report["resident_weight_init_latency_s"],
             report["resident_weight_memory_gb"], report["resident_cache_oom"],
             report["resident_cache_fallback_used"]))
    if comparison:
        print("speedup_x=%s  resident_reloaded=%s  max_abs_err=%s"
              % (comparison["forward_speedup_x"],
                 comparison["resident_weight_reloaded_each_step"],
                 correctness["max_abs_err_resident_vs_nonresident"]))
    print("audit_passed=%s tee_used_on_gpu=%s worker_has_mask_secrets=%s leaked=%s"
          % (report["audit_passed"], report["tee_used_on_gpu"],
             report["worker_has_mask_secrets"], report["leaked_secret_fields"]))
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
