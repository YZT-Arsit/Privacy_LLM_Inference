"""Trusted-setup builder for a folded Qwen weight package.

Runs in TDX (or another attested trusted setup environment). Reads the public
frozen base model + an internally-generated mask schedule and writes folded
operators ``W_tilde = N_in^{-1} W N_out`` to disk, sharded by layer, plus a
manifest. **Mask secrets are never written**; the package holds only folded
operators (``*_tilde``). Folding streams layer-by-layer so the full folded model
is never resident at once.

The untrusted H800 worker later loads this package locally and computes over
masked runtime tensors without ever holding ``N_in``/``N_out``. Folding is a
one-time setup cost; online decode reuses the provisioned weights.

Real Qwen2.5-7B needs CUDA + a local checkpoint. ``--dry-run`` builds a tiny
random Qwen2 package on CPU (never a paper result). ``--num-layers 1`` builds a
1-layer package; ``--estimate-only`` reports projected size without folding.

Examples::

    # 1-layer real package (cheap sanity build on the H800):
    python scripts/build_qwen7b_folded_package.py --model-path <MODEL_PATH> \\
        --output-dir packages/qwen7b_1layer --seq-len 128 --num-layers 1 \\
        --dtype bfloat16 --nonlinear-backend current --mask-schedule session \\
        --shard-by-layer true --write-manifest true
    # full 28-layer package (H800):
    python scripts/build_qwen7b_folded_package.py --model-path <MODEL_PATH> \\
        --output-dir packages/qwen7b_folded --seq-len 128 --num-layers 28 \\
        --dtype bfloat16 --nonlinear-backend current --mask-schedule session \\
        --shard-by-layer true --write-manifest true
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from pllo.deployment import (  # noqa: E402
    FoldedPackageWriter,
    build_manifest,
    compute_manifest_hash,
    write_manifest,
)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402


def _bool(s) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    mc = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                     num_hidden_layers=4, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=256,
                     rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                     tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(mc).eval(), mc


def _load_real(args):
    from transformers import AutoModelForCausalLM
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(args.dtype, torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dt, device_map=args.device,
        trust_remote_code=True, local_files_only=True).eval()
    return model, model.config


def _peak_mb():
    try:
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except Exception:
        pass
    return None


def _folding_runtime_hash(args, n, seed) -> str:
    payload = {"model_name": args.model_name, "num_layers": n,
               "dtype": args.dtype, "nonlinear_backend": args.nonlinear_backend,
               "mask_schedule": args.mask_schedule, "seed": seed,
               "seq_len": args.seq_len}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folded-weight-device", default=None)
    ap.add_argument("--mlp-down-chunk-size", type=int, default=512)
    ap.add_argument("--nonlinear-backend", default="current")
    ap.add_argument("--allow-unwired-nonlinear", action="store_true",
                    default=False,
                    help="allow a non-paper-facing PROTOTYPE build for a nonlinear "
                         "design not yet executed in the real path (tag-only)")
    ap.add_argument("--mask-schedule", default="session")
    ap.add_argument("--shard-by-layer", default="true")
    ap.add_argument("--write-manifest", default="true")
    ap.add_argument("--created-by", default="trusted_setup",
                    choices=["tdx_trusted_setup", "trusted_setup", "test"])
    ap.add_argument("--tee-type", default=None)
    ap.add_argument("--mr-td", default=None)
    ap.add_argument("--report-data", default=None)
    ap.add_argument("--created-at", default=None,
                    help="ISO timestamp for the manifest (else filled at runtime)")
    ap.add_argument("--seed", type=int, default=2035)
    # Linear-boundary additive padding is the MAIN paper scheme and is ON by
    # default. Use --no-linear-boundary-pad for a legacy/ablation mask-only build.
    ap.set_defaults(linear_boundary_pad=True)
    ap.add_argument("--linear-boundary-pad", dest="linear_boundary_pad",
                    action="store_true",
                    help="enable Linear-boundary additive input padding for every "
                    "folded Linear (q/k/v/o/gate/up/down/lm_head): the GPU matmul "
                    "operand becomes (X - T) N_in with a precomputed folded "
                    "compensation C_pad = T W N_out, output stays in the masked "
                    "basis Y N_out. Pads are boundary-local (not in nonlinear "
                    "cores, not persisted in the residual). Raw T/N never leave "
                    "trusted setup. THIS IS THE DEFAULT (main paper scheme).")
    ap.add_argument("--no-linear-boundary-pad", dest="linear_boundary_pad",
                    action="store_false",
                    help="disable the Linear-boundary pad and build a mask-only "
                    "package. This is a LEGACY/ABLATION build and is reported as "
                    "main_scheme=mask_only_legacy / paper_ready=false.")
    ap.add_argument("--linear-pad-scale", type=float, default=0.1,
                    help="magnitude of the (masked-basis) additive input pad; "
                    "output is mathematically invariant to it -- it only changes "
                    "the obfuscated matmul-operand view")
    ap.add_argument("--estimate-only", action="store_true",
                    help="report projected size/shards without folding/writing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    from pllo.experiments.nonlinear_designs import (  # noqa: E402
        assert_real_path_execution, NonlinearDesignNotWired,
        normalize_nonlinear_backend, nonlinear_design_report_fields)
    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)
    build_command = "python " + " ".join(sys.argv)

    dry_run = bool(args.dry_run or not args.model_path)

    # HONESTY GUARD: refuse a paper-facing build for a nonlinear design not yet
    # executed in the real path (it would be tag-only). dry-run / explicit
    # --allow-unwired-nonlinear prototypes are allowed.
    try:
        assert_real_path_execution(args.nonlinear_backend, dry_run=dry_run,
                                   allow_unwired=args.allow_unwired_nonlinear)
    except NonlinearDesignNotWired as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 3
    if dry_run:
        if not args.dry_run:
            print("NOTE: no --model-path; building --dry-run tiny package (NOT a "
                  "paper result).")
        model, mc = _tiny_model()
        device, dtype = "cpu", "float32"
    else:
        model, mc = _load_real(args)
        device, dtype = args.device, args.dtype

    cfg = MemoryOptimizedConfig(
        num_layers=min(args.num_layers, mc.num_hidden_layers), batch_size=1,
        seq_len=args.seq_len, max_new_tokens=1, device=device, dtype=dtype,
        folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size, seed=args.seed,
        use_linear_boundary_pad=bool(args.linear_boundary_pad),
        linear_pad_scale=float(args.linear_pad_scale))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    session = MaskedQwenSession(model, mc, cfg)
    n = session.n
    created_at = args.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())
    mask_schedule_id = f"{args.mask_schedule}-seed{args.seed}-n{n}"

    if args.estimate_only:
        # projected folded size from one layer's tensors (no full build).
        sample = session.export_folded_layer_tensors(0)
        layer_bytes = int(sum(v.numel() * v.element_size()
                              for v in sample.values()))
        head = session.export_folded_head_tensors()["w_lm_tilde"]
        head_bytes = int(head.numel() * head.element_size())
        total_gb = (layer_bytes * n + head_bytes) / (1024 ** 3)
        rep = {"stage": "folded_package_build", "estimate_only": True,
               "model_name": args.model_name, "num_layers": n,
               "projected_folded_weight_size_gb": round(total_gb, 4),
               "per_layer_bytes": layer_bytes, "head_bytes": head_bytes,
               "num_shards": n + 1, "contains_mask_secrets": False,
               "dry_run": dry_run}
        rep.update(nonlinear_design_report_fields(args.nonlinear_backend))
        print(json.dumps(rep, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(rep, indent=2))
        return 0

    shard_by_layer = _bool(args.shard_by_layer)
    writer = FoldedPackageWriter(args.output_dir)
    t0 = time.perf_counter()
    if shard_by_layer:
        for ell in range(n):
            writer.add_shard(f"layer_{ell:03d}",
                             session.export_folded_layer_tensors(ell))
    else:
        combined = {}
        for ell in range(n):
            for k, v in session.export_folded_layer_tensors(ell).items():
                combined[f"l{ell:03d}.{k}"] = v
        writer.add_shard("layers", combined)
    writer.add_shard("head", session.export_folded_head_tensors())
    gen_time_s = round(time.perf_counter() - t0, 4)
    peak_mb = _peak_mb()

    manifest = build_manifest(
        package_type="base_model", model_name=args.model_name,
        model_path_or_id=args.model_path, num_layers=n, dtype=args.dtype,
        nonlinear_backend=args.nonlinear_backend, created_by=args.created_by,
        shard_index=writer.shard_index,
        hidden_size=int(getattr(mc, "hidden_size")),
        vocab_size=int(getattr(mc, "vocab_size")),
        mask_schedule_id=mask_schedule_id,
        folding_runtime_hash=_folding_runtime_hash(args, n, args.seed),
        tee_type=args.tee_type, mr_td=args.mr_td, report_data=args.report_data,
        created_at=created_at, build_command=build_command)
    manifest_hash = compute_manifest_hash(manifest)
    if _bool(args.write_manifest):
        write_manifest(manifest, args.output_dir)

    folded_gb = writer.total_bytes() / (1024 ** 3)
    rep = {
        "stage": "folded_package_build",
        "estimate_only": False,
        "dry_run": dry_run,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "output_dir": str(args.output_dir),
        "num_layers": n,
        "dtype": args.dtype,
        "nonlinear_backend": args.nonlinear_backend,
        "mask_schedule_id": mask_schedule_id,
        "shard_by_layer": shard_by_layer,
        "num_shards": len(writer.shard_index),
        "shard_paths": [e["path"] for e in writer.shard_index],
        "folded_weight_size_gb": round(folded_gb, 6),
        "folded_weight_generation_time_s": gen_time_s,
        "peak_memory_mb": peak_mb,
        "manifest_hash": manifest_hash,
        "contains_mask_secrets": False,
        # H. provisioning report fields
        "folded_weight_setup_required": True,
        "folded_weight_source": ("test" if dry_run else args.created_by),
        "folded_package_path": str(args.output_dir),
        "folded_package_size_gb": round(folded_gb, 6),
        "folded_weight_transfer_required": True,
        "folded_weight_transfer_size_gb": round(folded_gb, 6),
        "worker_has_mask_secrets": False,
        "tee_used_on_gpu": False,
        "build_command": build_command,
    }
    rep.update(nonlinear_design_report_fields(args.nonlinear_backend))

    # ---- Linear-boundary additive pad audit + per-module coverage ----
    from pllo.deployment.linear_boundary_pad import (  # noqa: E402
        layer_pad_coverage, linear_boundary_pad_report_fields, ALL_PAD_MODULES)
    if bool(args.linear_boundary_pad):
        # coverage read back from the actually-written shard tensor names (a
        # module counts only if BOTH its xpad+cpad tensors are present)
        layer_names: list[str] = []
        head_names: list[str] = []
        for e in writer.shard_index:
            if e["name"].startswith("layer_") or e["name"] == "layers":
                layer_names += e.get("tensors", [])
            elif e["name"] == "head":
                head_names += e.get("tensors", [])
        cov = layer_pad_coverage(layer_names, head_names)
        rep.update(linear_boundary_pad_report_fields(
            enabled=True, coverage=cov, scale=float(args.linear_pad_scale)))
        rep["main_scheme"] = "linear_boundary_additive_pad"
        all_covered = all(cov.get(m, False) for m in ALL_PAD_MODULES)
        if not all_covered:
            rep["paper_ready"] = False
            rep["paper_ready_blocker"] = (
                "linear_boundary_pad enabled but these Linear modules lack pad "
                "coverage: %s" % [m for m in ALL_PAD_MODULES if not cov.get(m)])
    else:
        # Mask-only legacy / ablation build: NOT the main paper scheme.
        rep.update(linear_boundary_pad_report_fields(enabled=False))
        rep["main_scheme"] = "mask_only_legacy"
        rep["paper_ready"] = False
        rep["paper_ready_blocker"] = (
            "linear-boundary additive padding disabled; this is a "
            "legacy/ablation package, not the main paper scheme")
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"=== build folded Qwen package (dry_run={dry_run}) ===")
    print(f"output_dir={rep['output_dir']} num_layers={n} "
          f"num_shards={rep['num_shards']}")
    print(f"folded_weight_size_gb={rep['folded_weight_size_gb']} "
          f"generation_time_s={gen_time_s} peak_memory_mb={peak_mb}")
    print(f"manifest_hash={manifest_hash}")
    print(f"contains_mask_secrets={rep['contains_mask_secrets']} "
          f"worker_has_mask_secrets={rep['worker_has_mask_secrets']} "
          f"tee_used_on_gpu={rep['tee_used_on_gpu']}")
    print(f"main_scheme={rep.get('main_scheme')} "
          f"paper_ready={rep.get('paper_ready', True)}")
    print(f"linear_boundary_pad_enabled={rep['linear_boundary_pad_enabled']} "
          f"linear_input_form={rep['linear_input_form']!r} "
          f"online_extra_matmul_for_pad={rep['online_extra_matmul_for_pad']}")
    if rep["linear_boundary_pad_enabled"]:
        print(f"linear_pad_coverage={rep['linear_pad_coverage']}")
        print(f"raw_pad_visible_to_gpu={rep['raw_pad_visible_to_gpu']} "
              f"raw_mask_visible_to_gpu={rep['raw_mask_visible_to_gpu']} "
              f"c_pad_visible_to_gpu={rep['c_pad_visible_to_gpu']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
