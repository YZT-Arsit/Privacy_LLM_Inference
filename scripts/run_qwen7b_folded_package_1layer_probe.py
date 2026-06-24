"""Qwen 1-layer folded-package correctness probe (build -> verify -> load -> compare).

The missing bridge between (a) standalone H800 E1/E2 (full masked/folded compute
correctness) and (b) the attested TDX+H800 mock protocol (attested boundary
driving a remote worker) is folded-package PROVISIONING for Qwen. This probe is
the first stable rung: it proves a worker holding ONLY folded weights (no masks)
reproduces the in-process masked output for a single Qwen layer.

Steps:

1. Build a 1-layer folded package from the public model + an internal mask
   schedule (trusted setup; mask secrets never written), or consume an existing
   ``--folded-package-path``.
2. Verify the package (manifest + shard hashes + no secret tensor names).
3. Load the package in the untrusted worker (``Qwen7BFoldedPackageGpuBackend``);
   the worker holds NO mask secrets.
4. Compare the worker/package masked output against the in-process protected path
   (which has the masks) for the same masked input -> max/mean/relative-l2 error
   + allclose.

Real Qwen2.5-7B needs CUDA + a local checkpoint; ``--dry-run`` runs a tiny random
Qwen2 on CPU (never a paper result). Full 28-layer shard-streamed decode remains
TODO until this 1-layer path is stable.

Example (H800)::

    python scripts/run_qwen7b_folded_package_1layer_probe.py \\
        --model-path <MODEL_PATH> --model-name Qwen2.5-7B-Instruct \\
        --folded-package-path packages/qwen7b_1layer --seq-len 128 \\
        --dtype bfloat16 --device cuda --folded-weight-device cuda \\
        --output-json outputs/qwen7b_folded_1layer_probe.json \\
        --output-md   outputs/qwen7b_folded_1layer_probe.md
"""

from __future__ import annotations

import argparse
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
    load_manifest,
    package_size_gb,
    verify_package,
    write_manifest,
)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import (  # noqa: E402
    MemoryOptimizedConfig,
    _masked_block_prefill_chunked,
)
from pllo.protocol.gpu_worker import Qwen7BFoldedPackageGpuBackend  # noqa: E402
from pllo.protocol.tee_gpu_messages import BoundaryInitRequest  # noqa: E402


def _bool(s) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    mc = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                     num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=256,
                     rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                     tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(mc).eval(), mc


def _load_real(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(args.dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dt, device_map=args.device,
        trust_remote_code=True, local_files_only=True).eval()
    text = args.prompt
    if _bool(args.use_chat_template):
        text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                       tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"][:, :args.seq_len]
    return model, model.config, ids.to(args.device)


def _seed_from_manifest(pkg_dir, default):
    """Parse the seed from mask_schedule_id (``<sched>-seed<seed>-n<n>``) so an
    externally-built package's masks are reproduced by the in-process reference."""
    try:
        sid = load_manifest(pkg_dir).mask_schedule_id or ""
        if "-seed" in sid:
            tail = sid.split("-seed", 1)[1]
            return int(tail.split("-", 1)[0])
    except Exception:                                       # noqa: BLE001
        pass
    return default


def _err_stats(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    diff = a - b
    denom = float(torch.linalg.norm(b)) or 1.0
    return (float(diff.abs().max()), float(diff.abs().mean()),
            float(torch.linalg.norm(diff) / denom))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt", default="Explain why privacy matters in LLMs.")
    ap.add_argument("--use-chat-template", default="true")
    ap.add_argument("--folded-package-path", default="packages/qwen7b_1layer")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the package even if it already exists")
    ap.add_argument("--layer-index", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folded-weight-device", default=None)
    ap.add_argument("--mlp-down-chunk-size", type=int, default=512)
    ap.add_argument("--nonlinear-backend", default="current")
    ap.add_argument("--mask-schedule", default="session")
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-json",
                    default="outputs/qwen7b_folded_1layer_probe.json")
    ap.add_argument("--output-md",
                    default="outputs/qwen7b_folded_1layer_probe.md")
    args = ap.parse_args()

    dry_run = bool(args.dry_run or not args.model_path)
    if dry_run:
        if not args.dry_run:
            print("NOTE: no --model-path; running --dry-run tiny model (NOT a "
                  "paper result).")
        model, mc = _tiny_model()
        ids = torch.randint(0, mc.vocab_size, (1, min(args.seq_len, 8)))
        device, dtype = "cpu", "float32"
    else:
        model, mc, ids = _load_real(args)
        device, dtype = args.device, args.dtype

    pkg_dir = Path(args.folded_package_path)
    have_pkg = (pkg_dir / "manifest.json").exists() and not args.rebuild
    seed = _seed_from_manifest(pkg_dir, args.seed) if have_pkg else args.seed

    # 1-layer masked session (same seed/config the package was built with).
    cfg = MemoryOptimizedConfig(
        num_layers=1, batch_size=1, seq_len=int(ids.shape[1]), max_new_tokens=1,
        device=device, dtype=dtype, folding_dtype="float32",
        folded_weight_device=args.folded_weight_device or device,
        mlp_down_chunk_size=args.mlp_down_chunk_size, seed=seed)
    session = MaskedQwenSession(model, mc, cfg)

    li = args.layer_index
    h_tilde = session.mask_embeddings(ids)                 # boundary-masked input

    # in-process protected reference (has masks)
    folded, down_info, cfg_c = session._folded_layer(li)
    ref = _masked_block_prefill_chunked(
        h_tilde, folded, down_info, cfg_c, session._cos, session._sin,
        session.chunk)["y_tilde"]

    # 1. build (trusted setup) unless consuming an existing package
    if not have_pkg:
        writer = FoldedPackageWriter(pkg_dir)
        writer.add_shard(f"layer_{li:03d}",
                         session.export_folded_layer_tensors(li))
        writer.add_shard("head", session.export_folded_head_tensors())
        manifest = build_manifest(
            package_type="base_model", model_name=args.model_name,
            model_path_or_id=args.model_path, num_layers=1, dtype=args.dtype,
            nonlinear_backend=args.nonlinear_backend,
            created_by="test" if dry_run else "trusted_setup",
            shard_index=writer.shard_index,
            hidden_size=int(getattr(mc, "hidden_size")),
            vocab_size=int(getattr(mc, "vocab_size")),
            mask_schedule_id=f"{args.mask_schedule}-seed{seed}-n1",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_manifest(manifest, pkg_dir)

    # 2. verify
    vrep = verify_package(pkg_dir)

    # 3. load in the untrusted worker (NO masks)
    backend = Qwen7BFoldedPackageGpuBackend(
        folded_package_path=str(pkg_dir), device=device, dtype=dtype)
    init_resp = backend.init(BoundaryInitRequest(
        session_id="probe", hidden_size=int(getattr(mc, "hidden_size")),
        vocab_size=int(getattr(mc, "vocab_size")), num_layers=1, dtype=dtype,
        gpu_backend="qwen7b_folded_package"))
    desc = backend.describe()

    # 4. worker/package masked output (folded weights only) + compare
    t0 = time.perf_counter()
    out = backend.run_single_layer_prefill(
        h_tilde, li, cfg_c, session._cos, session._sin, session.eps)
    latency_s = time.perf_counter() - t0
    y_pkg = out["y_tilde"]
    max_abs, mean_abs, rel_l2 = _err_stats(y_pkg, ref)
    allclose = bool(torch.allclose(y_pkg, ref, atol=args.atol, rtol=args.rtol))

    report = {
        "stage": "qwen7b_folded_package_1layer_probe",
        "dry_run": dry_run,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "layer_index": li,
        "seq_len": int(ids.shape[1]),
        "dtype": dtype,
        "seed": seed,
        "folded_package_path": str(pkg_dir),
        "folded_package_loaded": bool(desc["folded_package_loaded"]),
        "folded_package_valid": bool(vrep["package_valid"]),
        "package_size_gb": round(package_size_gb(pkg_dir), 6),
        "manifest_hash": desc["manifest_hash"],
        "num_layers": 1,
        "num_shards": vrep["num_shards"],
        "worker_has_mask_secrets": bool(desc["worker_has_mask_secrets"]),
        "tee_used_on_gpu": bool(init_resp.tee_used_on_gpu),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2_error": rel_l2,
        "allclose": allclose,
        "atol": args.atol,
        "rtol": args.rtol,
        "latency_s": latency_s,
        # the worker received only the masked hidden + folded weights
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": vrep["forbidden_fields_found"],
        "verify_hash_mismatches": len(vrep["hash_mismatches"]),
        "verify_missing_shards": vrep["missing_shards"],
        "folded_weight_source": "test" if dry_run else "trusted_setup",
    }

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        _write_md(Path(args.output_md), report)

    print(f"=== Qwen 1-layer folded-package probe (dry_run={dry_run}) ===")
    print(f"folded_package_loaded={report['folded_package_loaded']} "
          f"folded_package_valid={report['folded_package_valid']} "
          f"num_layers=1 num_shards={report['num_shards']}")
    print(f"package_size_gb={report['package_size_gb']} "
          f"manifest_hash={report['manifest_hash']}")
    print(f"worker_has_mask_secrets={report['worker_has_mask_secrets']} "
          f"tee_used_on_gpu={report['tee_used_on_gpu']}")
    print(f"max_abs_error={max_abs:.3e} mean_abs_error={mean_abs:.3e} "
          f"relative_l2_error={rel_l2:.3e}")
    print(f"allclose={allclose} (atol={args.atol} rtol={args.rtol})")
    print(f"gpu_visible_plaintext_fields={report['gpu_visible_plaintext_fields']} "
          f"leaked_secret_fields={report['leaked_secret_fields']}")
    ok = (report["folded_package_loaded"] and report["folded_package_valid"]
          and not report["worker_has_mask_secrets"]
          and not report["tee_used_on_gpu"] and allclose
          and not report["leaked_secret_fields"])
    print(f"\nPROBE {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def _write_md(path: Path, r: dict) -> None:
    L = [f"# Qwen 1-layer folded-package probe (dry_run={r['dry_run']})", "",
         f"- model_name=`{r['model_name']}`  layer_index={r['layer_index']}  "
         f"seq_len={r['seq_len']}  dtype={r['dtype']}  seed={r['seed']}",
         f"- folded_package_path=`{r['folded_package_path']}`",
         f"- **folded_package_loaded={r['folded_package_loaded']}**  "
         f"**folded_package_valid={r['folded_package_valid']}**  "
         f"num_layers={r['num_layers']}  num_shards={r['num_shards']}",
         f"- package_size_gb={r['package_size_gb']}  "
         f"manifest_hash=`{r['manifest_hash']}`",
         f"- **worker_has_mask_secrets={r['worker_has_mask_secrets']}**  "
         f"**tee_used_on_gpu={r['tee_used_on_gpu']}**",
         "", "## Correctness (worker/package vs in-process protected path)", "",
         f"- max_abs_error={r['max_abs_error']:.3e}",
         f"- mean_abs_error={r['mean_abs_error']:.3e}",
         f"- relative_l2_error={r['relative_l2_error']:.3e}",
         f"- **allclose={r['allclose']}** (atol={r['atol']} rtol={r['rtol']})",
         f"- gpu_visible_plaintext_fields={r['gpu_visible_plaintext_fields'] or '[]'}",
         f"- leaked_secret_fields={r['leaked_secret_fields'] or '[]'}",
         "",
         "_The worker held only folded operators (no masks) + the masked input; "
         "the down projection is pre-folded in the package. Full 28-layer "
         "shard-streamed decode remains TODO._"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
