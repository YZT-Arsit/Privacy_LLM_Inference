"""Build a private folded-LoRA package (trusted setup).

Folds a private LoRA adapter into the SAME masked basis as a public base folded
package, producing low-rank folded operators ``a_tilde``/``b_tilde`` per (layer,
module). The untrusted GPU later merges ``W_tilde += a_tilde @ b_tilde`` and runs
the existing masked kernels -- it never sees raw A/B or the masks. See
``src/pllo/deployment/lora_folded_package.py`` for the folding formula.

Outputs: folded-LoRA shards + ``manifest.json`` (integrity) + ``lora_meta.json``
(rank, alpha, target_modules, adapter_hash, base_package_manifest_hash,
contains_*=False, trusted_setup=True).

``--dry-run`` uses the tiny base model + a synthetic adapter (never a paper
result). The base session seed is taken from the base package manifest so the
folded LoRA lands in the base package's masked basis.

Real HF/PEFT adapter (``adapter_model.safetensors`` + ``adapter_config.json``)::

    python scripts/build_qwen7b_lora_folded_package.py \\
        --model-path /root/.../Qwen2___5-7B-Instruct \\
        --base-folded-package-path /root/.../qwen7b_folded_full \\
        --raw-lora-adapter-path /path/to/adapter --adapter-format hf_peft \\
        --target-modules q_proj,k_proj,v_proj,o_proj --rank 8 --alpha 16 \\
        --output-dir /root/.../qwen7b_lora_folded \\
        --output-json outputs/qwen7b_lora_folded_build.json

``--adapter-path`` remains a backward-compatible alias of
``--raw-lora-adapter-path``. ``--dry-run`` uses the tiny base + a synthetic
adapter (never a paper result).
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

from pllo.deployment.lora_folded_package import (  # noqa: E402
    DEFAULT_TARGET_MODULES,
    build_lora_folded_package,
    load_hf_lora_adapter,
    read_hf_adapter_config,
    synthetic_lora_adapter,
)
from pllo.experiments.folded_probe_common import seed_from_manifest, tiny_model  # noqa: E402
from pllo.experiments.nonlinear_designs import (  # noqa: E402
    nonlinear_design_report_fields,
    normalize_nonlinear_backend,
)
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402


def _csv(s):
    return [p for p in str(s).replace(" ", "").split(",") if p]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--base-folded-package-path", default=None,
                    help="base package (for seed + base_package_manifest_hash)")
    ap.add_argument("--adapter-path", default=None,
                    help="raw LoRA adapter dir; omit for --dry-run synthetic")
    ap.add_argument("--raw-lora-adapter-path", default=None,
                    help="alias of --adapter-path (real HF/PEFT adapter dir)")
    ap.add_argument("--adapter-format", default="hf_peft",
                    choices=["hf_peft", "auto"],
                    help="raw adapter on-disk format (only hf_peft supported)")
    ap.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=2035,
                    help="mask seed (must match base package) if no base package")
    ap.add_argument("--rank-seed", type=int, default=None,
                    help="seed for the private rank masks (default: mask seed)")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--synthetic-scale", type=float, default=0.02)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-json", default=None,
                    help="write the build report JSON to this path (for pipelines)")
    ap.add_argument("--nonlinear-backend", default="current",
                    help="nonlinear design (current|trusted_shortcut, aliases ok)")
    args = ap.parse_args()
    args.nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)
    build_command = "python " + " ".join(sys.argv)

    dry_run = bool(args.dry_run or not args.model_path)
    target_modules = _csv(args.target_modules)
    adapter_path = args.raw_lora_adapter_path or args.adapter_path
    rank, alpha = args.rank, args.alpha

    # Real adapter: rank/alpha/target_modules MUST match the adapter, so read them
    # from adapter_config.json (public hyper-params) and override the CLI defaults.
    if adapter_path:
        try:
            acfg = read_hf_adapter_config(adapter_path)
            if acfg["rank"]:
                if acfg["rank"] != rank or float(acfg["alpha"]) != float(alpha):
                    print("NOTE: using adapter_config.json rank=%s alpha=%s "
                          "(CLI rank=%s alpha=%s)" % (acfg["rank"], acfg["alpha"],
                                                      rank, alpha))
                rank, alpha = acfg["rank"], acfg["alpha"]
            if acfg["target_modules"]:
                target_modules = acfg["target_modules"]
                print("NOTE: using adapter_config.json target_modules=%s"
                      % target_modules)
        except Exception as exc:                            # noqa: BLE001
            print("WARNING: could not read adapter_config.json (%s); using CLI "
                  "rank/alpha/target_modules" % exc)

    seed = args.seed
    base_manifest_hash = None
    if args.base_folded_package_path:
        seed = seed_from_manifest(args.base_folded_package_path, args.seed)
        try:
            from pllo.deployment import compute_manifest_hash, load_manifest
            base_manifest_hash = compute_manifest_hash(
                load_manifest(args.base_folded_package_path))
        except Exception as exc:                            # noqa: BLE001
            print("WARNING: could not read base manifest hash: %s" % exc)
    rank_seed = args.rank_seed if args.rank_seed is not None else seed

    if dry_run:
        if not args.dry_run:
            print("NOTE: no --model-path; --dry-run tiny base (NOT a paper result).")
        model, mc = tiny_model()
        device, dtype = "cpu", "float32"
    else:
        from transformers import AutoModelForCausalLM
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}.get(args.dtype, torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=dt, device_map=args.device,
            trust_remote_code=True, local_files_only=True).eval()
        mc = model.config
        device, dtype = args.device, args.dtype

    n_layers = (args.num_layers if args.num_layers is not None
                else int(len(model.model.layers)))
    cfg = MemoryOptimizedConfig(
        num_layers=n_layers, batch_size=1, seq_len=8, max_new_tokens=1,
        device=device, dtype=dtype, folding_dtype="float32",
        folded_weight_device=device, seed=seed)
    session = MaskedQwenSession(model, mc, cfg)

    if adapter_path:
        if args.adapter_format not in ("hf_peft", "auto"):
            raise SystemExit("unsupported --adapter-format %r" % args.adapter_format)
        lora = load_hf_lora_adapter(adapter_path, mc, session.n,
                                    target_modules)
    else:
        lora = synthetic_lora_adapter(mc, session.n, target_modules, rank,
                                      seed=rank_seed, scale=args.synthetic_scale)

    t0 = time.perf_counter()
    report = build_lora_folded_package(
        args.output_dir, session=session, lora=lora,
        target_modules=target_modules, rank=rank, alpha=alpha,
        rank_seed=rank_seed, base_manifest_hash=base_manifest_hash,
        model_name=args.model_name,
        created_by="trusted_setup" if not dry_run else "test",
        nonlinear_backend=args.nonlinear_backend, build_command=build_command)
    report["build_time_s"] = round(time.perf_counter() - t0, 3)
    report["dry_run"] = dry_run
    report["seed"] = seed
    report["num_layers"] = session.n
    report["stage"] = "qwen7b_lora_folded_build"
    report["adapter_source"] = ("hf_peft" if adapter_path else "synthetic")
    report["lora_package_built"] = True
    # surface the package's no-secret guarantees at the top level (the meta sidecar
    # carries the same flags) so pipelines/validators do not have to open the dir.
    for k in ("contains_raw_lora", "contains_optimizer_state",
              "contains_training_data", "contains_mask_secrets"):
        report[k] = bool(report["meta"][k])
    report["build_command"] = build_command
    report.update(nonlinear_design_report_fields(args.nonlinear_backend))

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "meta"}, indent=2,
                     default=str))
    print("\nFOLDED-LoRA PACKAGE BUILT: %s (%d shards, %.6f GB)"
          % (report["out_dir"], report["num_shards"], report["size_gb"]))
    print("rank=%s alpha=%s scaling=%s target_modules=%s adapter_hash=%s"
          % (report["rank"], report["alpha"], report["scaling"],
             report["target_modules"], report["adapter_hash"]))
    print("contains_raw_lora=False contains_optimizer_state=False "
          "contains_training_data=False contains_mask_secrets=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
