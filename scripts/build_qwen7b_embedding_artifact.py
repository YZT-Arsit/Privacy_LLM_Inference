"""Trusted-setup builder for the TDX-lite boundary embedding artifact.

Exports the SMALL trusted material a TDX guest needs to run masked prefill/decode
against a remote folded-package GPU worker WITHOUT loading the full Qwen
checkpoint or the 26GB folded package:

* the embedding table ``embed_tokens`` (~1GB at the model dtype),
* the shared residual mask ``N_0`` (``[H,H]``),
* the vocab logit mask (permutation + positive scale).

These are computed from the SAME mask schedule (seed + mask mode/strategy) the
folded package was built with -- so the masks STORED here match the masks the
package was folded against exactly (no device-dependent RNG reproduction on the
guest). Point ``--folded-package-path`` at the package to auto-sync the seed.

SECURITY: the artifact CONTAINS mask secrets (N_0 + vocab mask + the provenance
seed). It is the TRUSTED boundary's private material and stays inside the trusted
domain; it is NEVER sent to the GPU worker (the protocol audit enforces that only
masked embeddings + public metadata cross). This is the embedding/recovery
counterpart to the untrusted folded package.

``--dry-run`` builds the tiny model's artifact on CPU (matches
build_qwen7b_folded_package.py::_tiny_model so a dry-run folded package + this
artifact agree).

Example (trusted setup, has the checkpoint)::

    python scripts/build_qwen7b_embedding_artifact.py \\
        --model-path /root/autodl-tmp/modelscope_cache/Qwen/Qwen2___5-7B-Instruct \\
        --model-name Qwen2.5-7B-Instruct \\
        --folded-package-path /root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full \\
        --output-dir /root/autodl-tmp/privacy_llm_packages/qwen7b_boundary_artifact \\
        --dtype bfloat16 --device cpu
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

from pllo.deployment.embedding_artifact import build_embedding_artifact  # noqa: E402
from pllo.experiments.folded_probe_common import seed_from_manifest, tiny_model  # noqa: E402
from pllo.hf_wrappers.qwen_masked_session import MaskedQwenSession  # noqa: E402
from pllo.hf_wrappers.qwen_memory_optimized import MemoryOptimizedConfig  # noqa: E402


def _artifact_meta(session, mc, model_name: str, seed: int) -> dict:
    cfg0 = session.layer_configs[0]
    fold_dtype = {torch.float32: "float32", torch.float64: "float64",
                  torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(
        getattr(session, "fdtype", torch.float32), "float32")
    return {
        "model_name": str(model_name),
        "model_type": str(cfg0.model_type),
        "hidden_size": int(cfg0.hidden_size),
        "vocab_size": int(getattr(mc, "vocab_size")),
        "num_layers": int(session.n),
        "num_heads": int(cfg0.num_heads),
        "num_key_value_heads": int(cfg0.num_key_value_heads),
        "head_dim": int(cfg0.head_dim),
        "intermediate_size": int(cfg0.intermediate_size),
        "rope_theta": float(cfg0.rope_theta),
        "rms_norm_eps": float(session.eps),
        "attention_bias": bool(cfg0.attention_bias),
        "mlp_bias": bool(cfg0.mlp_bias),
        "mask_family": str(cfg0.mask_family),
        "fold_dtype": fold_dtype,
        "seed": int(seed),                       # trusted-only provenance
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--folded-package-path", default=None,
                    help="sync the mask seed from this package's manifest")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-layers", type=int, default=None,
                    help="layers used for mask generation (default: all; for the "
                         "default shared residual mask N_0+vocab are layer-"
                         "independent so this only matters for per_layer masks)")
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--store-embed-dtype", default="model",
                    help="'model' keeps the checkpoint dtype (smaller, lossless "
                         "cast to fold precision on load) or a dtype name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dry_run = bool(args.dry_run or not args.model_path)
    seed = args.seed
    if args.folded_package_path:
        seed = seed_from_manifest(args.folded_package_path, args.seed)

    if dry_run:
        if not args.dry_run:
            print("NOTE: no --model-path; building --dry-run tiny artifact "
                  "(NOT a paper result).")
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

    cfg = MemoryOptimizedConfig(
        num_layers=args.num_layers, batch_size=1, seq_len=8, max_new_tokens=1,
        device=device, dtype=dtype, folding_dtype="float32",
        folded_weight_device=device, seed=seed)
    t0 = time.perf_counter()
    session = MaskedQwenSession(model, mc, cfg)

    # Original-dtype embedding (lossless cast to fold precision on load) keeps the
    # artifact small; the masks are stored at fold precision (float32).
    base = session.base
    embed = base.embed_tokens.weight.detach().to("cpu")
    if args.store_embed_dtype != "model":
        ed = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}.get(args.store_embed_dtype)
        if ed is not None:
            embed = embed.to(ed)

    meta = _artifact_meta(session, mc, args.model_name, seed)
    info = build_embedding_artifact(
        args.output_dir, embed_tokens_weight=embed,
        residual_mask_n0=session._n0, vocab_mask=session._vocab_mask, meta=meta)
    build_time_s = time.perf_counter() - t0

    report = {
        "stage": "qwen7b_boundary_embedding_artifact",
        "dry_run": dry_run, "model_name": args.model_name,
        "output_dir": info["out_dir"], "tensors_file": info["tensors_file"],
        "tensors_format": info["tensors_format"],
        "tensors_sha256": info["tensors_sha256"],
        "size_gb": round(info["size_gb"], 6),
        "hidden_size": info["hidden_size"], "vocab_size": info["vocab_size"],
        "num_layers": meta["num_layers"], "seed": seed,
        "embed_dtype": str(embed.dtype).replace("torch.", ""),
        "contains_mask_secrets": True, "trusted_only": True,
        "build_time_s": round(build_time_s, 3),
    }
    print(json.dumps(report, indent=2))
    print("\nEMBEDDING ARTIFACT BUILT: %s (%.4f GB, %s)"
          % (info["out_dir"], info["size_gb"], info["tensors_format"]))
    print("TRUSTED-ONLY: contains N_0 + vocab mask + seed; NEVER send to GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
