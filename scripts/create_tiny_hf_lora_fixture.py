"""Create a tiny, self-contained HF/PEFT-style LoRA adapter fixture (no download).

Writes a directory that looks like a real PEFT LoRA adapter:

* ``adapter_config.json`` -- ``peft_type=LORA``, ``r``, ``lora_alpha``,
  ``target_modules``, ``bias=none`` (the public hyper-parameters the builder
  reads via ``read_hf_adapter_config``);
* ``adapter_model.safetensors`` -- ``lora_A`` ``[r, in]`` + ``lora_B`` ``[out, r]``
  per (layer, module), named exactly like PEFT
  (``base_model.model.model.layers.{ell}.<blk>.<proj>.lora_A.weight``).

This exercises the real ``load_hf_lora_adapter`` path (PEFT layout ->
``A [in, r]``, ``B [r, out]`` codebase convention) without any external
checkpoint. The fixture is deterministic for a fixed seed. NOT a paper result.

Example::

    python scripts/create_tiny_hf_lora_fixture.py \\
        --output-dir /tmp/tiny_hf_lora --rank 4 --alpha 8 \\
        --target-modules q_proj,k_proj,v_proj,o_proj --num-layers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.deployment.lora_folded_package import (  # noqa: E402
    ALL_TARGET_MODULES,
    DEFAULT_TARGET_MODULES,
    _module_in_out,
)

# PEFT submodule path per projection (for the safetensors key).
_PEFT_BLK = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn",
             "o_proj": "self_attn", "gate_proj": "mlp", "up_proj": "mlp",
             "down_proj": "mlp"}


def _csv(s):
    return [p for p in str(s).replace(" ", "").split(",") if p]


def create_tiny_hf_lora_fixture(out_dir, mc, *, num_layers, target_modules,
                                rank=4, alpha=8.0, seed=0, scale=0.05):
    """Write a PEFT-style fixture; return (path, raw_lora) where ``raw_lora`` is
    the canonical ``{ell: {module: (A[in,r], B[r,out])}}`` dict actually stored
    (so callers can cross-check the loader's transpose)."""
    import torch
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}
    raw_lora: dict = {}
    for ell in range(num_layers):
        raw_lora[ell] = {}
        for mi, m in enumerate(target_modules):
            din, dout = _module_in_out(mc, m)
            g = torch.Generator().manual_seed(seed + 7919 * (ell + 1) + 17 * mi)
            # canonical convention: A [in, r], B [r, out]
            a = torch.randn(din, rank, generator=g) * scale
            b = torch.randn(rank, dout, generator=g) * scale
            raw_lora[ell][m] = (a, b)
            key = "base_model.model.model.layers.%d.%s.%s" % (
                ell, _PEFT_BLK[m], m)
            # PEFT on-disk layout: lora_A [r, in], lora_B [out, r]
            tensors["%s.lora_A.weight" % key] = a.t().contiguous()
            tensors["%s.lora_B.weight" % key] = b.t().contiguous()

    save_file(tensors, str(out / "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": int(rank), "lora_alpha": float(alpha), "lora_dropout": 0.0,
        "bias": "none", "fan_in_fan_out": False,
        "target_modules": list(target_modules),
        "base_model_name_or_path": "tiny-qwen2-fixture",
        "inference_mode": True,
    }
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2),
                                             encoding="utf-8")
    return out, raw_lora


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.05)
    args = ap.parse_args()

    target_modules = _csv(args.target_modules)
    bad = [m for m in target_modules if m not in ALL_TARGET_MODULES]
    if bad:
        raise SystemExit("unknown target modules: %s" % bad)

    from pllo.experiments.folded_probe_common import tiny_model
    _model, mc = tiny_model()
    out, raw = create_tiny_hf_lora_fixture(
        args.output_dir, mc, num_layers=args.num_layers,
        target_modules=target_modules, rank=args.rank, alpha=args.alpha,
        seed=args.seed, scale=args.scale)
    n_mod = sum(len(v) for v in raw.values())
    print("=== tiny HF/PEFT LoRA fixture ===")
    print("out_dir=%s" % out)
    print("rank=%s alpha=%s target_modules=%s num_layers=%s tensors=%d"
          % (args.rank, args.alpha, target_modules, args.num_layers, 2 * n_mod))
    print("files: adapter_config.json, adapter_model.safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
