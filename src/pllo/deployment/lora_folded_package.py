"""Private folded-LoRA package: fold a private LoRA adapter into the SAME masked
basis as the public base folded package, so the untrusted GPU can add the LoRA
contribution over masked activations WITHOUT ever seeing raw A/B or the masks.

Conventions (row-vector, weight ``W`` is ``[in, out]`` so ``y = x @ W``):

* base folding (per module ``m``): ``W_tilde = L_m @ (rms_m * W) @ R_m`` where the
  left op ``L_m`` carries the input residual-mask inverse (or value-mask inverse /
  SwiGLU row-perm) and ``R_m`` carries the per-head / SwiGLU-col / residual output
  mask (and the q/k RoPE alignment). This mirrors
  ``fold_layer_attention_and_up`` + the folded down projection exactly.
* a LoRA branch is ``ΔW = scaling * A @ B`` with ``A`` ``[in, r]``, ``B`` ``[r, out]``
  (``scaling = alpha / r``). Folding is LINEAR and factors through the rank:
  ``L_m (A B) R_m = (L_m A)(B R_m)``. We therefore store low-rank folded
  operators ``a_tilde = (L_m A) @ Rk`` ``[in, r]`` and
  ``b_tilde = scaling * Rk^{-1} @ (B R_m)`` ``[r, out]``, where ``Rk`` is a private
  per-(layer, module) rank mask (a signed permutation, ``Rk^{-1} = Rk^T``). The
  rank mask cancels in the product (``a_tilde @ b_tilde == folded ΔW``) but keeps
  the rank-``r`` bottleneck masked if a worker multiplies the two factors over
  activations. The GPU MERGES ``W_tilde += a_tilde @ b_tilde`` and runs the
  existing masked kernels unchanged.

SECURITY: the GPU sees only ``a_tilde`` / ``b_tilde`` (folded with ``N`` masks +
``Rk``); recovering raw ``A``/``B`` needs ``N``, ``rms``, ``Rk`` (all trusted).
The package carries NO raw LoRA, NO optimizer state, NO training data, NO mask
secrets (enforced by name + the folded-package secret screen). ``tee_used`` n/a.

stdlib + torch (torch is used only by the folding functions).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_TARGET_MODULES", "MODULE_TO_WKEY", "ALL_TARGET_MODULES",
    "lora_scaling", "synthetic_lora_adapter", "load_hf_lora_adapter",
    "rank_mask", "fold_lora_for_layer", "merge_folded_lora",
    "adapter_hash", "build_lora_folded_package", "load_lora_meta",
    "load_folded_lora_layer", "verify_lora_folded_package",
    "apply_lora_to_model", "LORA_FOLD_FORMULA",
]

LORA_FOLD_FORMULA = (
    "a_tilde = (L_m @ A) @ Rk ; b_tilde = (alpha/r) * Rk^{-1} @ (B @ R_m) ; "
    "GPU merges W_tilde += a_tilde @ b_tilde (same masked basis as the base "
    "folded operator). L_m/R_m are the base per-module fold ops; Rk is a private "
    "rank mask (signed permutation, Rk^{-1}=Rk^T).")

ALL_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

MODULE_TO_WKEY = {
    "q_proj": "wq_tilde", "k_proj": "wk_tilde", "v_proj": "wv_tilde",
    "o_proj": "wo_tilde", "gate_proj": "wgate_tilde", "up_proj": "wup_tilde",
    "down_proj": "wdown_tilde",
}

LORA_MANIFEST_FILENAME = "lora_meta.json"


def lora_scaling(alpha: float, rank: int) -> float:
    return float(alpha) / float(rank)


# ---------------------------------------------------------------------------
# Adapter sources (codebase convention: A [in, r], B [r, out], dW = s * A @ B)
# ---------------------------------------------------------------------------


def _module_in_out(mc, module: str):
    H = int(mc.hidden_size)
    n_heads = int(getattr(mc, "num_attention_heads"))
    n_kv = int(getattr(mc, "num_key_value_heads", n_heads))
    head_dim = int(getattr(mc, "head_dim", H // n_heads))
    inter = int(mc.intermediate_size)
    return {
        "q_proj": (H, n_heads * head_dim), "k_proj": (H, n_kv * head_dim),
        "v_proj": (H, n_kv * head_dim), "o_proj": (n_heads * head_dim, H),
        "gate_proj": (H, inter), "up_proj": (H, inter), "down_proj": (inter, H),
    }[module]


def synthetic_lora_adapter(mc, num_layers: int, target_modules, rank: int,
                           seed: int = 7, scale: float = 0.02):
    """Deterministic synthetic LoRA (NOT a paper result): per (layer, module) a
    small random ``A``/``B`` in the codebase convention. Used for dry-run."""
    import torch
    out: dict[int, dict[str, tuple]] = {}
    for ell in range(num_layers):
        out[ell] = {}
        for mi, m in enumerate(target_modules):
            din, dout = _module_in_out(mc, m)
            g = torch.Generator().manual_seed(seed + 1009 * (ell + 1) + 31 * mi)
            a = torch.randn(din, rank, generator=g) * scale
            b = torch.randn(rank, dout, generator=g) * scale
            out[ell][m] = (a, b)
    return out


def load_hf_lora_adapter(adapter_path, mc, num_layers: int, target_modules,
                         dtype=None):
    """Best-effort load of a PEFT/HF LoRA adapter into the codebase convention
    ``A [in, r]``, ``B [r, out]`` per (layer, module). HF stores ``lora_A`` as
    ``[r, in]`` and ``lora_B`` as ``[out, r]`` (``dW_hf[out,in] = B_hf @ A_hf``);
    we transpose to ``A = A_hf^T``, ``B = B_hf^T``. Not exercised by dry-run."""
    import torch
    from safetensors.torch import load_file
    p = Path(adapter_path)
    sd = None
    for name in ("adapter_model.safetensors",):
        if (p / name).is_file():
            sd = load_file(str(p / name))
            break
    if sd is None and p.is_file():
        sd = load_file(str(p))
    if sd is None:
        raise FileNotFoundError("no adapter_model.safetensors under %s" % p)

    def find(ell, module, ab):
        for k, v in sd.items():
            if (".layers.%d." % ell) in k and module in k and \
                    ("lora_%s" % ab) in k.lower():
                return v
        return None

    out: dict[int, dict[str, tuple]] = {}
    for ell in range(num_layers):
        layer = {}
        for m in target_modules:
            a_hf = find(ell, m, "a")
            b_hf = find(ell, m, "b")
            if a_hf is None or b_hf is None:
                continue
            a = a_hf.t().contiguous().float()       # [in, r]
            b = b_hf.t().contiguous().float()       # [r, out]
            if dtype is not None:
                a, b = a.to(dtype), b.to(dtype)
            layer[m] = (a, b)
        if layer:
            out[ell] = layer
    return out


# ---------------------------------------------------------------------------
# Rank mask (private, signed permutation; never sent to the GPU)
# ---------------------------------------------------------------------------


def rank_mask(rank: int, seed: int, dtype=None, device="cpu"):
    """A private rank mask ``Rk`` ``[r, r]`` (signed permutation -> orthogonal,
    ``Rk^{-1} = Rk^T``). Trusted-only; used to mask the rank-r bottleneck."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(rank, generator=g)
    signs = torch.where(torch.rand(rank, generator=g) < 0.5,
                        torch.tensor(-1.0), torch.tensor(1.0))
    eye = torch.eye(rank)
    rk = (eye.index_select(1, perm) * signs)
    if dtype is not None:
        rk = rk.to(dtype)
    rk = rk.to(device)
    return rk, rk.transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Per-layer base fold ops + LoRA folding
# ---------------------------------------------------------------------------


def _layer_fold_ops(session, ell: int):
    """Per-module left/right fold ops for layer ``ell`` from the base session's
    masks (matches ``fold_layer_attention_and_up`` + folded down projection)."""
    import torch
    from pllo.hf_wrappers.llama_qwen_single_block import (
        block_diag_from_head_masks,
        extract_hf_single_block_weights,
    )
    from pllo.hf_wrappers.qwen_memory_optimized import hf_rope_interleave_index

    dev, fdt = session.compute_device, session.fdtype
    bm = session.masks.layer_block_masks[ell]
    n_res = bm["n_res"].to(dev, fdt)
    n_res_inv = bm["n_res_inv"].to(dev, fdt)
    perm = bm["perm"].to(dev)
    attn = bm["attn"]
    mq = block_diag_from_head_masks(attn["q_masks"]).to(dev, fdt)
    mk = block_diag_from_head_masks(attn["key_masks"]).to(dev, fdt)
    mv = block_diag_from_head_masks(attn["value_masks"]).to(dev, fdt)
    v_inv_qhead = attn["value_mask_inverses"].index_select(0, attn["kv_index"])
    sv_inv = block_diag_from_head_masks(v_inv_qhead).to(dev, fdt)

    w = extract_hf_single_block_weights(session.base.layers[ell], fdt, str(dev))
    rms1 = w.input_layernorm_weight.to(dev, fdt).unsqueeze(1)       # [H,1]
    rms2 = w.post_attention_layernorm_weight.to(dev, fdt).unsqueeze(1)

    c = session.layer_configs[ell]
    align = bool(session.config.align_rope_to_hf)
    gq = (hf_rope_interleave_index(c.num_heads, c.head_dim, dev) if align
          else None)
    gk = (hf_rope_interleave_index(c.num_key_value_heads, c.head_dim, dev)
          if align else None)
    return {"n_res": n_res, "n_res_inv": n_res_inv, "perm": perm, "mq": mq,
            "mk": mk, "mv": mv, "sv_inv": sv_inv, "rms1": rms1, "rms2": rms2,
            "gq": gq, "gk": gk}


def _fold_module(module: str, a, b, ops, scaling: float, rk, rk_inv):
    """Return (a_tilde [in,r], b_tilde [r,out]) for one module's LoRA branch."""
    n_res, n_res_inv = ops["n_res"], ops["n_res_inv"]
    perm = ops["perm"]
    if module == "q_proj":
        a_l = n_res_inv @ (ops["rms1"] * a)
        b_r = (b.index_select(1, ops["gq"]) if ops["gq"] is not None else b) \
            @ ops["mq"]
    elif module == "k_proj":
        a_l = n_res_inv @ (ops["rms1"] * a)
        b_r = (b.index_select(1, ops["gk"]) if ops["gk"] is not None else b) \
            @ ops["mk"]
    elif module == "v_proj":
        a_l = n_res_inv @ (ops["rms1"] * a)
        b_r = b @ ops["mv"]
    elif module == "o_proj":
        a_l = ops["sv_inv"] @ a
        b_r = b @ n_res
    elif module == "gate_proj":
        a_l = n_res_inv @ (ops["rms2"] * a)
        b_r = b.index_select(1, perm)
    elif module == "up_proj":
        a_l = n_res_inv @ (ops["rms2"] * a)
        b_r = b.index_select(1, perm)
    elif module == "down_proj":
        a_l = a.index_select(0, perm)
        b_r = b @ n_res
    else:
        raise ValueError("unknown target module %r" % module)
    a_tilde = a_l @ rk
    b_tilde = (scaling * rk_inv) @ b_r
    return a_tilde, b_tilde


def fold_lora_for_layer(session, ell: int, lora_layer: dict, *, scaling: float,
                        rank: int, rank_seed: int, target_modules):
    """Folded low-rank LoRA operators for layer ``ell`` (one shard's tensors).

    Returns ``{f"{m}_lora_a_tilde": ..., f"{m}_lora_b_tilde": ...}`` for each
    module present. ``rank_seed`` derives the private per-(layer, module) rank
    mask (never exported)."""
    ops = _layer_fold_ops(session, ell)
    dev, fdt = session.compute_device, session.fdtype
    out: dict[str, Any] = {}
    for mi, m in enumerate(target_modules):
        if m not in lora_layer:
            continue
        a, b = lora_layer[m]
        a = a.to(dev, fdt)
        b = b.to(dev, fdt)
        rk, rk_inv = rank_mask(rank, rank_seed + 911 * (ell + 1) + 13 * mi,
                               dtype=fdt, device=dev)
        a_tilde, b_tilde = _fold_module(m, a, b, ops, scaling, rk, rk_inv)
        out["%s_lora_a_tilde" % m] = a_tilde.contiguous()
        out["%s_lora_b_tilde" % m] = b_tilde.contiguous()
    return out


def merge_folded_lora(base_layer_tensors: dict, lora_layer_tensors: dict,
                      target_modules=None) -> dict:
    """Merge folded LoRA into a base folded layer dict (in place + returned):
    ``W_tilde += a_tilde @ b_tilde`` for each present module. The worker calls
    this; it never sees raw A/B or masks."""
    mods = target_modules or ALL_TARGET_MODULES
    for m in mods:
        ak, bk = "%s_lora_a_tilde" % m, "%s_lora_b_tilde" % m
        if ak in lora_layer_tensors and bk in lora_layer_tensors:
            wkey = MODULE_TO_WKEY[m]
            if wkey not in base_layer_tensors:
                raise KeyError("base layer missing %r for module %r"
                               % (wkey, m))
            a_t = lora_layer_tensors[ak]
            b_t = lora_layer_tensors[bk]
            delta = a_t.to(base_layer_tensors[wkey].dtype) @ \
                b_t.to(base_layer_tensors[wkey].dtype)
            base_layer_tensors[wkey] = base_layer_tensors[wkey] + delta
    return base_layer_tensors


# ---------------------------------------------------------------------------
# Trusted reference: apply raw LoRA to an HF model (for correctness comparison)
# ---------------------------------------------------------------------------


def apply_lora_to_model(model, lora: dict, target_modules, scaling: float):
    """Add ``scaling * (A @ B)`` to each target Linear's weight, in place (HF
    weight is ``[out, in]`` so the delta is ``(A @ B)^T``). Trusted-side only;
    used to build the 'base + raw LoRA' reference."""
    import torch
    base = model.model
    submod = {"q_proj": ("self_attn", "q_proj"), "k_proj": ("self_attn", "k_proj"),
              "v_proj": ("self_attn", "v_proj"), "o_proj": ("self_attn", "o_proj"),
              "gate_proj": ("mlp", "gate_proj"), "up_proj": ("mlp", "up_proj"),
              "down_proj": ("mlp", "down_proj")}
    with torch.no_grad():
        for ell, layer in enumerate(base.layers):
            if ell not in lora:
                continue
            for m in target_modules:
                if m not in lora[ell]:
                    continue
                a, b = lora[ell][m]
                blk, name = submod[m]
                lin = getattr(getattr(layer, blk), name)
                w = lin.weight                       # [out, in]
                delta = (scaling * (a @ b)).to(w.dtype).t().to(w.device)
                w.add_(delta)
    return model


# ---------------------------------------------------------------------------
# Package build / load / verify
# ---------------------------------------------------------------------------


def adapter_hash(lora: dict, target_modules) -> str:
    """Stable sha256 over the raw adapter tensors (provenance; computed trusted-
    side, NEVER stored in the package)."""
    import torch
    h = hashlib.sha256()
    for ell in sorted(lora):
        for m in target_modules:
            if m in lora[ell]:
                a, b = lora[ell][m]
                h.update(("%d:%s:" % (ell, m)).encode())
                h.update(torch.as_tensor(a).detach().to("cpu").float()
                         .numpy().tobytes())
                h.update(torch.as_tensor(b).detach().to("cpu").float()
                         .numpy().tobytes())
    return h.hexdigest()


def load_lora_meta(package_dir) -> dict:
    return json.loads((Path(package_dir) / LORA_MANIFEST_FILENAME)
                      .read_text(encoding="utf-8"))


def load_folded_lora_layer(package_dir, ell: int) -> dict:
    """Load one folded-LoRA layer shard (``lora_layer_{ell:03d}``)."""
    from pllo.deployment.folded_package import list_package_shards, load_shard
    name = "lora_layer_%03d" % ell
    for p in list_package_shards(Path(package_dir)):
        if p.stem == name:
            return load_shard(p)
    raise FileNotFoundError("no folded-LoRA shard %r in %s" % (name, package_dir))


def build_lora_folded_package(out_dir, *, session, lora: dict, target_modules,
                              rank: int, alpha: float, rank_seed: int,
                              base_manifest_hash: str | None,
                              model_name: str | None = None,
                              created_by: str = "trusted_setup") -> dict:
    """Fold the adapter layer-by-layer and stream folded-LoRA shards + a manifest
    + a ``lora_meta.json`` sidecar. Returns a build report."""
    from pllo.deployment.folded_package import FoldedPackageWriter
    from pllo.deployment.folded_package_manifest import build_manifest, write_manifest

    scaling = lora_scaling(alpha, rank)
    writer = FoldedPackageWriter(out_dir)
    covered: dict[int, list] = {}
    for ell in sorted(lora):
        tensors = fold_lora_for_layer(
            session, ell, lora[ell], scaling=scaling, rank=rank,
            rank_seed=rank_seed, target_modules=target_modules)
        if not tensors:
            continue
        writer.add_shard("lora_layer_%03d" % ell, tensors)   # screens names
        covered[ell] = sorted(m for m in target_modules if m in lora[ell])

    manifest = build_manifest(
        package_type="lora_adapter", model_name=model_name,
        model_path_or_id=None, num_layers=int(session.n),
        dtype=str(session.fdtype).replace("torch.", ""),
        nonlinear_backend="current", created_by=created_by,
        shard_index=writer.shard_index,
        mask_schedule_id=getattr(session.masks, "metadata", {}).get(
            "mask_schedule_id") if hasattr(session.masks, "metadata") else None)
    write_manifest(manifest, out_dir)

    a_hash = adapter_hash(lora, target_modules)
    meta = {
        "package_type": "lora_adapter", "trusted_setup": True,
        "rank": int(rank), "alpha": float(alpha), "scaling": float(scaling),
        "target_modules": list(target_modules),
        "covered_layers": {str(k): v for k, v in covered.items()},
        "num_covered_layers": len(covered),
        "adapter_hash": a_hash,
        "base_package_manifest_hash": base_manifest_hash,
        "model_name": model_name,
        "lora_fold_formula": LORA_FOLD_FORMULA,
        "contains_raw_lora": False, "contains_optimizer_state": False,
        "contains_training_data": False, "contains_mask_secrets": False,
    }
    (Path(out_dir) / LORA_MANIFEST_FILENAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    from pllo.deployment.folded_package import package_size_gb
    return {"out_dir": str(out_dir), "num_shards": len(writer.shard_index),
            "size_gb": round(package_size_gb(out_dir), 6),
            "adapter_hash": a_hash, "rank": rank, "alpha": alpha,
            "scaling": scaling, "target_modules": list(target_modules),
            "covered_layers": covered,
            "base_package_manifest_hash": base_manifest_hash, "meta": meta}


# raw-LoRA tensor-name patterns that must NEVER appear in a folded-LoRA shard
_RAW_LORA_NAME_HINTS = ("lora_a_raw", "lora_b_raw", "raw_lora", "lora_a.weight",
                        "lora_b.weight", "base_layer")


def verify_lora_folded_package(package_dir, *,
                               base_manifest_hash: str | None = None) -> dict:
    """Verify a folded-LoRA package: shard hashes, no forbidden / raw-LoRA tensor
    names, no optimizer/training/mask secrets, target coverage, and (optional)
    compatibility with the base folded package manifest hash."""
    from pllo.deployment.folded_package import (
        forbidden_tensor_names,
        list_package_shards,
        load_shard,
        verify_package,
    )
    package_dir = Path(package_dir)
    rep: dict[str, Any] = {"package_dir": str(package_dir),
                           "lora_package_valid": False, "problems": []}
    base_rep = verify_package(package_dir)                 # shard hashes + manifest
    rep["shard_integrity_valid"] = bool(base_rep["package_valid"])
    rep["num_shards"] = base_rep["num_shards"]
    rep["manifest_hash"] = base_rep["manifest_hash"]
    if not base_rep["package_valid"]:
        rep["problems"].append("shard/manifest integrity failed")

    forbidden_found: list[str] = []
    raw_lora_found: list[str] = []
    for p in list_package_shards(package_dir):
        names = list(load_shard(p).keys())
        forbidden_found += ["%s:%s" % (p.stem, n)
                            for n in forbidden_tensor_names(names)]
        for n in names:
            low = n.lower()
            if any(h in low for h in _RAW_LORA_NAME_HINTS):
                raw_lora_found.append("%s:%s" % (p.stem, n))
    rep["forbidden_fields_found"] = forbidden_found
    rep["raw_lora_tensor_names_found"] = raw_lora_found

    meta = {}
    try:
        meta = load_lora_meta(package_dir)
    except Exception as exc:                                # noqa: BLE001
        rep["problems"].append("missing/unreadable lora_meta.json: %s" % exc)
    rep["lora_meta"] = meta
    rep["rank"] = meta.get("rank")
    rep["alpha"] = meta.get("alpha")
    rep["target_modules"] = meta.get("target_modules")
    rep["adapter_hash"] = meta.get("adapter_hash")
    rep["base_package_manifest_hash"] = meta.get("base_package_manifest_hash")
    rep["contains_raw_lora"] = bool(meta.get("contains_raw_lora", False))
    rep["contains_optimizer_state"] = bool(
        meta.get("contains_optimizer_state", False))
    rep["contains_training_data"] = bool(
        meta.get("contains_training_data", False))
    rep["contains_mask_secrets"] = bool(meta.get("contains_mask_secrets", False))

    # target coverage: every declared target module appears in >=1 shard
    declared = set(meta.get("target_modules") or [])
    seen_modules: set = set()
    for p in list_package_shards(package_dir):
        for n in load_shard(p).keys():
            for m in ALL_TARGET_MODULES:
                if n.startswith(m + "_lora_"):
                    seen_modules.add(m)
    missing_cov = sorted(declared - seen_modules)
    rep["target_modules_covered"] = sorted(seen_modules)
    rep["target_modules_missing_coverage"] = missing_cov

    base_ok = True
    if base_manifest_hash is not None:
        base_ok = (meta.get("base_package_manifest_hash") == base_manifest_hash)
        rep["base_manifest_match"] = base_ok
        if not base_ok:
            rep["problems"].append("base_package_manifest_hash mismatch")

    rep["lora_package_valid"] = bool(
        rep["shard_integrity_valid"] and not forbidden_found
        and not raw_lora_found and not missing_cov
        and not rep["contains_raw_lora"] and not rep["contains_optimizer_state"]
        and not rep["contains_training_data"] and not rep["contains_mask_secrets"]
        and base_ok)
    return rep
