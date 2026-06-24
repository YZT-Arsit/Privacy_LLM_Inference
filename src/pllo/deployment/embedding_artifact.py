"""Trusted boundary artifact for TDX-lite remote package-backed decode.

The strict cross-machine deployment splits the model across two domains:

* the **untrusted** GPU worker holds the 26GB folded-weight package (folded
  layers + folded head over masked tensors, NO mask secrets);
* the **trusted** boundary (a TDX guest) holds only the small material needed to
  mask the prompt + recover logits: the embedding table, the shared residual
  mask ``N_0`` and the vocab logit mask.

This module reads/writes that small trusted artifact (a few GB: the ~1GB
embedding table at the model dtype + a tiny ``[H,H]`` residual mask + the vocab
permutation/scale). It lets a TDX guest run masked prefill/decode against the
remote worker WITHOUT loading the full Qwen checkpoint or the 26GB folded
package.

SECURITY: this artifact deliberately CONTAINS mask secrets (``N_0`` + the vocab
mask) -- it is the trusted boundary's private material and MUST stay inside the
trusted domain. It is NEVER sent to the GPU worker; only masked embeddings +
public metadata cross the boundary (the protocol audit enforces this). The masks
are STORED (not regenerated from the seed on the guest) so the device-dependent
mask RNG can never silently diverge from the masks the package was folded with.

Stored with safetensors (torch.save fallback). The mask-secret tensor names are
written directly here -- intentionally bypassing the folded-package secret
screen, which exists to keep secrets OUT of the *untrusted* package.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

ARTIFACT_TENSORS = "boundary.safetensors"
ARTIFACT_TENSORS_PT = "boundary.pt"
ARTIFACT_META = "boundary_meta.json"

# Tensor keys inside the artifact (trusted-only; never cross to the GPU).
EMBED_KEY = "embed_tokens_weight"
N0_KEY = "residual_mask_n0"
VOCAB_PERM_KEY = "vocab_perm"
VOCAB_INV_PERM_KEY = "vocab_inv_perm"
VOCAB_SCALE_KEY = "vocab_scale"
VOCAB_INV_SCALE_KEY = "vocab_inv_scale"

__all__ = [
    "ARTIFACT_TENSORS", "ARTIFACT_TENSORS_PT", "ARTIFACT_META",
    "build_embedding_artifact", "load_embedding_artifact",
    "embedding_artifact_size_gb",
]


def _have_safetensors() -> bool:
    try:
        import safetensors.torch  # noqa: F401
        return True
    except Exception:                                        # noqa: BLE001
        return False


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_embedding_artifact(out_dir: str | Path, *, embed_tokens_weight,
                             residual_mask_n0, vocab_mask, meta: dict[str, Any],
                             prefer_safetensors: bool = True) -> dict[str, Any]:
    """Write the trusted boundary artifact (embedding table + N_0 + vocab mask).

    ``embed_tokens_weight`` ``[V, H]`` is stored at its given dtype (typically the
    model dtype, e.g. bf16) -- casting bf16->fold_dtype on load is lossless, so
    this matches the full session's float32 embedding exactly while halving disk.
    ``residual_mask_n0`` ``[H, H]`` and the vocab mask are stored at fold
    precision (float32). ``meta`` carries the PUBLIC model/RoPE hyper-parameters
    plus a trusted-only provenance ``seed`` (used for provenance, NOT regenerated
    on the guest)."""
    import torch  # noqa: F401
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {
        EMBED_KEY: embed_tokens_weight.detach().to("cpu").contiguous(),
        N0_KEY: residual_mask_n0.detach().to("cpu").float().contiguous(),
        VOCAB_PERM_KEY: vocab_mask.permutation.detach().to("cpu").contiguous(),
        VOCAB_INV_PERM_KEY:
            vocab_mask.inverse_permutation.detach().to("cpu").contiguous(),
        VOCAB_SCALE_KEY: vocab_mask.scale.detach().to("cpu").float().contiguous(),
        VOCAB_INV_SCALE_KEY:
            vocab_mask.inverse_scale.detach().to("cpu").float().contiguous(),
    }
    use_st = prefer_safetensors and _have_safetensors()
    if use_st:
        from safetensors.torch import save_file
        tpath = out_dir / ARTIFACT_TENSORS
        save_file(tensors, str(tpath))
        fmt = "safetensors"
    else:
        warnings.warn("safetensors unavailable; using torch.save (.pt) fallback",
                      RuntimeWarning, stacklevel=2)
        tpath = out_dir / ARTIFACT_TENSORS_PT
        torch.save(tensors, str(tpath))
        fmt = "torch_save"

    meta_out = dict(meta)
    meta_out.update({
        "artifact_type": "trusted_boundary_embedding",
        "tensors_file": tpath.name, "tensors_format": fmt,
        "tensors_sha256": _file_sha256(tpath),
        "hidden_size": int(embed_tokens_weight.shape[1]),
        "vocab_size": int(embed_tokens_weight.shape[0]),
        "embed_dtype": str(embed_tokens_weight.dtype).replace("torch.", ""),
        "contains_mask_secrets": True,
        "trusted_only": True,
        "never_send_to_gpu": [N0_KEY, VOCAB_PERM_KEY, VOCAB_INV_PERM_KEY,
                              VOCAB_SCALE_KEY, VOCAB_INV_SCALE_KEY, "seed"],
    })
    mpath = out_dir / ARTIFACT_META
    mpath.write_text(json.dumps(meta_out, indent=2, default=str),
                     encoding="utf-8")
    return {"out_dir": str(out_dir), "tensors_file": tpath.name,
            "tensors_format": fmt, "tensors_sha256": meta_out["tensors_sha256"],
            "size_gb": embedding_artifact_size_gb(out_dir),
            "hidden_size": meta_out["hidden_size"],
            "vocab_size": meta_out["vocab_size"]}


def load_embedding_artifact(art_dir: str | Path, *, device: str = "cpu",
                            fdtype=None):
    """Load the trusted boundary artifact.

    Returns ``(embed_weight, residual_mask_n0, vocab_mask, meta)`` with the
    embedding + N_0 on ``device`` and (if ``fdtype`` given) the embedding +
    masks cast to ``fdtype`` (the fold precision). The returned mask material is
    trusted-only and must never be sent to the GPU worker."""
    import torch
    from pllo.ops.causal_lm_boundaries import VocabLogitMask
    art_dir = Path(art_dir)
    meta = json.loads((art_dir / ARTIFACT_META).read_text(encoding="utf-8"))
    tname = meta.get("tensors_file") or ARTIFACT_TENSORS
    tpath = art_dir / tname
    if tpath.suffix == ".safetensors":
        from safetensors.torch import load_file
        tensors = dict(load_file(str(tpath)))
    else:
        tensors = dict(torch.load(str(tpath), map_location="cpu"))

    dev = torch.device(device)
    embed = tensors[EMBED_KEY].to(dev)
    n0 = tensors[N0_KEY].to(dev)
    if fdtype is not None:
        embed = embed.to(fdtype)
        n0 = n0.to(fdtype)
    scale = tensors[VOCAB_SCALE_KEY].to(dev)
    inv_scale = tensors[VOCAB_INV_SCALE_KEY].to(dev)
    if fdtype is not None:
        scale = scale.to(fdtype)
        inv_scale = inv_scale.to(fdtype)
    vocab_mask = VocabLogitMask(
        permutation=tensors[VOCAB_PERM_KEY].to(dev),
        inverse_permutation=tensors[VOCAB_INV_PERM_KEY].to(dev),
        scale=scale, inverse_scale=inv_scale)
    return embed, n0, vocab_mask, meta


def embedding_artifact_size_gb(art_dir: str | Path) -> float:
    art_dir = Path(art_dir)
    total = 0
    for name in (ARTIFACT_TENSORS, ARTIFACT_TENSORS_PT, ARTIFACT_META):
        p = art_dir / name
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 ** 3)
