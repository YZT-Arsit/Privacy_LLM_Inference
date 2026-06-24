"""Shared helpers for the folded-package execution probes.

Centralises model loading + tokenisation, the seed-from-manifest parse, and error
statistics so the prefill / one-step-logits / decode probes stay consistent. The
tiny dry-run model MUST match ``build_qwen7b_folded_package.py::_tiny_model`` so a
dry-run package built there and consumed by a probe uses the identical model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def tiny_model():
    """Tiny random Qwen2 -- MUST match build_qwen7b_folded_package.py::_tiny_model."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    mc = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                     num_hidden_layers=4, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=256,
                     rms_norm_eps=1e-6, rope_theta=1_000_000.0,
                     tie_word_embeddings=False)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(mc).eval(), mc


def _bool(s) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_model_and_ids(args, dry_run: bool):
    """Return (model, model_config, input_ids, device, dtype).

    Real path: load the HF checkpoint + tokenizer (optional chat template),
    truncate to ``--seq-len``. Dry-run: tiny random Qwen2 on CPU with a short
    random prompt (never a paper result)."""
    if dry_run:
        if not getattr(args, "dry_run", False):
            print("NOTE: no --model-path; running --dry-run tiny model (NOT a "
                  "paper result).")
        model, mc = tiny_model()
        ids = torch.randint(0, mc.vocab_size, (1, min(args.seq_len, 8)))
        return model, mc, ids, "cpu", "float32"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}.get(args.dtype, torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dt, device_map=args.device,
        trust_remote_code=True, local_files_only=True).eval()
    text = args.prompt
    if _bool(getattr(args, "use_chat_template", "true")):
        text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                       tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"][:, :args.seq_len]
    return model, model.config, ids.to(args.device), args.device, args.dtype


def folded_exec_metadata(session, *, model_name: str, num_layers: int,
                         seq_len: int, max_new_tokens: int, vocab_size: int
                         ) -> dict:
    """PUBLIC model + RoPE metadata a remote folded-package worker needs to
    rebuild its per-layer config + RoPE caches and execute the folded shards.

    Contains ONLY public hyper-parameters (head counts, head_dim, RoPE theta,
    rms_norm_eps, biases, mask_family, dtype, sizes). It deliberately omits the
    seed and every mask secret -- the worker reconstructs the deterministic
    public artifacts (config + cos/sin) and never the masks."""
    cfg0 = session.layer_configs[0]
    fold_dtype = {torch.float32: "float32", torch.float64: "float64",
                  torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(
        getattr(session, "fdtype", torch.float32), "float32")
    rope_max_pos = int(seq_len) + max(0, int(max_new_tokens) - 1) + 1
    return {
        "model_name": str(model_name),
        "model_type": str(cfg0.model_type),
        "hidden_size": int(cfg0.hidden_size),
        "intermediate_size": int(cfg0.intermediate_size),
        "num_heads": int(cfg0.num_heads),
        "num_key_value_heads": int(cfg0.num_key_value_heads),
        "head_dim": int(cfg0.head_dim),
        "rope_theta": float(cfg0.rope_theta),
        "rms_norm_eps": float(session.eps),
        "attention_bias": bool(cfg0.attention_bias),
        "mlp_bias": bool(cfg0.mlp_bias),
        "mask_family": str(cfg0.mask_family),
        "fold_dtype": fold_dtype,
        "rope_max_pos": rope_max_pos,
        "num_layers": int(num_layers),
        "vocab_size": int(vocab_size),
        "seq_len": int(seq_len),
        "max_new_tokens": int(max_new_tokens),
    }


def folded_exec_metadata_from_meta(meta: dict, *, seq_len: int,
                                   max_new_tokens: int) -> dict:
    """Build the PUBLIC worker exec metadata from a boundary-artifact meta dict
    (the lite/TDX path, where there is no full session). Copies the public config
    fields, adds the runtime-derived ``rope_max_pos`` / ``seq_len`` /
    ``max_new_tokens``, and DROPS the trusted-only ``seed``."""
    rope_max_pos = int(seq_len) + max(0, int(max_new_tokens) - 1) + 1
    out = {
        "model_name": str(meta.get("model_name", "unknown")),
        "model_type": str(meta.get("model_type", "qwen2")),
        "hidden_size": int(meta["hidden_size"]),
        "intermediate_size": int(meta["intermediate_size"]),
        "num_heads": int(meta["num_heads"]),
        "num_key_value_heads": int(meta["num_key_value_heads"]),
        "head_dim": int(meta["head_dim"]),
        "rope_theta": float(meta["rope_theta"]),
        "rms_norm_eps": float(meta["rms_norm_eps"]),
        "attention_bias": bool(meta.get("attention_bias", False)),
        "mlp_bias": bool(meta.get("mlp_bias", False)),
        "mask_family": str(meta.get("mask_family", "pairwise_complex_scaling")),
        "fold_dtype": str(meta.get("fold_dtype", "float32")),
        "rope_max_pos": rope_max_pos,
        "num_layers": int(meta["num_layers"]),
        "vocab_size": int(meta["vocab_size"]),
        "seq_len": int(seq_len),
        "max_new_tokens": int(max_new_tokens),
    }
    assert "seed" not in out, "exec metadata must never carry the mask seed"
    return out


class LiteBoundary:
    """Lightweight trusted boundary for TDX-lite remote package-backed decode.

    Holds ONLY the small trusted material from a boundary embedding artifact (the
    embedding table + shared residual mask ``N_0`` + vocab mask) -- NOT the full
    Qwen checkpoint, NOT the 26GB folded package. It can embed+mask the prompt /
    each sampled token and recover masked logits, which is everything the boundary
    needs to drive a remote folded-package worker. The masks are trusted-only and
    never leave this object (only masked embeddings cross to the GPU)."""

    def __init__(self, embed_weight, n0, vocab_mask, meta: dict,
                 device: str = "cpu", fdtype=None) -> None:
        self.compute_device = torch.device(device)
        self.fdtype = fdtype if fdtype is not None else torch.float32
        self.embed = embed_weight.to(self.compute_device, self.fdtype)
        self._n0 = n0.to(self.compute_device, self.fdtype)
        self._vocab_mask = vocab_mask
        self.meta = dict(meta)
        self.eps = float(meta["rms_norm_eps"])

    @classmethod
    def from_artifact(cls, art_dir, *, device: str = "cpu", fdtype=None):
        from pllo.deployment.embedding_artifact import load_embedding_artifact
        fdt = fdtype if fdtype is not None else torch.float32
        embed, n0, vocab_mask, meta = load_embedding_artifact(
            art_dir, device=device, fdtype=fdt)
        return cls(embed, n0, vocab_mask, meta, device=device, fdtype=fdt)

    def mask_embeddings(self, input_ids):
        from pllo.ops.causal_lm_boundaries import trusted_embedding_lookup
        x = trusted_embedding_lookup(input_ids.to(self.compute_device),
                                     self.embed)
        return x @ self._n0

    def mask_token_embedding(self, token_ids):
        from pllo.ops.causal_lm_boundaries import trusted_embedding_lookup
        ids = token_ids.reshape(-1, 1).to(self.compute_device)
        return trusted_embedding_lookup(ids, self.embed) @ self._n0

    def recover(self, logits_tilde):
        from pllo.ops.causal_lm_boundaries import recover_vocab_logits
        return recover_vocab_logits(logits_tilde, self._vocab_mask)

    def exec_metadata(self, *, seq_len: int, max_new_tokens: int) -> dict:
        return folded_exec_metadata_from_meta(
            self.meta, seq_len=seq_len, max_new_tokens=max_new_tokens)


def seed_from_manifest(pkg_dir, default: int) -> int:
    """Parse the seed from a package manifest's mask_schedule_id
    (``<sched>-seed<seed>-n<n>``) so a probe's reference masks match the package."""
    try:
        from pllo.deployment import load_manifest
        sid = load_manifest(pkg_dir).mask_schedule_id or ""
        if "-seed" in sid:
            return int(sid.split("-seed", 1)[1].split("-", 1)[0])
    except Exception:                                    # noqa: BLE001
        pass
    return int(default)


def err_stats(a: torch.Tensor, b: torch.Tensor):
    """(max_abs, mean_abs, relative_l2) between two tensors (float)."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    diff = a - b
    denom = float(torch.linalg.norm(b)) or 1.0
    return (float(diff.abs().max()), float(diff.abs().mean()),
            float(torch.linalg.norm(diff) / denom))


def topk_overlap(a_logits: torch.Tensor, b_logits: torch.Tensor,
                 k: int = 5) -> float:
    """Mean per-row top-k index overlap |A∩B|/k."""
    k = min(k, a_logits.shape[-1])
    ta = a_logits.topk(k, dim=-1).indices.reshape(-1, k)
    tb = b_logits.topk(k, dim=-1).indices.reshape(-1, k)
    ov = [len(set(ta[i].tolist()) & set(tb[i].tolist())) / k
          for i in range(ta.shape[0])]
    return float(sum(ov) / len(ov)) if ov else 0.0
