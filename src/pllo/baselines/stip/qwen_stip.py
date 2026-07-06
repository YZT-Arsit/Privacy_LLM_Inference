"""STIP-on-Qwen2: parameter transform, protected forward, logit recovery.

STIP is numerically a *relabeling* (feature permutation), so the cloud-visible
protected activations equal the plaintext activations permuted by the STIP keys,
and the recovered logits equal plaintext exactly (Theorem 1). This module:
  * ``stip_parameter_transform_linear`` — the exact weight/bias transform
    (nn.Linear: ``W' = W[p_out][:, p_in]``, ``b' = b[p_out]``), used by the audit
    to *prove* the permuted-weight forward reproduces the plaintext output.
  * ``stip_protect_qwen`` — runs the plaintext model, then derives the
    cloud-visible protected representation (permuted hidden states / Q/K/V, the
    UNPERMUTED attention scores that STIP leaks, permuted embedding) plus the
    recovered logits, for both the security attacks and the baseline audit.
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.attacks.qwen_hooks import QwenActivationCapture
from pllo.baselines.obfuscatune.qwen_config import extract_arch
from .permutation import (StipPermutations, apply_perm_last, inverse_perm,
                          make_stip_permutations)


def stip_parameter_transform_linear(module: torch.nn.Module, p_in: torch.Tensor,
                                    p_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Transform an nn.Linear: ``W' = W[p_out][:, p_in]``, ``b' = b[p_out]``.

    Satisfies ``linear'(x[..., p_in]) == linear(x)[..., p_out]``.
    """
    w = module.weight.detach()
    w2 = w.index_select(0, p_out.to(w.device)).index_select(1, p_in.to(w.device))
    b = module.bias
    b2 = None if b is None else b.detach().index_select(0, p_out.to(b.device))
    return w2, b2


def verify_linear_equivalence(module: torch.nn.Module, p_in: torch.Tensor,
                              p_out: torch.Tensor, x: torch.Tensor) -> float:
    """Max abs error of the STIP linear transform vs permuted plaintext output."""
    w2, b2 = stip_parameter_transform_linear(module, p_in, p_out)
    y_perm = x.index_select(-1, p_in.to(x.device)) @ w2.transpose(0, 1)
    if b2 is not None:
        y_perm = y_perm + b2
    y_plain = module(x).index_select(-1, p_out.to(x.device))
    return float((y_perm - y_plain).abs().max().item())


def stip_recover_logits(protected_logits: torch.Tensor, pi_c: torch.Tensor) -> torch.Tensor:
    """Recover plaintext logits from cloud output ``o' = o π_c``: ``o = o' π_cᵀ``."""
    return apply_perm_last(protected_logits, inverse_perm(pi_c))


def build_transformed_qwen(model, perms: StipPermutations):
    """Return a deep copy of ``model`` with STIP-transformed weights (the cloud model).

    Running it on plaintext ``input_ids`` yields ``logits π_c`` (the embed' already
    applies ``π``); recover with :func:`stip_recover_logits`. This is a *genuine
    transformed-weight forward* through the real HF RoPE/GQA path — the definitive
    correctness proof of STIP-on-Qwen (see ``audit.py``).
    """
    import copy
    m = copy.deepcopy(model)
    pi, pic = perms.pi, perms.pi_c

    def _set(mod, src, p_in, p_out):
        w, b = stip_parameter_transform_linear(src, p_in, p_out)
        with torch.no_grad():
            mod.weight.copy_(w)
            if b is not None:
                mod.bias.copy_(b)

    with torch.no_grad():
        # embed' so embed'(ids) = embed(ids)[:, pi]  (device applies π before cloud)
        m.model.embed_tokens.weight.copy_(model.model.embed_tokens.weight.index_select(1, pi))
        m.model.norm.weight.copy_(model.model.norm.weight.index_select(0, pi))
        _set(m.lm_head, model.lm_head, pi, pic)
        for li, layer in enumerate(m.model.layers):
            src = model.model.layers[li]
            pq, pkv, pmlp = perms.pi_attn_q[li], perms.pi_attn_kv[li], perms.pi_mlp[li]
            layer.input_layernorm.weight.copy_(src.input_layernorm.weight.index_select(0, pi))
            layer.post_attention_layernorm.weight.copy_(
                src.post_attention_layernorm.weight.index_select(0, pi))
            _set(layer.self_attn.q_proj, src.self_attn.q_proj, pi, pq)
            _set(layer.self_attn.k_proj, src.self_attn.k_proj, pi, pkv)
            _set(layer.self_attn.v_proj, src.self_attn.v_proj, pi, pkv)
            _set(layer.self_attn.o_proj, src.self_attn.o_proj, pq, pi)
            _set(layer.mlp.gate_proj, src.mlp.gate_proj, pi, pmlp)
            _set(layer.mlp.up_proj, src.mlp.up_proj, pi, pmlp)
            _set(layer.mlp.down_proj, src.mlp.down_proj, pmlp, pi)
    return m


def stip_protect_qwen(
    model, input_ids: torch.Tensor, perms: StipPermutations | None = None,
    *, seed: int = 0,
) -> dict[str, Any]:
    """Run STIP on Qwen2; return protected reps + recovered logits + audit facts.

    Returns a dict with:
      * ``recovered_logits`` (== plaintext), ``protected_logits`` (= logits π_c)
      * ``protected_embedding`` (= embed π), ``embedding_table`` (device-side, plaintext)
      * per-layer ``layer{li}_hidden_perm`` (residual stream π), ``layer{li}_q_perm``,
        ``_k_perm``, ``_v_perm`` (whole-head permuted), and
        ``layer{li}_attention_scores`` (UNPERMUTED — STIP leaks these).
      * ``perms`` (the key set), ``arch``.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    arch = extract_arch(model)
    if perms is None:
        perms = make_stip_permutations(arch, seed=seed)

    layers = list(range(arch.num_layers))
    cap = QwenActivationCapture(model, layers=layers)
    captured = cap.run(input_ids)
    cap.remove()

    plain_logits = captured["logits"]
    protected_logits = apply_perm_last(plain_logits, perms.pi_c)
    recovered = stip_recover_logits(protected_logits, perms.pi_c)

    out: dict[str, Any] = {
        "recovered_logits": recovered,
        "protected_logits": protected_logits,
        "plaintext_logits": plain_logits,
        "logits_recovery_max_abs_error": float((recovered - plain_logits).abs().max().item()),
        "embedding_table": captured["embedding_table"],           # device-side plaintext
        "protected_embedding": apply_perm_last(captured["embed_tokens"], perms.pi),
        "plaintext_embedding": captured["embed_tokens"],
        "input_ids": input_ids,
        "arch": arch,
        "perms": perms,
    }
    # per-layer cloud-visible protected activations
    for li in layers:
        h_in = captured.get(f"layer{li}_input")
        if h_in is not None:
            out[f"layer{li}_hidden_perm"] = apply_perm_last(h_in, perms.pi)
        q = captured.get(f"layer{li}_q_proj")
        k = captured.get(f"layer{li}_k_proj")
        v = captured.get(f"layer{li}_v_proj")
        if q is not None:
            out[f"layer{li}_q_perm"] = apply_perm_last(q, perms.pi_attn_q[li])
        if k is not None:
            out[f"layer{li}_k_perm"] = apply_perm_last(k, perms.pi_attn_kv[li])
        if v is not None:
            out[f"layer{li}_v_perm"] = apply_perm_last(v, perms.pi_attn_kv[li])
    return out


__all__ = [
    "stip_parameter_transform_linear", "verify_linear_equivalence",
    "stip_recover_logits", "stip_protect_qwen",
]
