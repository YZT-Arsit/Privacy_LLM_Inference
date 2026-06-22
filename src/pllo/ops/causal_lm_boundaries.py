"""Stage 6.7 -- masked CausalLM input/output boundaries (CPU, correctness).

Validates the trusted boundaries that wrap a masked decoder stack. The
intended deployment boundary (no intermediate TEE inside the decoder):

  TEE: input_ids -> embedding lookup -> X_plain
       X_tilde = (X_plain - T_in) @ N_res        (release ONLY X_tilde)
  GPU: run masked decoder; produce masked hidden / masked logits;
       never sees input_ids, plaintext embeddings, or plaintext logits.
  TEE: recover plaintext logits, sample / stop-token / penalties,
       keep the next token, look up + mask its embedding, release only the
       masked next embedding for the following decode step.

Output boundary: with orthogonal ``N_res``, ``rmsnorm_core`` commutes with
the mask (``rmsnorm_core(H @ N_res) = rmsnorm_core(H) @ N_res``). A
lightweight vocab-logit mask ``M_vocab = P_vocab @ D_vocab`` (permutation +
positive diagonal scaling -- NOT a dense ``vocab x vocab`` matrix) is folded
together with the final-norm affine and ``N_res^{-1}`` into the LM head, so
the GPU computes ``L_tilde = L @ M_vocab`` and the TEE recovers
``L = L_tilde @ M_vocab^{-1}``.

Permutation+scaling hides direct token-index/logit alignment but is weaker
than dense vocab masking and gives no semantic security. No full decoder
stack, no generation loop, no tokenizer, no network. No formal,
cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pllo.ops.nonlinear_islands import rmsnorm_core

__all__ = [
    "CausalLMBoundaryConfig",
    "CausalLMBoundaryWeights",
    "VocabLogitMask",
    "apply_temperature",
    "apply_vocab_logit_mask",
    "embedding_boundary_forward",
    "final_norm_lm_head_masked",
    "final_norm_lm_head_plain",
    "fold_final_norm_lm_head_with_vocab_mask",
    "greedy_sample",
    "init_causal_lm_boundary_weights",
    "make_vocab_logit_mask",
    "mask_embedding_output",
    "recover_vocab_logits",
    "run_output_boundary_on_hidden",
    "sample_from_logits",
    "top_k_filter",
    "top_p_filter",
    "trusted_embedding_lookup",
    "trusted_next_token_to_masked_embedding",
    "trusted_sample_from_masked_logits",
]


# ---------------------------------------------------------------------------
# Config + weights
# ---------------------------------------------------------------------------


@dataclass
class CausalLMBoundaryConfig:
    batch_size: int = 2
    seq_len: int = 8
    vocab_size: int = 128
    hidden_size: int = 32
    tie_word_embeddings: bool = False
    use_input_pad: bool = True
    rms_norm_eps: float = 1e-5
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 2030


@dataclass
class CausalLMBoundaryWeights:
    embed_tokens_weight: torch.Tensor   # [vocab, hidden]
    final_norm_weight: torch.Tensor     # [hidden]
    lm_head_weight: torch.Tensor        # [hidden, vocab]


def init_causal_lm_boundary_weights(
    config: CausalLMBoundaryConfig, generator: torch.Generator,
) -> CausalLMBoundaryWeights:
    """Deterministic, modestly-scaled boundary weights."""
    dtype = config.dtype
    device = torch.device(config.device)
    h, v = config.hidden_size, config.vocab_size
    embed = torch.randn(v, h, generator=generator, dtype=dtype,
                        device=device) * (1.0 / (h ** 0.5))
    final_norm = 1.0 + 0.1 * torch.randn(h, generator=generator, dtype=dtype,
                                         device=device)
    if config.tie_word_embeddings:
        lm_head = embed.t().contiguous()                       # [hidden, vocab]
    else:
        lm_head = torch.randn(h, v, generator=generator, dtype=dtype,
                              device=device) * (1.0 / (h ** 0.5))
    return CausalLMBoundaryWeights(
        embed_tokens_weight=embed,
        final_norm_weight=final_norm,
        lm_head_weight=lm_head,
    )


# ---------------------------------------------------------------------------
# Input boundary (trusted embedding lookup -> pad -> mask -> release)
# ---------------------------------------------------------------------------


def trusted_embedding_lookup(
    input_ids: torch.Tensor, embed_tokens_weight: torch.Tensor,
) -> torch.Tensor:
    """Trusted-side embedding lookup. ``input_ids`` are trusted-only.

    ``input_ids`` ``[B, T]`` -> ``x_plain`` ``[B, T, hidden]``.
    """
    if input_ids.dtype not in (torch.int64, torch.int32, torch.long):
        raise ValueError("input_ids must be an integer/long tensor")
    return embed_tokens_weight[input_ids]


def mask_embedding_output(
    x_plain: torch.Tensor, n_res: torch.Tensor,
    pad_in: torch.Tensor | None = None,
) -> torch.Tensor:
    """Release the masked embedding ``(x_plain - pad_in) @ n_res`` (pad
    optional). Only this masked tensor is ever sent to the GPU."""
    shifted = x_plain if pad_in is None else x_plain - pad_in
    return shifted @ n_res


def embedding_boundary_forward(
    input_ids: torch.Tensor, weights: CausalLMBoundaryWeights,
    n_res: torch.Tensor, pad_in: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Full input boundary: lookup (trusted) -> pad -> mask -> release."""
    x_plain = trusted_embedding_lookup(input_ids, weights.embed_tokens_weight)
    x_tilde = mask_embedding_output(x_plain, n_res, pad_in)
    shifted = x_plain if pad_in is None else x_plain - pad_in
    return {
        "x_plain": x_plain,
        "x_tilde": x_tilde,
        "expected_x_tilde": shifted @ n_res,
        "pad_in": pad_in,
        "metadata": {
            "input_boundary":
                "trusted_embedding_lookup_then_masked_embedding_release",
            "input_ids_visible_to_gpu": False,
            "plaintext_embedding_visible_to_gpu": False,
            "released_to_gpu": "masked_embeddings_only",
            "used_input_pad": pad_in is not None,
        },
    }


# ---------------------------------------------------------------------------
# Vocab-logit mask (permutation + positive diagonal scaling)
# ---------------------------------------------------------------------------


@dataclass
class VocabLogitMask:
    permutation: torch.Tensor          # [vocab]
    inverse_permutation: torch.Tensor  # [vocab]
    scale: torch.Tensor                # [vocab], strictly positive
    inverse_scale: torch.Tensor        # [vocab]


def make_vocab_logit_mask(
    vocab_size: int, dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    scale_low: float = 0.5, scale_high: float = 2.0,
) -> VocabLogitMask:
    """Build ``M_vocab = P_vocab @ D_vocab`` as permutation + positive scale."""
    device = torch.device(device)
    perm = torch.randperm(vocab_size, generator=generator, device=device)
    inv_perm = torch.argsort(perm)
    u = torch.rand(vocab_size, generator=generator, dtype=dtype, device=device)
    scale = scale_low + u * (scale_high - scale_low)  # strictly positive
    return VocabLogitMask(
        permutation=perm, inverse_permutation=inv_perm,
        scale=scale, inverse_scale=1.0 / scale,
    )


def apply_vocab_logit_mask(
    logits: torch.Tensor, vocab_mask: VocabLogitMask,
) -> torch.Tensor:
    """``L_tilde = L[..., permutation] * scale`` (== ``L @ M_vocab``)."""
    return logits.index_select(-1, vocab_mask.permutation) * vocab_mask.scale


def recover_vocab_logits(
    masked_logits: torch.Tensor, vocab_mask: VocabLogitMask,
) -> torch.Tensor:
    """Invert :func:`apply_vocab_logit_mask` (TEE-side)."""
    tmp = masked_logits * vocab_mask.inverse_scale
    return tmp.index_select(-1, vocab_mask.inverse_permutation)


# ---------------------------------------------------------------------------
# Output boundary (final RMSNorm + masked LM head)
# ---------------------------------------------------------------------------


def final_norm_lm_head_plain(
    h_plain: torch.Tensor, final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor, eps: float,
) -> dict[str, torch.Tensor]:
    """Plain final RMSNorm + LM head."""
    core = rmsnorm_core(h_plain, eps)
    normed = core * final_norm_weight
    logits = normed @ lm_head_weight
    return {"core": core, "normed": normed, "logits": logits}


def fold_final_norm_lm_head_with_vocab_mask(
    final_norm_weight: torch.Tensor, lm_head_weight: torch.Tensor,
    n_res_inv: torch.Tensor, vocab_mask: VocabLogitMask,
) -> torch.Tensor:
    """``W_lm_tilde = n_res^{-1} @ diag(gamma) @ W_lm @ M_vocab`` without
    materialising the dense vocab mask."""
    w_lm_fold = final_norm_weight.unsqueeze(1) * lm_head_weight  # diag @ W_lm
    w_lm_unmasked = n_res_inv @ w_lm_fold                        # [hidden,vocab]
    return w_lm_unmasked.index_select(1, vocab_mask.permutation) \
        * vocab_mask.scale.unsqueeze(0)


def final_norm_lm_head_masked(
    h_tilde: torch.Tensor, h_plain: torch.Tensor,
    weights: CausalLMBoundaryWeights, n_res_inv: torch.Tensor,
    vocab_mask: VocabLogitMask, eps: float,
) -> dict[str, Any]:
    """Masked final norm + folded LM head; GPU produces masked logits only."""
    n_res = n_res_inv.transpose(-2, -1)  # orthogonal: (n_res^{-1})^T = n_res
    plain = final_norm_lm_head_plain(
        h_plain, weights.final_norm_weight, weights.lm_head_weight, eps)
    core_tilde = rmsnorm_core(h_tilde, eps)
    w_lm_tilde = fold_final_norm_lm_head_with_vocab_mask(
        weights.final_norm_weight, weights.lm_head_weight, n_res_inv,
        vocab_mask)
    logits_tilde = core_tilde @ w_lm_tilde
    logits_recovered = recover_vocab_logits(logits_tilde, vocab_mask)
    expected_logits_tilde = apply_vocab_logit_mask(plain["logits"], vocab_mask)

    def mx(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max().item())

    return {
        "core_tilde": core_tilde,
        "expected_core_tilde": plain["core"] @ n_res,
        "logits_plain": plain["logits"],
        "logits_tilde": logits_tilde,
        "expected_logits_tilde": expected_logits_tilde,
        "logits_recovered": logits_recovered,
        "folded_lm_head_weight": w_lm_tilde,
        "metrics": {
            "final_norm_core_max_abs_error": mx(core_tilde, plain["core"] @ n_res),
            "masked_logits_max_abs_error": mx(logits_tilde, expected_logits_tilde),
            "recovered_logits_max_abs_error": mx(logits_recovered, plain["logits"]),
        },
        "metadata": {
            "plaintext_logits_visible_to_gpu": False,
            "masked_logits_visible_to_gpu": True,
            "logits_recovered_in_tee": True,
            "sampling_runs_in_tee": True,
        },
    }


def run_output_boundary_on_hidden(
    h_plain: torch.Tensor, h_tilde: torch.Tensor,
    weights: CausalLMBoundaryWeights, n_res: torch.Tensor,
    n_res_inv: torch.Tensor, vocab_mask: VocabLogitMask, eps: float,
) -> dict[str, Any]:
    """Integration helper: run the output boundary on decoder hidden states
    (e.g. Stage 6.5/6.6 ``h_plain`` with ``h_tilde = h_plain @ n_res``)."""
    masked = final_norm_lm_head_masked(
        h_tilde, h_plain, weights, n_res_inv, vocab_mask, eps)
    sampled = trusted_sample_from_masked_logits(
        masked["logits_tilde"], vocab_mask, mode="greedy")
    return {
        "logits_plain": masked["logits_plain"],
        "logits_tilde": masked["logits_tilde"],
        "logits_recovered": masked["logits_recovered"],
        "greedy_tokens": sampled["tokens"],
        "metrics": masked["metrics"],
    }


# ---------------------------------------------------------------------------
# Trusted-side sampling boundary
# ---------------------------------------------------------------------------


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over the vocab (last) dimension -> ``[...]`` token ids."""
    return logits.argmax(dim=-1)


def apply_temperature(
    logits: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """Scale logits by ``1/temperature``. ``temperature > 0`` required."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep the top-``k`` logits per row; set the rest to ``-inf``.

    ``k is None`` returns logits unchanged. ``k <= 0`` or ``k > vocab_size``
    raises ``ValueError``.
    """
    if k is None:
        return logits.clone()
    vocab = logits.shape[-1]
    if k <= 0 or k > vocab:
        raise ValueError(f"top_k must be in [1, {vocab}], got {k}")
    if k == vocab:
        return logits.clone()
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    """Nucleus filter: keep the smallest descending prefix with cumulative
    probability ``>= p`` (always >= 1 token). ``p is None`` -> unchanged.
    ``p <= 0`` or ``p > 1`` raises ``ValueError``."""
    if p is None:
        return logits.clone()
    if p <= 0 or p > 1:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cdf = probs.cumsum(dim=-1)
    remove_sorted = (cdf - probs) >= p   # cumulative-before-token already >= p
    remove_sorted[..., 0] = False        # always keep the top token
    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def sample_from_logits(
    logits: torch.Tensor, temperature: float = 1.0,
    top_k: int | None = None, top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Trusted-side sampling: temperature -> top-k -> top-p -> multinomial.
    Returns ``[...]`` token ids (vocab dim removed)."""
    work = logits if temperature == 1.0 else apply_temperature(
        logits, temperature)
    work = top_k_filter(work, top_k)
    work = top_p_filter(work, top_p)
    probs = torch.softmax(work, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1, generator=generator)
    return sampled.reshape(*logits.shape[:-1])


def trusted_sample_from_masked_logits(
    masked_logits: torch.Tensor, vocab_mask: VocabLogitMask,
    mode: str = "greedy", temperature: float = 1.0,
    top_k: int | None = None, top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Recover logits in the TEE, then sample. ``mode`` in {greedy, sample}."""
    logits = recover_vocab_logits(masked_logits, vocab_mask)
    if mode == "greedy":
        tokens = greedy_sample(logits)
    elif mode == "sample":
        tokens = sample_from_logits(logits, temperature, top_k, top_p,
                                    generator)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected greedy|sample")
    return {
        "tokens": tokens,
        "recovered_logits": logits,
        "metadata": {
            "logits_recovered_in_tee": True,
            "sampling_boundary": "trusted_side",
        },
    }


# ---------------------------------------------------------------------------
# One-step decode boundary (next token -> masked next embedding)
# ---------------------------------------------------------------------------


def trusted_next_token_to_masked_embedding(
    next_token_ids: torch.Tensor, weights: CausalLMBoundaryWeights,
    n_res: torch.Tensor, pad_next: torch.Tensor | None = None,
) -> dict[str, Any]:
    """TEE: next token -> embedding lookup -> mask -> release masked emb only.

    Validates the iterative-generation boundary without running a decoder.
    """
    x_next_plain = trusted_embedding_lookup(next_token_ids,
                                            weights.embed_tokens_weight)
    shifted = x_next_plain if pad_next is None else x_next_plain - pad_next
    x_next_tilde = shifted @ n_res
    return {
        "x_next_plain": x_next_plain,
        "x_next_tilde": x_next_tilde,
        "expected_x_next_tilde": shifted @ n_res,
        "metadata": {
            "next_token_ids_visible_to_gpu": False,
            "next_embedding_visible_to_gpu": False,
            "released_to_gpu": "masked_next_embedding_only",
        },
    }
