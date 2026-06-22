"""Stage 6.7 -- masked CausalLM input/output boundary probe.

Isolates boundary correctness from decoder correctness: the "decoder
output" is a synthetic hidden state ``h_plain`` with ``h_tilde = h_plain @
n_res``. Validates the trusted embedding input boundary (with optional input
pad), the masked-logits output boundary (vocab permutation+scaling mask,
TEE-side recovery), trusted sampling, and the one-step next-token boundary.

No full decoder stack, no generation loop, no tokenizer, no network. The
GPU never sees ``input_ids``, plaintext embeddings, or plaintext logits in
this design. No formal, cryptographic, or semantic security is claimed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pllo.ops.causal_lm_boundaries import (
    CausalLMBoundaryConfig,
    embedding_boundary_forward,
    final_norm_lm_head_masked,
    greedy_sample,
    init_causal_lm_boundary_weights,
    make_vocab_logit_mask,
    sample_from_logits,
    trusted_next_token_to_masked_embedding,
    trusted_sample_from_masked_logits,
)

_REQUIRED_STATEMENT = (
    "This stage validates trusted embedding input and masked-logits output "
    "boundaries for CausalLM inference. It does not validate full-model "
    "generation or claim semantic security."
)

_CAVEATS = [
    "Boundary probe only; no full decoder stack",
    "Synthetic embeddings and LM head unless integrated with HF model",
    "GPU sees masked logits in the probe, not plaintext logits",
    "Permutation+scaling vocab mask is weaker than dense vocab masking",
    "Full vocab LM head cost is not optimized",
    "No end-to-end generation loop",
    "No tokenizer or chat template integration",
    "Final output text semantics are not protected once returned to the user",
]


@dataclass
class CausalLMBoundaryProbeConfig:
    batch_size: int = 2
    seq_len: int = 8
    vocab_size: int = 128
    hidden_size: int = 32
    tie_word_embeddings: bool = False
    use_input_pad: bool = True
    rms_norm_eps: float = 1e-5
    dtype: str = "float64"
    device: str = "cpu"
    seed: int = 2030
    run_sampling: bool = True


def _dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _orthogonal(dim: int, dtype: torch.dtype, device: torch.device,
                g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=dtype,
                                       device=device))
    return q


def _to_boundary_config(
    cfg: CausalLMBoundaryProbeConfig,
) -> CausalLMBoundaryConfig:
    return CausalLMBoundaryConfig(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
        tie_word_embeddings=cfg.tie_word_embeddings,
        use_input_pad=cfg.use_input_pad, rms_norm_eps=cfg.rms_norm_eps,
        dtype=_dtype(cfg.dtype), device=cfg.device, seed=cfg.seed,
    )


def run_causal_lm_boundary_probe(
    config: CausalLMBoundaryProbeConfig,
) -> dict[str, Any]:
    bcfg = _to_boundary_config(config)
    dtype = bcfg.dtype
    device = torch.device(bcfg.device)
    g = torch.Generator(device=device).manual_seed(bcfg.seed)
    eps = bcfg.rms_norm_eps

    # 1. synthetic (trusted-only) input_ids
    input_ids = torch.randint(0, bcfg.vocab_size,
                              (bcfg.batch_size, bcfg.seq_len), generator=g,
                              device=device)
    # 2. boundary weights
    weights = init_causal_lm_boundary_weights(bcfg, g)
    # 3. orthogonal residual mask
    n_res = _orthogonal(bcfg.hidden_size, dtype, device, g)
    n_res_inv = n_res.transpose(-2, -1).contiguous()
    # 4. optional input pad
    pad_in = None
    if bcfg.use_input_pad:
        pad_in = torch.randn(bcfg.hidden_size, generator=g, dtype=dtype,
                             device=device)

    # 5. input embedding boundary
    emb = embedding_boundary_forward(input_ids, weights, n_res, pad_in)
    embedding_mask_err = float(
        (emb["x_tilde"] - emb["expected_x_tilde"]).abs().max().item())

    # 6. simulate decoder output (isolated from decoder correctness)
    h_plain = torch.randn(bcfg.batch_size, bcfg.seq_len, bcfg.hidden_size,
                          generator=g, dtype=dtype, device=device)
    h_tilde = h_plain @ n_res

    # 7. vocab logit mask
    vocab_mask = make_vocab_logit_mask(bcfg.vocab_size, dtype, device, g)

    # 8. final norm + masked LM head
    out = final_norm_lm_head_masked(h_tilde, h_plain, weights, n_res_inv,
                                    vocab_mask, eps)
    om = out["metrics"]
    logits_recovered_allclose = bool(torch.allclose(
        out["logits_recovered"], out["logits_plain"], atol=1e-8, rtol=1e-8))

    # 9. trusted greedy sampling from masked logits
    greedy_plain = greedy_sample(out["logits_plain"])
    trusted = trusted_sample_from_masked_logits(
        out["logits_tilde"], vocab_mask, mode="greedy")
    greedy_tilde_recovered = greedy_sample(out["logits_recovered"])
    greedy_match_rate = float(
        (greedy_plain == greedy_tilde_recovered).to(dtype).mean().item())
    trusted_greedy_match_rate = float(
        (greedy_plain == trusted["tokens"]).to(dtype).mean().item())

    # 10. optional seeded stochastic sampling determinism
    sampled_shape_ok = True
    seeded_sampling_deterministic = None
    if config.run_sampling:
        g1 = torch.Generator(device=device).manual_seed(777)
        g2 = torch.Generator(device=device).manual_seed(777)
        s1 = sample_from_logits(out["logits_recovered"], temperature=0.8,
                                top_k=20, top_p=0.95, generator=g1)
        s2 = sample_from_logits(out["logits_recovered"], temperature=0.8,
                                top_k=20, top_p=0.95, generator=g2)
        sampled_shape_ok = bool(
            tuple(s1.shape) == (bcfg.batch_size, bcfg.seq_len))
        seeded_sampling_deterministic = bool(torch.equal(s1, s2))

    # 11. one-step next-token boundary
    next_tokens = trusted["tokens"][:, -1]  # [B]
    pad_next = pad_in
    nxt = trusted_next_token_to_masked_embedding(
        next_tokens, weights, n_res, pad_next)
    next_emb_err = float(
        (nxt["x_next_tilde"] - nxt["expected_x_next_tilde"]).abs().max().item())

    metrics = {
        "embedding_mask_max_abs_error": embedding_mask_err,
        "final_norm_core_max_abs_error": om["final_norm_core_max_abs_error"],
        "masked_logits_max_abs_error": om["masked_logits_max_abs_error"],
        "recovered_logits_max_abs_error": om["recovered_logits_max_abs_error"],
        "logits_recovered_allclose": logits_recovered_allclose,
        "greedy_token_match_rate": greedy_match_rate,
        "trusted_greedy_from_masked_match_rate": trusted_greedy_match_rate,
        "sampled_tokens_shape_ok": sampled_shape_ok,
        "seeded_sampling_deterministic": seeded_sampling_deterministic,
        "next_embedding_mask_max_abs_error": next_emb_err,
    }
    all_allclose = bool(
        embedding_mask_err <= 1e-8
        and om["final_norm_core_max_abs_error"] <= 1e-8
        and om["masked_logits_max_abs_error"] <= 1e-8
        and om["recovered_logits_max_abs_error"] <= 1e-8
        and logits_recovered_allclose
        and greedy_match_rate == 1.0
        and trusted_greedy_match_rate == 1.0
        and next_emb_err <= 1e-8)

    return {
        "stage": "6.7_causal_lm_boundaries",
        "experiment": "causal_lm_boundary_probe",
        "status": "ok",
        "statement": _REQUIRED_STATEMENT,
        "config": asdict(config),
        "metrics": metrics,
        "all_allclose": all_allclose,
        "metadata": {
            "stage": "6.7_causal_lm_boundaries",
            "no_intermediate_tee": True,
            "input_ids_visible_to_gpu": False,
            "plaintext_embedding_visible_to_gpu": False,
            "released_to_gpu_at_input": "masked_embeddings_only",
            "plaintext_logits_visible_to_gpu": False,
            "masked_logits_visible_to_gpu": True,
            "logits_recovered_in_tee": True,
            "sampling_boundary": "trusted_side",
            "output_boundary":
                "masked_logits_to_trusted_recovery_and_sampling",
            "next_token_ids_visible_to_gpu": False,
            "vocab_mask_family": "permutation_positive_diagonal_scaling",
            "dense_vocab_mask_used": False,
            "greedy_supported": True,
            "temperature_supported": True,
            "top_k_supported": True,
            "top_p_supported": True,
            "security_status":
                "operator_compatible_leakage_reduction_not_semantic_security",
            "semantic_security_claimed": False,
            "formal_security_claimed": False,
            "cryptographic_security_claimed": False,
            "caveats": _CAVEATS,
        },
        "limitations": _CAVEATS,
    }


__all__ = [
    "CausalLMBoundaryProbeConfig",
    "run_causal_lm_boundary_probe",
]
