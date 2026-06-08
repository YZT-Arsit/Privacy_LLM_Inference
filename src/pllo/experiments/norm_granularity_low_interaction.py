"""Stage 7.6h -- RMSNorm-compatible mask granularity (sequence / chunk / token).

The Stage 7.6g rope-safe low-interaction path masks every token row of
the residual stream by a *single* per-layer orthogonal ``Q_l``. This
keeps the H_hat boundary numerically convenient but exactly preserves
the Gram matrix ``H H^T`` (the token-pair similarity structure leaks on
the accelerator boundary).

Stage 7.6h adds a granularity knob:

* ``sequence`` -- baseline (Stage 7.6g behaviour). One Q per layer per
  call, full Gram preserved.

* ``chunk(k)`` -- partition the token rows into chunks of size ``k``;
  each chunk gets its own orthogonal ``Q_chunk``. Within-chunk Gram is
  preserved, cross-chunk Gram is disrupted.

* ``token`` -- every token row gets its own orthogonal ``Q_i``. Row
  norms are still mathematically preserved (RMSNorm correctness
  requires it), but the *off-diagonal* Gram leakage is disrupted: for
  ``i != j``, ``(H_hat H_hat^T)_{ij} = h_i Q_i Q_j^T h_j^T`` no longer
  equals ``h_i h_j^T``.

This is a security--efficiency knob, not formal cryptographic security:
token-wise masking removes the sequence-shared Gram structure on the
accelerator boundary but the per-row L2 norm is still observable
because RMSNorm correctness requires it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pllo.models.tiny_modern_decoder import (
    TinyModernDecoderConfig,
    TinyModernDecoderForCausalLM,
)
from pllo.wrappers.low_interaction_modern_decoder_generation_wrapper import (
    LowInteractionDiagnostics,
    LowInteractionTinyModernDecoderWrapper,
)


# ---------------------------------------------------------------------------
# Config + tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormGranularityConfig:
    seed: int = 2026
    weights_seed: int = 2026
    prompt_seed: int = 2027
    mask_seed_a: int = 2028
    mask_seed_b: int = 2029
    batch_size: int = 2
    prompt_len: int = 6
    max_new_tokens: int = 3
    num_layers: int = 1
    chunk_size: int = 2  # only used by "chunk" granularity row
    use_pad: bool = True
    fresh_pad: bool = True
    fresh_mask: bool = True


LOGITS_TOLERANCE = 1e-9
KV_CACHE_TOLERANCE = 1e-11
INVARIANT_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------
# Norm + Gram leakage audit
# ---------------------------------------------------------------------------


def _row_norms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(dim=-1).clamp_min(0).sqrt()


def _gram(x: torch.Tensor) -> torch.Tensor:
    """Token-pair Gram matrix ``x @ x^T`` per batch."""
    return x @ x.transpose(-2, -1)


def _h_hat_sequence(plain_h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Apply a single orthogonal ``Q`` uniformly across all rows."""
    return plain_h @ q


def _h_hat_chunk(
    plain_h: torch.Tensor, qs: List[torch.Tensor], chunk_size: int
) -> torch.Tensor:
    """Apply per-chunk orthogonal Q to each token row."""
    b, s, hidden = plain_h.shape
    out = torch.zeros_like(plain_h)
    for i in range(s):
        out[:, i, :] = plain_h[:, i, :] @ qs[i // chunk_size]
    return out


def _h_hat_token(plain_h: torch.Tensor, qs: List[torch.Tensor]) -> torch.Tensor:
    """Apply per-token orthogonal Q[i] to each token row."""
    return _h_hat_chunk(plain_h, qs, chunk_size=1)


def _sample_orthogonals(
    n: int, dim: int, generator: torch.Generator, dtype: torch.dtype, device: str
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for _ in range(n):
        raw = torch.randn(dim, dim, dtype=dtype, device=device, generator=generator)
        q, r = torch.linalg.qr(raw)
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        out.append(q * signs.unsqueeze(0))
    return out


def _chunked_indicator(s: int, chunk_size: int) -> torch.Tensor:
    """[S, S] indicator: 1 if rows i, j are in the same chunk, else 0."""
    out = torch.zeros(s, s)
    for i in range(s):
        for j in range(s):
            if i // chunk_size == j // chunk_size:
                out[i, j] = 1.0
    return out


def _norm_and_gram_audit(
    model: TinyModernDecoderForCausalLM,
    cfg: NormGranularityConfig,
    input_ids_main: torch.Tensor,
    input_ids_alt: torch.Tensor,
) -> Dict[str, Any]:
    """Compute row-norm, full-Gram, off-diagonal-Gram, within-/cross-chunk
    Gram errors for sequence / chunk / token granularities on the layer-
    entry boundary ``H_hat = H Q``.
    """
    cfg_md = model.cfg
    plain_h = model.embed_tokens(input_ids_main)
    plain_h_alt = model.embed_tokens(input_ids_alt)

    b, s, hidden = plain_h.shape
    chunk_size = cfg.chunk_size

    # Sample Q matrices for two runs (A and B) per granularity so the
    # same-prompt-fresh-Q comparison is meaningful.
    g_a = torch.Generator(device="cpu").manual_seed(cfg.mask_seed_a)
    g_b = torch.Generator(device="cpu").manual_seed(cfg.mask_seed_b)

    # Sequence mode: one Q per run.
    q_seq_a = _sample_orthogonals(1, hidden, g_a, cfg_md.dtype, cfg_md.device)[0]
    q_seq_b = _sample_orthogonals(1, hidden, g_b, cfg_md.dtype, cfg_md.device)[0]
    # Chunk mode: ceil(s/k) Q's per run.
    num_chunks = (s + chunk_size - 1) // chunk_size
    qs_chunk_a = _sample_orthogonals(
        num_chunks, hidden, g_a, cfg_md.dtype, cfg_md.device
    )
    qs_chunk_b = _sample_orthogonals(
        num_chunks, hidden, g_b, cfg_md.dtype, cfg_md.device
    )
    # Token mode: s Q's per run.
    qs_token_a = _sample_orthogonals(
        s, hidden, g_a, cfg_md.dtype, cfg_md.device
    )
    qs_token_b = _sample_orthogonals(
        s, hidden, g_b, cfg_md.dtype, cfg_md.device
    )

    h_hat_seq_a = _h_hat_sequence(plain_h, q_seq_a)
    h_hat_seq_b = _h_hat_sequence(plain_h, q_seq_b)
    h_hat_chunk_a = _h_hat_chunk(plain_h, qs_chunk_a, chunk_size)
    h_hat_chunk_b = _h_hat_chunk(plain_h, qs_chunk_b, chunk_size)
    h_hat_token_a = _h_hat_token(plain_h, qs_token_a)
    h_hat_token_b = _h_hat_token(plain_h, qs_token_b)

    # Same-prompt different prompt distance is computed in sequence mode
    # for cross-prompt Gram distance baseline (different inputs).
    h_hat_seq_alt = _h_hat_sequence(plain_h_alt, q_seq_a)

    gram_plain = _gram(plain_h)
    eye_mask = torch.eye(s, dtype=cfg_md.dtype, device=cfg_md.device).unsqueeze(0)
    offdiag_mask = 1.0 - eye_mask

    chunk_indicator = _chunked_indicator(s, chunk_size).to(
        dtype=cfg_md.dtype, device=cfg_md.device
    ).unsqueeze(0)
    within_chunk_mask = chunk_indicator
    cross_chunk_mask = 1.0 - chunk_indicator

    def _max_abs(t: torch.Tensor) -> float:
        return float(t.abs().max().item())

    def _max_abs_masked(diff: torch.Tensor, mask: torch.Tensor) -> float:
        return float((diff * mask).abs().max().item())

    audit: Dict[str, Any] = {
        "sequence": {
            "row_norm_error": _max_abs(_row_norms(h_hat_seq_a) - _row_norms(plain_h)),
            "full_gram_error": _max_abs(_gram(h_hat_seq_a) - gram_plain),
            "off_diagonal_gram_error": _max_abs_masked(
                _gram(h_hat_seq_a) - gram_plain, offdiag_mask
            ),
            "within_chunk_gram_error": _max_abs_masked(
                _gram(h_hat_seq_a) - gram_plain, within_chunk_mask
            ),
            "cross_chunk_gram_error": _max_abs_masked(
                _gram(h_hat_seq_a) - gram_plain, cross_chunk_mask
            ),
            "same_prompt_fresh_Q_gram_distance": _max_abs(
                _gram(h_hat_seq_a) - _gram(h_hat_seq_b)
            ),
            "same_prompt_fresh_Q_offdiag_distance": _max_abs_masked(
                _gram(h_hat_seq_a) - _gram(h_hat_seq_b), offdiag_mask
            ),
        },
        "chunk": {
            "row_norm_error": _max_abs(_row_norms(h_hat_chunk_a) - _row_norms(plain_h)),
            "full_gram_error": _max_abs(_gram(h_hat_chunk_a) - gram_plain),
            "off_diagonal_gram_error": _max_abs_masked(
                _gram(h_hat_chunk_a) - gram_plain, offdiag_mask
            ),
            "within_chunk_gram_error": _max_abs_masked(
                _gram(h_hat_chunk_a) - gram_plain, within_chunk_mask
            ),
            "cross_chunk_gram_error": _max_abs_masked(
                _gram(h_hat_chunk_a) - gram_plain, cross_chunk_mask
            ),
            "same_prompt_fresh_Q_gram_distance": _max_abs(
                _gram(h_hat_chunk_a) - _gram(h_hat_chunk_b)
            ),
            "same_prompt_fresh_Q_offdiag_distance": _max_abs_masked(
                _gram(h_hat_chunk_a) - _gram(h_hat_chunk_b), offdiag_mask
            ),
        },
        "token": {
            "row_norm_error": _max_abs(_row_norms(h_hat_token_a) - _row_norms(plain_h)),
            "full_gram_error": _max_abs(_gram(h_hat_token_a) - gram_plain),
            "off_diagonal_gram_error": _max_abs_masked(
                _gram(h_hat_token_a) - gram_plain, offdiag_mask
            ),
            "within_chunk_gram_error": _max_abs_masked(
                _gram(h_hat_token_a) - gram_plain, within_chunk_mask
            ),
            "cross_chunk_gram_error": _max_abs_masked(
                _gram(h_hat_token_a) - gram_plain, cross_chunk_mask
            ),
            "same_prompt_fresh_Q_gram_distance": _max_abs(
                _gram(h_hat_token_a) - _gram(h_hat_token_b)
            ),
            "same_prompt_fresh_Q_offdiag_distance": _max_abs_masked(
                _gram(h_hat_token_a) - _gram(h_hat_token_b), offdiag_mask
            ),
        },
        "different_prompt_gram_distance": _max_abs(
            _gram(plain_h) - _gram(plain_h_alt)
        ),
        "chunk_size": chunk_size,
        "explanation": (
            "Sequence mode: full Gram preserved exactly (the leakage "
            "Stage 7.6g already reports). Chunk mode: within-chunk Gram "
            "preserved, cross-chunk Gram disrupted by independent "
            "Q_chunk. Token mode: only row L2 norms preserved (required "
            "by exact RMSNorm correctness); the full Gram off-diagonal "
            "is disrupted by independent Q_i."
        ),
    }
    return audit


# ---------------------------------------------------------------------------
# Per-mode greedy + diagnostics
# ---------------------------------------------------------------------------


def _diag_to_summary(diag: LowInteractionDiagnostics) -> Dict[str, Any]:
    return {
        "norm_mask_granularity": diag.norm_mask_granularity,
        "norm_chunk_size": diag.norm_chunk_size,
        "norm_q_is_per_row": diag.norm_q_is_per_row,
        "main_layer_invariant": diag.main_layer_invariant,
        "rmsnorm_mode": diag.rmsnorm_mode,
        "rope_mask_mode": diag.rope_mask_mode,
        "rope_transient_plain_qk_visible": diag.rope_transient_plain_qk_visible,
        "rope_transient_plain_v_visible": diag.rope_transient_plain_v_visible,
        "qkv_projection_outputs_masked_directly": diag.qkv_projection_outputs_masked_directly,
        "trusted_fallback_used_in_main_path": diag.trusted_fallback_used_in_main_path,
        "intermediate_tee_reentry": diag.intermediate_tee_reentry,
        "online_boundary_round_trips_per_decode_step": diag.online_boundary_round_trips_per_decode_step,
        "use_pad": diag.use_pad,
        "fresh_pad_used_at_linear_boundaries": diag.fresh_pad_used_at_linear_boundaries,
        "pad_enters_rmsnorm_core": diag.pad_enters_rmsnorm_core,
        "pad_enters_swiglu_core": diag.pad_enters_swiglu_core,
        "pad_enters_rope_core": diag.pad_enters_rope_core,
        "pad_enters_softmax": diag.pad_enters_softmax,
        "h_hat_layer_entry_invariant_max_abs_error": diag.h_hat_layer_entry_invariant_max_abs_error,
        "rmsnorm_core_orthogonal_commutation_max_abs_error": diag.rmsnorm_core_orthogonal_commutation_max_abs_error,
        "transition_trick_max_abs_error": diag.transition_trick_max_abs_error,
        "swiglu_paired_permutation_max_abs_error": diag.swiglu_paired_permutation_max_abs_error,
        "o_proj_recovery_max_abs_error": diag.o_proj_recovery_max_abs_error,
        "down_proj_recovery_max_abs_error": diag.down_proj_recovery_max_abs_error,
        "qk_constraint_max_error": diag.qk_constraint_max_error,
        "rope_commutation_max_abs_error": diag.rope_commutation_max_abs_error,
        "qk_score_invariant_max_abs_error": diag.qk_score_invariant_max_abs_error,
        "kv_cache_invariant_max_abs_error": diag.kv_cache_invariant_max_abs_error,
        "prefill_logits_max_abs_error": diag.prefill_logits_max_abs_error,
        "decode_step_logits_max_abs_error_max": diag.decode_step_logits_max_abs_error_max,
        "lm_head_recovery_max_abs_error": diag.lm_head_recovery_max_abs_error,
        "masked_boundary_fingerprints": dict(diag.masked_boundary_fingerprints),
    }


def _run_one_mode(
    model: TinyModernDecoderForCausalLM,
    cfg: NormGranularityConfig,
    granularity: str,
    chunk_size: int,
    input_ids: torch.Tensor,
    mask_seed: int,
    fingerprint_prefix: str,
) -> Tuple[torch.Tensor, LowInteractionDiagnostics]:
    wrapper = LowInteractionTinyModernDecoderWrapper(
        model,
        use_pad=cfg.use_pad,
        fresh_pad=cfg.fresh_pad,
        fresh_mask=cfg.fresh_mask,
        rope_mask_mode="pre_rope_block_diagonal_rotation",
        norm_mask_granularity=granularity,
        norm_chunk_size=chunk_size,
    )
    g = torch.Generator(device="cpu").manual_seed(mask_seed)
    diag = LowInteractionDiagnostics()
    tokens, diag = wrapper.low_interaction_generate(
        input_ids, cfg.max_new_tokens, generator=g, diagnostics=diag,
        fingerprint_keys={
            "layer_entry_h_hat": f"{fingerprint_prefix}_layer_entry_h_hat",
            "lm_head_logits_tilde": f"{fingerprint_prefix}_lm_head_logits_tilde",
        },
    )
    return tokens, diag


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------


def run_norm_granularity_low_interaction(
    *, cfg: Optional[NormGranularityConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = NormGranularityConfig()

    torch.manual_seed(cfg.seed)

    decoder_cfg = TinyModernDecoderConfig(num_layers=cfg.num_layers)
    decoder_cfg.validate()
    model = TinyModernDecoderForCausalLM(decoder_cfg)
    model.init_random_weights(torch.Generator(device="cpu").manual_seed(cfg.weights_seed))

    g_prompt = torch.Generator(device="cpu").manual_seed(cfg.prompt_seed)
    input_ids = torch.randint(
        0, decoder_cfg.vocab_size,
        (cfg.batch_size, cfg.prompt_len),
        generator=g_prompt,
    )
    input_ids_alt = torch.randint(
        0, decoder_cfg.vocab_size,
        (cfg.batch_size, cfg.prompt_len),
        generator=torch.Generator(device="cpu").manual_seed(cfg.prompt_seed + 1),
    )
    plain_tokens = model.greedy_generate(input_ids, cfg.max_new_tokens)

    # Run all three granularity modes end-to-end.
    modes: List[Tuple[str, int]] = [
        ("sequence", 1),
        ("chunk", cfg.chunk_size),
        ("token", 1),
    ]
    per_mode_results: Dict[str, Any] = {}
    for granularity, chunk_size in modes:
        tokens, diag = _run_one_mode(
            model, cfg, granularity, chunk_size, input_ids,
            cfg.mask_seed_a, granularity,
        )
        # Two-run fingerprint check.
        _, diag_b = _run_one_mode(
            model, cfg, granularity, chunk_size, input_ids,
            cfg.mask_seed_b, granularity + "_b",
        )
        same_input_two_runs_same_output = bool(torch.equal(tokens, plain_tokens))
        same_input_two_runs_different_masked_fingerprints = (
            diag.masked_boundary_fingerprints[f"{granularity}_layer_entry_h_hat"]
            != diag_b.masked_boundary_fingerprints[f"{granularity}_b_layer_entry_h_hat"]
        )
        per_mode_results[granularity] = {
            "granularity": granularity,
            "chunk_size": chunk_size,
            "greedy_token_match_rate": float(
                (plain_tokens == tokens).float().mean().item()
            ),
            "sequence_exact_match": bool(torch.equal(plain_tokens, tokens)),
            "same_input_two_runs_same_output": same_input_two_runs_same_output,
            "same_input_two_runs_different_masked_fingerprints":
                same_input_two_runs_different_masked_fingerprints,
            "diagnostics": _diag_to_summary(diag),
            "plain_token_sequence": plain_tokens.tolist(),
            "masked_token_sequence": tokens.tolist(),
        }

    # Norm + Gram leakage audit (boundary-level, layer-entry H_hat).
    leakage = _norm_and_gram_audit(model, cfg, input_ids, input_ids_alt)

    report: Dict[str, Any] = {
        "status": "ok",
        "stage": "7.6h",
        "main_mode": "norm_mask_granularity_low_interaction",
        "modes_evaluated": [m for m, _ in modes],
        "device": "cpu",
        "dtype": "float64",
        "model_type": "synthetic_tiny_modern_decoder",
        "config": {
            "vocab_size": decoder_cfg.vocab_size,
            "hidden_size": decoder_cfg.hidden_size,
            "intermediate_size": decoder_cfg.intermediate_size,
            "num_layers": decoder_cfg.num_layers,
            "num_query_heads": decoder_cfg.num_query_heads,
            "num_kv_heads": decoder_cfg.num_kv_heads,
            "head_dim": decoder_cfg.head_dim,
            "batch_size": cfg.batch_size,
            "prompt_len": cfg.prompt_len,
            "max_new_tokens": cfg.max_new_tokens,
            "chunk_size": cfg.chunk_size,
            "weights_seed": cfg.weights_seed,
            "prompt_seed": cfg.prompt_seed,
            "mask_seed_a": cfg.mask_seed_a,
            "mask_seed_b": cfg.mask_seed_b,
        },
        "stage_7_6g_inherited": {
            "use_pad": True,
            "rope_mask_mode": "pre_rope_block_diagonal_rotation",
            "rope_transient_plain_qk_visible": False,
            "qkv_projection_outputs_masked_directly": True,
            "intermediate_tee_reentry": False,
            "online_boundary_round_trips_per_decode_step": 1,
            "trusted_fallback_used_in_main_path": False,
        },
        "per_mode_results": per_mode_results,
        "norm_and_gram_leakage_audit": leakage,
        "tolerances": {
            "logits_tolerance": LOGITS_TOLERANCE,
            "kv_cache_invariant_tolerance": KV_CACHE_TOLERANCE,
            "invariant_tolerance": INVARIANT_TOLERANCE,
        },
        "limitations": [
            "Token-wise orthogonal masking does not hide row norms, because exact RMSNorm requires row-norm preservation. Row L2 norms remain observable on every boundary.",
            "Token / chunk modes reduce full Gram leakage by avoiding a sequence-shared orthogonal basis; off-diagonal Gram terms become per-row Q_i Q_j^T mixtures rather than identity.",
            "Per-token / per-chunk Q sampling and transition tables introduce per-row matmuls (einsum over the token axis); this is a security-efficiency knob, not formal cryptographic security.",
            "Sequence mode (chunk size = full sequence) is the Stage 7.6g baseline; full Gram is preserved (carried-over leakage surface).",
            "CPU local emulation only; no real TEE / GPU deployment.",
            "Synthetic tiny modern decoder; num_layers default = 1.",
            "Attention scores / probs are plain by construction of the QK invariant (carried over from Stage 7.6g).",
            "RoPE-pair 2D norms are still preserved (Stage 7.6g RoPE-safe leakage surface, carried over).",
            "This is NOT formal cryptographic / semantic / differential-privacy security.",
        ],
        "paper_safe_wording": (
            "We add a granularity knob to the orthogonal RMSNorm-compatible "
            "mask. Token-wise and chunk-wise modes preserve per-row L2 norms "
            "(required by exact RMSNorm correctness) but disrupt the full-"
            "sequence Gram-matrix leakage that sequence-shared Q exhibits. "
            "Per-decode-step token-wise masking trades fresh per-row "
            "transition cost for reduced Gram leakage on the accelerator "
            "boundary; we present this as a security-efficiency knob, "
            "not a formal security guarantee."
        ),
        "unsafe_wording_to_avoid": [
            "Token-wise masking hides row norms.",
            "Token-wise masking cryptographically hides hidden states.",
            "Per-token Q eliminates all RMSNorm-compatible leakage.",
            "Gram leakage is zero.",
            "This is formal cryptographic security.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    if x == 0.0:
        return "0.0"
    if abs(x) >= 1e-3:
        return f"{x:.6g}"
    return f"{x:.3e}"


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Norm-Mask Granularity Low-Interaction Correctness")
    w()
    w(
        "_Stage 7.6h: tighten the RMSNorm-compatible orthogonal mask "
        "granularity (sequence / chunk / token) on top of Stage 7.6g's "
        "rope-safe no-reentry path._"
    )
    w()

    w("## Inherited Stage 7.6g Guarantees")
    w()
    inh = report["stage_7_6g_inherited"]
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "use_pad", "rope_mask_mode", "rope_transient_plain_qk_visible",
        "qkv_projection_outputs_masked_directly",
        "intermediate_tee_reentry",
        "online_boundary_round_trips_per_decode_step",
        "trusted_fallback_used_in_main_path",
    ):
        w(f"| {k} | {inh[k]} |")
    w()

    w("## Configuration")
    w()
    cfg = report["config"]
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "vocab_size", "hidden_size", "intermediate_size", "num_layers",
        "num_query_heads", "num_kv_heads", "head_dim",
        "batch_size", "prompt_len", "max_new_tokens", "chunk_size",
    ):
        w(f"| {k} | {cfg[k]} |")
    w(f"| dtype | {report['dtype']} |")
    w(f"| device | {report['device']} |")
    w()

    w("## Per-Mode Correctness")
    w()
    w("| Granularity | chunk_size | greedy_token_match_rate | sequence_exact_match | h_hat invariant max | qk_score invariant max | kv_cache invariant max | lm_head recovery max |")
    w("|---|---|---|---|---|---|---|---|")
    for mode_key in report["modes_evaluated"]:
        m = report["per_mode_results"][mode_key]
        d = m["diagnostics"]
        w(f"| `{mode_key}` | {m['chunk_size']} | {m['greedy_token_match_rate']} | "
          f"{m['sequence_exact_match']} | "
          f"{_fmt(d['h_hat_layer_entry_invariant_max_abs_error'])} | "
          f"{_fmt(d['qk_score_invariant_max_abs_error'])} | "
          f"{_fmt(d['kv_cache_invariant_max_abs_error'])} | "
          f"{_fmt(d['lm_head_recovery_max_abs_error'])} |")
    w()

    w("## Per-Mode Stage 7.6g Carry-Over Diagnostics")
    w()
    w("| Granularity | norm_q_is_per_row | use_pad | rope_transient_plain_qk_visible | qkv_projection_outputs_masked_directly | intermediate_tee_reentry | trusted_fallback_used_in_main_path | online_boundary_round_trips_per_decode_step |")
    w("|---|---|---|---|---|---|---|---|")
    for mode_key in report["modes_evaluated"]:
        m = report["per_mode_results"][mode_key]
        d = m["diagnostics"]
        w(f"| `{mode_key}` | {d['norm_q_is_per_row']} | {d['use_pad']} | "
          f"{d['rope_transient_plain_qk_visible']} | "
          f"{d['qkv_projection_outputs_masked_directly']} | "
          f"{d['intermediate_tee_reentry']} | "
          f"{d['trusted_fallback_used_in_main_path']} | "
          f"{d['online_boundary_round_trips_per_decode_step']} |")
    w()

    w("## Norm + Gram Leakage Audit (Layer-Entry Boundary)")
    w()
    leak = report["norm_and_gram_leakage_audit"]
    w(
        "Boundary tensor: ``H_hat = H @ Q`` (embedded prompt, layer 0). "
        "All errors are ``max | metric(H_hat) - metric(H) |``. Row L2 "
        "norms are mathematically preserved in every mode (RMSNorm "
        "correctness requirement)."
    )
    w()
    w("| Mode | row_norm_error | full_gram_error | off_diagonal_gram_error | within_chunk_gram_error | cross_chunk_gram_error | same_prompt_fresh_Q_gram_distance |")
    w("|---|---|---|---|---|---|---|")
    for mode_key in ("sequence", "chunk", "token"):
        m = leak[mode_key]
        w(f"| `{mode_key}` | {_fmt(m['row_norm_error'])} | "
          f"{_fmt(m['full_gram_error'])} | "
          f"{_fmt(m['off_diagonal_gram_error'])} | "
          f"{_fmt(m['within_chunk_gram_error'])} | "
          f"{_fmt(m['cross_chunk_gram_error'])} | "
          f"{_fmt(m['same_prompt_fresh_Q_gram_distance'])} |")
    w()
    w(f"`different_prompt_gram_distance` (cross-prompt baseline): "
      f"**{_fmt(leak['different_prompt_gram_distance'])}**.")
    w()
    w(leak["explanation"])
    w()

    w("## Limitations")
    w()
    for item in report["limitations"]:
        w(f"- {item}")
    w()

    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()

    w("## Unsafe Wording to Avoid")
    w()
    for item in report["unsafe_wording_to_avoid"]:
        w(f"- {item}")
    w()

    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any],
    *,
    outputs_dir: Path,
    json_filename: str = "norm_granularity_low_interaction.json",
    md_filename: str = "norm_granularity_low_interaction.md",
) -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / json_filename
    md_path = outputs_dir / md_filename
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "NormGranularityConfig",
    "render_markdown",
    "run_norm_granularity_low_interaction",
    "write_reports",
    "LOGITS_TOLERANCE",
    "KV_CACHE_TOLERANCE",
    "INVARIANT_TOLERANCE",
]
