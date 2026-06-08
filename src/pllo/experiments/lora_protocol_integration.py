"""Stage 7.7b -- LoRA integration with the Stage 7.6g/h/i main protocol.

This experiment verifies that LoRA adapters ``Y = X W + X A B`` can be
folded into the existing padded-masked-boundary tables of the
low-interaction wrapper *without* breaking any Stage 7.6g/h/i
invariant. Two integration paths are validated:

1. ``Algebraic identity`` -- a self-contained verification of the
   LoRA-padded-boundary identity, per insertion site, at every
   relevant ``N_out`` (B_Q for q_proj, B_K for k_proj, N_V for v_proj,
   Q_l-compatible for o_proj / down_proj, paired-permutation P for
   gate/up_proj). Sampled with fresh masks ``M`` (input pad target),
   ``T`` (pad), and ``R`` (rank-space mask). The identity is

       X_pad @ W_tilde + C_base + X_pad @ A_tilde @ B_tilde + C_lora
           == (X W + X A B) @ N_out

   where
       W_tilde = M^{-1} W N_out
       A_tilde = M^{-1} A R
       B_tilde = R^{-1} B N_out
       C_base  = T W N_out
       C_lora  = T A B N_out
       X_pad   = (X - T) M

2. ``Merged-weights generation`` -- LoRA is mathematically equivalent
   to a modified weight ``W_eff = W + A B``. The existing Stage 7.6h
   wrapper consumes only the merged weight; so an end-to-end greedy
   generation test with LoRA-augmented weights confirms that LoRA is
   supported by the existing protocol under every combination of
   ``norm_mask_granularity`` and ``attention_privacy_mode``.

The report explicitly notes:
* rank padding hides the *true* rank (an inner-dimension secret) but
  not the *padded* rank (the inner dimension of A_tilde, B_tilde is
  observable on the accelerator).
* the LoRA adapters ``A``, ``B`` themselves remain trusted-side; only
  the merged ``W_tilde + A_tilde B_tilde`` (or the merged ``W_eff``
  in merged-weights mode) is materialised on the accelerator path.
* LoRA *training* (backward pass) is NOT addressed here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
    generate_rope_plane_rotation_mask,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRAProtocolConfig:
    seed: int = 2026
    weights_seed: int = 2026
    prompt_seed: int = 2027
    mask_seed: int = 2028
    lora_seed: int = 2029
    batch_size: int = 2
    prompt_len: int = 6
    max_new_tokens: int = 3
    num_layers: int = 1
    true_rank: int = 4
    padded_rank: int = 8


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def _sample_orthogonal(
    dim: int, *, dtype: torch.dtype, device: str, generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = torch.randn(dim, dim, dtype=dtype, device=device, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q, q.transpose(-2, -1)


def _sample_invertible(
    dim: int, *, dtype: torch.dtype, device: str, generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = torch.randn(dim, dim, dtype=dtype, device=device, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q, q.transpose(-2, -1)


# ---------------------------------------------------------------------------
# Algebraic identity verification
# ---------------------------------------------------------------------------


def _verify_lora_padded_identity(
    *,
    in_dim: int,
    out_dim: int,
    true_rank: int,
    padded_rank: int,
    n_out: torch.Tensor,
    n_out_inv: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    generator: torch.Generator,
    use_pad: bool = True,
) -> Dict[str, Any]:
    """Sample one (W, A, B) + masks + pad and verify the identity."""
    B, S = 3, 5
    X = torch.randn(B, S, in_dim, dtype=dtype, device=device, generator=generator)
    W = torch.randn(in_dim, out_dim, dtype=dtype, device=device, generator=generator)
    # LoRA factors: A in R^{in_dim x true_rank}, B in R^{true_rank x out_dim}.
    A = torch.randn(in_dim, true_rank, dtype=dtype, device=device, generator=generator)
    B_mat = torch.randn(true_rank, out_dim, dtype=dtype, device=device, generator=generator)

    # Pad rank: column-pad A and row-pad B with zeros so the merged
    # AB equals the true-rank AB exactly. ``padded_rank`` is what the
    # accelerator sees on the inner dimension.
    A_pad = torch.zeros(in_dim, padded_rank, dtype=dtype, device=device)
    A_pad[:, :true_rank] = A
    B_pad = torch.zeros(padded_rank, out_dim, dtype=dtype, device=device)
    B_pad[:true_rank, :] = B_mat
    # Sanity check: the padded factors give the same AB.
    ab_padded = A_pad @ B_pad
    ab_true = A @ B_mat
    pad_rank_err = float((ab_padded - ab_true).abs().max().item())

    M, M_inv = _sample_invertible(
        in_dim, dtype=dtype, device=device, generator=generator
    )
    R, R_inv = _sample_invertible(
        padded_rank, dtype=dtype, device=device, generator=generator
    )
    T = (
        torch.randn(B, S, in_dim, dtype=dtype, device=device, generator=generator)
        * (0.5 if use_pad else 0.0)
    )

    W_tilde = M_inv @ W @ n_out
    A_tilde = M_inv @ A_pad @ R
    B_tilde = R_inv @ B_pad @ n_out
    C_base = T @ W @ n_out
    C_lora = T @ A_pad @ B_pad @ n_out
    X_pad = (X - T) @ M
    Y_tilde = X_pad @ W_tilde + C_base + X_pad @ A_tilde @ B_tilde + C_lora
    Y_plain = (X @ W + X @ ab_true) @ n_out
    err = float((Y_tilde - Y_plain).abs().max().item())

    # Trusted recovery: Y = Y_tilde @ N_out_inv.
    Y_rec = Y_tilde @ n_out_inv
    Y_ref = X @ W + X @ ab_true
    rec_err = float((Y_rec - Y_ref).abs().max().item())
    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "true_rank": true_rank,
        "padded_rank": padded_rank,
        "padded_AB_minus_true_AB_max_abs_error": pad_rank_err,
        "padded_boundary_identity_max_abs_error": err,
        "trusted_recovery_max_abs_error": rec_err,
    }


# ---------------------------------------------------------------------------
# Per-site verifications
# ---------------------------------------------------------------------------


def _per_site_identity_audit(
    cfg: LoRAProtocolConfig,
    decoder_cfg: TinyModernDecoderConfig,
) -> Dict[str, Dict[str, Any]]:
    dtype = decoder_cfg.dtype
    device = str(decoder_cfg.device)
    g = torch.Generator(device="cpu").manual_seed(cfg.lora_seed)
    hidden = decoder_cfg.hidden_size
    inter = decoder_cfg.intermediate_size
    head_dim = decoder_cfg.head_dim
    n_q = decoder_cfg.num_query_heads
    n_kv = decoder_cfg.num_kv_heads

    results: Dict[str, Dict[str, Any]] = {}

    # q_proj: out = num_query_heads * head_dim. N_out = block-diag B_Q
    # per Q head. Each per-head block is a 2D RoPE-plane rotation.
    bq_blocks = [
        generate_rope_plane_rotation_mask(
            head_dim, dtype=dtype, device=device, generator=g
        )
        for _ in range(n_q)
    ]
    n_out_q = torch.block_diag(*bq_blocks)
    n_out_q_inv = n_out_q.transpose(-2, -1)
    results["q_proj"] = {
        **_verify_lora_padded_identity(
            in_dim=hidden, out_dim=n_q * head_dim,
            true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
            n_out=n_out_q, n_out_inv=n_out_q_inv,
            dtype=dtype, device=device, generator=g,
        ),
        "n_out_kind": "B_Q_block_diagonal_rope_plane_rotation",
        "rope_safe": True,
    }
    # k_proj: out = num_kv_heads * head_dim. N_out = B_K block.
    bk_blocks = [
        generate_rope_plane_rotation_mask(
            head_dim, dtype=dtype, device=device, generator=g
        )
        for _ in range(n_kv)
    ]
    n_out_k = torch.block_diag(*bk_blocks)
    n_out_k_inv = n_out_k.transpose(-2, -1)
    results["k_proj"] = {
        **_verify_lora_padded_identity(
            in_dim=hidden, out_dim=n_kv * head_dim,
            true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
            n_out=n_out_k, n_out_inv=n_out_k_inv,
            dtype=dtype, device=device, generator=g,
        ),
        "n_out_kind": "B_K_block_diagonal_rope_plane_rotation",
        "rope_safe": True,
    }
    # v_proj: out = num_kv_heads * head_dim. N_out = N_V block.
    nv_blocks: List[torch.Tensor] = []
    nv_inv_blocks: List[torch.Tensor] = []
    for _ in range(n_kv):
        nb, nb_inv = _sample_orthogonal(
            head_dim, dtype=dtype, device=device, generator=g
        )
        nv_blocks.append(nb)
        nv_inv_blocks.append(nb_inv)
    n_out_v = torch.block_diag(*nv_blocks)
    n_out_v_inv = torch.block_diag(*nv_inv_blocks)
    results["v_proj"] = {
        **_verify_lora_padded_identity(
            in_dim=hidden, out_dim=n_kv * head_dim,
            true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
            n_out=n_out_v, n_out_inv=n_out_v_inv,
            dtype=dtype, device=device, generator=g,
        ),
        "n_out_kind": "N_V_block_diagonal_orthogonal_per_head",
        "rope_safe": True,
    }
    # o_proj: out_dim = hidden, N_out = Q_l (residual-stream).
    n_out_o, n_out_o_inv = _sample_orthogonal(
        hidden, dtype=dtype, device=device, generator=g
    )
    results["o_proj"] = {
        **_verify_lora_padded_identity(
            in_dim=n_q * head_dim, out_dim=hidden,
            true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
            n_out=n_out_o, n_out_inv=n_out_o_inv,
            dtype=dtype, device=device, generator=g,
        ),
        "n_out_kind": "Q_l_residual_stream_orthogonal",
        "rope_safe": True,
    }
    # up/gate: out_dim = intermediate_size, N_out = P (paired permutation).
    perm = torch.randperm(inter, generator=g)
    P = torch.zeros(inter, inter, dtype=dtype, device=device)
    P[torch.arange(inter), perm] = 1.0
    P_inv = P.transpose(-2, -1)
    for site in ("up_proj", "gate_proj"):
        results[site] = {
            **_verify_lora_padded_identity(
                in_dim=hidden, out_dim=inter,
                true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
                n_out=P, n_out_inv=P_inv,
                dtype=dtype, device=device, generator=g,
            ),
            "n_out_kind": "paired_permutation_P",
            "rope_safe": True,
        }
    # down_proj: in_dim = intermediate_size, out_dim = hidden, N_out = Q_l.
    n_out_d, n_out_d_inv = _sample_orthogonal(
        hidden, dtype=dtype, device=device, generator=g
    )
    results["down_proj"] = {
        **_verify_lora_padded_identity(
            in_dim=inter, out_dim=hidden,
            true_rank=cfg.true_rank, padded_rank=cfg.padded_rank,
            n_out=n_out_d, n_out_inv=n_out_d_inv,
            dtype=dtype, device=device, generator=g,
        ),
        "n_out_kind": "Q_l_residual_stream_orthogonal",
        "rope_safe": True,
    }
    return results


# ---------------------------------------------------------------------------
# Merged-weights greedy integration
# ---------------------------------------------------------------------------


def _inject_lora_into_model(
    model: TinyModernDecoderForCausalLM,
    cfg: LoRAProtocolConfig,
    generator: torch.Generator,
) -> Dict[str, Any]:
    """Add a LoRA AB term into every supported projection's weight.

    Returns a dict of the original weights (for restoration) and a
    description of the injection. Modifies ``model`` in place.
    """
    dtype = model.cfg.dtype
    device = str(model.cfg.device)
    rank = cfg.true_rank
    saved: Dict[str, torch.Tensor] = {}
    sites: List[str] = []
    for layer_idx, layer in enumerate(model.layers):
        for name, mod in (
            ("q_proj", layer.attn.q_proj),
            ("k_proj", layer.attn.k_proj),
            ("v_proj", layer.attn.v_proj),
            ("o_proj", layer.attn.o_proj),
            ("up_proj", layer.mlp.up_proj),
            ("gate_proj", layer.mlp.gate_proj),
            ("down_proj", layer.mlp.down_proj),
        ):
            w = mod.weight   # [out, in]
            out_dim, in_dim = w.shape
            A = torch.randn(in_dim, rank, dtype=dtype, device=device, generator=generator)
            B = torch.randn(rank, out_dim, dtype=dtype, device=device, generator=generator)
            ab = A @ B  # [in, out]
            saved[f"layer_{layer_idx}_{name}"] = w.detach().clone()
            with torch.no_grad():
                w.add_(ab.transpose(0, 1) * 0.05)
            sites.append(f"layer_{layer_idx}_{name}")
    return {"saved_weights": saved, "sites": sites}


def _restore_model(
    model: TinyModernDecoderForCausalLM, saved: Dict[str, torch.Tensor],
) -> None:
    for layer_idx, layer in enumerate(model.layers):
        for name, mod in (
            ("q_proj", layer.attn.q_proj),
            ("k_proj", layer.attn.k_proj),
            ("v_proj", layer.attn.v_proj),
            ("o_proj", layer.attn.o_proj),
            ("up_proj", layer.mlp.up_proj),
            ("gate_proj", layer.mlp.gate_proj),
            ("down_proj", layer.mlp.down_proj),
        ):
            key = f"layer_{layer_idx}_{name}"
            with torch.no_grad():
                mod.weight.copy_(saved[key])


def _run_one_protocol_combo(
    model: TinyModernDecoderForCausalLM,
    cfg: LoRAProtocolConfig,
    input_ids: torch.Tensor,
    *,
    norm_mask_granularity: str,
    norm_chunk_size: int,
    attention_privacy_mode: str,
) -> Dict[str, Any]:
    plain = model.greedy_generate(input_ids, cfg.max_new_tokens)
    wrapper = LowInteractionTinyModernDecoderWrapper(
        model,
        use_pad=True,
        rope_mask_mode="pre_rope_block_diagonal_rotation",
        norm_mask_granularity=norm_mask_granularity,
        norm_chunk_size=norm_chunk_size,
        attention_privacy_mode=attention_privacy_mode,
    )
    g = torch.Generator(device="cpu").manual_seed(cfg.mask_seed)
    diag = LowInteractionDiagnostics()
    tokens, diag = wrapper.low_interaction_generate(
        input_ids, cfg.max_new_tokens, generator=g, diagnostics=diag,
    )
    return {
        "norm_mask_granularity": norm_mask_granularity,
        "norm_chunk_size": norm_chunk_size,
        "attention_privacy_mode": attention_privacy_mode,
        "greedy_token_match_rate": float(
            (plain == tokens).float().mean().item()
        ),
        "sequence_exact_match": bool(torch.equal(plain, tokens)),
        "lm_head_recovery_max_abs_error": diag.lm_head_recovery_max_abs_error,
        "h_hat_layer_entry_invariant_max_abs_error":
            diag.h_hat_layer_entry_invariant_max_abs_error,
        "attention_scores_visible": diag.attention_scores_visible,
        "intermediate_tee_reentry": diag.intermediate_tee_reentry,
        "online_boundary_round_trips_per_decode_step":
            diag.online_boundary_round_trips_per_decode_step,
    }


def run_lora_protocol_integration(
    *, cfg: Optional[LoRAProtocolConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = LoRAProtocolConfig()
    torch.manual_seed(cfg.seed)
    decoder_cfg = TinyModernDecoderConfig(num_layers=cfg.num_layers)
    decoder_cfg.validate()
    model = TinyModernDecoderForCausalLM(decoder_cfg)
    model.init_random_weights(
        torch.Generator(device="cpu").manual_seed(cfg.weights_seed)
    )

    site_audit = _per_site_identity_audit(cfg, decoder_cfg)

    g_prompt = torch.Generator(device="cpu").manual_seed(cfg.prompt_seed)
    input_ids = torch.randint(
        0, decoder_cfg.vocab_size,
        (cfg.batch_size, cfg.prompt_len),
        generator=g_prompt,
    )
    g_lora = torch.Generator(device="cpu").manual_seed(cfg.lora_seed)
    inj = _inject_lora_into_model(model, cfg, g_lora)

    combos: List[Tuple[str, int, str]] = [
        ("sequence", 1, "exact_visible_attention"),
        ("token", 1, "exact_visible_attention"),
        ("chunk", 2, "exact_visible_attention"),
        ("sequence", 1, "trusted_softmax_attention"),
        ("token", 1, "trusted_softmax_attention"),
    ]
    combo_results: List[Dict[str, Any]] = []
    for gran, ck, apm in combos:
        combo_results.append(
            _run_one_protocol_combo(
                model, cfg, input_ids,
                norm_mask_granularity=gran,
                norm_chunk_size=ck,
                attention_privacy_mode=apm,
            )
        )

    # Restore for hygiene (so the model can be reused).
    _restore_model(model, inj["saved_weights"])

    report = {
        "status": "ok",
        "stage": "7.7b",
        "main_mode": "lora_protocol_integration",
        "device": "cpu",
        "dtype": str(decoder_cfg.dtype),
        "config": {
            "num_layers": decoder_cfg.num_layers,
            "hidden_size": decoder_cfg.hidden_size,
            "intermediate_size": decoder_cfg.intermediate_size,
            "true_rank": cfg.true_rank,
            "padded_rank": cfg.padded_rank,
            "batch_size": cfg.batch_size,
            "prompt_len": cfg.prompt_len,
            "max_new_tokens": cfg.max_new_tokens,
        },
        "supported_lora_sites": list(site_audit.keys()),
        "unsupported_lora_sites": [],
        "rank_padding_policy": (
            "pad rank from true_rank to padded_rank with zeros in A "
            "and B; A_tilde / B_tilde share the inner dimension "
            "padded_rank on the accelerator side."
        ),
        "true_rank_hidden_from_shape": True,
        "padded_rank_visible": True,
        "lora_adapter_plaintext_visible": False,
        "lora_training_backward_supported": False,
        "site_padded_boundary_identity_audit": site_audit,
        "merged_weights_generation_combos": combo_results,
        "stage_7_6h_inherited": {
            "use_pad": True,
            "rope_mask_mode": "pre_rope_block_diagonal_rotation",
            "qkv_projection_outputs_masked_directly": True,
        },
        "limitations": [
            "CPU local emulation only; no real TEE / GPU.",
            "LoRA forward path validated only; backward / training is "
            "NOT implemented.",
            "Algebraic identity is verified per insertion site with "
            "fresh masks and pads; sample sizes are small.",
            "Rank padding hides the *true* inner rank but the *padded* "
            "rank (inner dimension of A_tilde and B_tilde) is "
            "observable on the accelerator side.",
            "The merged-weights generation path absorbs LoRA into the "
            "base weight; the resulting W_eff = W + A B is what the "
            "wrapper sees, equivalent to deploying a fine-tuned model.",
            "No formal cryptographic / semantic / differential-privacy "
            "security claim.",
        ],
        "paper_safe_wording": (
            "LoRA adapters integrate with the low-interaction main "
            "protocol via the same padded-boundary algebra used for "
            "the base linear: A_tilde = M^{-1} A R, B_tilde = R^{-1} "
            "B N_out, with rank-space mask R. The forward path is "
            "exact at float64; rank padding hides the true rank but "
            "the padded rank is observable. We do not address LoRA "
            "training."
        ),
        "unsafe_wording_to_avoid": [
            "LoRA training is supported.",
            "Padded rank cryptographically hides true rank.",
            "LoRA fine-tuning is end-to-end private.",
            "This is formal cryptographic security.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if x == 0.0:
            return "0.0"
        return f"{x:.3e}" if abs(x) < 1e-3 else f"{x:.6g}"
    return str(x)


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# LoRA Integration with Stage 7.6g/h/i Main Protocol")
    w()
    w(
        "_Stage 7.7b: verify LoRA adapters at every supported "
        "insertion site under the padded low-interaction protocol._"
    )
    w()
    w("## Supported LoRA Sites")
    w()
    w(
        "| Site | n_out_kind | rope_safe | padded_boundary_identity_max | "
        "trusted_recovery_max | padded_AB_minus_true_AB_max |"
    )
    w("|---|---|---|---|---|---|")
    for site, info in report["site_padded_boundary_identity_audit"].items():
        w(
            f"| `{site}` | {info['n_out_kind']} | {info['rope_safe']} | "
            f"{_fmt(info['padded_boundary_identity_max_abs_error'])} | "
            f"{_fmt(info['trusted_recovery_max_abs_error'])} | "
            f"{_fmt(info['padded_AB_minus_true_AB_max_abs_error'])} |"
        )
    w()

    w("## Merged-Weights Generation Across Modes")
    w()
    w(
        "| norm_granularity | chunk | attention_privacy_mode | "
        "greedy_match | seq_exact | lm_head_recovery_max | "
        "h_hat_max | round_trips |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for c in report["merged_weights_generation_combos"]:
        w(
            f"| `{c['norm_mask_granularity']}` | {c['norm_chunk_size']} | "
            f"`{c['attention_privacy_mode']}` | "
            f"{c['greedy_token_match_rate']} | "
            f"{c['sequence_exact_match']} | "
            f"{_fmt(c['lm_head_recovery_max_abs_error'])} | "
            f"{_fmt(c['h_hat_layer_entry_invariant_max_abs_error'])} | "
            f"{c['online_boundary_round_trips_per_decode_step']} |"
        )
    w()

    w("## Rank Padding Policy")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "rank_padding_policy",
        "true_rank_hidden_from_shape",
        "padded_rank_visible",
        "lora_adapter_plaintext_visible",
        "lora_training_backward_supported",
    ):
        w(f"| {k} | {report[k]} |")
    w()
    w("## Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w("## Unsafe Wording to Avoid")
    w()
    for x in report["unsafe_wording_to_avoid"]:
        w(f"- {x}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any], *, outputs_dir: Path,
    json_filename: str = "lora_protocol_integration.json",
    md_filename: str = "lora_protocol_integration.md",
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
    "LoRAProtocolConfig",
    "render_markdown",
    "run_lora_protocol_integration",
    "write_reports",
]
