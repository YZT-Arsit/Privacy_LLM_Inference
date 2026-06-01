"""Stage 6.4b — Modern decoder-only block-level obfuscation wrapper.

Single-block wrapper for LLaMA / TinyLlama / Qwen / Qwen2 style decoders.
Consumes a transformer block module (or extracted weights), runs both a
plain reference forward and a Stage 5.2a / 5.3e compatible-islands
forward in parallel, and verifies the recovered output matches the plain
reference.

Scope is intentionally narrow:

* **One block only.** No full Qwen / TinyLlama model-level wrapper, no
  generation, no LM head, no tokenizer.
* **No KV-cache runtime.** Attention is computed per-call with no past
  key/value reuse.
* **Conservative RoPE handling.** The compatible path applies RoPE
  *first* and then masks with the Stage 6.4 post-RoPE per-head
  invariant. Pre-RoPE dense-mask commutation is NOT assumed.
* **RMSNorm via folded-affine island.** RMSNorm gamma is folded into the
  following q/k/v and gate/up projections via Stage 5.2a's
  ``fold_rmsnorm_affine_into_linear``. The norm core (no affine) runs in
  the trusted side under the input mask, which is orthogonal so the
  RMSNorm denominator is preserved.

The feature flag ``nonlinear_mode`` and the mitigation bundle enum
``mitigation_bundle`` are reused exactly as in Stage 5.3a / 5.3e. Default
mode remains ``"trusted"``; default bundle remains ``"fresh_perm_only"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from pllo.experiments.gqa_probe import repeat_kv
from pllo.experiments.rope_probe import apply_rope
from pllo.hf_wrappers.nonlinear_modes import (
    DEFAULT_NONLINEAR_MODE,
    VALID_NONLINEAR_MODES,
    normalize_nonlinear_mode,
)
from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.model_zoo.modern_decoder_spec import (
    ModernDecoderBlockSpec,
    extract_linear_row_weights,
    extract_rmsnorm_params,
)
from pllo.ops.compatible_masks import (
    generate_orthogonal,
    generate_permutation,
)
from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    VALID_MITIGATION_BUNDLES,
    bundle_metadata,
    describe_mitigation_bundle,
    normalize_mitigation_bundle,
)
from pllo.ops.nonlinear_islands import rmsnorm_core, run_swiglu_mlp_island


# ---------------------------------------------------------------------------
# Extracted block weights (block-level container, no HF dependency)
# ---------------------------------------------------------------------------


@dataclass
class ModernDecoderBlockWeights:
    """All weights / params needed for a forward pass on one block.

    ``W_*`` use row-vector convention: ``Y = X @ W + b``. Bias may be
    ``None`` (LLaMA / Qwen typically have ``bias=False`` on attention
    projections and SwiGLU MLP). RMSNorm has no bias.
    """

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_base: float

    # input RMSNorm
    input_norm_weight: torch.Tensor
    input_norm_eps: float
    # attention projections
    w_q: torch.Tensor
    b_q: torch.Tensor | None
    w_k: torch.Tensor
    b_k: torch.Tensor | None
    w_v: torch.Tensor
    b_v: torch.Tensor | None
    w_o: torch.Tensor
    b_o: torch.Tensor | None
    # post-attention RMSNorm
    post_attention_norm_weight: torch.Tensor
    post_attention_norm_eps: float
    # SwiGLU MLP projections
    w_gate: torch.Tensor
    b_gate: torch.Tensor | None
    w_up: torch.Tensor
    b_up: torch.Tensor | None
    w_down: torch.Tensor
    b_down: torch.Tensor | None

    @classmethod
    def from_synthetic(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        rope_base: float = 10000.0,
        seed: int = 0,
        norm_eps: float = 1e-6,
    ) -> "ModernDecoderBlockWeights":
        """Generate small random weights for a synthetic LLaMA-style block."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        device_obj = torch.device(device)

        def randn(*shape: int) -> torch.Tensor:
            return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.1).to(
                dtype=dtype, device=device_obj
            )

        kv_total = num_key_value_heads * head_dim
        q_total = num_attention_heads * head_dim
        return cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_base=rope_base,
            input_norm_weight=(
                0.9
                + (torch.rand(hidden_size, generator=g) * 0.2)
            ).to(dtype=dtype, device=device_obj),
            input_norm_eps=norm_eps,
            w_q=randn(hidden_size, q_total),
            b_q=None,
            w_k=randn(hidden_size, kv_total),
            b_k=None,
            w_v=randn(hidden_size, kv_total),
            b_v=None,
            w_o=randn(q_total, hidden_size),
            b_o=None,
            post_attention_norm_weight=(
                0.9
                + (torch.rand(hidden_size, generator=g) * 0.2)
            ).to(dtype=dtype, device=device_obj),
            post_attention_norm_eps=norm_eps,
            w_gate=randn(hidden_size, intermediate_size),
            b_gate=None,
            w_up=randn(hidden_size, intermediate_size),
            b_up=None,
            w_down=randn(intermediate_size, hidden_size),
            b_down=None,
        )

    @classmethod
    def from_hf_block(
        cls,
        block: torch.nn.Module,
        spec: ModernDecoderBlockSpec,
    ) -> "ModernDecoderBlockWeights":
        """Extract weights from a real HF LLaMA / Qwen block."""
        sa = block.self_attn
        mlp = block.mlp
        w_q, b_q = extract_linear_row_weights(sa.q_proj)
        w_k, b_k = extract_linear_row_weights(sa.k_proj)
        w_v, b_v = extract_linear_row_weights(sa.v_proj)
        w_o, b_o = extract_linear_row_weights(sa.o_proj)
        w_gate, b_gate = extract_linear_row_weights(mlp.gate_proj)
        w_up, b_up = extract_linear_row_weights(mlp.up_proj)
        w_down, b_down = extract_linear_row_weights(mlp.down_proj)
        in_w, in_eps = extract_rmsnorm_params(block.input_layernorm)
        pa_w, pa_eps = extract_rmsnorm_params(block.post_attention_layernorm)
        return cls(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            rope_base=float(spec.rope_base or 10000.0),
            input_norm_weight=in_w,
            input_norm_eps=in_eps,
            w_q=w_q,
            b_q=b_q,
            w_k=w_k,
            b_k=b_k,
            w_v=w_v,
            b_v=b_v,
            w_o=w_o,
            b_o=b_o,
            post_attention_norm_weight=pa_w,
            post_attention_norm_eps=pa_eps,
            w_gate=w_gate,
            b_gate=b_gate,
            w_up=w_up,
            b_up=b_up,
            w_down=w_down,
            b_down=b_down,
        )


# ---------------------------------------------------------------------------
# Plain reference block forward (no HF dependency at runtime)
# ---------------------------------------------------------------------------


def _rmsnorm_with_gamma(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return rmsnorm_core(x, eps=eps) * gamma


def _reshape_heads(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``[B, S, num_heads*head_dim] -> [B, num_heads, S, head_dim]`` (LLaMA layout)."""
    B, S, _ = t.shape
    return t.view(B, S, num_heads, head_dim).transpose(1, 2).contiguous()


def _merge_heads(t: torch.Tensor) -> torch.Tensor:
    """``[B, num_heads, S, head_dim] -> [B, S, num_heads*head_dim]``."""
    B, H, S, D = t.shape
    return t.transpose(1, 2).contiguous().view(B, S, H * D)


def _causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Additive causal mask: ``0`` on/below diagonal, ``-inf`` above."""
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    neg_inf = torch.finfo(dtype).min
    return torch.where(
        torch.arange(seq_len, device=device).unsqueeze(-1)
        < torch.arange(seq_len, device=device).unsqueeze(0),
        torch.full_like(mask, neg_inf),
        mask,
    )


def plain_block_forward(
    x: torch.Tensor, w: ModernDecoderBlockWeights
) -> dict[str, torch.Tensor]:
    """Reference forward using extracted weights (no HF block call).

    Returns intermediate tensors for diagnostic comparison with the
    obfuscated path.
    """
    B, S, H = x.shape
    assert H == w.hidden_size
    head_dim = w.head_dim
    num_q = w.num_attention_heads
    num_kv = w.num_key_value_heads
    group = num_q // num_kv

    # 1. input RMSNorm.
    h1 = _rmsnorm_with_gamma(x, w.input_norm_weight, w.input_norm_eps)
    # 2. q/k/v projections.
    q = h1 @ w.w_q + (w.b_q if w.b_q is not None else 0)
    k = h1 @ w.w_k + (w.b_k if w.b_k is not None else 0)
    v = h1 @ w.w_v + (w.b_v if w.b_v is not None else 0)
    # 3. reshape heads to [B, H, S, D].
    q = _reshape_heads(q, num_q, head_dim)
    k = _reshape_heads(k, num_kv, head_dim)
    v = _reshape_heads(v, num_kv, head_dim)
    # 4. RoPE on q and k.
    q_rope = apply_rope(q, base=w.rope_base)
    k_rope = apply_rope(k, base=w.rope_base)
    # 5. repeat_kv for GQA / MQA.
    k_rep = repeat_kv(k_rope, group)
    v_rep = repeat_kv(v, group)
    # 6. causal attention.
    scores = q_rope @ k_rep.transpose(-2, -1) / math.sqrt(head_dim)
    scores = scores + _causal_mask(S, scores.dtype, scores.device)
    probs = F.softmax(scores, dim=-1)
    attn = probs @ v_rep   # [B, num_q, S, D]
    attn = _merge_heads(attn)
    # 7. o_proj.
    attn_out = attn @ w.w_o + (w.b_o if w.b_o is not None else 0)
    # 8. residual.
    h_mid = x + attn_out
    # 9. post-attention RMSNorm.
    h2 = _rmsnorm_with_gamma(
        h_mid, w.post_attention_norm_weight, w.post_attention_norm_eps
    )
    # 10. SwiGLU MLP.
    gate = h2 @ w.w_gate + (w.b_gate if w.b_gate is not None else 0)
    up = h2 @ w.w_up + (w.b_up if w.b_up is not None else 0)
    hidden = F.silu(gate) * up
    mlp_out = hidden @ w.w_down + (w.b_down if w.b_down is not None else 0)
    # 11. residual.
    y = h_mid + mlp_out
    return {
        "y": y,
        "h_mid": h_mid,
        "attn_out": attn_out,
        "mlp_out": mlp_out,
        "h1": h1,
        "h2": h2,
    }


# ---------------------------------------------------------------------------
# Obfuscated block wrapper
# ---------------------------------------------------------------------------


@dataclass
class BlockHandlingFlags:
    """Honest report of which sub-paths were obfuscated vs. trusted shortcut."""

    rmsnorm_handling: str
    rope_attention_handling: str
    gqa_handling: str
    swiglu_handling: str
    residual_alignment: str
    notes: list[str] = field(default_factory=list)


class ObfuscatedModernDecoderBlockWrapper:
    """Block-level obfuscated forward with allclose recovery against plain.

    Constructed with ``ModernDecoderBlockWeights`` (already extracted from
    a real HF block or generated synthetically). The wrapper runs a
    matching plain reference and an obfuscated path, applies the inverse
    masks where needed, and exposes the result + ``correctness_report``
    so smoke / probe scripts can publish allclose numbers.
    """

    def __init__(
        self,
        weights: ModernDecoderBlockWeights,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        use_pad: bool = False,
        nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
        mitigation_bundle: str = DEFAULT_MITIGATION_BUNDLE,
    ) -> None:
        self.weights = weights
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_pad = bool(use_pad)
        self.nonlinear_mode = normalize_nonlinear_mode(nonlinear_mode)
        self.mitigation_bundle = normalize_mitigation_bundle(mitigation_bundle)
        self._desc = describe_mitigation_bundle(self.mitigation_bundle)
        self._build_handling_report()

    def _build_handling_report(self) -> None:
        if self.nonlinear_mode == "trusted":
            self.handling = BlockHandlingFlags(
                rmsnorm_handling="trusted_shortcut",
                rope_attention_handling="trusted_shortcut",
                gqa_handling="trusted_shortcut",
                swiglu_handling="trusted_shortcut",
                residual_alignment="trusted_shortcut",
                notes=[
                    "nonlinear_mode='trusted' replays the plain reference;"
                    " no compatible-island math runs.",
                ],
            )
            return
        self.handling = BlockHandlingFlags(
            rmsnorm_handling="orthogonal_island_with_gamma_folded_into_qkv",
            rope_attention_handling="rope_post_mask_only",
            gqa_handling="per_kv_head_mask_with_repeat_kv",
            swiglu_handling="compatible_island_paired_permutation",
            residual_alignment="consistent_mask_space_across_branches",
            notes=[
                "RoPE handled by masking AFTER RoPE; pre-RoPE dense-mask"
                " commutation is not assumed.",
                "Residual paths align under the same orthogonal N_res mask"
                " on both branches.",
                "At the SwiGLU boundary the residual N_res is removed"
                " (trusted-side; N_res is orthogonal so this is exact);"
                " the SwiGLU island then applies its own freshly-sampled"
                " N_in / P / N_out per the Stage 5.3e bundle.",
            ],
        )

    # ------------------------------------------------------------------ utils
    def _atol_rtol(self) -> tuple[float, float]:
        return (1e-4, 1e-4) if self.dtype is torch.float32 else (1e-8, 1e-6)

    def _make_correctness_report(
        self,
        plain: torch.Tensor,
        recovered: torch.Tensor,
        intermediates: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        atol, rtol = self._atol_rtol()
        diff = (recovered - plain).abs()
        ref_norm = float(plain.norm().clamp_min(1e-30).item())
        rel_l2 = float(
            ((recovered - plain).norm() / max(ref_norm, 1e-30)).item()
        )
        cos = float(
            (recovered.flatten() @ plain.flatten()
             / (recovered.norm() * plain.norm()).clamp_min(1e-30)).item()
        )
        bundle_meta = bundle_metadata(
            self.mitigation_bundle, use_pad=self.use_pad,
            online_extra_matmul_count=0,
        )
        return {
            "nonlinear_mode": self.nonlinear_mode,
            "valid_nonlinear_modes": list(VALID_NONLINEAR_MODES),
            "mitigation_bundle": self.mitigation_bundle,
            "valid_mitigation_bundles": list(VALID_MITIGATION_BUNDLES),
            "use_pad": self.use_pad,
            "max_abs_error": float(diff.max().item()),
            "relative_l2_error": rel_l2,
            "cosine_similarity": cos,
            "allclose": bool(
                torch.allclose(plain, recovered, atol=atol, rtol=rtol)
            ),
            "online_extra_matmul_count": 0,
            "rmsnorm_status": self.handling.rmsnorm_handling,
            "rope_attention_status": self.handling.rope_attention_handling,
            "gqa_status": self.handling.gqa_handling,
            "swiglu_status": self.handling.swiglu_handling,
            "residual_alignment_status": self.handling.residual_alignment,
            "intermediate_metrics": intermediates,
            "mitigation_bundle_metadata": bundle_meta,
            "dense_sandwich_enabled": bundle_meta["dense_sandwich_enabled"],
            "boundary_pad_enabled": bundle_meta["boundary_pad_enabled"],
            "boundary_pad_required": bundle_meta["boundary_pad_required"],
            "default_on_candidate_under_stage_5_4": bundle_meta[
                "default_on_candidate_under_stage_5_4"
            ],
            "fresh_permutation_enabled": bundle_meta["fresh_permutation_enabled"],
            "activation_input_form": bundle_meta["activation_input_form"],
            "handling_notes": list(self.handling.notes),
            "caveats": [
                "Block-level integration only; not a full model-level wrapper.",
                "No generation / decode_step / KV cache runtime is implemented.",
                "RoPE uses post-RoPE per-head masking; mask-before-RoPE"
                " commutation is not assumed.",
                "This is not a real TEE measurement.",
                "This is not formal security.",
            ],
        }

    # ----------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run plain and obfuscated forwards; return ``(recovered_y, report)``.

        In ``trusted`` mode the obfuscated branch is just the plain
        reference; the report reflects that.
        """
        recovered, report, _ = self.forward_with_traces(x, collect_traces=False)
        return recovered, report

    def forward_with_traces(
        self, x: torch.Tensor, *, collect_traces: bool = True
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
        """Run the block and additionally return GPU-visible / plaintext traces.

        Returns ``(recovered_y, correctness_report, traces)``. ``traces`` is
        a flat dict of ``{name: tensor}`` covering both plaintext references
        (``*_plain``) and the attacker-visible obfuscated tensors
        (``*_visible``). In ``trusted`` mode the visible tensors are equal
        to their plaintext counterparts and the dict reflects that.

        ``collect_traces=False`` is the fast path used by ``forward()`` —
        the obfuscated branch still runs the full math, the traces are just
        not assembled into the return value.
        """
        x = x.to(dtype=self.dtype, device=self.device)
        plain = plain_block_forward(x, self.weights)

        if self.nonlinear_mode == "trusted":
            recovered = plain["y"]
            inter = {
                "attn_branch": {
                    "max_abs_error": 0.0,
                    "allclose": True,
                },
                "mlp_branch": {
                    "max_abs_error": 0.0,
                    "allclose": True,
                },
                "rmsnorm_attn": {
                    "max_abs_error": 0.0,
                    "allclose": True,
                },
                "rmsnorm_mlp": {
                    "max_abs_error": 0.0,
                    "allclose": True,
                },
            }
            report = self._make_correctness_report(plain["y"], recovered, inter)
            traces: dict[str, torch.Tensor] = {}
            if collect_traces:
                traces = {
                    "boundary_input_plain": x.detach(),
                    "boundary_input_visible": x.detach(),
                    "q_plain": plain["h1"].detach(),
                    "q_visible": plain["h1"].detach(),
                    "final_plain": plain["y"].detach(),
                    "final_visible": plain["y"].detach(),
                }
            return recovered, report, traces

        recovered, inter, traces = self._obfuscated_block_forward(
            x, plain, collect_traces=collect_traces
        )
        report = self._make_correctness_report(plain["y"], recovered, inter)
        return recovered, report, traces

    # ------------------------------------------------- obfuscated branch
    def _obfuscated_block_forward(
        self,
        x: torch.Tensor,
        plain: dict[str, torch.Tensor],
        *,
        collect_traces: bool = False,
    ) -> tuple[torch.Tensor, dict[str, dict[str, float]], dict[str, torch.Tensor]]:
        """Run the compatible-islands forward.

        Returns ``(recovered_y, intermediates, traces)``. ``traces`` is empty
        when ``collect_traces=False``; otherwise it contains both plaintext
        references and the attacker-visible obfuscated tensors used by
        Stage 5.5's real-activation adaptive attacker.
        """
        w = self.weights
        B, S, H = x.shape
        device = x.device
        dtype = x.dtype
        atol, rtol = self._atol_rtol()
        head_dim = w.head_dim
        num_q = w.num_attention_heads
        num_kv = w.num_key_value_heads
        group = num_q // num_kv

        # ---------------- 1. Attention branch ----------------
        # Use orthogonal N_res for residual alignment so that
        # rmsnorm_core(X N_res) == rmsnorm_core(X) N_res.
        n_res = generate_orthogonal(H, dtype, device)
        x_tilde = x @ n_res

        # RMSNorm core in the masked space; recover gamma via fold into q/k/v.
        h1_core_tilde = rmsnorm_core(x_tilde, eps=w.input_norm_eps)
        # Fold input_norm_weight into q/k/v projection weights.
        w_q_folded = w.input_norm_weight.unsqueeze(-1) * w.w_q
        w_k_folded = w.input_norm_weight.unsqueeze(-1) * w.w_k
        w_v_folded = w.input_norm_weight.unsqueeze(-1) * w.w_v
        # q/k/v projections need ``W_tilde = N_res^T @ W_folded`` so that
        # ``h1_core_tilde @ W_tilde = h1_core @ W_folded``.
        w_q_tilde = n_res.T @ w_q_folded
        w_k_tilde = n_res.T @ w_k_folded
        w_v_tilde = n_res.T @ w_v_folded
        q_full = h1_core_tilde @ w_q_tilde + (w.b_q if w.b_q is not None else 0)
        k_full = h1_core_tilde @ w_k_tilde + (w.b_k if w.b_k is not None else 0)
        v_full = h1_core_tilde @ w_v_tilde + (w.b_v if w.b_v is not None else 0)

        q = _reshape_heads(q_full, num_q, head_dim)
        k = _reshape_heads(k_full, num_kv, head_dim)
        v = _reshape_heads(v_full, num_kv, head_dim)
        q_rope = apply_rope(q, base=w.rope_base)
        k_rope = apply_rope(k, base=w.rope_base)

        # Post-RoPE per-head Q/K masks satisfying N_Q[i] N_K[k]^T = I, one
        # K-mask per kv head.
        N_K_list, N_K_inv_list = [], []
        for _ in range(num_kv):
            n_k, n_k_inv = generate_invertible_matrix(head_dim, dtype, device)
            N_K_list.append(n_k)
            N_K_inv_list.append(n_k_inv)
        N_K_stack = torch.stack(N_K_list, dim=0)            # [num_kv, D, D]
        N_Q_per_q = torch.stack(
            [N_K_inv_list[i // group].transpose(-2, -1) for i in range(num_q)], dim=0
        )                                                   # [num_q, D, D]
        # Per-kv-head N_V; output projection folds N_V_rep^{-1} back.
        N_V_list, N_V_inv_list = [], []
        for _ in range(num_kv):
            n_v, n_v_inv = generate_invertible_matrix(head_dim, dtype, device)
            N_V_list.append(n_v)
            N_V_inv_list.append(n_v_inv)
        N_V_stack = torch.stack(N_V_list, dim=0)            # [num_kv, D, D]

        q_tilde = q_rope @ N_Q_per_q.unsqueeze(0)
        k_tilde = k_rope @ N_K_stack.unsqueeze(0)
        v_tilde = v @ N_V_stack.unsqueeze(0)

        k_tilde_rep = repeat_kv(k_tilde, group)
        v_tilde_rep = repeat_kv(v_tilde, group)
        N_V_per_q = torch.stack(
            [N_V_list[i // group] for i in range(num_q)], dim=0
        )                                                   # [num_q, D, D]
        N_V_inv_per_q = torch.stack(
            [N_V_inv_list[i // group] for i in range(num_q)], dim=0
        )

        scores_tilde = q_tilde @ k_tilde_rep.transpose(-2, -1) / math.sqrt(head_dim)
        scores_tilde = scores_tilde + _causal_mask(S, scores_tilde.dtype, device)
        probs_tilde = F.softmax(scores_tilde, dim=-1)
        av_tilde = probs_tilde @ v_tilde_rep                # [B, num_q, S, D]
        # Recover per-q-head: av_tilde @ N_V[i]^{-1} = av_plain.
        av_recovered = av_tilde @ N_V_inv_per_q.unsqueeze(0)
        attn_merged_plain = _merge_heads(av_recovered)
        # Output projection in the residual mask space: rebuild attn_out in
        # the N_res-masked space directly:
        # attn_out_tilde = attn_merged_plain @ (W_o @ N_res)
        w_o_tilde = w.w_o @ n_res
        attn_out_tilde = attn_merged_plain @ w_o_tilde
        if w.b_o is not None:
            attn_out_tilde = attn_out_tilde + (w.b_o @ n_res)
        h_mid_tilde = x_tilde + attn_out_tilde
        # n_res is orthogonal ⇒ n_res^{-1} == n_res.T (exact, no inv() needed).
        n_res_inv = n_res.T
        attn_out_recovered = attn_out_tilde @ n_res_inv
        h_mid_recovered = h_mid_tilde @ n_res_inv

        # ---------------- 2. MLP branch ----------------
        # post-attention RMSNorm core in N_res-masked space (orthogonal ⇒
        # rms invariant). Fold post-attention gamma into gate/up.
        h2_core_tilde = rmsnorm_core(h_mid_tilde, eps=w.post_attention_norm_eps)
        w_gate_folded = w.post_attention_norm_weight.unsqueeze(-1) * w.w_gate
        w_up_folded = w.post_attention_norm_weight.unsqueeze(-1) * w.w_up

        # Bring h2_core into plain space at the SwiGLU boundary. Because
        # n_res is orthogonal this is an exact one-matmul unmask on the
        # trusted side; the island then applies its own freshly-sampled
        # N_in / P / N_out (the Stage 5.3e dense sandwich) over the gate /
        # up / down projections.
        h2_plain_for_island = h2_core_tilde @ n_res_inv     # = rmsnorm_core(h_mid)
        n_in_island, n_in_island_inv = generate_invertible_matrix(H, dtype, device)
        n_out_island, n_out_island_inv = generate_invertible_matrix(H, dtype, device)
        perm = generate_permutation(
            w.intermediate_size, dtype=dtype, device=device
        )["perm"]
        pad_in = None
        if self.use_pad:
            pad_in = torch.randn(
                h2_plain_for_island.reshape(-1, H).shape, dtype=dtype, device=device
            )
        island = run_swiglu_mlp_island(
            x=h2_plain_for_island.reshape(-1, H),
            w_up=w_up_folded,
            b_up=w.b_up,
            w_gate=w_gate_folded,
            b_gate=w.b_gate,
            w_down=w.w_down,
            b_down=w.b_down,
            n_in=n_in_island,
            n_in_inv=n_in_island_inv,
            permutation=perm,
            n_out=n_out_island,
            pad_in=pad_in,
            mitigation_bundle=self.mitigation_bundle,
        )
        mlp_out_island_tilde = island["y_tilde"].reshape(B, S, H)
        # Recover plain mlp_out from the island's N_out, then re-mask into
        # N_res space for the residual addition.
        mlp_out_plain = mlp_out_island_tilde @ n_out_island_inv
        mlp_out_in_res = mlp_out_plain @ n_res

        # ---------------- 3. Residual add ----------------
        y_tilde = h_mid_tilde + mlp_out_in_res
        y_recovered = y_tilde @ n_res_inv

        # ---------------- 3b. Stage 5.5 attacker-visible traces ----------------
        traces: dict[str, torch.Tensor] = {}
        if collect_traces:
            # SwiGLU sub-tensors. The island uses A = x@W_up (i.e. "up") and
            # B = x@W_gate (i.e. "gate"); ``g_plain_permuted`` is G[:, perm].
            h2_plain = h2_plain_for_island
            up_plain = h2_plain @ w_up_folded
            if w.b_up is not None:
                up_plain = up_plain + w.b_up
            gate_plain = h2_plain @ w_gate_folded
            if w.b_gate is not None:
                gate_plain = gate_plain + w.b_gate
            up_visible = up_plain.index_select(dim=-1, index=perm)
            gate_visible = gate_plain.index_select(dim=-1, index=perm)
            g_visible = island["g_tilde"].reshape(
                B, S, w.intermediate_size
            )                                                   # = G[:, perm]
            g_plain = F.silu(gate_plain) * up_plain             # plain SwiGLU
            traces.update(
                {
                    # Block boundary.
                    "boundary_input_plain": x.detach(),
                    "boundary_input_visible": x_tilde.detach(),
                    # Q / K / V (post-RoPE for q/k, pre-attn for v).
                    "q_plain": q_rope.detach(),
                    "q_visible": q_tilde.detach(),
                    "k_plain": k_rope.detach(),
                    "k_visible": k_tilde.detach(),
                    "v_plain": v.detach(),
                    "v_visible": v_tilde.detach(),
                    # SwiGLU intermediates (per-token row).
                    "gate_plain": gate_plain.detach(),
                    "gate_visible": gate_visible.detach(),
                    "up_plain": up_plain.detach(),
                    "up_visible": up_visible.detach(),
                    "swiglu_intermediate_plain": g_plain.detach(),
                    "swiglu_intermediate_visible": g_visible.detach(),
                    # Down-projection output: visible is mlp_out @ N_out_island.
                    "post_island_plain": mlp_out_plain.detach(),
                    "post_island_visible": mlp_out_island_tilde.detach(),
                    # Block final output.
                    "final_plain": plain["y"].detach(),
                    "final_visible": y_tilde.detach(),
                }
            )

        # ---------------- 4. Intermediate diagnostics ----------------
        def _metrics(plain_t: torch.Tensor, rec_t: torch.Tensor) -> dict[str, float]:
            diff = (rec_t - plain_t).abs()
            ref_norm = float(plain_t.norm().clamp_min(1e-30).item())
            return {
                "max_abs_error": float(diff.max().item()),
                "relative_l2_error": float(
                    ((rec_t - plain_t).norm() / max(ref_norm, 1e-30)).item()
                ),
                "allclose": bool(
                    torch.allclose(plain_t, rec_t, atol=atol, rtol=rtol)
                ),
            }

        intermediates = {
            "attn_branch": _metrics(plain["attn_out"], attn_out_recovered),
            "h_mid": _metrics(plain["h_mid"], h_mid_recovered),
            "mlp_branch": _metrics(plain["mlp_out"], mlp_out_plain),
        }
        return y_recovered, intermediates, traces


__all__ = [
    "BlockHandlingFlags",
    "ModernDecoderBlockWeights",
    "ObfuscatedModernDecoderBlockWrapper",
    "plain_block_forward",
]
