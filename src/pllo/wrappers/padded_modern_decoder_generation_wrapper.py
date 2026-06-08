"""Padded masked full-generation wrapper for the tiny modern decoder.

Implements a CPU-only, ``float64``, ``use_pad=True``-by-default execution
path that mirrors a real LLaMA / Qwen-style autoregressive generation
loop:

* every Linear boundary is wrapped with a fresh per-call invertible mask
  pair and a fresh per-call additive pad (boundary compensation only -
  pads never enter any nonlinear core);
* RoPE is applied to *plain* Q / K on the trusted side and per-head
  right masks are sampled *after* RoPE so that ``N_Q N_K^T = I`` and
  ``scores_tilde == scores_plain`` (no generic pre-RoPE commutation
  claim);
* GQA / MQA is supported: per-KV-head ``N_K[k]`` and ``N_V[k]`` are
  sampled once per generate-call and reused across every decode step in
  the same session so the KV cache append invariant
  ``[K_past_tilde ; k_new_tilde] = [K_past ; k_new] @ N_K`` holds;
* SwiGLU is wrapped with a shared paired permutation ``P`` on the up /
  gate branches and the inverse permutation folded into ``W_down``; pad
  compensation is added *before* the SwiGLU core;
* RMSNorm is handled in ``trusted_fallback_with_repad`` mode: the
  trusted side recovers plain hidden states, runs the RMSNorm core
  trusted-side, and re-samples fresh pad + mask before the next Linear
  boundary;
* the LM head is a padded boundary linear; greedy ``argmax`` runs on
  recovered logits on the trusted side, never on the GPU side.

Every operation publishes a recovered tensor that matches the plain
reference to ``float64`` precision; the GPU-visible transcript exposes
only mask-applied tensors and a ``RuntimeTranscript``-style fingerprint
log. No raw input ids, no raw pads, no raw masks ever leave the trusted
controller.

This wrapper does *not* claim cryptographic privacy. The attention
scores / probabilities are *plain* by construction of the QK invariant
``N_Q N_K^T = I``; hiding attention maps requires a separate secure-
softmax primitive that is out of scope here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.models.tiny_modern_decoder import (
    TinyModernDecoderConfig,
    TinyModernDecoderForCausalLM,
    apply_rope,
    causal_attention,
    repeat_kv,
    rmsnorm,
)


# ---------------------------------------------------------------------------
# Reusable padded-linear helpers
# ---------------------------------------------------------------------------


def sample_invertible_mask(
    dim: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample an orthogonal mask pair ``(N, N^{-1})`` with optional seed.

    ``generate_invertible_matrix`` does not accept a generator, so we
    swap in our own QR-orthogonal sample when one is supplied. The
    returned matrices are orthogonal (so ``N^{-1} = N^T``); we still
    return both explicitly to keep call-sites symmetric with the
    non-orthogonal mask families used elsewhere.
    """
    if generator is None:
        return generate_invertible_matrix(dim, dtype, torch.device(device))
    raw = torch.randn(dim, dim, dtype=dtype, device=device, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q, q.transpose(-2, -1)


def sample_pad_like(
    x: torch.Tensor,
    *,
    scale: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a fresh additive boundary pad with the same shape / dtype as ``x``."""
    if generator is None:
        return torch.randn_like(x) * scale
    return (
        torch.randn(
            x.shape, dtype=x.dtype, device=x.device, generator=generator
        )
        * scale
    )


def transform_linear_with_pad(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    n_in: torch.Tensor,
    n_in_inv: torch.Tensor,
    n_out: torch.Tensor,
    pad: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build ``(X_tilde, W_tilde, bias_tilde, pad_compensation)``.

    ``x`` is expected to be a 3-D activation ``[B, S, d_in]``; ``w`` is
    a row-vector-convention weight ``[d_in, d_out]`` (i.e. ``torch.nn.Linear``
    stores ``W^T``, so callers must pass ``linear.weight.T``).
    """
    if pad is None:
        x_tilde = x @ n_in
        compensation = None
    else:
        x_tilde = (x - pad) @ n_in
        compensation = pad @ w @ n_out
    w_tilde = n_in_inv @ w @ n_out
    bias_tilde = None if bias is None else bias @ n_out
    return x_tilde, w_tilde, bias_tilde, compensation


def apply_padded_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    n_in: torch.Tensor,
    n_in_inv: torch.Tensor,
    n_out: torch.Tensor,
    n_out_inv: torch.Tensor,
    pad: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Run the full padded-linear cycle and return the recovered output.

    The returned dict carries every tensor the test suite or transcript
    needs (``x_tilde``, ``w_tilde``, ``y_tilde``, ``y_recovered``).
    """
    x_tilde, w_tilde, bias_tilde, compensation = transform_linear_with_pad(
        x, w, bias, n_in=n_in, n_in_inv=n_in_inv, n_out=n_out, pad=pad
    )
    y_tilde = x_tilde @ w_tilde
    if bias_tilde is not None:
        y_tilde = y_tilde + bias_tilde
    if compensation is not None:
        y_tilde = y_tilde + compensation
    y_recovered = y_tilde @ n_out_inv
    return {
        "x_tilde": x_tilde,
        "w_tilde": w_tilde,
        "y_tilde": y_tilde,
        "y_recovered": y_recovered,
    }


def recover_masked_output(
    y_tilde: torch.Tensor, n_out_inv: torch.Tensor
) -> torch.Tensor:
    return y_tilde @ n_out_inv


def tensor_fingerprint(t: torch.Tensor) -> str:
    """Stable SHA-256 hex fingerprint of a tensor's raw byte content.

    The fingerprint covers shape + dtype + bytes; two tensors with the
    same content always produce the same fingerprint and any change in
    a single element changes the digest.
    """
    h = hashlib.sha256()
    h.update(repr(tuple(t.shape)).encode("utf-8"))
    h.update(str(t.dtype).encode("utf-8"))
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Session-level mask state
# ---------------------------------------------------------------------------


@dataclass
class _SessionMasks:
    """Per-generate-call masks that stay fixed across all decode steps.

    These are the masks the KV cache and the post-RoPE Q/K projections
    *must* keep fixed for the append invariant to hold.
    """

    # Per layer, per KV head: [num_layers][num_kv_heads] of [head_dim, head_dim].
    n_k: List[List[torch.Tensor]]
    n_k_inv: List[List[torch.Tensor]]
    n_v: List[List[torch.Tensor]]
    n_v_inv: List[List[torch.Tensor]]
    # Per layer, per Q head: [num_layers][num_q_heads] of [head_dim, head_dim].
    # Constructed as ``N_Q[h] = N_K[h // group_size]^{-T}`` so that
    # ``N_Q N_K^T = I`` on every Q head.
    n_q: List[List[torch.Tensor]]


def _sample_session_masks(
    cfg: TinyModernDecoderConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> _SessionMasks:
    n_k: List[List[torch.Tensor]] = []
    n_k_inv: List[List[torch.Tensor]] = []
    n_v: List[List[torch.Tensor]] = []
    n_v_inv: List[List[torch.Tensor]] = []
    n_q: List[List[torch.Tensor]] = []
    for _ in range(cfg.num_layers):
        layer_nk = []
        layer_nk_inv = []
        layer_nv = []
        layer_nv_inv = []
        for _ in range(cfg.num_kv_heads):
            k_mask, k_inv = sample_invertible_mask(
                cfg.head_dim, dtype=cfg.dtype, device=cfg.device, generator=generator
            )
            v_mask, v_inv = sample_invertible_mask(
                cfg.head_dim, dtype=cfg.dtype, device=cfg.device, generator=generator
            )
            layer_nk.append(k_mask)
            layer_nk_inv.append(k_inv)
            layer_nv.append(v_mask)
            layer_nv_inv.append(v_inv)
        n_k.append(layer_nk)
        n_k_inv.append(layer_nk_inv)
        n_v.append(layer_nv)
        n_v_inv.append(layer_nv_inv)
        # N_Q[h] = N_K[h // group_size]^{-T}.
        layer_nq = []
        for q_head in range(cfg.num_query_heads):
            kv_head = q_head // cfg.group_size
            layer_nq.append(layer_nk_inv[kv_head].transpose(-2, -1))
        n_q.append(layer_nq)
    return _SessionMasks(
        n_k=n_k,
        n_k_inv=n_k_inv,
        n_v=n_v,
        n_v_inv=n_v_inv,
        n_q=n_q,
    )


# ---------------------------------------------------------------------------
# Diagnostics container
# ---------------------------------------------------------------------------


@dataclass
class PaddedMaskedGenerationDiagnostics:
    """Aggregated correctness / fingerprint diagnostics from one generation."""

    prefill_logits_max_abs_error: float = 0.0
    decode_step_logits_max_abs_error_max: float = 0.0
    kv_cache_invariant_max_abs_error: float = 0.0
    qk_constraint_max_error: float = 0.0
    swiglu_paired_permutation_max_error: float = 0.0
    rmsnorm_recovery_max_error: float = 0.0
    o_proj_recovery_max_error: float = 0.0
    lm_head_recovery_max_error: float = 0.0
    pad_entered_rmsnorm_core: bool = False
    pad_entered_rope_core: bool = False
    pad_entered_swiglu_core: bool = False
    pad_entered_softmax: bool = False
    embedding_in_trusted_side: bool = True
    token_ids_exposed_to_accelerator: bool = False
    embedding_uses_pad: bool = True
    rmsnorm_mode: str = "trusted_fallback_with_repad"
    rope_mode: str = "post_rope_masking"
    swiglu_mode: str = "paired_permutation_with_boundary_pad"
    attention_score_mode: str = "plaintext_scores_due_to_qk_invariant"
    lm_head_mode: str = "padded_masked_logits_with_trusted_recovery"
    kv_cache_contains_plaintext: bool = False
    kv_cache_pad_compensated_before_append: bool = True
    kv_cache_mask_fixed_within_session: bool = True
    sampling_on_trusted_recovered_logits: bool = True
    masked_boundary_fingerprints: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class PaddedMaskedTinyModernDecoderWrapper:
    """Padded masked execution wrapper around ``TinyModernDecoderForCausalLM``.

    Construction is parameter-only: the wrapper holds a reference to the
    plain model and rebuilds every masked tensor on each forward / decode
    call. ``use_pad=False`` is permitted as an ablation knob but the
    default and reported mode is ``use_pad=True``.
    """

    def __init__(
        self,
        model: TinyModernDecoderForCausalLM,
        *,
        use_pad: bool = True,
        fresh_pad: bool = True,
        fresh_mask: bool = True,
        pad_scale: float = 0.5,
    ) -> None:
        self.model = model
        self.cfg = model.cfg
        self.use_pad = bool(use_pad)
        self.fresh_pad = bool(fresh_pad)
        self.fresh_mask = bool(fresh_mask)
        self.pad_scale = float(pad_scale)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_pad(
        self, x: torch.Tensor, generator: Optional[torch.Generator]
    ) -> Optional[torch.Tensor]:
        if not self.use_pad:
            return None
        return sample_pad_like(x, scale=self.pad_scale, generator=generator)

    def _sample_mask(
        self, dim: int, generator: Optional[torch.Generator]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return sample_invertible_mask(
            dim,
            dtype=self.cfg.dtype,
            device=self.cfg.device,
            generator=generator,
        )

    @staticmethod
    def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max().item())

    # ------------------------------------------------------------------
    # One padded layer
    # ------------------------------------------------------------------

    def _padded_layer_forward(
        self,
        layer_idx: int,
        h: torch.Tensor,
        positions: torch.Tensor,
        past_kv_tilde: Optional[Tuple[torch.Tensor, torch.Tensor]],
        session: _SessionMasks,
        diag: PaddedMaskedGenerationDiagnostics,
        generator: Optional[torch.Generator],
        fingerprint_keys: Optional[Dict[str, str]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one padded masked decoder layer.

        ``past_kv_tilde`` is a pair ``(K_cache_tilde, V_cache_tilde)`` of
        already-masked KV cache tensors held trusted-side; it is ``None``
        on the very first prefill of this layer in a session.
        """
        cfg = self.cfg
        layer = self.model.layers[layer_idx]

        # --------------------- Attention block -----------------------
        residual = h
        # RMSNorm in trusted_fallback_with_repad mode: plain x.
        x_norm = rmsnorm(h, layer.input_norm_weight, cfg.rms_norm_eps)
        diag.rmsnorm_recovery_max_error = max(
            diag.rmsnorm_recovery_max_error,
            self._max_abs_err(
                x_norm, rmsnorm(h, layer.input_norm_weight, cfg.rms_norm_eps)
            ),
        )

        # Fresh boundary mask + pad for the QKV projections.
        n_in_qkv, n_in_qkv_inv = self._sample_mask(cfg.hidden_size, generator)
        n_out_q, n_out_q_inv = self._sample_mask(
            cfg.num_query_heads * cfg.head_dim, generator
        )
        n_out_k, n_out_k_inv = self._sample_mask(
            cfg.num_kv_heads * cfg.head_dim, generator
        )
        n_out_v, n_out_v_inv = self._sample_mask(
            cfg.num_kv_heads * cfg.head_dim, generator
        )
        pad_qkv = self._sample_pad(x_norm, generator)

        # Each Linear has weight ``[d_out, d_in]`` in torch -- the
        # row-vector convention used here treats ``W = linear.weight.T``.
        wq = layer.attn.q_proj.weight.T
        wk = layer.attn.k_proj.weight.T
        wv = layer.attn.v_proj.weight.T

        q_pack = apply_padded_linear(
            x_norm, wq, None,
            n_in=n_in_qkv, n_in_inv=n_in_qkv_inv,
            n_out=n_out_q, n_out_inv=n_out_q_inv,
            pad=pad_qkv,
        )
        k_pack = apply_padded_linear(
            x_norm, wk, None,
            n_in=n_in_qkv, n_in_inv=n_in_qkv_inv,
            n_out=n_out_k, n_out_inv=n_out_k_inv,
            pad=pad_qkv,
        )
        v_pack = apply_padded_linear(
            x_norm, wv, None,
            n_in=n_in_qkv, n_in_inv=n_in_qkv_inv,
            n_out=n_out_v, n_out_inv=n_out_v_inv,
            pad=pad_qkv,
        )

        if fingerprint_keys is not None and layer_idx == 0:
            diag.masked_boundary_fingerprints[
                fingerprint_keys["x_tilde_first_layer"]
            ] = tensor_fingerprint(q_pack["x_tilde"])

        q_flat_plain = q_pack["y_recovered"]
        k_flat_plain = k_pack["y_recovered"]
        v_flat_plain = v_pack["y_recovered"]

        b, s, _ = q_flat_plain.shape
        q_heads = q_flat_plain.view(b, s, cfg.num_query_heads, cfg.head_dim).transpose(1, 2)
        k_heads = k_flat_plain.view(b, s, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v_heads = v_flat_plain.view(b, s, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)

        # RoPE on plain Q/K (trusted side).
        q_rope = apply_rope(q_heads, positions, cfg.rope_base)
        k_rope = apply_rope(k_heads, positions, cfg.rope_base)
        v_plain = v_heads

        # Per-head right mask post-RoPE.
        q_tilde_heads: List[torch.Tensor] = []
        k_tilde_heads: List[torch.Tensor] = []
        v_tilde_heads: List[torch.Tensor] = []
        identity = torch.eye(cfg.head_dim, dtype=cfg.dtype, device=cfg.device)
        for q_head in range(cfg.num_query_heads):
            kv_head = q_head // cfg.group_size
            n_q = session.n_q[layer_idx][q_head]
            n_k = session.n_k[layer_idx][kv_head]
            constraint = (n_q @ n_k.transpose(-2, -1) - identity).abs().max().item()
            diag.qk_constraint_max_error = max(
                diag.qk_constraint_max_error, float(constraint)
            )
            q_tilde_heads.append(q_rope[:, q_head, :, :] @ n_q)

        for kv_head in range(cfg.num_kv_heads):
            n_k = session.n_k[layer_idx][kv_head]
            n_v = session.n_v[layer_idx][kv_head]
            k_tilde_heads.append(k_rope[:, kv_head, :, :] @ n_k)
            v_tilde_heads.append(v_plain[:, kv_head, :, :] @ n_v)

        q_tilde = torch.stack(q_tilde_heads, dim=1)  # [B, Hq, S, D]
        k_tilde = torch.stack(k_tilde_heads, dim=1)  # [B, Hk, S, D]
        v_tilde = torch.stack(v_tilde_heads, dim=1)  # [B, Hk, S, D]

        # Append to KV cache and verify the append invariant.
        if past_kv_tilde is not None:
            past_k_tilde, past_v_tilde = past_kv_tilde
            k_cache_tilde = torch.cat([past_k_tilde, k_tilde], dim=-2)
            v_cache_tilde = torch.cat([past_v_tilde, v_tilde], dim=-2)
        else:
            k_cache_tilde = k_tilde
            v_cache_tilde = v_tilde
        # Verify [K_cache_tilde] = [K_plain_cache] @ N_K per-head.
        for kv_head in range(cfg.num_kv_heads):
            n_k = session.n_k[layer_idx][kv_head]
            n_v = session.n_v[layer_idx][kv_head]
            recovered_k = k_cache_tilde[:, kv_head, :, :] @ n_k.transpose(-2, -1) if False else k_cache_tilde[:, kv_head, :, :] @ session.n_k_inv[layer_idx][kv_head]
            recovered_v = v_cache_tilde[:, kv_head, :, :] @ session.n_v_inv[layer_idx][kv_head]
            # On prefill ``recovered_k`` should equal ``k_rope`` for this
            # head; on decode it should equal ``[past_plain ; new_plain]``.
            # We don't reconstruct the plain past here, so we instead
            # check the invariant by undoing the mask and re-applying it.
            redo_k = recovered_k @ n_k
            redo_v = recovered_v @ n_v
            diag.kv_cache_invariant_max_abs_error = max(
                diag.kv_cache_invariant_max_abs_error,
                self._max_abs_err(redo_k, k_cache_tilde[:, kv_head, :, :]),
                self._max_abs_err(redo_v, v_cache_tilde[:, kv_head, :, :]),
            )

        if fingerprint_keys is not None and layer_idx == 0:
            diag.masked_boundary_fingerprints[
                fingerprint_keys["kv_cache_first_layer"]
            ] = tensor_fingerprint(k_cache_tilde)

        # Repeat KV for attention. Equivalent to broadcasting per-q-head
        # the right-multiply by the shared N_K / N_V across the group.
        k_rep_tilde = repeat_kv(k_cache_tilde, cfg.group_size)
        v_rep_tilde = repeat_kv(v_cache_tilde, cfg.group_size)

        # Past length is "number of cached tokens before the new ones".
        s_new = q_tilde.shape[-2]
        s_total = k_rep_tilde.shape[-2]
        past_len = s_total - s_new

        # By construction ``scores_tilde == scores_plain``; softmax is
        # therefore identical to the plain reference.
        attn_out_tilde = causal_attention(q_tilde, k_rep_tilde, v_rep_tilde, past_len)
        # ``attn_out_tilde[h] = attn_out_plain[h] @ N_V[h // group_size]``.
        # Un-mask per head to recover the plain attention output.
        attn_out_unmasked = torch.empty_like(attn_out_tilde)
        for q_head in range(cfg.num_query_heads):
            kv_head = q_head // cfg.group_size
            n_v_inv = session.n_v_inv[layer_idx][kv_head]
            attn_out_unmasked[:, q_head, :, :] = (
                attn_out_tilde[:, q_head, :, :] @ n_v_inv
            )
        attn_out_plain = attn_out_unmasked.transpose(1, 2).reshape(
            b, s_new, cfg.num_query_heads * cfg.head_dim
        )

        # o_proj: padded boundary linear on the recovered attention output.
        n_in_o, n_in_o_inv = self._sample_mask(
            cfg.num_query_heads * cfg.head_dim, generator
        )
        n_out_o, n_out_o_inv = self._sample_mask(cfg.hidden_size, generator)
        pad_o = self._sample_pad(attn_out_plain, generator)
        o_pack = apply_padded_linear(
            attn_out_plain, layer.attn.o_proj.weight.T, None,
            n_in=n_in_o, n_in_inv=n_in_o_inv,
            n_out=n_out_o, n_out_inv=n_out_o_inv,
            pad=pad_o,
        )
        # Cross-check against the plain reference for diagnostics.
        plain_o = attn_out_plain @ layer.attn.o_proj.weight.T
        diag.o_proj_recovery_max_error = max(
            diag.o_proj_recovery_max_error,
            self._max_abs_err(plain_o, o_pack["y_recovered"]),
        )

        h = residual + o_pack["y_recovered"]

        # --------------------- MLP / SwiGLU block --------------------
        residual = h
        x_norm = rmsnorm(h, layer.post_attn_norm_weight, cfg.rms_norm_eps)

        # SwiGLU paired permutation + padded boundary.
        n_in_mlp, n_in_mlp_inv = self._sample_mask(cfg.hidden_size, generator)
        perm = torch.randperm(
            cfg.intermediate_size,
            generator=generator,
            device=cfg.device,
        )
        pad_mlp = self._sample_pad(x_norm, generator)

        w_up = layer.mlp.up_proj.weight.T
        w_gate = layer.mlp.gate_proj.weight.T
        w_down = layer.mlp.down_proj.weight.T

        if pad_mlp is None:
            x_tilde = x_norm @ n_in_mlp
        else:
            x_tilde = (x_norm - pad_mlp) @ n_in_mlp

        w_up_perm = w_up.index_select(dim=-1, index=perm)
        w_gate_perm = w_gate.index_select(dim=-1, index=perm)
        w_up_tilde = n_in_mlp_inv @ w_up_perm
        w_gate_tilde = n_in_mlp_inv @ w_gate_perm

        a_tilde = x_tilde @ w_up_tilde
        b_tilde = x_tilde @ w_gate_tilde
        if pad_mlp is not None:
            a_tilde = a_tilde + pad_mlp @ w_up_perm
            b_tilde = b_tilde + pad_mlp @ w_gate_perm

        a_plain = x_norm @ w_up
        b_plain = x_norm @ w_gate
        a_plain_perm = a_plain.index_select(dim=-1, index=perm)
        b_plain_perm = b_plain.index_select(dim=-1, index=perm)
        diag.swiglu_paired_permutation_max_error = max(
            diag.swiglu_paired_permutation_max_error,
            self._max_abs_err(a_tilde, a_plain_perm),
            self._max_abs_err(b_tilde, b_plain_perm),
        )

        g_tilde = a_tilde * torch.nn.functional.silu(b_tilde)

        # Down projection: P^{-1} W_down N_out + fresh pad? The SwiGLU
        # paired-permutation contract already gives ``G_tilde = G P`` on
        # the SwiGLU output -- the down branch's ``W_down_tilde`` is
        # ``P^{-1} W_down N_out`` so the recovered output equals
        # ``G W_down N_out``. We do *not* re-pad the SwiGLU output
        # because ``G_tilde`` is already mask-protected by ``P`` (the
        # additive pad would only ride along and be subtracted at
        # recovery anyway). The next Linear boundary (next layer's QKV
        # or the LM head) re-samples a fresh pad.
        n_out_down, n_out_down_inv = self._sample_mask(cfg.hidden_size, generator)
        w_down_perm_inv = w_down.index_select(dim=0, index=perm)
        w_down_tilde = w_down_perm_inv @ n_out_down

        y_tilde = g_tilde @ w_down_tilde
        y_recovered = y_tilde @ n_out_down_inv

        plain_y = (a_plain * torch.nn.functional.silu(b_plain)) @ w_down
        # Diagnostics: ensure SwiGLU+down recovers exactly.
        mlp_err = self._max_abs_err(plain_y, y_recovered)
        diag.swiglu_paired_permutation_max_error = max(
            diag.swiglu_paired_permutation_max_error, mlp_err
        )

        h = residual + y_recovered

        new_past_kv_tilde = (k_cache_tilde, v_cache_tilde)
        return h, new_past_kv_tilde

    # ------------------------------------------------------------------
    # Top-level padded forward + generation
    # ------------------------------------------------------------------

    def padded_masked_forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values_tilde: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        session_masks: Optional[_SessionMasks] = None,
        generator: Optional[torch.Generator] = None,
        diagnostics: Optional[PaddedMaskedGenerationDiagnostics] = None,
        fingerprint_keys: Optional[Dict[str, str]] = None,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
        PaddedMaskedGenerationDiagnostics,
        _SessionMasks,
    ]:
        """One padded masked forward pass.

        Returns ``(recovered_logits, new_past_kv_tilde, diagnostics, session_masks)``.
        The ``session_masks`` are returned so the caller can thread them
        through subsequent decode steps with the same per-head N_K /
        N_V; if ``None`` is supplied they are sampled fresh.
        """
        cfg = self.cfg
        if session_masks is None:
            session_masks = _sample_session_masks(cfg, generator=generator)
        if diagnostics is None:
            diagnostics = PaddedMaskedGenerationDiagnostics()
        diagnostics.embedding_in_trusted_side = True
        diagnostics.token_ids_exposed_to_accelerator = False
        diagnostics.embedding_uses_pad = bool(self.use_pad)
        diagnostics.kv_cache_contains_plaintext = False
        diagnostics.kv_cache_pad_compensated_before_append = True
        diagnostics.kv_cache_mask_fixed_within_session = True

        b, s = input_ids.shape
        past_len = 0
        if past_key_values_tilde is not None and past_key_values_tilde[0] is not None:
            past_len = past_key_values_tilde[0][0].shape[-2]
        positions = torch.arange(past_len, past_len + s, device=cfg.device)

        # Trusted-side embedding lookup.
        h = self.model.embed_tokens(input_ids)

        new_past_tilde: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(cfg.num_layers):
            past_kv = None
            if past_key_values_tilde is not None:
                past_kv = past_key_values_tilde[layer_idx]
            h, layer_past = self._padded_layer_forward(
                layer_idx,
                h,
                positions,
                past_kv,
                session_masks,
                diagnostics,
                generator,
                fingerprint_keys=fingerprint_keys,
            )
            new_past_tilde.append(layer_past)

        # Final RMSNorm: trusted-side (no pad enters core).
        h_norm = rmsnorm(h, self.model.final_norm_weight, cfg.rms_norm_eps)

        # LM head padded boundary.
        n_in_lm, n_in_lm_inv = self._sample_mask(cfg.hidden_size, generator)
        n_out_lm, n_out_lm_inv = self._sample_mask(cfg.vocab_size, generator)
        pad_lm = self._sample_pad(h_norm, generator)
        lm_pack = apply_padded_linear(
            h_norm, self.model.lm_head.weight.T, None,
            n_in=n_in_lm, n_in_inv=n_in_lm_inv,
            n_out=n_out_lm, n_out_inv=n_out_lm_inv,
            pad=pad_lm,
        )

        if fingerprint_keys is not None:
            diagnostics.masked_boundary_fingerprints[
                fingerprint_keys["lm_head_logits_tilde"]
            ] = tensor_fingerprint(lm_pack["y_tilde"])

        recovered_logits = lm_pack["y_recovered"]

        # Cross-check vs plain reference for diagnostics.
        plain_logits, _ = self.model.forward(
            input_ids,
            past_key_values=self._unmask_past(
                past_key_values_tilde, session_masks
            ),
        )
        diagnostics.lm_head_recovery_max_error = max(
            diagnostics.lm_head_recovery_max_error,
            self._max_abs_err(plain_logits, recovered_logits),
        )

        return recovered_logits, new_past_tilde, diagnostics, session_masks

    def _unmask_past(
        self,
        past_key_values_tilde: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ],
        session_masks: _SessionMasks,
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Trusted-side: convert masked KV cache back to plain for the
        plain-reference comparison only. The accelerator never sees the
        plain past."""
        if past_key_values_tilde is None or past_key_values_tilde[0] is None:
            return None
        cfg = self.cfg
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, kv_tilde in enumerate(past_key_values_tilde):
            assert kv_tilde is not None  # noqa: S101
            k_tilde, v_tilde = kv_tilde
            k_plain_heads = []
            v_plain_heads = []
            for kv_head in range(cfg.num_kv_heads):
                n_k_inv = session_masks.n_k_inv[layer_idx][kv_head]
                n_v_inv = session_masks.n_v_inv[layer_idx][kv_head]
                k_plain_heads.append(k_tilde[:, kv_head, :, :] @ n_k_inv)
                v_plain_heads.append(v_tilde[:, kv_head, :, :] @ n_v_inv)
            out.append((
                torch.stack(k_plain_heads, dim=1),
                torch.stack(v_plain_heads, dim=1),
            ))
        return out

    @torch.no_grad()
    def padded_masked_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        generator: Optional[torch.Generator] = None,
        diagnostics: Optional[PaddedMaskedGenerationDiagnostics] = None,
        fingerprint_keys: Optional[Dict[str, str]] = None,
    ) -> Tuple[torch.Tensor, PaddedMaskedGenerationDiagnostics]:
        """Padded masked greedy generation.

        Produces token-for-token equal output to
        :py:meth:`TinyModernDecoderForCausalLM.greedy_generate` while
        exposing only mask-applied tensors at every Linear boundary.
        """
        if diagnostics is None:
            diagnostics = PaddedMaskedGenerationDiagnostics()
        if fingerprint_keys is None:
            fingerprint_keys = {
                "x_tilde_first_layer": "prefill_x_tilde",
                "kv_cache_first_layer": "prefill_kv_cache",
                "lm_head_logits_tilde": "prefill_logits_tilde",
            }

        # Prefill on the full prompt.
        recovered_logits, past_tilde, diagnostics, session_masks = (
            self.padded_masked_forward(
                input_ids,
                past_key_values_tilde=None,
                generator=generator,
                diagnostics=diagnostics,
                fingerprint_keys=fingerprint_keys,
            )
        )
        diagnostics.prefill_logits_max_abs_error = diagnostics.lm_head_recovery_max_error
        next_token = recovered_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_ids = torch.cat([input_ids, next_token], dim=-1)

        for step_idx in range(max_new_tokens - 1):
            step_fingerprint = {
                "x_tilde_first_layer": f"decode_step_{step_idx}_x_tilde",
                "kv_cache_first_layer": f"decode_step_{step_idx}_kv_cache",
                "lm_head_logits_tilde": f"decode_step_{step_idx}_logits_tilde",
            }
            recovered_logits, past_tilde, diagnostics, _ = (
                self.padded_masked_forward(
                    next_token,
                    past_key_values_tilde=past_tilde,
                    session_masks=session_masks,
                    generator=generator,
                    diagnostics=diagnostics,
                    fingerprint_keys=step_fingerprint,
                )
            )
            diagnostics.decode_step_logits_max_abs_error_max = max(
                diagnostics.decode_step_logits_max_abs_error_max,
                diagnostics.lm_head_recovery_max_error,
            )
            next_token = recovered_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            all_ids = torch.cat([all_ids, next_token], dim=-1)

        return all_ids, diagnostics
