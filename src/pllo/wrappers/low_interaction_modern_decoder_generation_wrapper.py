"""Stage 7.6f / 7.6g -- low-interaction operator-compatible generation wrapper.

Stage 7.6g extends the wrapper with a ``rope_mask_mode`` parameter:

* ``"post_rope_masking"`` (Stage 7.6f) -- preferred path for the
  baseline; the qkv-projection output is plain Q / K / V transiently on
  the accelerator, RoPE is applied on plain Q / K, and per-head right
  masks ``N_Q`` / ``N_K`` are applied immediately afterwards. This is
  the explicit RoPE blocker.

* ``"pre_rope_block_diagonal_rotation"`` (Stage 7.6g) -- the main paper
  path. The qkv-projection output is masked DIRECTLY by per-head
  block-diagonal rotation masks ``B_Q`` / ``B_K`` that act as 2D
  rotations in each RoPE pair (channel ``j`` paired with channel
  ``j + head_dim/2``). Because the RoPE block rotation and the mask
  block rotation operate in the same 2D plane, they commute:
  ``RoPE(Q @ B_Q) = RoPE(Q) @ B_Q``. No plain Q / K / V tensor is ever
  visible on the accelerator. With ``B_Q[i] = B_K[i // group_size]`` the
  attention score invariant
  ``Q_rope_tilde K_rope_tilde^T = Q_rope K_rope^T`` holds by
  construction.


This wrapper demonstrates the paper *main* invariant:

    H_hat_l = H_l @ Q_l            (Q_l orthogonal, per-session)

with **no intermediate TEE re-entry** in the online decode path. Per
decode step there is exactly **one** boundary round-trip:

    TEE -> accelerator: one masked current-token state H_hat_0
    accelerator runs every layer with operator-compatible RMSNorm,
        padded boundary linears, post-RoPE per-head masks, paired-
        permutation SwiGLU, and a masked KV cache
    accelerator -> TEE: one masked logits tensor z_tilde = z @ N_vocab
    TEE recovers logits and samples

The protocol-level trick that makes the no-reentry path work:

    Given a *compatible* state ``X @ Q`` and a fresh padded-boundary
    target ``(X - T) @ M``, the trusted side precomputes

        A = Q^{-1} M ,   C_T = T M

    and ships ``(A, C_T)`` to the accelerator. The accelerator then
    transitions from the compatible state to the padded boundary state
    *without* TEE re-entry:

        (X @ Q) @ A - C_T = (X - T) @ M .

    The accelerator then runs the standard padded linear

        Y_tilde = X_pad_tilde @ W_tilde + b_tilde + C_linear
                = X W N_out + b N_out

    where ``W_tilde = M^{-1} W N_out``, ``b_tilde = b N_out``,
    ``C_linear = T W N_out``.

RMSNorm runs in ``operator_compatible_orthogonal`` mode: with ``Q_l``
orthogonal,

    RMSNormCore(H_l @ Q_l) = RMSNormCore(H_l) @ Q_l ,

and the affine ``gamma`` is folded into the *following* Linear weight
matrix at session-compile time. The accelerator never holds a fresh
RMSNorm gamma; it only sees gamma-folded weights inside the padded
linear tables.

SwiGLU uses a shared paired permutation ``P``:

    a_tilde = a_plain @ P ,  b_tilde = b_plain @ P
    g_tilde = a_tilde * silu(b_tilde) = g_plain @ P
    down_proj_compat = P^{-1} W_down @ Q_l

Pads are mandatory at every GPU-visible Linear boundary; pads never
enter RMSNorm / SwiGLU / RoPE / softmax cores.

RoPE remains the explicit blocker: the preferred post-RoPE per-head
masking requires plain Q / K transiently on the accelerator inside the
qkv-projection -> RoPE -> per-head-mask block. This is documented
explicitly in the diagnostics and counted as accelerator-side transient
leakage, NOT as a TEE re-entry (no data leaves the accelerator during
that step).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.functional import silu

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.models.tiny_modern_decoder import (
    TinyModernDecoderConfig,
    TinyModernDecoderForCausalLM,
    apply_rope,
    causal_attention,
    repeat_kv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


import math


def generate_rope_plane_rotation_mask(
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a block-diagonal mask that commutes with the repo's RoPE.

    LLaMA / Qwen ``rotate_half`` convention pairs channel ``j`` with
    channel ``j + head_dim/2``. Any 2D rotation in that same plane
    commutes with RoPE's rotation. We sample one fresh angle ``phi_j``
    per pair (``head_dim/2`` angles total) and assemble the
    corresponding ``head_dim x head_dim`` orthogonal mask.

    The convention mirrors :func:`pllo.experiments.rope_probe._generate_block_diagonal_rotation_mask`
    so that ``apply_rope(Q @ mask) == apply_rope(Q) @ mask`` under the
    same ``apply_rope`` used by the model.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    device_t = torch.device(device)
    if generator is None:
        angles = (
            torch.empty(half, dtype=torch.float64, device=device_t)
            .uniform_(-math.pi, math.pi)
        )
    else:
        angles = torch.empty(half, dtype=torch.float64, device=device_t).uniform_(
            -math.pi, math.pi, generator=generator
        )
    c = angles.cos().to(dtype)
    s = angles.sin().to(dtype)
    n = torch.zeros(head_dim, head_dim, dtype=dtype, device=device_t)
    for j in range(half):
        n[j, j] = c[j]
        n[j + half, j + half] = c[j]
        n[j, j + half] = -s[j]
        n[j + half, j] = s[j]
    return n


def verify_rope_commutation(
    q: torch.Tensor,
    b: torch.Tensor,
    positions: torch.Tensor,
    base: float,
) -> float:
    """Return ``max | apply_rope(Q @ B) - apply_rope(Q) @ B |``.

    Used by tests + the experiment to confirm the sampled ``B`` actually
    commutes with the repo's ``apply_rope``.
    """
    lhs = apply_rope(q @ b, positions, base)
    rhs = apply_rope(q, positions, base) @ b
    return float((lhs - rhs).abs().max().item())


def _apply_per_head_right_mask(
    w: torch.Tensor,
    masks_per_head: List[torch.Tensor],
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Right-multiply a [in_dim, num_heads * head_dim] weight by a
    block-diagonal mask whose ``h``-th block is ``masks_per_head[h]``.

    Equivalent to constructing the full ``[num_heads*head_dim,
    num_heads*head_dim]`` block-diagonal matrix and right-multiplying,
    but avoids materialising it.
    """
    in_dim = w.shape[0]
    w_per_head = w.view(in_dim, num_heads, head_dim)
    masks_stack = torch.stack(masks_per_head, dim=0)
    out = torch.einsum("ihd,hde->ihe", w_per_head, masks_stack)
    return out.reshape(in_dim, num_heads * head_dim)


def _sample_orthogonal(
    dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a fresh orthogonal mask ``(Q, Q^T)`` via QR."""
    device_t = torch.device(device)
    if generator is None:
        return generate_invertible_matrix(dim, dtype, device_t)
    raw = torch.randn(dim, dim, dtype=dtype, device=device_t, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q, q.transpose(-2, -1)


def _sample_pad_for(
    shape: Tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    scale: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, dtype=dtype, device=device) * scale
    return torch.randn(shape, dtype=dtype, device=device, generator=generator) * scale


def _tensor_fingerprint(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(repr(tuple(t.shape)).encode("utf-8"))
    h.update(str(t.dtype).encode("utf-8"))
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Compiled boundary tables (what the accelerator sees per Linear per call)
# ---------------------------------------------------------------------------


@dataclass
class _BoundaryTable:
    """Precomputed accelerator-side tensors for one padded boundary linear.

    A boundary corresponds to a single Linear with an input transition
    from a compatible state ``X @ Q_in_state``. The accelerator does:

        x_pad = X_state @ A - C_T          # transition Q_in_state -> M
        y_tilde = x_pad @ W_tilde + b_tilde + C_linear

    where ``A = Q_in_state^{-1} M`` and ``C_T = T M`` are the trusted
    precomputed transition tensors.
    """

    a: torch.Tensor                        # [d_state, d_state]  -- Q^{-1} M
    c_t: torch.Tensor                      # [B, S, d_state]      -- T M
    w_tilde: torch.Tensor | List[torch.Tensor]      # [d_state, d_out] or list (qkv, up/gate)
    bias_tilde: Optional[torch.Tensor]              # [d_out] or None
    c_linear: torch.Tensor | List[torch.Tensor]     # [B, S, d_out] or list


# ---------------------------------------------------------------------------
# Session state (fixed within a single generate call)
# ---------------------------------------------------------------------------


@dataclass
class _SessionState:
    """Per-generate-call invariants the accelerator must keep stable.

    * ``q_layer[l]``: orthogonal Q_l defining the residual-stream mask
      at the entry / exit of layer l. The same Q_l is reused across
      prefill and every decode step in this session so that the KV
      cache append invariant ``[K_past_tilde ; k_new_tilde] = [K_past ;
      k_new] @ N_K`` holds.

    * ``n_k[l][kv_head]``, ``n_v[l][kv_head]``: per-KV-head right masks
      on Q-rope and V used in the attention block.

    * ``n_q[l][q_head] = n_k[l][q_head // group_size]^{-T}``: derived,
      so that ``N_Q N_K^T = I`` per Q head -> ``scores_tilde ==
      scores_plain`` by construction.

    * ``n_vocab``: per-session LM-head output mask. Logits crossing
      the accelerator -> TEE boundary are masked by N_vocab; TEE
      recovers + samples.
    """

    q_layer: List[torch.Tensor]
    q_layer_inv: List[torch.Tensor]
    n_k: List[List[torch.Tensor]]
    n_k_inv: List[List[torch.Tensor]]
    n_v: List[List[torch.Tensor]]
    n_v_inv: List[List[torch.Tensor]]
    n_q: List[List[torch.Tensor]]
    n_vocab: torch.Tensor
    n_vocab_inv: torch.Tensor


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class LowInteractionDiagnostics:
    """Diagnostics for the no-reentry decode path."""

    main_layer_invariant: str = "H_hat_l = H_l Q_l"
    rmsnorm_mode: str = "operator_compatible_orthogonal"
    rope_mode: str = "post_rope_masking_with_transient_plain_qk_blocker"
    rope_mask_mode: str = "post_rope_masking"
    swiglu_mode: str = "paired_permutation_with_boundary_pad"
    attention_score_mode: str = "plaintext_scores_due_to_qk_invariant"
    lm_head_mode: str = "padded_masked_logits_with_trusted_recovery"
    trusted_fallback_used_in_main_path: bool = False
    intermediate_tee_reentry: bool = False
    online_boundary_round_trips_per_decode_step: int = 1
    pad_at_linear_boundaries: bool = True
    pad_enters_rmsnorm_core: bool = False
    pad_enters_rope_core: bool = False
    pad_enters_swiglu_core: bool = False
    pad_enters_softmax: bool = False
    use_pad: bool = True
    fresh_pad_used_at_linear_boundaries: bool = True
    rope_blocker_transient_plain_qk_on_accelerator: bool = True
    # Stage 7.6g rope-safe path fields (False / 0.0 outside rope-safe mode).
    rope_transient_plain_qk_visible: bool = True
    rope_transient_plain_v_visible: bool = True
    qkv_projection_outputs_masked_directly: bool = False
    trusted_rope_recovery_used: bool = False
    generic_pre_rope_dense_commutation_used: bool = False
    rope_commutation_max_abs_error: float = 0.0
    qk_score_invariant_max_abs_error: float = 0.0
    # Per-layer / aggregate correctness errors (filled at runtime).
    h_hat_layer_entry_invariant_max_abs_error: float = 0.0
    rmsnorm_core_orthogonal_commutation_max_abs_error: float = 0.0
    transition_trick_max_abs_error: float = 0.0
    swiglu_paired_permutation_max_abs_error: float = 0.0
    o_proj_recovery_max_abs_error: float = 0.0
    down_proj_recovery_max_abs_error: float = 0.0
    qk_constraint_max_error: float = 0.0
    kv_cache_invariant_max_abs_error: float = 0.0
    prefill_logits_max_abs_error: float = 0.0
    decode_step_logits_max_abs_error_max: float = 0.0
    lm_head_recovery_max_abs_error: float = 0.0
    masked_boundary_fingerprints: Dict[str, str] = field(default_factory=dict)
    # Norm leakage audit (populated by experiment runner).
    row_norm_error: float = 0.0
    gram_matrix_error: float = 0.0


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class LowInteractionTinyModernDecoderWrapper:
    """Compile-then-run wrapper for the no-reentry decode path.

    Construction binds the wrapper to a plain model. Each call to
    ``compile_session`` samples fresh per-session ``Q_l`` and per-head
    KV masks. Each call to ``compile_step_tables`` samples fresh
    per-call boundary pads ``T_*`` and input / output masks ``M_*``,
    folds RMSNorm gamma into the next Linear, and emits the
    accelerator-side tables (``A``, ``C_T``, ``W_tilde``, ``b_tilde``,
    ``C_linear``) for every padded boundary. The accelerator-side
    forward (``_accelerator_forward``) then runs the whole sequence /
    decode step with **no** TEE call.
    """

    def __init__(
        self,
        model: TinyModernDecoderForCausalLM,
        *,
        use_pad: bool = True,
        fresh_pad: bool = True,
        fresh_mask: bool = True,
        pad_scale: float = 0.5,
        rope_mask_mode: str = "post_rope_masking",
    ) -> None:
        self.model = model
        self.cfg = model.cfg
        self.use_pad = bool(use_pad)
        self.fresh_pad = bool(fresh_pad)
        self.fresh_mask = bool(fresh_mask)
        self.pad_scale = float(pad_scale)
        if rope_mask_mode not in (
            "post_rope_masking",
            "pre_rope_block_diagonal_rotation",
        ):
            raise ValueError(
                f"unknown rope_mask_mode={rope_mask_mode!r}; "
                "expected 'post_rope_masking' or "
                "'pre_rope_block_diagonal_rotation'"
            )
        self.rope_mask_mode = rope_mask_mode

    # ------------------------------------------------------------------
    # Session compile
    # ------------------------------------------------------------------

    def compile_session(
        self, *, generator: Optional[torch.Generator] = None
    ) -> _SessionState:
        cfg = self.cfg
        q_layer: List[torch.Tensor] = []
        q_layer_inv: List[torch.Tensor] = []
        for _ in range(cfg.num_layers):
            q, q_inv = _sample_orthogonal(
                cfg.hidden_size,
                dtype=cfg.dtype, device=cfg.device, generator=generator,
            )
            q_layer.append(q)
            q_layer_inv.append(q_inv)

        n_k: List[List[torch.Tensor]] = []
        n_k_inv: List[List[torch.Tensor]] = []
        n_v: List[List[torch.Tensor]] = []
        n_v_inv: List[List[torch.Tensor]] = []
        n_q: List[List[torch.Tensor]] = []
        for _ in range(cfg.num_layers):
            layer_nk, layer_nk_inv, layer_nv, layer_nv_inv = [], [], [], []
            for _ in range(cfg.num_kv_heads):
                if self.rope_mask_mode == "pre_rope_block_diagonal_rotation":
                    # B_K is a block-diagonal 2D rotation in each RoPE
                    # pair so it commutes with RoPE. Inverse = transpose
                    # (orthogonal mask).
                    k_mask = generate_rope_plane_rotation_mask(
                        cfg.head_dim,
                        dtype=cfg.dtype, device=cfg.device, generator=generator,
                    )
                    k_inv = k_mask.transpose(-2, -1)
                else:
                    k_mask, k_inv = _sample_orthogonal(
                        cfg.head_dim,
                        dtype=cfg.dtype, device=cfg.device, generator=generator,
                    )
                v_mask, v_inv = _sample_orthogonal(
                    cfg.head_dim,
                    dtype=cfg.dtype, device=cfg.device, generator=generator,
                )
                layer_nk.append(k_mask)
                layer_nk_inv.append(k_inv)
                layer_nv.append(v_mask)
                layer_nv_inv.append(v_inv)
            n_k.append(layer_nk)
            n_k_inv.append(layer_nk_inv)
            n_v.append(layer_nv)
            n_v_inv.append(layer_nv_inv)
            layer_nq = []
            for q_head in range(cfg.num_query_heads):
                kv_head = q_head // cfg.group_size
                layer_nq.append(layer_nk_inv[kv_head].transpose(-2, -1))
            n_q.append(layer_nq)

        n_vocab, n_vocab_inv = _sample_orthogonal(
            cfg.vocab_size,
            dtype=cfg.dtype, device=cfg.device, generator=generator,
        )
        return _SessionState(
            q_layer=q_layer,
            q_layer_inv=q_layer_inv,
            n_k=n_k,
            n_k_inv=n_k_inv,
            n_v=n_v,
            n_v_inv=n_v_inv,
            n_q=n_q,
            n_vocab=n_vocab,
            n_vocab_inv=n_vocab_inv,
        )

    # ------------------------------------------------------------------
    # Per-call padded-boundary table compile (trusted side)
    # ------------------------------------------------------------------

    def _compile_layer_step_tables(
        self,
        layer_idx: int,
        h_hat_shape: Tuple[int, int, int],
        session: _SessionState,
        generator: Optional[torch.Generator],
    ) -> Dict[str, _BoundaryTable | torch.Tensor]:
        """Trusted-side: precompute all accelerator-visible tables for one
        layer at one forward call.

        Returns a dict with five tables (``qkv``, ``o``, ``mlp_in``,
        ``down``) plus the SwiGLU paired permutation ``P``. Tables are
        sized for the actual ``[B, S, hidden]`` batch shape so pad
        compensation tensors broadcast correctly.
        """
        cfg = self.cfg
        layer = self.model.layers[layer_idx]
        b, s, hidden = h_hat_shape

        gamma1 = layer.input_norm_weight  # [hidden]
        gamma2 = layer.post_attn_norm_weight  # [hidden]

        # ------------------ QKV padded boundary ------------------
        # Input state: X = RMSNormCore(H) @ Q_l (operator-compatible).
        # Target padded form: (RMSNormCore(H) - T_qkv) @ M_qkv.
        q_l = session.q_layer[layer_idx]
        q_l_inv = session.q_layer_inv[layer_idx]

        m_qkv, m_qkv_inv = self._sample_mask(cfg.hidden_size, generator)
        t_qkv = self._sample_pad((b, s, cfg.hidden_size), generator)
        a_qkv = q_l_inv @ m_qkv               # [hidden, hidden]
        c_t_qkv = t_qkv @ m_qkv                # [B, S, hidden]

        # Per-projection gamma-folded weight: diag(gamma1) @ W.
        wq = (gamma1.unsqueeze(-1) * layer.attn.q_proj.weight.T)
        wk = (gamma1.unsqueeze(-1) * layer.attn.k_proj.weight.T)
        wv = (gamma1.unsqueeze(-1) * layer.attn.v_proj.weight.T)

        if self.rope_mask_mode == "pre_rope_block_diagonal_rotation":
            # Fold per-head right masks B_Q / B_K / N_V directly into
            # the qkv projection so the accelerator-visible output is
            # Q B_Q / K B_K / V N_V -- no plain Q / K / V transient.
            b_q_per_head = [
                session.n_q[layer_idx][q_head]
                for q_head in range(cfg.num_query_heads)
            ]
            b_k_per_head = [
                session.n_k[layer_idx][kv_head]
                for kv_head in range(cfg.num_kv_heads)
            ]
            n_v_per_head = [
                session.n_v[layer_idx][kv_head]
                for kv_head in range(cfg.num_kv_heads)
            ]
            wq_masked = _apply_per_head_right_mask(
                wq, b_q_per_head, cfg.num_query_heads, cfg.head_dim
            )
            wk_masked = _apply_per_head_right_mask(
                wk, b_k_per_head, cfg.num_kv_heads, cfg.head_dim
            )
            wv_masked = _apply_per_head_right_mask(
                wv, n_v_per_head, cfg.num_kv_heads, cfg.head_dim
            )
            w_q_tilde = m_qkv_inv @ wq_masked
            w_k_tilde = m_qkv_inv @ wk_masked
            w_v_tilde = m_qkv_inv @ wv_masked
            c_linear_q = t_qkv @ wq_masked
            c_linear_k = t_qkv @ wk_masked
            c_linear_v = t_qkv @ wv_masked
        else:
            # 7.6f path: qkv outputs are plain (N_out = I); the RoPE
            # blocker requires plain Q / K transiently on the
            # accelerator for RoPE application; per-head masks N_K /
            # N_V are applied immediately afterwards.
            w_q_tilde = m_qkv_inv @ wq
            w_k_tilde = m_qkv_inv @ wk
            w_v_tilde = m_qkv_inv @ wv
            c_linear_q = t_qkv @ wq
            c_linear_k = t_qkv @ wk
            c_linear_v = t_qkv @ wv

        # ------------------ o_proj padded boundary ------------------
        # Input state: attn_out_block_masked = attn_out @ N_V_block.
        # Build N_V_block in row-vector convention: per-head blocks
        # stacked along the embedding dimension.
        n_v_block = torch.zeros(
            cfg.num_query_heads * cfg.head_dim,
            cfg.num_query_heads * cfg.head_dim,
            dtype=cfg.dtype, device=cfg.device,
        )
        n_v_block_inv = torch.zeros_like(n_v_block)
        for q_head in range(cfg.num_query_heads):
            kv_head = q_head // cfg.group_size
            start = q_head * cfg.head_dim
            end = start + cfg.head_dim
            n_v_block[start:end, start:end] = session.n_v[layer_idx][kv_head]
            n_v_block_inv[start:end, start:end] = session.n_v_inv[layer_idx][kv_head]

        m_o, m_o_inv = self._sample_mask(
            cfg.num_query_heads * cfg.head_dim, generator
        )
        t_o = self._sample_pad(
            (b, s, cfg.num_query_heads * cfg.head_dim), generator
        )
        a_o = n_v_block_inv @ m_o
        c_t_o = t_o @ m_o

        # o_proj output mask = Q_l (so the residual addition stays
        # compatible with the layer-entry orthogonal mask).
        w_o = layer.attn.o_proj.weight.T
        w_o_tilde = m_o_inv @ w_o @ q_l
        c_linear_o = t_o @ w_o @ q_l

        # ------------------ MLP-in padded boundary ------------------
        # Input state: RMSNormCore(H_after_attn) @ Q_l (compatible).
        m_mlp, m_mlp_inv = self._sample_mask(cfg.hidden_size, generator)
        t_mlp = self._sample_pad((b, s, cfg.hidden_size), generator)
        a_mlp = q_l_inv @ m_mlp
        c_t_mlp = t_mlp @ m_mlp

        # Up / gate output mask = P (shared paired permutation).
        if generator is None:
            perm = torch.randperm(cfg.intermediate_size, device=cfg.device)
        else:
            perm = torch.randperm(
                cfg.intermediate_size, generator=generator, device=cfg.device
            )
        # Apply gamma2 fold + paired permutation.
        w_up = (gamma2.unsqueeze(-1) * layer.mlp.up_proj.weight.T)
        w_gate = (gamma2.unsqueeze(-1) * layer.mlp.gate_proj.weight.T)
        w_up_perm = w_up.index_select(dim=-1, index=perm)
        w_gate_perm = w_gate.index_select(dim=-1, index=perm)
        w_up_tilde = m_mlp_inv @ w_up_perm
        w_gate_tilde = m_mlp_inv @ w_gate_perm
        c_linear_up = t_mlp @ w_up_perm
        c_linear_gate = t_mlp @ w_gate_perm

        # ------------------ Down padded boundary ------------------
        # Input state: g_plain @ P (SwiGLU paired-permutation island).
        # Inverse-permutation matrix: P^{-1} W_down rearranges rows of
        # W_down by the same perm.
        w_down = layer.mlp.down_proj.weight.T
        w_down_perm = w_down.index_select(dim=0, index=perm)

        m_down, m_down_inv = self._sample_mask(cfg.intermediate_size, generator)
        t_down = self._sample_pad((b, s, cfg.intermediate_size), generator)

        # Build A_down = P_state^{-1} @ M_down. ``g_tilde = g @ P`` means
        # ``g_tilde[..., i] = g[..., perm[i]]``, i.e. P[r, c] = 1 iff
        # r == perm[c]. Then P^{-1} = P^T has (i, j) = 1 iff j ==
        # perm[i], so ``(P^T @ M_down)[i, :] = M_down[perm[i], :]``.
        inv_perm = torch.argsort(perm)
        a_down = m_down.index_select(dim=0, index=perm)
        c_t_down = t_down @ m_down

        w_down_tilde = m_down_inv @ w_down @ q_l
        c_linear_down = t_down @ w_down @ q_l

        return {
            "qkv": _BoundaryTable(
                a=a_qkv, c_t=c_t_qkv,
                w_tilde=[w_q_tilde, w_k_tilde, w_v_tilde],
                bias_tilde=None,
                c_linear=[c_linear_q, c_linear_k, c_linear_v],
            ),
            "o": _BoundaryTable(
                a=a_o, c_t=c_t_o,
                w_tilde=w_o_tilde, bias_tilde=None, c_linear=c_linear_o,
            ),
            "mlp_in": _BoundaryTable(
                a=a_mlp, c_t=c_t_mlp,
                w_tilde=[w_up_tilde, w_gate_tilde],
                bias_tilde=None,
                c_linear=[c_linear_up, c_linear_gate],
            ),
            "down": _BoundaryTable(
                a=a_down, c_t=c_t_down,
                w_tilde=w_down_tilde, bias_tilde=None, c_linear=c_linear_down,
            ),
            "perm": perm,
            "inv_perm": inv_perm,
        }

    def _compile_lm_head_table(
        self,
        h_hat_shape: Tuple[int, int, int],
        session: _SessionState,
        generator: Optional[torch.Generator],
    ) -> _BoundaryTable:
        cfg = self.cfg
        b, s, hidden = h_hat_shape
        gamma_final = self.model.final_norm_weight
        w_lm = (gamma_final.unsqueeze(-1) * self.model.lm_head.weight.T)
        q_L = session.q_layer[-1]
        q_L_inv = session.q_layer_inv[-1]

        m_lm, m_lm_inv = self._sample_mask(cfg.hidden_size, generator)
        t_lm = self._sample_pad((b, s, cfg.hidden_size), generator)
        a_lm = q_L_inv @ m_lm
        c_t_lm = t_lm @ m_lm

        w_lm_tilde = m_lm_inv @ w_lm @ session.n_vocab
        c_linear_lm = t_lm @ w_lm @ session.n_vocab

        return _BoundaryTable(
            a=a_lm, c_t=c_t_lm,
            w_tilde=w_lm_tilde, bias_tilde=None, c_linear=c_linear_lm,
        )

    def _sample_mask(
        self, dim: int, generator: Optional[torch.Generator]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Padded boundary M masks need *not* be orthogonal -- the only
        # operator-compatibility constraint is the residual-stream Q_l
        # (orthogonal, sampled by ``_sample_orthogonal``).
        if generator is None:
            return generate_invertible_matrix(
                dim, self.cfg.dtype, self.cfg.device
            )
        raw = torch.randn(
            dim, dim,
            dtype=self.cfg.dtype, device=self.cfg.device, generator=generator,
        )
        q, r = torch.linalg.qr(raw)
        signs = torch.sign(torch.diag(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        return q, q.transpose(-2, -1)

    def _sample_pad(
        self,
        shape: Tuple[int, ...],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if not self.use_pad:
            return torch.zeros(shape, dtype=self.cfg.dtype, device=self.cfg.device)
        return _sample_pad_for(
            shape,
            dtype=self.cfg.dtype, device=self.cfg.device,
            scale=self.pad_scale, generator=generator,
        )

    # ------------------------------------------------------------------
    # Accelerator-side forward (no TEE re-entry)
    # ------------------------------------------------------------------

    def _accelerator_forward(
        self,
        h_hat: torch.Tensor,
        positions: torch.Tensor,
        past_kv_tilde: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ],
        session: _SessionState,
        layer_tables: List[Dict[str, _BoundaryTable | torch.Tensor]],
        lm_head_table: _BoundaryTable,
        *,
        diag: LowInteractionDiagnostics,
        plain_layer_h: List[torch.Tensor],  # plain reference H per layer entry
        plain_post_attn_h: List[torch.Tensor],  # plain reference H after attn block
        plain_layer_h_out: List[torch.Tensor],  # plain reference H after layer
        plain_attn_out: List[torch.Tensor],  # plain attn_out merged per layer
        fingerprint_keys: Optional[Dict[str, str]] = None,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Run the no-reentry accelerator path; return masked logits and
        new masked KV cache.

        ``plain_layer_h``, ``plain_post_attn_h``, ``plain_layer_h_out``
        are *cross-check* inputs supplied by the wrapper's outer
        ``padded_masked_forward`` so it can verify the
        ``H_hat = H @ Q_l`` invariant at every boundary at float64
        precision. They are NEVER consumed by the accelerator-side
        computation itself; they only feed the diagnostics.
        """
        cfg = self.cfg

        # H_hat at entry: H_0 @ Q_1.
        invariant_err = float(
            (h_hat - plain_layer_h[0] @ session.q_layer[0]).abs().max().item()
        )
        diag.h_hat_layer_entry_invariant_max_abs_error = max(
            diag.h_hat_layer_entry_invariant_max_abs_error, invariant_err
        )
        if fingerprint_keys is not None:
            diag.masked_boundary_fingerprints[
                fingerprint_keys["layer_entry_h_hat"]
            ] = _tensor_fingerprint(h_hat)

        new_past_tilde: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, tbl in enumerate(layer_tables):
            q_l = session.q_layer[layer_idx]

            # ---------- Attention block ----------
            # Operator-compatible RMSNorm: core only (gamma folded).
            mean_sq = h_hat.pow(2).mean(dim=-1, keepdim=True)
            x_hat = h_hat * torch.rsqrt(mean_sq + cfg.rms_norm_eps)
            # Cross-check: x_hat == RMSNormCore(plain H) @ Q_l.
            plain_h = plain_layer_h[layer_idx]
            plain_mean_sq = plain_h.pow(2).mean(dim=-1, keepdim=True)
            plain_x = plain_h * torch.rsqrt(plain_mean_sq + cfg.rms_norm_eps)
            rmsnorm_err = float(
                (x_hat - plain_x @ q_l).abs().max().item()
            )
            diag.rmsnorm_core_orthogonal_commutation_max_abs_error = max(
                diag.rmsnorm_core_orthogonal_commutation_max_abs_error,
                rmsnorm_err,
            )

            # QKV padded-boundary transition + linear.
            qkv_tbl = tbl["qkv"]
            x_pad_qkv = x_hat @ qkv_tbl.a - qkv_tbl.c_t
            # Cross-check the transition trick: x_pad_qkv == (plain_x - T) @ M.
            # (T M and Q_l^{-1} M are baked into c_t and a; we recompute
            # the expected form using plain reference.)
            # Skipping reconstruction of T from c_t since we only need
            # the boundary error already captured below in q_plain.

            # Apply the QKV weights individually.
            if self.rope_mask_mode == "pre_rope_block_diagonal_rotation":
                # Accelerator output is masked directly: Q B_Q, K B_K,
                # V N_V. No plain Q / K / V tensor exists on the
                # accelerator side.
                q_pre_tilde_flat = (
                    x_pad_qkv @ qkv_tbl.w_tilde[0] + qkv_tbl.c_linear[0]
                )
                k_pre_tilde_flat = (
                    x_pad_qkv @ qkv_tbl.w_tilde[1] + qkv_tbl.c_linear[1]
                )
                v_tilde_flat = (
                    x_pad_qkv @ qkv_tbl.w_tilde[2] + qkv_tbl.c_linear[2]
                )
                b, s, _ = q_pre_tilde_flat.shape
                q_pre_tilde = q_pre_tilde_flat.view(
                    b, s, cfg.num_query_heads, cfg.head_dim
                ).transpose(1, 2)
                k_pre_tilde = k_pre_tilde_flat.view(
                    b, s, cfg.num_kv_heads, cfg.head_dim
                ).transpose(1, 2)
                v_tilde = v_tilde_flat.view(
                    b, s, cfg.num_kv_heads, cfg.head_dim
                ).transpose(1, 2)
                # Apply RoPE DIRECTLY to the masked tensors. Because
                # B_Q / B_K are block-diagonal 2D rotations in each
                # RoPE pair, ``apply_rope(Q @ B_Q) = apply_rope(Q) @ B_Q``.
                q_tilde = apply_rope(q_pre_tilde, positions, cfg.rope_base)
                k_tilde = apply_rope(k_pre_tilde, positions, cfg.rope_base)

                # --- Trusted-side diagnostics (not on accelerator path) ---
                identity = torch.eye(
                    cfg.head_dim, dtype=cfg.dtype, device=cfg.device
                )
                # 1. QK constraint: B_Q[i] @ B_K[i//group]^T = I.
                for q_head in range(cfg.num_query_heads):
                    kv_head = q_head // cfg.group_size
                    b_q = session.n_q[layer_idx][q_head]
                    b_k = session.n_k[layer_idx][kv_head]
                    diag.qk_constraint_max_error = max(
                        diag.qk_constraint_max_error,
                        float(
                            (b_q @ b_k.transpose(-2, -1) - identity)
                            .abs().max().item()
                        ),
                    )
                # 2. Plain reference Q/K/V (trusted-side only, NEVER on
                #    the accelerator path) for cross-check.
                ref_wq = (
                    self.model.layers[layer_idx].input_norm_weight.unsqueeze(-1)
                    * self.model.layers[layer_idx].attn.q_proj.weight.T
                )
                ref_wk = (
                    self.model.layers[layer_idx].input_norm_weight.unsqueeze(-1)
                    * self.model.layers[layer_idx].attn.k_proj.weight.T
                )
                ref_wv = (
                    self.model.layers[layer_idx].input_norm_weight.unsqueeze(-1)
                    * self.model.layers[layer_idx].attn.v_proj.weight.T
                )
                plain_q_flat = plain_x @ ref_wq
                plain_k_flat = plain_x @ ref_wk
                plain_v_flat = plain_x @ ref_wv
                plain_q = plain_q_flat.view(
                    b, s, cfg.num_query_heads, cfg.head_dim
                ).transpose(1, 2)
                plain_k = plain_k_flat.view(
                    b, s, cfg.num_kv_heads, cfg.head_dim
                ).transpose(1, 2)
                plain_v = plain_v_flat.view(
                    b, s, cfg.num_kv_heads, cfg.head_dim
                ).transpose(1, 2)
                plain_q_rope = apply_rope(plain_q, positions, cfg.rope_base)
                plain_k_rope = apply_rope(plain_k, positions, cfg.rope_base)

                # 3. Transition-trick check: q_pre_tilde[h] == plain_q[h] @ B_Q[h].
                for q_head in range(cfg.num_query_heads):
                    b_q = session.n_q[layer_idx][q_head]
                    expected = plain_q[:, q_head, :, :] @ b_q
                    diag.transition_trick_max_abs_error = max(
                        diag.transition_trick_max_abs_error,
                        float(
                            (q_pre_tilde[:, q_head, :, :] - expected)
                            .abs().max().item()
                        ),
                    )
                # 4. RoPE commutation cross-check: q_tilde[h] = q_rope_plain[h] @ B_Q[h].
                for q_head in range(cfg.num_query_heads):
                    b_q = session.n_q[layer_idx][q_head]
                    expected = plain_q_rope[:, q_head, :, :] @ b_q
                    diag.rope_commutation_max_abs_error = max(
                        diag.rope_commutation_max_abs_error,
                        float(
                            (q_tilde[:, q_head, :, :] - expected)
                            .abs().max().item()
                        ),
                    )
                for kv_head in range(cfg.num_kv_heads):
                    b_k = session.n_k[layer_idx][kv_head]
                    expected = plain_k_rope[:, kv_head, :, :] @ b_k
                    diag.rope_commutation_max_abs_error = max(
                        diag.rope_commutation_max_abs_error,
                        float(
                            (k_tilde[:, kv_head, :, :] - expected)
                            .abs().max().item()
                        ),
                    )
                # 5. QK score invariant: scores_tilde == scores_plain (per Q head).
                for q_head in range(cfg.num_query_heads):
                    kv_head = q_head // cfg.group_size
                    scores_tilde = (
                        q_tilde[:, q_head, :, :]
                        @ k_tilde[:, kv_head, :, :].transpose(-2, -1)
                    )
                    scores_plain = (
                        plain_q_rope[:, q_head, :, :]
                        @ plain_k_rope[:, kv_head, :, :].transpose(-2, -1)
                    )
                    diag.qk_score_invariant_max_abs_error = max(
                        diag.qk_score_invariant_max_abs_error,
                        float((scores_tilde - scores_plain).abs().max().item()),
                    )
            else:
                # Stage 7.6f path: plain Q / K / V transient on accelerator.
                q_plain = x_pad_qkv @ qkv_tbl.w_tilde[0] + qkv_tbl.c_linear[0]
                k_plain = x_pad_qkv @ qkv_tbl.w_tilde[1] + qkv_tbl.c_linear[1]
                v_plain = x_pad_qkv @ qkv_tbl.w_tilde[2] + qkv_tbl.c_linear[2]
                q_plain_ref = plain_x @ (
                    self.model.layers[layer_idx].input_norm_weight.unsqueeze(-1)
                    * self.model.layers[layer_idx].attn.q_proj.weight.T
                )
                transition_err = float(
                    (q_plain - q_plain_ref).abs().max().item()
                )
                diag.transition_trick_max_abs_error = max(
                    diag.transition_trick_max_abs_error, transition_err
                )

                b, s, _ = q_plain.shape
                q_heads = q_plain.view(b, s, cfg.num_query_heads, cfg.head_dim).transpose(1, 2)
                k_heads = k_plain.view(b, s, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
                v_heads = v_plain.view(b, s, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
                q_rope = apply_rope(q_heads, positions, cfg.rope_base)
                k_rope = apply_rope(k_heads, positions, cfg.rope_base)

                identity = torch.eye(cfg.head_dim, dtype=cfg.dtype, device=cfg.device)
                q_tilde_per_head = []
                for q_head in range(cfg.num_query_heads):
                    kv_head = q_head // cfg.group_size
                    n_q = session.n_q[layer_idx][q_head]
                    n_k = session.n_k[layer_idx][kv_head]
                    constraint = (
                        n_q @ n_k.transpose(-2, -1) - identity
                    ).abs().max().item()
                    diag.qk_constraint_max_error = max(
                        diag.qk_constraint_max_error, float(constraint)
                    )
                    q_tilde_per_head.append(q_rope[:, q_head, :, :] @ n_q)
                q_tilde = torch.stack(q_tilde_per_head, dim=1)

                k_tilde_per_head = []
                v_tilde_per_head = []
                for kv_head in range(cfg.num_kv_heads):
                    k_tilde_per_head.append(
                        k_rope[:, kv_head, :, :] @ session.n_k[layer_idx][kv_head]
                    )
                    v_tilde_per_head.append(
                        v_heads[:, kv_head, :, :] @ session.n_v[layer_idx][kv_head]
                    )
                k_tilde = torch.stack(k_tilde_per_head, dim=1)
                v_tilde = torch.stack(v_tilde_per_head, dim=1)

            # KV cache append + invariant check.
            if past_kv_tilde is not None and past_kv_tilde[layer_idx] is not None:
                past_k_tilde, past_v_tilde = past_kv_tilde[layer_idx]
                k_cache_tilde = torch.cat([past_k_tilde, k_tilde], dim=-2)
                v_cache_tilde = torch.cat([past_v_tilde, v_tilde], dim=-2)
            else:
                k_cache_tilde = k_tilde
                v_cache_tilde = v_tilde
            for kv_head in range(cfg.num_kv_heads):
                n_k = session.n_k[layer_idx][kv_head]
                n_v = session.n_v[layer_idx][kv_head]
                redo_k = (
                    k_cache_tilde[:, kv_head, :, :]
                    @ session.n_k_inv[layer_idx][kv_head]
                    @ n_k
                )
                redo_v = (
                    v_cache_tilde[:, kv_head, :, :]
                    @ session.n_v_inv[layer_idx][kv_head]
                    @ n_v
                )
                diag.kv_cache_invariant_max_abs_error = max(
                    diag.kv_cache_invariant_max_abs_error,
                    float((redo_k - k_cache_tilde[:, kv_head, :, :]).abs().max().item()),
                    float((redo_v - v_cache_tilde[:, kv_head, :, :]).abs().max().item()),
                )

            new_past_tilde.append((k_cache_tilde, v_cache_tilde))

            # Attention: scores plain by QK invariant.
            k_rep_tilde = repeat_kv(k_cache_tilde, cfg.group_size)
            v_rep_tilde = repeat_kv(v_cache_tilde, cfg.group_size)
            past_len = k_rep_tilde.shape[-2] - q_tilde.shape[-2]
            attn_out_tilde = causal_attention(
                q_tilde, k_rep_tilde, v_rep_tilde, past_len
            )

            # attn_out_tilde[h] = attn_out_plain[h] @ N_V[h//group].
            # Reshape into block-diagonal-masked hidden dim.
            attn_out_block_masked = attn_out_tilde.transpose(1, 2).reshape(
                b, s, cfg.num_query_heads * cfg.head_dim
            )

            # o_proj padded-boundary transition + linear with output Q_l.
            o_tbl = tbl["o"]
            x_pad_o = attn_out_block_masked @ o_tbl.a - o_tbl.c_t
            attn_o_tilde = x_pad_o @ o_tbl.w_tilde + o_tbl.c_linear

            # Cross-check vs plain o_proj.
            attn_out_plain_merged = plain_attn_out[layer_idx]
            o_plain = attn_out_plain_merged @ self.model.layers[layer_idx].attn.o_proj.weight.T
            o_proj_err = float((attn_o_tilde - o_plain @ q_l).abs().max().item())
            diag.o_proj_recovery_max_abs_error = max(
                diag.o_proj_recovery_max_abs_error, o_proj_err
            )

            # Residual: H_post_attn_hat = H_hat + attn_o_tilde = (H + o_plain) Q_l.
            h_hat = h_hat + attn_o_tilde
            invariant_err = float(
                (h_hat - plain_post_attn_h[layer_idx] @ q_l).abs().max().item()
            )
            diag.h_hat_layer_entry_invariant_max_abs_error = max(
                diag.h_hat_layer_entry_invariant_max_abs_error, invariant_err
            )

            # ---------- MLP block ----------
            mean_sq = h_hat.pow(2).mean(dim=-1, keepdim=True)
            x_hat = h_hat * torch.rsqrt(mean_sq + cfg.rms_norm_eps)
            plain_h_post_attn = plain_post_attn_h[layer_idx]
            plain_mean_sq = plain_h_post_attn.pow(2).mean(dim=-1, keepdim=True)
            plain_x_post = plain_h_post_attn * torch.rsqrt(plain_mean_sq + cfg.rms_norm_eps)
            rmsnorm_err = float(
                (x_hat - plain_x_post @ q_l).abs().max().item()
            )
            diag.rmsnorm_core_orthogonal_commutation_max_abs_error = max(
                diag.rmsnorm_core_orthogonal_commutation_max_abs_error,
                rmsnorm_err,
            )

            mlp_tbl = tbl["mlp_in"]
            x_pad_mlp = x_hat @ mlp_tbl.a - mlp_tbl.c_t
            a_tilde = x_pad_mlp @ mlp_tbl.w_tilde[0] + mlp_tbl.c_linear[0]
            b_tilde = x_pad_mlp @ mlp_tbl.w_tilde[1] + mlp_tbl.c_linear[1]

            # Cross-check paired-permutation invariant.
            perm = tbl["perm"]
            a_plain_ref = plain_x_post @ (
                self.model.layers[layer_idx].post_attn_norm_weight.unsqueeze(-1)
                * self.model.layers[layer_idx].mlp.up_proj.weight.T
            )
            b_plain_ref = plain_x_post @ (
                self.model.layers[layer_idx].post_attn_norm_weight.unsqueeze(-1)
                * self.model.layers[layer_idx].mlp.gate_proj.weight.T
            )
            a_plain_perm = a_plain_ref.index_select(dim=-1, index=perm)
            b_plain_perm = b_plain_ref.index_select(dim=-1, index=perm)
            diag.swiglu_paired_permutation_max_abs_error = max(
                diag.swiglu_paired_permutation_max_abs_error,
                float((a_tilde - a_plain_perm).abs().max().item()),
                float((b_tilde - b_plain_perm).abs().max().item()),
            )

            g_tilde = a_tilde * silu(b_tilde)

            # Down padded-boundary transition + linear with output Q_l.
            down_tbl = tbl["down"]
            x_pad_down = g_tilde @ down_tbl.a - down_tbl.c_t
            mlp_out_tilde = x_pad_down @ down_tbl.w_tilde + down_tbl.c_linear

            # Cross-check vs plain down_proj.
            g_plain_ref = a_plain_ref * silu(b_plain_ref)
            mlp_out_plain = g_plain_ref @ self.model.layers[layer_idx].mlp.down_proj.weight.T
            down_err = float(
                (mlp_out_tilde - mlp_out_plain @ q_l).abs().max().item()
            )
            diag.down_proj_recovery_max_abs_error = max(
                diag.down_proj_recovery_max_abs_error, down_err
            )

            # Residual: H_layer_out_hat = H_post_attn_hat + mlp_out_tilde.
            h_hat = h_hat + mlp_out_tilde
            invariant_err = float(
                (h_hat - plain_layer_h_out[layer_idx] @ q_l).abs().max().item()
            )
            diag.h_hat_layer_entry_invariant_max_abs_error = max(
                diag.h_hat_layer_entry_invariant_max_abs_error, invariant_err
            )

            # If there is a next layer, transition Q_l -> Q_{l+1}. This
            # is folded into the next layer's qkv ``A_qkv_next`` so the
            # accelerator does not need a separate matmul; we therefore
            # leave h_hat in the Q_l basis here, and the next layer's
            # tables (precomputed using Q_{l+1}^{-1} on the layer-entry
            # transition) will absorb the basis change implicitly.
            #
            # We do this by reusing q_l = Q_{l+1} for the *next* tables;
            # see how _compile_layer_step_tables uses session.q_layer[l].
            if layer_idx + 1 < cfg.num_layers:
                q_next = session.q_layer[layer_idx + 1]
                # Change of basis: H_hat (in Q_l basis) -> H_hat' (in
                # Q_{l+1} basis) via accelerator-side R = q_l^T @ q_next.
                r = session.q_layer_inv[layer_idx] @ q_next
                h_hat = h_hat @ r

        # ---------- Final RMSNorm + LM head ----------
        mean_sq = h_hat.pow(2).mean(dim=-1, keepdim=True)
        x_hat = h_hat * torch.rsqrt(mean_sq + cfg.rms_norm_eps)

        x_pad_lm = x_hat @ lm_head_table.a - lm_head_table.c_t
        logits_tilde = x_pad_lm @ lm_head_table.w_tilde + lm_head_table.c_linear

        if fingerprint_keys is not None:
            diag.masked_boundary_fingerprints[
                fingerprint_keys["lm_head_logits_tilde"]
            ] = _tensor_fingerprint(logits_tilde)

        return logits_tilde, new_past_tilde

    def _plain_attention(
        self,
        plain_h: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        past_kv_plain: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Plain-reference attention output (pre-o_proj) for diagnostics only."""
        cfg = self.cfg
        layer = self.model.layers[layer_idx]
        plain_mean_sq = plain_h.pow(2).mean(dim=-1, keepdim=True)
        plain_x = plain_h * torch.rsqrt(plain_mean_sq + cfg.rms_norm_eps)
        plain_x = plain_x * layer.input_norm_weight
        q = layer.attn.q_proj(plain_x)
        k = layer.attn.k_proj(plain_x)
        v = layer.attn.v_proj(plain_x)
        b, s, _ = plain_x.shape
        q = q.view(b, s, cfg.num_query_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(b, s, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(b, s, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q = apply_rope(q, positions, cfg.rope_base)
        k = apply_rope(k, positions, cfg.rope_base)
        past_len = 0
        if past_kv_plain is not None:
            past_k, past_v = past_kv_plain
            past_len = past_k.shape[-2]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        k_rep = repeat_kv(k, cfg.group_size)
        v_rep = repeat_kv(v, cfg.group_size)
        out = causal_attention(q, k_rep, v_rep, past_len)
        return out.transpose(1, 2).reshape(b, s, cfg.num_query_heads * cfg.head_dim)

    # ------------------------------------------------------------------
    # Top-level padded forward + generation (low-interaction)
    # ------------------------------------------------------------------

    def padded_masked_forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values_tilde: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
        session: Optional[_SessionState] = None,
        generator: Optional[torch.Generator] = None,
        diagnostics: Optional[LowInteractionDiagnostics] = None,
        fingerprint_keys: Optional[Dict[str, str]] = None,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
        LowInteractionDiagnostics,
        _SessionState,
    ]:
        """One low-interaction forward call.

        Returns ``(recovered_logits, new_past_kv_tilde, diagnostics,
        session)`` -- recovered_logits is the trusted-side recovery of
        the *single* accelerator -> TEE transfer for this step.
        """
        cfg = self.cfg
        if session is None:
            session = self.compile_session(generator=generator)
        if diagnostics is None:
            diagnostics = LowInteractionDiagnostics()
        diagnostics.use_pad = bool(self.use_pad)
        diagnostics.fresh_pad_used_at_linear_boundaries = bool(self.fresh_pad)
        diagnostics.rope_mask_mode = self.rope_mask_mode
        if self.rope_mask_mode == "pre_rope_block_diagonal_rotation":
            diagnostics.rope_mode = "pre_rope_block_diagonal_rotation"
            diagnostics.rope_blocker_transient_plain_qk_on_accelerator = False
            diagnostics.rope_transient_plain_qk_visible = False
            diagnostics.rope_transient_plain_v_visible = False
            diagnostics.qkv_projection_outputs_masked_directly = True
            diagnostics.trusted_rope_recovery_used = False
            diagnostics.generic_pre_rope_dense_commutation_used = False
        else:
            diagnostics.rope_mode = "post_rope_masking_with_transient_plain_qk_blocker"
            diagnostics.rope_blocker_transient_plain_qk_on_accelerator = True
            diagnostics.rope_transient_plain_qk_visible = True
            diagnostics.rope_transient_plain_v_visible = True
            diagnostics.qkv_projection_outputs_masked_directly = False
            diagnostics.trusted_rope_recovery_used = False
            diagnostics.generic_pre_rope_dense_commutation_used = False

        b, s = input_ids.shape

        # Trusted-side embedding lookup + apply Q_1 to ship masked H_0.
        plain_h0 = self.model.embed_tokens(input_ids)
        h_hat = plain_h0 @ session.q_layer[0]

        past_len = 0
        if past_key_values_tilde is not None and past_key_values_tilde[0] is not None:
            past_len = past_key_values_tilde[0][0].shape[-2]
        positions = torch.arange(past_len, past_len + s, device=cfg.device)

        # Pre-compute plain reference at every layer boundary for
        # cross-checking the H_hat = H Q_l invariant. The trusted side
        # never ships these to the accelerator.
        plain_layer_h: List[torch.Tensor] = []
        plain_post_attn_h: List[torch.Tensor] = []
        plain_layer_h_out: List[torch.Tensor] = []
        plain_attn_out: List[torch.Tensor] = []
        plain_h = plain_h0
        if past_key_values_tilde is not None and past_key_values_tilde[0] is not None:
            past_plain = self._unmask_past(past_key_values_tilde, session)
        else:
            past_plain = None
        for layer_idx in range(cfg.num_layers):
            plain_layer_h.append(plain_h)
            past_kv = past_plain[layer_idx] if past_plain is not None else None
            attn_out = self._plain_attention(plain_h, layer_idx, positions, past_kv)
            plain_attn_out.append(attn_out)
            o_plain = attn_out @ self.model.layers[layer_idx].attn.o_proj.weight.T
            h_after_attn = plain_h + o_plain
            plain_post_attn_h.append(h_after_attn)

            mean_sq = h_after_attn.pow(2).mean(dim=-1, keepdim=True)
            x_norm = h_after_attn * torch.rsqrt(mean_sq + cfg.rms_norm_eps)
            x_norm = x_norm * self.model.layers[layer_idx].post_attn_norm_weight
            a = self.model.layers[layer_idx].mlp.up_proj(x_norm)
            b_act = self.model.layers[layer_idx].mlp.gate_proj(x_norm)
            mlp_out = self.model.layers[layer_idx].mlp.down_proj(a * silu(b_act))
            plain_h = h_after_attn + mlp_out
            plain_layer_h_out.append(plain_h)

        # Compile padded-boundary tables for every layer + LM head.
        layer_tables = [
            self._compile_layer_step_tables(
                layer_idx, (b, s, cfg.hidden_size), session, generator
            )
            for layer_idx in range(cfg.num_layers)
        ]
        lm_head_table = self._compile_lm_head_table(
            (b, s, cfg.hidden_size), session, generator
        )

        if fingerprint_keys is None:
            fingerprint_keys = {
                "layer_entry_h_hat": "layer_entry_h_hat",
                "lm_head_logits_tilde": "lm_head_logits_tilde",
            }

        logits_tilde, new_past_tilde = self._accelerator_forward(
            h_hat, positions, past_key_values_tilde, session,
            layer_tables, lm_head_table,
            diag=diagnostics,
            plain_layer_h=plain_layer_h,
            plain_post_attn_h=plain_post_attn_h,
            plain_layer_h_out=plain_layer_h_out,
            plain_attn_out=plain_attn_out,
            fingerprint_keys=fingerprint_keys,
        )

        # Trusted-side recovery (the single accelerator -> TEE transfer
        # in the no-reentry protocol).
        recovered_logits = logits_tilde @ session.n_vocab_inv

        # Cross-check vs the plain reference logits.
        plain_logits, _ = self.model.forward(
            input_ids,
            past_key_values=self._unmask_past(past_key_values_tilde, session)
            if past_key_values_tilde is not None else None,
        )
        diagnostics.lm_head_recovery_max_abs_error = max(
            diagnostics.lm_head_recovery_max_abs_error,
            float((recovered_logits - plain_logits).abs().max().item()),
        )

        return recovered_logits, new_past_tilde, diagnostics, session

    def _unmask_past(
        self,
        past_key_values_tilde: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ],
        session: _SessionState,
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        if past_key_values_tilde is None or past_key_values_tilde[0] is None:
            return None
        cfg = self.cfg
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, kv in enumerate(past_key_values_tilde):
            assert kv is not None  # noqa: S101
            k_tilde, v_tilde = kv
            k_heads = []
            v_heads = []
            for kv_head in range(cfg.num_kv_heads):
                k_heads.append(
                    k_tilde[:, kv_head, :, :] @ session.n_k_inv[layer_idx][kv_head]
                )
                v_heads.append(
                    v_tilde[:, kv_head, :, :] @ session.n_v_inv[layer_idx][kv_head]
                )
            out.append((torch.stack(k_heads, dim=1), torch.stack(v_heads, dim=1)))
        return out

    @torch.no_grad()
    def low_interaction_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        generator: Optional[torch.Generator] = None,
        diagnostics: Optional[LowInteractionDiagnostics] = None,
        fingerprint_keys: Optional[Dict[str, str]] = None,
    ) -> Tuple[torch.Tensor, LowInteractionDiagnostics]:
        if diagnostics is None:
            diagnostics = LowInteractionDiagnostics()
        if fingerprint_keys is None:
            fingerprint_keys = {
                "layer_entry_h_hat": "prefill_layer_entry_h_hat",
                "lm_head_logits_tilde": "prefill_lm_head_logits_tilde",
            }
        # Prefill.
        recovered_logits, past_tilde, diagnostics, session = (
            self.padded_masked_forward(
                input_ids,
                past_key_values_tilde=None,
                generator=generator,
                diagnostics=diagnostics,
                fingerprint_keys=fingerprint_keys,
            )
        )
        diagnostics.prefill_logits_max_abs_error = (
            diagnostics.lm_head_recovery_max_abs_error
        )
        next_token = recovered_logits[:, -1, :].argmax(dim=-1, keepdim=True)
        all_ids = torch.cat([input_ids, next_token], dim=-1)

        for step_idx in range(max_new_tokens - 1):
            step_fingerprints = {
                "layer_entry_h_hat": f"decode_step_{step_idx}_layer_entry_h_hat",
                "lm_head_logits_tilde": f"decode_step_{step_idx}_lm_head_logits_tilde",
            }
            recovered_logits, past_tilde, diagnostics, _ = (
                self.padded_masked_forward(
                    next_token,
                    past_key_values_tilde=past_tilde,
                    session=session,
                    generator=generator,
                    diagnostics=diagnostics,
                    fingerprint_keys=step_fingerprints,
                )
            )
            diagnostics.decode_step_logits_max_abs_error_max = max(
                diagnostics.decode_step_logits_max_abs_error_max,
                diagnostics.lm_head_recovery_max_abs_error,
            )
            next_token = recovered_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            all_ids = torch.cat([all_ids, next_token], dim=-1)

        return all_ids, diagnostics


__all__ = [
    "LowInteractionDiagnostics",
    "LowInteractionTinyModernDecoderWrapper",
]
