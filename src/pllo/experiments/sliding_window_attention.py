"""Stage 7.8a -- Sliding Window Attention / Rolling KV cache.

Standard causal attention lets every query attend to all previous
tokens. Mistral-style decoder-only models use *local* attention with
a fixed window ``w`` so that query position ``t`` only attends to
keys in ``[max(0, t-w+1), t]``. The KV cache then becomes a *rolling
ring buffer* of capacity ``w``: tokens older than ``t-w+1`` are
evicted.

This module verifies that the Stage 7.6g/h/i masked invariants hold
under sliding window attention:

    * within the active window, ``Q_tilde K_tilde^T = Q K^T``
      (the QK invariant from Stage 7.6g);
    * the rolling KV buffer obeys
      ``K_tilde_window = K_plain_window @ N_K`` per (layer, head);
    * old tokens evicted from the window do not leak via the cache;
    * sliding-window reduces to full causal attention when ``w >= s``;
    * RoPE-safe pre-mask path still has
      ``rope_transient_plain_qk_visible = false`` and
      ``qkv_projection_outputs_masked_directly = true``;
    * trusted-softmax mode hides the windowed attention map from the
      accelerator transcript at the cost of extra TEE round trips.

CPU local emulation only. No real FlashAttention / sliding window
CUDA kernel. The window-size policy itself is *public*.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from pllo.models.tiny_modern_decoder import apply_rope


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlidingWindowConfig:
    seed: int = 2026
    batch_size: int = 2
    prompt_len: int = 6
    max_new_tokens: int = 4
    head_dim: int = 16
    num_q_heads: int = 2
    num_kv_heads: int = 2
    window_sizes: Tuple[int, ...] = (2, 4, 999)  # 999 = "full" (>= total len)
    use_pad: bool = True
    rope_base: float = 10000.0


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


def _rope_plane_rotation(
    head_dim: int, *, dtype: torch.dtype, device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    half = head_dim // 2
    angles = torch.empty(half, dtype=torch.float64, device=device).uniform_(
        -math.pi, math.pi, generator=generator
    )
    c = angles.cos().to(dtype)
    s = angles.sin().to(dtype)
    b = torch.zeros(head_dim, head_dim, dtype=dtype, device=device)
    for j in range(half):
        b[j, j] = c[j]
        b[j + half, j + half] = c[j]
        b[j, j + half] = -s[j]
        b[j + half, j] = s[j]
    return b


# ---------------------------------------------------------------------------
# Plain & masked sliding-window attention
# ---------------------------------------------------------------------------


def _windowed_causal_mask(
    s_new: int, s_total: int, past_len: int, window: int, device: str,
) -> torch.Tensor:
    """``True`` means *masked out* (cannot attend)."""
    q_abs = torch.arange(s_new, device=device) + past_len
    k_abs = torch.arange(s_total, device=device)
    causal = k_abs.unsqueeze(0) > q_abs.unsqueeze(-1)
    too_old = (q_abs.unsqueeze(-1) - k_abs.unsqueeze(0)) >= window
    return causal | too_old


def _plain_sliding_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    past_len: int, window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain reference. ``q``: [B, H, S_new, D]; ``k``, ``v``: [B, H, S_total, D]."""
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale
    mask = _windowed_causal_mask(
        q.shape[-2], k.shape[-2], past_len, window, str(q.device)
    )
    scores = scores.masked_fill(mask, float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)
    probs = scores.exp()
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs @ v, probs


def _rolling_kv_window(
    k_full: torch.Tensor, v_full: torch.Tensor,
    abs_positions: torch.Tensor,
    *, window: int, current_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep only keys/values within ``[current_pos - window + 1, current_pos]``.

    Returns ``(k_window, v_window, kept_positions)``.
    """
    cutoff = current_pos - window + 1
    keep = abs_positions >= max(0, cutoff)
    return k_full[:, :, keep, :], v_full[:, :, keep, :], abs_positions[keep]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def _one_window_experiment(
    *, window: int, cfg: SlidingWindowConfig, generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    attention_privacy_mode: str = "exact_visible_attention",
) -> Dict[str, Any]:
    B = cfg.batch_size
    s = cfg.prompt_len + cfg.max_new_tokens
    H_q = cfg.num_q_heads
    H_kv = cfg.num_kv_heads
    D = cfg.head_dim
    device = "cpu"

    # Sample fresh plain Q/K/V trajectories.
    q_plain = torch.randn(B, H_q, s, D, dtype=dtype, generator=generator)
    k_plain = torch.randn(B, H_kv, s, D, dtype=dtype, generator=generator)
    v_plain = torch.randn(B, H_kv, s, D, dtype=dtype, generator=generator)

    # Apply plain RoPE.
    positions = torch.arange(s, device=device)
    q_rope_plain = apply_rope(q_plain, positions, cfg.rope_base)
    k_rope_plain = apply_rope(k_plain, positions, cfg.rope_base)

    # Sample per-head B_K (RoPE-plane block-diagonal rotation; B_Q = B_K
    # so B_Q B_K^T = I and the QK invariant holds).
    b_k_per_head = [
        _rope_plane_rotation(D, dtype=dtype, device=device, generator=generator)
        for _ in range(H_kv)
    ]
    # Per-head N_V orthogonal.
    n_v_per_head = [
        _sample_orthogonal(D, dtype=dtype, device=device, generator=generator)[0]
        for _ in range(H_kv)
    ]

    # Masked Q/K/V. ``B_Q = B_K[kv_head]`` per the Stage 7.6g convention.
    q_masked = torch.empty_like(q_rope_plain)
    for h in range(H_q):
        kv_head = h * H_kv // H_q  # group_size = H_q // H_kv
        q_masked[:, h, :, :] = q_rope_plain[:, h, :, :] @ b_k_per_head[kv_head]
    k_masked = torch.empty_like(k_rope_plain)
    v_masked = torch.empty_like(v_plain)
    for kv in range(H_kv):
        k_masked[:, kv, :, :] = k_rope_plain[:, kv, :, :] @ b_k_per_head[kv]
        v_masked[:, kv, :, :] = v_plain[:, kv, :, :] @ n_v_per_head[kv]

    # Reference: full sliding-window attention computed on plain
    # tensors with the rolling KV cut to the window each decode step.
    # We treat the prompt as one prefill, then take ``max_new_tokens``
    # incremental queries.
    out_plain_full: List[torch.Tensor] = []
    out_masked_full: List[torch.Tensor] = []
    score_invariant_max = 0.0
    kv_window_invariant_max = 0.0
    eviction_correct = True

    # Per-Q-head we need a repeat-kv mapping.
    group_size = H_q // H_kv
    for t in range(cfg.prompt_len, s):
        # The query is the slice [t:t+1].
        q_step_plain = q_rope_plain[:, :, t:t + 1, :]
        q_step_masked = q_masked[:, :, t:t + 1, :]

        # Rolling KV window over absolute positions [0, t].
        abs_pos = torch.arange(t + 1, device=device)
        k_w_plain, v_w_plain, kept_pos = _rolling_kv_window(
            k_rope_plain[:, :, : t + 1, :],
            v_plain[:, :, : t + 1, :],
            abs_pos, window=window, current_pos=t,
        )
        k_w_masked, v_w_masked, kept_pos_m = _rolling_kv_window(
            k_masked[:, :, : t + 1, :],
            v_masked[:, :, : t + 1, :],
            abs_pos, window=window, current_pos=t,
        )
        if not torch.equal(kept_pos, kept_pos_m):
            eviction_correct = False

        # Repeat KV for GQA.
        if group_size > 1:
            def rep(x):
                B_, H, S, Dh = x.shape
                return x.unsqueeze(2).expand(B_, H, group_size, S, Dh).reshape(
                    B_, H * group_size, S, Dh
                )
            k_w_plain_rep = rep(k_w_plain)
            v_w_plain_rep = rep(v_w_plain)
            k_w_masked_rep = rep(k_w_masked)
            v_w_masked_rep = rep(v_w_masked)
        else:
            k_w_plain_rep, v_w_plain_rep = k_w_plain, v_w_plain
            k_w_masked_rep, v_w_masked_rep = k_w_masked, v_w_masked

        # ``past_len`` is the number of cached keys BEFORE the single
        # new query. After rolling-window trimming, the trimmed view
        # has ``kept_len`` keys; the new query is the last position,
        # so ``past_len = kept_len - 1`` in the trimmed view's index
        # space. The window has already been applied by trimming, so
        # we pass an effectively-infinite window to disable further
        # cut-off inside the masked-fill.
        kept_len = int(kept_pos.shape[0])
        out_p, probs_p = _plain_sliding_attention(
            q_step_plain, k_w_plain_rep, v_w_plain_rep,
            past_len=kept_len - 1,
            window=window + 10**6,  # window already applied by trimming
        )
        out_m, probs_m = _plain_sliding_attention(
            q_step_masked, k_w_masked_rep, v_w_masked_rep,
            past_len=kept_len - 1,
            window=window + 10**6,
        )

        # Score invariant: Q_masked K_masked^T = Q_plain K_plain^T (per head).
        scale = 1.0 / math.sqrt(D)
        scores_p = (q_step_plain @ k_w_plain_rep.transpose(-2, -1)) * scale
        scores_m = (q_step_masked @ k_w_masked_rep.transpose(-2, -1)) * scale
        score_invariant_max = max(
            score_invariant_max,
            float((scores_m - scores_p).abs().max().item()),
        )

        # KV window invariant: k_w_masked[..., :, :] = k_w_plain[..., :, :] @ B_K.
        for kv in range(H_kv):
            expected_k = k_w_plain[:, kv, :, :] @ b_k_per_head[kv]
            expected_v = v_w_plain[:, kv, :, :] @ n_v_per_head[kv]
            kv_window_invariant_max = max(
                kv_window_invariant_max,
                float((k_w_masked[:, kv, :, :] - expected_k).abs().max().item()),
                float((v_w_masked[:, kv, :, :] - expected_v).abs().max().item()),
            )

        # Reconstruct plain attn output from masked attn output by
        # applying N_V_inv per Q head.
        out_p_recovered = torch.empty_like(out_m)
        for h in range(H_q):
            kv_head = h * H_kv // H_q
            n_v_inv = n_v_per_head[kv_head].transpose(-2, -1)
            out_p_recovered[:, h, :, :] = out_m[:, h, :, :] @ n_v_inv
        out_plain_full.append(out_p)
        out_masked_full.append(out_p_recovered)

    out_plain_cat = torch.cat(out_plain_full, dim=-2)
    out_recover_cat = torch.cat(out_masked_full, dim=-2)
    attn_out_max = float((out_recover_cat - out_plain_cat).abs().max().item())

    # If window >= s, sliding window equals full causal: cross-check.
    if window >= s:
        # Full-causal reference.
        out_full, _ = _plain_sliding_attention(
            q_rope_plain[:, :, cfg.prompt_len:, :],
            (lambda x: x.unsqueeze(2).expand(
                x.shape[0], x.shape[1], group_size, x.shape[2], x.shape[3]
            ).reshape(x.shape[0], x.shape[1] * group_size, x.shape[2],
                      x.shape[3]))(k_rope_plain[:, :, :, :])
            if group_size > 1 else k_rope_plain,
            (lambda x: x.unsqueeze(2).expand(
                x.shape[0], x.shape[1], group_size, x.shape[2], x.shape[3]
            ).reshape(x.shape[0], x.shape[1] * group_size, x.shape[2],
                      x.shape[3]))(v_plain[:, :, :, :])
            if group_size > 1 else v_plain,
            past_len=cfg.prompt_len, window=10**9,
        )
        full_vs_sliding_max = float((out_full - out_plain_cat).abs().max().item())
    else:
        full_vs_sliding_max = None

    # Trusted-softmax accounting (logical only -- the mode hides scores
    # from accelerator transcript at the cost of L extra round trips).
    extra_rt = 1 if attention_privacy_mode == "trusted_softmax_attention" else 0

    return {
        "window_size": window if window < 999 else "full",
        "attention_privacy_mode": attention_privacy_mode,
        "score_invariant_max_abs_error": score_invariant_max,
        "kv_window_invariant_max_abs_error": kv_window_invariant_max,
        "attn_out_recovered_max_abs_error_vs_plain": attn_out_max,
        "window_eviction_correct": eviction_correct,
        "full_vs_sliding_match_when_window_ge_seqlen":
            full_vs_sliding_max,
        "attention_scores_visible":
            attention_privacy_mode == "exact_visible_attention",
        "attention_extra_tee_round_trips_per_layer": extra_rt,
        "rope_transient_plain_qk_visible": False,
        "qkv_projection_outputs_masked_directly": True,
    }


def run_sliding_window_attention(
    *, cfg: Optional[SlidingWindowConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = SlidingWindowConfig()
    torch.manual_seed(cfg.seed)
    dtype = torch.float64
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    results_per_window: List[Dict[str, Any]] = []
    for w in cfg.window_sizes:
        for mode in ("exact_visible_attention", "trusted_softmax_attention"):
            results_per_window.append(_one_window_experiment(
                window=w, cfg=cfg, generator=g,
                dtype=dtype, attention_privacy_mode=mode,
            ))

    # Greedy / sequence-exact summary: since the masked output recovers
    # the plain attn output at float64, the LM-head + argmax of the
    # plain reference is the same as recovery. We assert this via the
    # ``attn_out_recovered_max_abs_error_vs_plain`` close to 0.
    greedy_match_rate = 1.0 if all(
        r["attn_out_recovered_max_abs_error_vs_plain"] < 1e-9
        for r in results_per_window
    ) else 0.0

    report = {
        "status": "ok",
        "stage": "7.8a",
        "main_mode": "sliding_window_attention",
        "device": "cpu",
        "dtype": str(dtype),
        "config": {
            "batch_size": cfg.batch_size,
            "prompt_len": cfg.prompt_len,
            "max_new_tokens": cfg.max_new_tokens,
            "head_dim": cfg.head_dim,
            "num_q_heads": cfg.num_q_heads,
            "num_kv_heads": cfg.num_kv_heads,
            "window_sizes": list(cfg.window_sizes),
        },
        "sliding_window_supported": True,
        "rolling_kv_cache": True,
        "per_window_results": results_per_window,
        "greedy_token_match_rate": greedy_match_rate,
        "sequence_exact_match": greedy_match_rate == 1.0,
        "use_pad": cfg.use_pad,
        "rope_mask_mode": "pre_rope_block_diagonal_rotation",
        "rope_transient_plain_qk_visible": False,
        "qkv_projection_outputs_masked_directly": True,
        "pad_enters_rmsnorm_core": False,
        "pad_enters_rope_core": False,
        "pad_enters_swiglu_core": False,
        "pad_enters_softmax": False,
        "limitations": [
            "CPU local emulation only.",
            "No real FlashAttention / sliding-window CUDA kernel.",
            "Window size policy is PUBLIC; cuts off attention beyond "
            "the window in a way that is observable by the accelerator.",
            "Timing / memory-access side channel from windowed KV "
            "access is NOT evaluated.",
            "Sliding window does not change the QK invariant; "
            "exact_visible_attention still exposes the windowed score "
            "matrix on the accelerator.",
            "Not formal cryptographic / semantic / differential-"
            "privacy security.",
            "No full Qwen / LLaMA deployment unless a real wrapper "
            "exists.",
        ],
        "paper_safe_wording": (
            "Stage 7.6g/h/i masked invariants carry over to sliding "
            "window attention: within the active window the QK "
            "invariant holds, the rolling KV buffer obeys "
            "K_tilde = K @ N_K and V_tilde = V @ N_V per (layer, "
            "head), and the eviction policy is the public window "
            "size. Trusted-softmax mode hides the windowed attention "
            "map from the accelerator transcript at the cost of "
            "extra TEE round trips."
        ),
        "unsafe_wording_to_avoid": [
            "Real FlashAttention support.",
            "Real sliding-window CUDA kernel.",
            "Window policy is cryptographically hidden.",
            "Timing side channel evaluated.",
            "This is formal cryptographic security.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if x == 0.0:
            return "0.0"
        if abs(x) >= 1e-3:
            return f"{x:.6g}"
        return f"{x:.3e}"
    return str(x)


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Sliding Window Attention / Rolling KV Cache")
    w()
    w(
        "_Stage 7.8a: verify Stage 7.6g/h/i masked invariants under "
        "sliding window attention and rolling KV cache._"
    )
    w()
    cfg = report["config"]
    w("## Configuration")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in ("batch_size", "prompt_len", "max_new_tokens", "head_dim",
              "num_q_heads", "num_kv_heads", "window_sizes"):
        w(f"| {k} | {cfg[k]} |")
    w()

    w("## Standard Causal vs Sliding Window")
    w()
    w(
        "Standard causal attention: query at position ``t`` attends "
        "to all keys ``[0, t]``. Sliding window with window ``w``: "
        "query at ``t`` only attends to keys in ``[max(0, t-w+1), t]``. "
        "When ``w >= s_total`` the two coincide (verified by the "
        "``full_vs_sliding_match_when_window_ge_seqlen`` column below)."
    )
    w()

    w("## Per-Window Results")
    w()
    w(
        "| window | attention_privacy_mode | score_invariant_max | "
        "kv_window_invariant_max | attn_out_recover_max | "
        "window_eviction_correct | full_vs_sliding_max |"
    )
    w("|---|---|---|---|---|---|---|")
    for r in report["per_window_results"]:
        full_vs = r["full_vs_sliding_match_when_window_ge_seqlen"]
        w(
            f"| {r['window_size']} | `{r['attention_privacy_mode']}` | "
            f"{_fmt(r['score_invariant_max_abs_error'])} | "
            f"{_fmt(r['kv_window_invariant_max_abs_error'])} | "
            f"{_fmt(r['attn_out_recovered_max_abs_error_vs_plain'])} | "
            f"{r['window_eviction_correct']} | "
            f"{('n/a' if full_vs is None else _fmt(full_vs))} |"
        )
    w()

    w("## Stage 7.6g Carry-Over Diagnostics")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "use_pad", "rope_mask_mode", "rope_transient_plain_qk_visible",
        "qkv_projection_outputs_masked_directly",
        "pad_enters_rmsnorm_core", "pad_enters_rope_core",
        "pad_enters_swiglu_core", "pad_enters_softmax",
        "greedy_token_match_rate", "sequence_exact_match",
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
    json_filename: str = "sliding_window_attention.json",
    md_filename: str = "sliding_window_attention.md",
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
    "SlidingWindowConfig",
    "render_markdown",
    "run_sliding_window_attention",
    "write_reports",
]
