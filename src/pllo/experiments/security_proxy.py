"""Stage 6.3 — Security proxy experiments (NOT a formal security proof).

Four lightweight proxies for downstream paper sections:

1. **Pad linkability** — pairwise cosine / L2 distances over four mask + pad
   policies, showing that fresh-mask + fresh-pad makes the GPU-visible
   tensor stream linkability-resistant under a naive observer model.
2. **Mask freshness / uniqueness audit** — counts fingerprint uniqueness for
   each mask kind under its declared reuse policy, *without* leaking real
   mask contents into the report.
3. **Boundary leakage accounting** — a static table partitioning every
   simulated tensor into ``gpu_visible`` vs ``trusted_only`` with a short
   note per item.
4. **Cache leakage proxy** — direct nearest-neighbour matching of plaintext
   K / V against obfuscated K_tilde = K N_K. Top-1 match rate vs the
   plaintext-to-plaintext baseline.

All four proxies are *upper bounds on adversary success under a naive
observer model*, not full security proofs. Adaptive attacks, learned
inversion attacks, side channels, LoRA-adapter extraction, and real TEE
isolation are explicitly out of scope.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.pad_generator import generate_pad
from pllo.model_zoo.base import torch_dtype_from_string


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SecurityProxyConfig:
    output_dir: str = "outputs"
    num_trials: int = 32
    batch_size: int = 2
    seq_len: int = 8
    hidden_size: int = 64
    pad_scale: float = 1.0
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 1337


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Hash of tensor bytes — used to count uniqueness without leaking contents."""
    buf = tensor.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(buf).hexdigest()


def _pairwise_cosine(stack: torch.Tensor) -> dict[str, float]:
    """Pairwise cosine similarity over ``stack[i]`` flattened to vectors."""
    if stack.shape[0] < 2:
        return {
            "mean_pairwise_cosine": 1.0,
            "max_pairwise_cosine": 1.0,
            "min_pairwise_cosine": 1.0,
        }
    flat = stack.reshape(stack.shape[0], -1).to(torch.float64)
    norm = flat / (flat.norm(dim=-1, keepdim=True).clamp_min(1e-30))
    sim = norm @ norm.T
    n = sim.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    vals = sim[iu[0], iu[1]]
    return {
        "mean_pairwise_cosine": float(vals.mean().item()),
        "max_pairwise_cosine": float(vals.max().item()),
        "min_pairwise_cosine": float(vals.min().item()),
    }


def _pairwise_l2(stack: torch.Tensor) -> dict[str, float]:
    if stack.shape[0] < 2:
        return {
            "mean_pairwise_l2": 0.0,
            "max_pairwise_l2": 0.0,
            "min_pairwise_l2": 0.0,
        }
    flat = stack.reshape(stack.shape[0], -1).to(torch.float64)
    n = flat.shape[0]
    diffs: list[float] = []
    for i, j in itertools.combinations(range(n), 2):
        diffs.append(float((flat[i] - flat[j]).norm().item()))
    diffs_t = torch.tensor(diffs, dtype=torch.float64)
    return {
        "mean_pairwise_l2": float(diffs_t.mean().item()),
        "max_pairwise_l2": float(diffs_t.max().item()),
        "min_pairwise_l2": float(diffs_t.min().item()),
    }


def _condition_numbers(masks: list[torch.Tensor]) -> dict[str, float]:
    conds = [
        float(torch.linalg.cond(m.to(torch.float64)).item()) for m in masks
    ]
    if not conds:
        return {
            "condition_number_mean": 0.0,
            "condition_number_max": 0.0,
            "condition_number_min": 0.0,
        }
    t = torch.tensor(conds, dtype=torch.float64)
    return {
        "condition_number_mean": float(t.mean().item()),
        "condition_number_max": float(t.max().item()),
        "condition_number_min": float(t.min().item()),
    }


# ---------------------------------------------------------------------------
# Proxy 1: pad vs no-pad linkability
# ---------------------------------------------------------------------------


def _pad_linkability(
    H: torch.Tensor,
    num_trials: int,
    hidden_size: int,
    pad_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Run the four pad/mask policy combinations and report pairwise stats."""
    H_flat = H.reshape(-1, hidden_size)

    n_fixed, _ = generate_invertible_matrix(hidden_size, dtype, device)

    def trials_fixed_mask_no_pad() -> torch.Tensor:
        stacks = [H_flat @ n_fixed for _ in range(num_trials)]
        return torch.stack(stacks, dim=0)

    def trials_fresh_mask_no_pad() -> torch.Tensor:
        stacks = []
        for _ in range(num_trials):
            n_i, _ = generate_invertible_matrix(hidden_size, dtype, device)
            stacks.append(H_flat @ n_i)
        return torch.stack(stacks, dim=0)

    def trials_fixed_mask_fresh_pad() -> torch.Tensor:
        stacks = []
        for _ in range(num_trials):
            t_i = generate_pad(
                tuple(H_flat.shape), dtype, device, pad_scale
            )
            stacks.append((H_flat - t_i) @ n_fixed)
        return torch.stack(stacks, dim=0)

    def trials_fresh_mask_fresh_pad() -> torch.Tensor:
        stacks = []
        for _ in range(num_trials):
            n_i, _ = generate_invertible_matrix(hidden_size, dtype, device)
            t_i = generate_pad(
                tuple(H_flat.shape), dtype, device, pad_scale
            )
            stacks.append((H_flat - t_i) @ n_i)
        return torch.stack(stacks, dim=0)

    strategies = {
        "fixed_mask_no_pad": trials_fixed_mask_no_pad,
        "fresh_mask_no_pad": trials_fresh_mask_no_pad,
        "fixed_mask_fresh_pad": trials_fixed_mask_fresh_pad,
        "fresh_mask_fresh_pad": trials_fresh_mask_fresh_pad,
    }

    per_strategy: dict[str, dict[str, Any]] = {}
    for name, fn in strategies.items():
        stack = fn()
        cos = _pairwise_cosine(stack)
        l2 = _pairwise_l2(stack)
        per_strategy[name] = {
            "num_trials": num_trials,
            **cos,
            **l2,
            "interpretation": _link_interpretation(name),
        }
    return {
        "per_strategy": per_strategy,
        "ranking_by_mean_cosine_descending": sorted(
            per_strategy.keys(),
            key=lambda k: per_strategy[k]["mean_pairwise_cosine"],
            reverse=True,
        ),
        "summary_note": (
            "fixed_mask_no_pad is the highest naive-linkability risk; fresh"
            " pad reduces the stability of the GPU-visible tensor across"
            " requests with identical plaintext input. This is a proxy under"
            " a naive observer model only, not a security proof."
        ),
    }


def _link_interpretation(name: str) -> str:
    return {
        "fixed_mask_no_pad": "Highest naive-linkability — same plaintext maps to identical GPU tensor.",
        "fresh_mask_no_pad": "Fresh mask alone removes the identity equivalence but preserves linear structure.",
        "fixed_mask_fresh_pad": "Fresh pad alone scrambles values but reuses the same mask across requests.",
        "fresh_mask_fresh_pad": "Fresh mask + fresh pad — lowest naive-linkability under this proxy.",
    }.get(name, "")


# ---------------------------------------------------------------------------
# Proxy 2: mask freshness / uniqueness audit
# ---------------------------------------------------------------------------


MASK_AUDIT_SPECS: tuple[dict[str, str], ...] = (
    {
        "mask_name": "input_mask",
        "expected_policy": "fresh_across_trials",
    },
    {
        "mask_name": "output_mask",
        "expected_policy": "fresh_across_trials",
    },
    {
        "mask_name": "pad",
        "expected_policy": "fresh_across_trials",
    },
    {
        "mask_name": "kv_cache_mask",
        "expected_policy": "reused_within_session_fresh_across_sessions",
    },
    {
        "mask_name": "encoder_memory_mask",
        "expected_policy": "reused_within_encoder_memory_fresh_across_sessions",
    },
)


def _audit_mask_set(
    name: str,
    expected_policy: str,
    generated_masks: list[torch.Tensor],
    reuse_signature_groups: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Count fingerprint uniqueness without leaking tensor contents.

    ``reuse_signature_groups`` lets the caller assert which trial indices
    are *expected* to share a fingerprint (e.g. within a generation
    session). Any duplicate outside those groups is reported as
    ``unexpected_reuse_count``.
    """
    fingerprints = [_tensor_fingerprint(t) for t in generated_masks]
    num_generated = len(fingerprints)
    unique_fingerprints = set(fingerprints)

    allowed_pairs: set[tuple[int, int]] = set()
    if reuse_signature_groups:
        for group in reuse_signature_groups:
            for i, j in itertools.combinations(sorted(group), 2):
                allowed_pairs.add((i, j))

    unexpected_reuse_count = 0
    for i, j in itertools.combinations(range(num_generated), 2):
        if fingerprints[i] == fingerprints[j] and (i, j) not in allowed_pairs:
            unexpected_reuse_count += 1

    cond_stats = _condition_numbers(generated_masks)

    return {
        "mask_name": name,
        "expected_policy": expected_policy,
        "num_generated": num_generated,
        "num_unique_fingerprints": len(unique_fingerprints),
        "unexpected_reuse_count": unexpected_reuse_count,
        **cond_stats,
        "fingerprint_algorithm": "sha256(float32_bytes)",
        "leakage_note": "Only fingerprint counts are recorded; no mask tensor contents are emitted.",
    }


def _mask_freshness_audit(
    num_trials: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    pad_scale: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    """Build a uniqueness audit for every declared mask kind."""
    input_masks = [
        generate_invertible_matrix(hidden_size, dtype, device)[0]
        for _ in range(num_trials)
    ]
    output_masks = [
        generate_invertible_matrix(hidden_size, dtype, device)[0]
        for _ in range(num_trials)
    ]
    pads = [
        generate_pad((batch_size * seq_len, hidden_size), dtype, device, pad_scale)
        for _ in range(num_trials)
    ]

    # KV cache mask: per-session reuse — simulate ``num_trials`` sessions, each
    # reusing the same mask twice (prefill + first decode), and require
    # uniqueness across sessions.
    kv_masks: list[torch.Tensor] = []
    kv_groups: list[list[int]] = []
    for s in range(num_trials):
        m = generate_invertible_matrix(hidden_size, dtype, device)[0]
        kv_masks.append(m)
        kv_masks.append(m.clone())
        kv_groups.append([2 * s, 2 * s + 1])

    # Encoder memory mask: per-encoder-memory reuse — simulate ``num_trials``
    # sessions each reusing the same encoder K/V mask twice.
    enc_masks: list[torch.Tensor] = []
    enc_groups: list[list[int]] = []
    for s in range(num_trials):
        m = generate_invertible_matrix(hidden_size, dtype, device)[0]
        enc_masks.append(m)
        enc_masks.append(m.clone())
        enc_groups.append([2 * s, 2 * s + 1])

    mask_sources: dict[str, dict[str, Any]] = {
        "input_mask": {"masks": input_masks, "groups": None},
        "output_mask": {"masks": output_masks, "groups": None},
        "pad": {"masks": pads, "groups": None},
        "kv_cache_mask": {"masks": kv_masks, "groups": kv_groups},
        "encoder_memory_mask": {"masks": enc_masks, "groups": enc_groups},
    }
    per_mask: list[dict[str, Any]] = []
    for spec in MASK_AUDIT_SPECS:
        src = mask_sources[spec["mask_name"]]
        per_mask.append(
            _audit_mask_set(
                spec["mask_name"],
                spec["expected_policy"],
                src["masks"],
                src["groups"],
            )
        )
    return {
        "per_mask": per_mask,
        "summary_note": (
            "Only sha256 fingerprints of mask tensors are kept. The condition"
            " number stats are aggregate-only and do not expose any single"
            " mask. Unexpected reuse counts duplicates outside the declared"
            " session-reuse policy."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 3: boundary leakage accounting (static)
# ---------------------------------------------------------------------------


GPU_VISIBLE_TENSORS: tuple[dict[str, str], ...] = (
    {
        "name": "obfuscated_input",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "X_tilde = (X - T) N_in or X N_in — never the plaintext input.",
    },
    {
        "name": "transformed_linear_weight",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "W_tilde = N_in_inv W N_out — depends on masks only.",
    },
    {
        "name": "transformed_lora_adapter",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "decoder_only",
        "leakage_note": "A_tilde, B_tilde — adapter content depends on rank mask.",
    },
    {
        "name": "compensation_terms",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "C = T W N_out is GPU-visible; depends on pad and masks, not on plaintext X.",
    },
    {
        "name": "obfuscated_q",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "Q_tilde = Q N_Q (per-head block-diag mask).",
    },
    {
        "name": "obfuscated_k",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "K_tilde = K N_K.",
    },
    {
        "name": "obfuscated_v",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "V_tilde = V N_V.",
    },
    {
        "name": "obfuscated_kv_cache",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "decoder_only",
        "leakage_note": "Stage 4.8 ObfuscatedGPT2KVCache: GPU keeps only masked K/V.",
    },
    {
        "name": "obfuscated_encoder_memory_cache",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "encoder_decoder",
        "leakage_note": "Stage 6.2 EncoderMemoryCache (probe-only) — GPU never sees K_plain/V_plain.",
    },
    {
        "name": "obfuscated_logits",
        "visibility": "gpu_visible",
        "contains_plaintext": "false",
        "architecture_scope": "decoder_only",
        "leakage_note": "logits_tilde = logits N_vocab; vocab mask is diagonal in Stage 4.7+.",
    },
)


TRUSTED_ONLY_TENSORS: tuple[dict[str, str], ...] = (
    {
        "name": "plaintext_input",
        "visibility": "trusted_only",
        "contains_plaintext": "true",
        "architecture_scope": "all",
        "leakage_note": "Plaintext input never leaves SimulatedTEE.",
    },
    {
        "name": "plaintext_hidden_state",
        "visibility": "trusted_only",
        "contains_plaintext": "true",
        "architecture_scope": "all",
        "leakage_note": "Intermediate plaintext hidden states stay on trusted side.",
    },
    {
        "name": "plaintext_logits",
        "visibility": "trusted_only",
        "contains_plaintext": "true",
        "architecture_scope": "all",
        "leakage_note": "Recovered logits Y = Y_tilde N_out_inv stay trusted-only.",
    },
    {
        "name": "sampling_result",
        "visibility": "trusted_only",
        "contains_plaintext": "true",
        "architecture_scope": "decoder_only",
        "leakage_note": "Argmax / token id produced by greedy decode is trusted-only.",
    },
    {
        "name": "masks",
        "visibility": "trusted_only",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "N_in, N_out, N_Q, N_K, N_V, N_vocab never sent to GPU.",
    },
    {
        "name": "mask_inverses",
        "visibility": "trusted_only",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "N_*_inv held by SimulatedTEE for input transform and output recovery.",
    },
    {
        "name": "pads",
        "visibility": "trusted_only",
        "contains_plaintext": "false",
        "architecture_scope": "all",
        "leakage_note": "T held trusted-only; only T W N_out is sent across as compensation.",
    },
    {
        "name": "plaintext_lora_adapter",
        "visibility": "trusted_only",
        "contains_plaintext": "true",
        "architecture_scope": "decoder_only",
        "leakage_note": "Plain LoRA A, B held trusted-only; only A_tilde, B_tilde cross to GPU.",
    },
    {
        "name": "optimizer_state",
        "visibility": "trusted_only",
        "contains_plaintext": "true",
        "architecture_scope": "all",
        "leakage_note": "No optimizer state crosses the boundary in this inference-only stage.",
    },
)


def _boundary_accounting() -> dict[str, Any]:
    gpu_visible = list(GPU_VISIBLE_TENSORS)
    trusted_only = list(TRUSTED_ONLY_TENSORS)
    return {
        "gpu_visible": gpu_visible,
        "trusted_only": trusted_only,
        "num_gpu_visible": len(gpu_visible),
        "num_trusted_only": len(trusted_only),
        "summary_notes": [
            "compensation_terms are GPU-visible transcript objects.",
            "security proxy does not prove semantic security.",
            "real TEE isolation is not implemented in this stage.",
        ],
    }


# ---------------------------------------------------------------------------
# Proxy 4: cache leakage proxy
# ---------------------------------------------------------------------------


def _cache_matching(
    plain: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    """Per-row nearest-neighbour matching of ``candidate`` against ``plain``.

    Both tensors are shaped ``[N, D]`` and the correct answer is
    ``candidate[i] ↔ plain[i]``. Cosine similarity is used as the
    matching metric.
    """
    plain64 = plain.to(torch.float64)
    cand64 = candidate.to(torch.float64)
    n = plain64.shape[0]

    pn = plain64 / plain64.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    cn = cand64 / cand64.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    sim = cn @ pn.T  # [N, N]
    correct = sim.diagonal()
    # Build a mask of off-diagonal entries to find the best wrong score per row.
    mask = torch.ones_like(sim, dtype=torch.bool)
    mask.fill_diagonal_(False)
    best_wrong = sim.masked_fill(~mask, float("-inf")).max(dim=-1).values

    top1_indices = sim.argmax(dim=-1)
    top1_match = (top1_indices == torch.arange(n)).to(torch.float64)

    # Mean rank of the correct answer (0 = top), averaged over rows.
    ranks = (sim > correct.unsqueeze(-1)).sum(dim=-1).to(torch.float64)
    return {
        "top1_match_rate": float(top1_match.mean().item()),
        "mean_correct_rank": float(ranks.mean().item()),
        "mean_cosine_correct_pair": float(correct.mean().item()),
        "mean_cosine_best_wrong_pair": float(best_wrong.mean().item()),
        "num_queries": int(n),
    }


def _cache_leakage_proxy(
    num_trials: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    """KV-cache and encoder-memory-cache leakage proxies."""
    # KV cache shapes: [num_trials, batch, heads, seq, head_dim] flattened to
    # [num_trials*batch*heads*seq, head_dim] per row.
    num_heads = max(1, hidden_size // 16)
    head_dim = hidden_size // num_heads
    n_k = generate_invertible_matrix(hidden_size, dtype, device)[0]
    n_v = generate_invertible_matrix(hidden_size, dtype, device)[0]

    k_plain_flat_rows: list[torch.Tensor] = []
    v_plain_flat_rows: list[torch.Tensor] = []
    k_tilde_flat_rows: list[torch.Tensor] = []
    v_tilde_flat_rows: list[torch.Tensor] = []
    for _ in range(num_trials):
        k = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=dtype, device=device, generator=generator,
        )
        v = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=dtype, device=device, generator=generator,
        )
        k_plain_flat_rows.append(k.reshape(-1, hidden_size))
        v_plain_flat_rows.append(v.reshape(-1, hidden_size))
        k_tilde_flat_rows.append(k.reshape(-1, hidden_size) @ n_k)
        v_tilde_flat_rows.append(v.reshape(-1, hidden_size) @ n_v)
    k_plain_flat = torch.cat(k_plain_flat_rows, dim=0)
    v_plain_flat = torch.cat(v_plain_flat_rows, dim=0)
    k_tilde_flat = torch.cat(k_tilde_flat_rows, dim=0)
    v_tilde_flat = torch.cat(v_tilde_flat_rows, dim=0)

    # Encoder memory cache uses different shapes (no autoregressive growth).
    e_plain_flat_rows: list[torch.Tensor] = []
    e_tilde_flat_rows: list[torch.Tensor] = []
    n_enc = generate_invertible_matrix(hidden_size, dtype, device)[0]
    for _ in range(num_trials):
        e = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=dtype, device=device, generator=generator,
        )
        e_plain_flat_rows.append(e.reshape(-1, hidden_size))
        e_tilde_flat_rows.append(e.reshape(-1, hidden_size) @ n_enc)
    e_plain_flat = torch.cat(e_plain_flat_rows, dim=0)
    e_tilde_flat = torch.cat(e_tilde_flat_rows, dim=0)

    results = {
        "kv_cache": {
            "plain_to_plain_baseline": _cache_matching(k_plain_flat, k_plain_flat),
            "obfuscated_to_plain": _cache_matching(k_plain_flat, k_tilde_flat),
            "obfuscated_to_plain_v": _cache_matching(v_plain_flat, v_tilde_flat),
            "shape_per_trial": [batch_size, num_heads, seq_len, head_dim],
            "num_trials": num_trials,
        },
        "encoder_memory_cache": {
            "plain_to_plain_baseline": _cache_matching(e_plain_flat, e_plain_flat),
            "obfuscated_to_plain": _cache_matching(e_plain_flat, e_tilde_flat),
            "shape_per_trial": [batch_size, seq_len, hidden_size],
            "num_trials": num_trials,
        },
        "interpretation_note": (
            "This is a direct nearest-neighbour matching proxy under cosine"
            " similarity. It does not implement adaptive or learned"
            " inversion attacks. A low top1_match_rate for obfuscated→plain"
            " only bounds the naive observer; the absence of a stronger"
            " attack here is not a guarantee."
        ),
    }
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_security_proxy_experiments(
    config: SecurityProxyConfig,
) -> dict[str, Any]:
    """Run the four security proxy experiments and return a structured report."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    H = torch.randn(
        config.batch_size,
        config.seq_len,
        config.hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    pad_linkability = _pad_linkability(
        H=H,
        num_trials=config.num_trials,
        hidden_size=config.hidden_size,
        pad_scale=config.pad_scale,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    mask_freshness = _mask_freshness_audit(
        num_trials=config.num_trials,
        hidden_size=config.hidden_size,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pad_scale=config.pad_scale,
        dtype=dtype,
        device=device,
    )
    boundary = _boundary_accounting()
    cache_leakage = _cache_leakage_proxy(
        num_trials=config.num_trials,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        hidden_size=config.hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    return {
        "config": asdict(config),
        "pad_linkability_proxy": pad_linkability,
        "mask_freshness_audit": mask_freshness,
        "boundary_leakage_accounting": boundary,
        "cache_leakage_proxy": cache_leakage,
        "global_limitations": [
            "These experiments are security proxies, not formal security proofs.",
            "They do not implement adaptive attacks.",
            "They do not implement learned inversion attacks.",
            "They do not evaluate real TEE isolation.",
            "They do not cover side channels.",
            "They do not prove LoRA adapter extraction resistance.",
        ],
    }


__all__ = [
    "GPU_VISIBLE_TENSORS",
    "MASK_AUDIT_SPECS",
    "SecurityProxyConfig",
    "TRUSTED_ONLY_TENSORS",
    "run_security_proxy_experiments",
]
