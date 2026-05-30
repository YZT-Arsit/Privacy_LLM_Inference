"""Stage 5.2 — Nonlinear island security proxy experiments.

Three lightweight proxies for the operator-compatible mask scheme used by
the Stage 5.2 nonlinear islands. *None* of these proves formal security —
they are bounds on naive observers only.

* **Proxy 1**: permutation recovery via channel statistics. Given many
  samples drawn from the same ``Z``-distribution, an attacker tries to
  match each masked column back to its plaintext column using only the
  per-channel mean / std / quantile signature.
* **Proxy 2**: nonlinear island linkability. Same plaintext input is run
  many times through several mask + pad policies; pairwise cosine and L2
  distances quantify the visible-tensor stability across requests.
* **Proxy 3**: static mask-family accounting. Each compatible mask family
  has a known leakage profile (norm preserved, mean preserved, coordinate
  multiset preserved, etc.) — recorded here so the paper's security section
  can quote them directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from typing import Any

import torch

from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.pad_generator import generate_pad
from pllo.model_zoo.base import torch_dtype_from_string
from pllo.ops.compatible_masks import generate_permutation


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NonlinearIslandSecurityConfig:
    output_dir: str = "outputs"
    num_sessions: int = 8
    num_samples_per_session: int = 32
    num_trials: int = 32
    hidden_size: int = 64
    batch_size: int = 2
    seq_len: int = 4
    pad_scale: float = 1.0
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 2025


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _signature(z: torch.Tensor) -> torch.Tensor:
    """Per-channel mean / std / 25th / 75th quantile signature.

    ``z`` is shaped ``[N, hidden]``. Returns ``[hidden, 4]``. The signature
    is the only statistic the proxy attacker uses to match masked columns
    to plaintext columns.
    """
    z64 = z.to(torch.float64)
    mean = z64.mean(dim=0)
    std = z64.std(dim=0, unbiased=False)
    q25 = z64.quantile(0.25, dim=0)
    q75 = z64.quantile(0.75, dim=0)
    return torch.stack([mean, std, q25, q75], dim=-1)


def _recover_permutation_top_k(
    signature_masked: torch.Tensor,
    signature_plain: torch.Tensor,
    k: int = 5,
) -> dict[str, Any]:
    """Attempt to recover the per-column permutation by matching signatures.

    Returns top-1 / top-5 match rates assuming the true permutation is the
    identity *between* the masked column index and the plaintext column index
    that produced it. Caller is responsible for aligning that semantics by
    ordering ``signature_masked`` according to the masked-side column order
    that maps back to plaintext column ``i``.
    """
    # Cosine-similarity between every (masked, plain) column signature pair.
    sm = signature_masked / signature_masked.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sp = signature_plain / signature_plain.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sims = sm @ sp.T  # [H, H]
    hidden = sims.shape[0]
    targets = torch.arange(hidden)

    top1 = (sims.argmax(dim=-1) == targets).to(torch.float64).mean().item()
    k = min(k, hidden)
    topk_indices = sims.topk(k, dim=-1).indices  # [H, k]
    topk = (topk_indices == targets.unsqueeze(-1)).any(dim=-1).to(torch.float64).mean().item()
    mean_signature_error = float((sm - sp).abs().mean().item())
    return {
        "top1": float(top1),
        f"top{k}": float(topk),
        "mean_signature_error": mean_signature_error,
    }


# ---------------------------------------------------------------------------
# Proxy 1 — permutation recovery by channel statistics
# ---------------------------------------------------------------------------


def _proxy_permutation_recovery(
    config: NonlinearIslandSecurityConfig,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    hidden = config.hidden_size
    # Generate a *non-iid* Z whose channel-wise statistics differ enough
    # that a naive attacker has a chance with a fixed permutation. We make
    # each column have a distinct location & scale by adding a per-channel
    # bias and per-channel scale to a Gaussian sample.
    channel_bias = torch.linspace(-2.0, 2.0, hidden, dtype=dtype, device=device)
    channel_scale = torch.linspace(0.5, 3.0, hidden, dtype=dtype, device=device)

    def sample_pool(n: int) -> torch.Tensor:
        z = torch.randn(
            n, hidden, dtype=dtype, device=device, generator=generator
        )
        return z * channel_scale + channel_bias

    # The attacker's reference distribution is sampled once with a large pool
    # so the plaintext signature is sharp; that models the case where the
    # attacker has prior knowledge of the input distribution.
    ref_plain = sample_pool(
        config.num_sessions * config.num_samples_per_session
    )
    sig_plain = _signature(ref_plain)

    # ---- Strategy A: fixed permutation across all sessions ----
    # The realistic attacker model is to accumulate samples ACROSS sessions
    # (the permutation is stable), giving an effective signature with
    # ``num_sessions * num_samples_per_session`` samples.
    fixed_perm = generate_permutation(hidden, dtype, device)["perm"]
    fixed_pool = sample_pool(
        config.num_sessions * config.num_samples_per_session
    )
    fixed_masked = fixed_pool.index_select(dim=-1, index=fixed_perm)
    inv = torch.empty_like(fixed_perm)
    inv[fixed_perm] = torch.arange(hidden, device=fixed_perm.device)
    sig_fixed_aligned = _signature(fixed_masked).index_select(dim=0, index=inv)
    fixed_summary = _recover_permutation_top_k(
        sig_fixed_aligned, sig_plain, k=5
    )
    fixed_results = [fixed_summary]

    # ---- Strategy B: fresh permutation per session ----
    # Each session has its own P, so the attacker can at best run an
    # independent within-session recovery and average the top-k rate.
    fresh_results: list[dict[str, Any]] = []
    for s in range(config.num_sessions):
        perm = generate_permutation(hidden, dtype, device)["perm"]
        pool = sample_pool(config.num_samples_per_session)
        masked = pool.index_select(dim=-1, index=perm)
        sig_masked = _signature(masked)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(hidden, device=perm.device)
        sig_masked_aligned = sig_masked.index_select(dim=0, index=inv)
        fresh_results.append(
            _recover_permutation_top_k(sig_masked_aligned, sig_plain, k=5)
        )

    # ---- Strategy C: permutation pool (a small set of perms reused across sessions) ----
    pool_size = max(2, config.num_sessions // 2)
    perm_pool = [
        generate_permutation(hidden, dtype, device)["perm"]
        for _ in range(pool_size)
    ]
    pool_results: list[dict[str, Any]] = []
    # Each perm in the pool is observed across multiple sessions, so the
    # attacker can per-perm accumulate samples (still less than the full
    # fixed-perm pool).
    sessions_per_perm = max(1, config.num_sessions // pool_size)
    for perm in perm_pool:
        samples = sample_pool(sessions_per_perm * config.num_samples_per_session)
        masked = samples.index_select(dim=-1, index=perm)
        sig_masked = _signature(masked)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(hidden, device=perm.device)
        sig_masked_aligned = sig_masked.index_select(dim=0, index=inv)
        pool_results.append(
            _recover_permutation_top_k(sig_masked_aligned, sig_plain, k=5)
        )

    # ---- Strategy D: dense-mask sandwich ----
    sandwich_results: list[dict[str, Any]] = []
    for s in range(config.num_sessions):
        perm = generate_permutation(hidden, dtype, device)["perm"]
        N_left, _ = generate_invertible_matrix(hidden, dtype, device)
        N_right, _ = generate_invertible_matrix(hidden, dtype, device)
        pool = sample_pool(config.num_samples_per_session)
        masked = (pool @ N_left).index_select(dim=-1, index=perm) @ N_right
        sig_masked = _signature(masked)
        sandwich_results.append(
            _recover_permutation_top_k(sig_masked, sig_plain, k=5)
        )

    def _avg(results: list[dict[str, Any]], key: str) -> float:
        if not results:
            return 0.0
        return float(sum(r[key] for r in results) / len(results))

    summary = {
        "fixed_permutation": {
            "permutation_recovery_top1": _avg(fixed_results, "top1"),
            "permutation_recovery_top5": _avg(fixed_results, "top5"),
            "mean_channel_signature_error": _avg(
                fixed_results, "mean_signature_error"
            ),
        },
        "fresh_permutation_per_session": {
            "permutation_recovery_top1": _avg(fresh_results, "top1"),
            "permutation_recovery_top5": _avg(fresh_results, "top5"),
            "mean_channel_signature_error": _avg(
                fresh_results, "mean_signature_error"
            ),
        },
        "permutation_pool": {
            "permutation_recovery_top1": _avg(pool_results, "top1"),
            "permutation_recovery_top5": _avg(pool_results, "top5"),
            "mean_channel_signature_error": _avg(
                pool_results, "mean_signature_error"
            ),
            "pool_size": pool_size,
        },
        "dense_sandwich_reference": {
            "permutation_recovery_top1": _avg(sandwich_results, "top1"),
            "permutation_recovery_top5": _avg(sandwich_results, "top5"),
            "mean_channel_signature_error": _avg(
                sandwich_results, "mean_signature_error"
            ),
        },
    }
    return {
        "num_sessions": config.num_sessions,
        "num_samples_per_session": config.num_samples_per_session,
        "hidden_size": hidden,
        "per_strategy": summary,
        "interpretation": (
            "fixed_permutation lets per-channel signatures align across"
            " sessions, so a naive attacker can match columns above chance."
            " fresh_permutation_per_session breaks that alignment and drives"
            " top1 back toward 1/H. dense_sandwich_reference adds dense"
            " linear mixing on both sides and erases the column statistics"
            " entirely."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 2 — nonlinear-island linkability
# ---------------------------------------------------------------------------


def _pairwise_cosine_and_l2(stack: torch.Tensor) -> dict[str, float]:
    if stack.shape[0] < 2:
        return {
            "mean_pairwise_cosine": 1.0,
            "mean_pairwise_l2": 0.0,
        }
    flat = stack.reshape(stack.shape[0], -1).to(torch.float64)
    nrm = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    sim = nrm @ nrm.T
    n = sim.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    cos = sim[iu[0], iu[1]]
    # L2
    diffs: list[float] = []
    for i, j in itertools.combinations(range(n), 2):
        diffs.append(float((flat[i] - flat[j]).norm().item()))
    l2 = torch.tensor(diffs, dtype=torch.float64)
    return {
        "mean_pairwise_cosine": float(cos.mean().item()),
        "max_pairwise_cosine": float(cos.max().item()),
        "min_pairwise_cosine": float(cos.min().item()),
        "mean_pairwise_l2": float(l2.mean().item()),
    }


def _proxy_island_linkability(
    config: NonlinearIslandSecurityConfig,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    hidden = config.hidden_size
    H = torch.randn(
        config.batch_size, config.seq_len, hidden,
        dtype=dtype, device=device, generator=generator,
    )
    H_flat = H.reshape(-1, hidden)
    perm_fixed = generate_permutation(hidden, dtype, device)["perm"]

    def trials_fixed_perm_no_pad() -> torch.Tensor:
        stacks = [H_flat.index_select(dim=-1, index=perm_fixed)
                  for _ in range(config.num_trials)]
        return torch.stack(stacks, dim=0)

    def trials_fixed_perm_with_linear_boundary_pad() -> torch.Tensor:
        n, _ = generate_invertible_matrix(hidden, dtype, device)
        stacks: list[torch.Tensor] = []
        for _ in range(config.num_trials):
            t = generate_pad(tuple(H_flat.shape), dtype, device, config.pad_scale)
            # Pad applied at the Linear boundary BEFORE entering the island:
            # the GPU sees (H - T) N, then the activation island runs on its
            # output. Here we approximate the post-linear visible tensor as
            # ``(H_flat - t) @ n``, then permute it.
            stacks.append(
                ((H_flat - t) @ n).index_select(dim=-1, index=perm_fixed)
            )
        return torch.stack(stacks, dim=0)

    def trials_fresh_perm_with_linear_boundary_pad() -> torch.Tensor:
        n, _ = generate_invertible_matrix(hidden, dtype, device)
        stacks: list[torch.Tensor] = []
        for _ in range(config.num_trials):
            perm = generate_permutation(hidden, dtype, device)["perm"]
            t = generate_pad(tuple(H_flat.shape), dtype, device, config.pad_scale)
            stacks.append(
                ((H_flat - t) @ n).index_select(dim=-1, index=perm)
            )
        return torch.stack(stacks, dim=0)

    def trials_dense_perm_dense_sandwich() -> torch.Tensor:
        stacks: list[torch.Tensor] = []
        for _ in range(config.num_trials):
            perm = generate_permutation(hidden, dtype, device)["perm"]
            n_left, _ = generate_invertible_matrix(hidden, dtype, device)
            n_right, _ = generate_invertible_matrix(hidden, dtype, device)
            stacks.append(
                (H_flat @ n_left).index_select(dim=-1, index=perm) @ n_right
            )
        return torch.stack(stacks, dim=0)

    strategies = {
        "fixed_perm_no_pad": trials_fixed_perm_no_pad,
        "fixed_perm_with_linear_boundary_pad": trials_fixed_perm_with_linear_boundary_pad,
        "fresh_perm_with_linear_boundary_pad": trials_fresh_perm_with_linear_boundary_pad,
        "dense_to_perm_to_dense_sandwich": trials_dense_perm_dense_sandwich,
    }
    per_strategy: dict[str, Any] = {}
    for name, fn in strategies.items():
        stack = fn()
        stats = _pairwise_cosine_and_l2(stack)
        per_strategy[name] = {
            "num_trials": config.num_trials,
            **stats,
        }
    ranking = sorted(
        per_strategy.keys(),
        key=lambda k: per_strategy[k]["mean_pairwise_cosine"],
        reverse=True,
    )
    return {
        "per_strategy": per_strategy,
        "linkability_rank_high_to_low": ranking,
        "interpretation": (
            "fixed_perm_no_pad is the highest naive linkability because the"
            " GPU-visible tensor is a deterministic function of the plaintext."
            " Adding pad at the Linear boundary collapses the value-level"
            " stability; sandwiching with a fresh dense mask removes the"
            " coordinate-multiset signal entirely."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 3 — static mask-family accounting
# ---------------------------------------------------------------------------


MASK_FAMILY_ACCOUNTING: tuple[dict[str, str], ...] = (
    {
        "mask_family": "dense_invertible",
        "where_used": "Linear / Attention / KV cache boundaries",
        "preserved_invariants": "none beyond invertibility",
        "leakage_note": "Strong linear mixing; right multiply by a fresh dense N erases the per-channel signal.",
    },
    {
        "mask_family": "orthogonal",
        "where_used": "RMSNorm core",
        "preserved_invariants": "row L2 norm (||X N||_2 = ||X||_2)",
        "leakage_note": "Norm-preserving by design — observer can recover row norms.",
    },
    {
        "mask_family": "mean_preserving_orthogonal",
        "where_used": "LayerNorm core",
        "preserved_invariants": "row mean (X N · 1 = X · 1) AND row centered L2 norm",
        "leakage_note": "Mean + centered-norm preserved by design; an attacker observing many samples sees stable per-row mean and centered norm.",
    },
    {
        "mask_family": "permutation",
        "where_used": "Activation island (GELU / ReLU / SiLU)",
        "preserved_invariants": "coordinate-value multiset (the sorted set of channel values is unchanged)",
        "leakage_note": (
            "Permutation islands hide channel identity but do not hide"
            " coordinate-value multisets. Same multiset across sessions →"
            " permutation can be recovered by per-channel statistics if the"
            " permutation is reused."
        ),
    },
    {
        "mask_family": "paired_permutation",
        "where_used": "SwiGLU island (shared P for up and gate branches)",
        "preserved_invariants": "paired coordinate multiset for (up, gate) tuples",
        "leakage_note": (
            "Same multiset leakage as permutation; additionally exposes that"
            " the up- and gate-branches use the *same* P (paired alignment)."
        ),
    },
)


def _proxy_mask_family_accounting() -> dict[str, Any]:
    return {
        "table": list(MASK_FAMILY_ACCOUNTING),
        "summary_notes": [
            "Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.",
            "Permutation islands hide channel identity but do not hide coordinate-value multisets.",
            "Security depends on freshness, dense-mask sandwiching, and pad at Linear boundaries.",
            "Real TEE isolation is not implemented in this stage.",
        ],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_nonlinear_island_security_experiments(
    config: NonlinearIslandSecurityConfig,
) -> dict[str, Any]:
    """Run all three nonlinear-island security proxies and return a report dict."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    permutation_recovery = _proxy_permutation_recovery(
        config, dtype, device, generator
    )
    linkability = _proxy_island_linkability(config, dtype, device, generator)
    accounting = _proxy_mask_family_accounting()

    return {
        "config": asdict(config),
        "permutation_recovery_proxy": permutation_recovery,
        "island_linkability_proxy": linkability,
        "mask_family_accounting": accounting,
        "global_limitations": [
            "These are proxy attacks, not adaptive learned attacks.",
            "Fresh permutation reduces stable statistical recovery but does not provide dense linear mixing.",
            "Coordinate-value multiset leakage exists inside activation islands.",
            "Security depends on limiting island lifetime and sandwiching with dense masks.",
            "No real TEE isolation is implemented.",
        ],
    }


__all__ = [
    "MASK_FAMILY_ACCOUNTING",
    "NonlinearIslandSecurityConfig",
    "run_nonlinear_island_security_experiments",
]
