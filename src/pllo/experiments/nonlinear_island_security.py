"""Stage 5.2b — Nonlinear island security proxy experiments.

Three lightweight proxies over the operator-compatible mask scheme used by
the Stage 5.2a nonlinear islands. None of these constitute a formal
security proof — each is a *naive-observer upper bound* recorded so the
paper's security section can quote it directly.

* **Proxy 1 — Permutation recovery by channel statistics.** Build a
  per-channel signature ``(mean, std, median, q25, q75, mean_abs)`` over a
  reference activation distribution, then attempt to recover the masked
  permutation by greedy nearest-neighbour matching on those signatures.
  Compared across ``fixed_permutation`` / ``fresh_permutation_per_session``
  / ``permutation_pool`` / ``dense_sandwich_reference``.
* **Proxy 2 — Nonlinear island linkability.** Run the *same* plaintext
  input many times through four mask + pad policies; report pairwise
  cosine and L2 distance over the GPU-visible tensor. For the
  ``fixed_perm_with_linear_boundary_pad`` strategy, both views are
  recorded: ``boundary_input_visible = (X - T) N_in`` (where fresh
  pad / mask collapse linkability) and ``activation_input_visible = Z P``
  (where a fixed P keeps the visible tensor stable across requests).
* **Proxy 3 — Mask family security accounting (static).** Catalogues the
  preserved invariants and known leakage profile per mask family.

Outputs are aggregate metrics, sha-256 fingerprints, or short text only.
No mask tensors are emitted to JSON / CSV / Markdown.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
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
    num_sessions: int = 16
    samples_per_session: int = 32
    batch_size: int = 2
    seq_len: int = 8
    hidden_size: int = 64
    pad_scale: float = 1.0
    permutation_pool_size: int = 4
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 2026


# ---------------------------------------------------------------------------
# Threat model (static text used by the markdown emitter and tests)
# ---------------------------------------------------------------------------


THREAT_MODEL = (
    "Naive-observer adversary: sees only GPU-visible tensors produced by"
    " the masked forward (post-mask Linear inputs / outputs and the"
    " activation island's permuted input). Has prior knowledge of the"
    " plaintext channel distribution but does NOT execute adaptive or"
    " learned inversion attacks, does NOT observe trusted-side tensors,"
    " and does NOT use side channels."
)


# ---------------------------------------------------------------------------
# Signature + greedy permutation recovery
# ---------------------------------------------------------------------------


def compute_channel_signature(x: torch.Tensor) -> torch.Tensor:
    """Per-channel signature over a sample pool.

    ``x`` is shaped ``[num_samples, hidden_size]``. Returns
    ``[hidden_size, 6]`` with features ``(mean, std, median, q25, q75,
    mean_abs)``. The signature is the only statistic the proxy attacker
    is allowed to use.
    """
    x64 = x.to(torch.float64)
    mean = x64.mean(dim=0)
    std = x64.std(dim=0, unbiased=False)
    median = x64.quantile(0.5, dim=0)
    q25 = x64.quantile(0.25, dim=0)
    q75 = x64.quantile(0.75, dim=0)
    mean_abs = x64.abs().mean(dim=0)
    return torch.stack([mean, std, median, q25, q75, mean_abs], dim=-1)


def recover_permutation_by_signature(
    ref_signature: torch.Tensor,
    visible_signature: torch.Tensor,
) -> dict[str, float]:
    """Greedy nearest-neighbour recovery by cosine on the channel signature.

    Both inputs are shaped ``[hidden_size, num_features]`` and are assumed
    to be *aligned* — i.e. ``visible_signature[i]`` is the signature for the
    masked column that was originally plaintext column ``i``. The caller is
    responsible for the alignment (when permutation is ``perm``, align by
    ``visible_signature[inv_perm]``).

    Returns ``top1_recovery_rate``, ``top5_recovery_rate``,
    ``mean_correct_rank``, ``mean_signature_error``.
    """
    if ref_signature.shape != visible_signature.shape:
        raise ValueError(
            f"signature shape mismatch: ref {tuple(ref_signature.shape)} "
            f"vs visible {tuple(visible_signature.shape)}"
        )
    ref = ref_signature.to(torch.float64)
    vis = visible_signature.to(torch.float64)
    ref_n = ref / ref.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    vis_n = vis / vis.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    sim = vis_n @ ref_n.T  # [H, H]
    hidden = sim.shape[0]

    # For each visible column i, rank ref columns by descending similarity.
    sorted_idx = sim.argsort(dim=-1, descending=True)
    targets = torch.arange(hidden)

    top1 = (sorted_idx[:, 0] == targets).to(torch.float64).mean().item()
    k = min(5, hidden)
    topk_idx = sorted_idx[:, :k]
    topk = (topk_idx == targets.unsqueeze(-1)).any(dim=-1).to(torch.float64).mean().item()
    # mean rank of the true target across the visible columns
    ranks = (sorted_idx == targets.unsqueeze(-1)).int().argmax(dim=-1).to(torch.float64)
    mean_rank = ranks.mean().item()
    mean_signature_error = float((vis_n - ref_n).abs().mean().item())

    return {
        "top1_recovery_rate": float(top1),
        "top5_recovery_rate": float(topk),
        "mean_correct_rank": float(mean_rank),
        "mean_signature_error": mean_signature_error,
    }


# ---------------------------------------------------------------------------
# Synthetic activation distribution with channel-specific (mean, scale)
# ---------------------------------------------------------------------------


def _channel_profile(
    hidden: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean offset + scale that the attacker has *prior knowledge* of."""
    channel_mean = torch.linspace(-2.0, 2.0, hidden, dtype=dtype, device=device)
    channel_scale = torch.linspace(0.5, 3.0, hidden, dtype=dtype, device=device)
    return channel_mean, channel_scale


def _sample_pool(
    n: int,
    hidden: int,
    channel_mean: torch.Tensor,
    channel_scale: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """``Z[:, j] = noise * scale_j + mean_j`` per spec."""
    noise = torch.randn(
        n, hidden, dtype=dtype, device=device, generator=generator
    )
    return noise * channel_scale + channel_mean


# ---------------------------------------------------------------------------
# Proxy 1 — Permutation recovery by channel statistics
# ---------------------------------------------------------------------------


def _proxy_permutation_recovery(
    config: NonlinearIslandSecurityConfig,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    hidden = config.hidden_size
    channel_mean, channel_scale = _channel_profile(hidden, dtype, device)

    def sample(n: int) -> torch.Tensor:
        return _sample_pool(n, hidden, channel_mean, channel_scale, dtype, device, generator)

    # Reference: large pool so the attacker's plaintext signature is sharp.
    ref_plain = sample(config.num_sessions * config.samples_per_session)
    sig_plain = compute_channel_signature(ref_plain)

    # ---- Strategy A: fixed_permutation across all sessions ----
    # Attacker accumulates across sessions because the perm is stable.
    fixed_perm = generate_permutation(hidden, dtype, device)["perm"]
    fixed_pool = sample(config.num_sessions * config.samples_per_session)
    fixed_visible = fixed_pool.index_select(dim=-1, index=fixed_perm)
    inv = torch.empty_like(fixed_perm)
    inv[fixed_perm] = torch.arange(hidden, device=fixed_perm.device)
    sig_fixed_aligned = compute_channel_signature(fixed_visible).index_select(
        dim=0, index=inv
    )
    fixed_metrics = recover_permutation_by_signature(sig_plain, sig_fixed_aligned)

    # ---- Strategy B: fresh permutation per session ----
    # Attacker can only recover within each session (M samples each), then average.
    fresh_metrics_per_session: list[dict[str, float]] = []
    for _ in range(config.num_sessions):
        perm = generate_permutation(hidden, dtype, device)["perm"]
        pool = sample(config.samples_per_session)
        visible = pool.index_select(dim=-1, index=perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(hidden, device=perm.device)
        sig_aligned = compute_channel_signature(visible).index_select(dim=0, index=inv)
        fresh_metrics_per_session.append(
            recover_permutation_by_signature(sig_plain, sig_aligned)
        )
    fresh_metrics = {
        key: float(sum(m[key] for m in fresh_metrics_per_session) / len(fresh_metrics_per_session))
        for key in (
            "top1_recovery_rate",
            "top5_recovery_rate",
            "mean_correct_rank",
            "mean_signature_error",
        )
    }

    # ---- Strategy C: permutation pool ----
    pool_size = max(2, config.permutation_pool_size)
    perm_pool = [
        generate_permutation(hidden, dtype, device)["perm"]
        for _ in range(pool_size)
    ]
    sessions_per_perm = max(1, config.num_sessions // pool_size)
    pool_metrics_per_perm: list[dict[str, float]] = []
    for perm in perm_pool:
        samples = sample(sessions_per_perm * config.samples_per_session)
        visible = samples.index_select(dim=-1, index=perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(hidden, device=perm.device)
        sig_aligned = compute_channel_signature(visible).index_select(dim=0, index=inv)
        pool_metrics_per_perm.append(
            recover_permutation_by_signature(sig_plain, sig_aligned)
        )
    pool_metrics = {
        key: float(sum(m[key] for m in pool_metrics_per_perm) / len(pool_metrics_per_perm))
        for key in (
            "top1_recovery_rate",
            "top5_recovery_rate",
            "mean_correct_rank",
            "mean_signature_error",
        )
    } | {"pool_size": pool_size, "sessions_per_perm": sessions_per_perm}

    # ---- Strategy D: dense → permutation → dense sandwich ----
    # Different N_left / N_right / P per session; attacker has nothing stable.
    sandwich_metrics_per_session: list[dict[str, float]] = []
    for _ in range(config.num_sessions):
        perm = generate_permutation(hidden, dtype, device)["perm"]
        N_left, _ = generate_invertible_matrix(hidden, dtype, device)
        N_right, _ = generate_invertible_matrix(hidden, dtype, device)
        pool = sample(config.samples_per_session)
        visible = (pool @ N_left).index_select(dim=-1, index=perm) @ N_right
        # Attacker has no perm to undo — signature is read column-by-column
        # against the plaintext signature without alignment.
        sig_visible = compute_channel_signature(visible)
        sandwich_metrics_per_session.append(
            recover_permutation_by_signature(sig_plain, sig_visible)
        )
    sandwich_metrics = {
        key: float(sum(m[key] for m in sandwich_metrics_per_session) / len(sandwich_metrics_per_session))
        for key in (
            "top1_recovery_rate",
            "top5_recovery_rate",
            "mean_correct_rank",
            "mean_signature_error",
        )
    }

    random_chance = 1.0 / hidden
    expected_risk = {
        "fixed_permutation": "high (stable across sessions; aggregate signature is sharp)",
        "fresh_permutation_per_session": "moderate (per-session recovery only; no cross-session alignment)",
        "permutation_pool": "moderate-to-high (per-perm signature is sharper than fresh but weaker than fixed)",
        "dense_sandwich_reference": "near random chance (column statistics destroyed by dense mixing)",
    }
    interpretations = {
        "fixed_permutation": (
            "Fixed permutation lets the attacker accumulate samples across"
            " sessions; the per-channel signature converges to the true"
            " permuted distribution and greedy matching recovers above chance."
        ),
        "fresh_permutation_per_session": (
            "Fresh permutation breaks cross-session alignment, so the attacker"
            " only has M = samples_per_session points per per-session signature."
        ),
        "permutation_pool": (
            "Each pool entry gets sessions_per_perm × M samples — between fixed"
            " and fresh in signature sharpness."
        ),
        "dense_sandwich_reference": (
            "Dense linear mixing on both sides of the permutation destroys the"
            " per-channel signature; recovery drops to ~ 1/H random chance."
        ),
    }
    per_strategy: dict[str, dict[str, Any]] = {}
    for name, metrics in (
        ("fixed_permutation", fixed_metrics),
        ("fresh_permutation_per_session", fresh_metrics),
        ("permutation_pool", pool_metrics),
        ("dense_sandwich_reference", sandwich_metrics),
    ):
        per_strategy[name] = {
            "strategy": name,
            "num_sessions": config.num_sessions,
            "samples_per_session": config.samples_per_session,
            "hidden_size": hidden,
            "expected_risk_level": expected_risk[name],
            "interpretation": interpretations[name],
            **metrics,
        }
    return {
        "random_chance_top1": random_chance,
        "per_strategy": per_strategy,
        "ranking_by_top1_descending": sorted(
            per_strategy.keys(),
            key=lambda k: per_strategy[k]["top1_recovery_rate"],
            reverse=True,
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 2 — Nonlinear island linkability
# ---------------------------------------------------------------------------


def _pairwise_stats(stack: torch.Tensor) -> dict[str, float]:
    """Pairwise cosine + L2 over ``stack[i].flatten()`` rows."""
    n = stack.shape[0]
    if n < 2:
        return {
            "mean_pairwise_cosine": 1.0,
            "max_pairwise_cosine": 1.0,
            "min_pairwise_cosine": 1.0,
            "mean_pairwise_l2": 0.0,
            "max_pairwise_l2": 0.0,
            "min_pairwise_l2": 0.0,
        }
    flat = stack.reshape(n, -1).to(torch.float64)
    nrm = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    sim = nrm @ nrm.T
    iu = torch.triu_indices(n, n, offset=1)
    cos = sim[iu[0], iu[1]]
    diffs: list[float] = []
    for i, j in itertools.combinations(range(n), 2):
        diffs.append(float((flat[i] - flat[j]).norm().item()))
    l2 = torch.tensor(diffs, dtype=torch.float64)
    return {
        "mean_pairwise_cosine": float(cos.mean().item()),
        "max_pairwise_cosine": float(cos.max().item()),
        "min_pairwise_cosine": float(cos.min().item()),
        "mean_pairwise_l2": float(l2.mean().item()),
        "max_pairwise_l2": float(l2.max().item()),
        "min_pairwise_l2": float(l2.min().item()),
    }


def _proxy_island_linkability(
    config: NonlinearIslandSecurityConfig,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    hidden = config.hidden_size
    # Fixed plaintext input X (shared across all trials of all strategies).
    X = torch.randn(
        config.batch_size, config.seq_len, hidden,
        dtype=dtype, device=device, generator=generator,
    )
    X_flat = X.reshape(-1, hidden)
    # Z = activation island input ≈ X (we don't run a Linear here — only the
    # permutation behaviour is being measured), so Z = X_flat.
    Z_flat = X_flat
    # Number of repeated trials per strategy.
    num_trials = config.num_sessions
    perm_fixed = generate_permutation(hidden, dtype, device)["perm"]

    # Strategy 1: fixed_perm_no_pad
    s1_visible = torch.stack(
        [Z_flat.index_select(dim=-1, index=perm_fixed) for _ in range(num_trials)],
        dim=0,
    )

    # Strategy 2: fixed_perm_with_linear_boundary_pad
    s2_boundary_stacks: list[torch.Tensor] = []
    s2_activation_stacks: list[torch.Tensor] = []
    for _ in range(num_trials):
        N_in, _ = generate_invertible_matrix(hidden, dtype, device)
        T = generate_pad(tuple(X_flat.shape), dtype, device, config.pad_scale)
        boundary_visible = (X_flat - T) @ N_in
        # Activation input is Z P (no pad — see Stage 5.2a placement rule).
        activation_visible = Z_flat.index_select(dim=-1, index=perm_fixed)
        s2_boundary_stacks.append(boundary_visible)
        s2_activation_stacks.append(activation_visible)
    s2_boundary = torch.stack(s2_boundary_stacks, dim=0)
    s2_activation = torch.stack(s2_activation_stacks, dim=0)

    # Strategy 3: fresh_perm_with_linear_boundary_pad
    s3_boundary_stacks: list[torch.Tensor] = []
    s3_activation_stacks: list[torch.Tensor] = []
    for _ in range(num_trials):
        N_in, _ = generate_invertible_matrix(hidden, dtype, device)
        T = generate_pad(tuple(X_flat.shape), dtype, device, config.pad_scale)
        perm = generate_permutation(hidden, dtype, device)["perm"]
        boundary_visible = (X_flat - T) @ N_in
        activation_visible = Z_flat.index_select(dim=-1, index=perm)
        s3_boundary_stacks.append(boundary_visible)
        s3_activation_stacks.append(activation_visible)
    s3_boundary = torch.stack(s3_boundary_stacks, dim=0)
    s3_activation = torch.stack(s3_activation_stacks, dim=0)

    # Strategy 4: dense_to_perm_to_dense_sandwich
    s4_pre_stacks: list[torch.Tensor] = []
    s4_island_stacks: list[torch.Tensor] = []
    s4_post_stacks: list[torch.Tensor] = []
    for _ in range(num_trials):
        N_pre, _ = generate_invertible_matrix(hidden, dtype, device)
        N_post, _ = generate_invertible_matrix(hidden, dtype, device)
        perm = generate_permutation(hidden, dtype, device)["perm"]
        pre_island = Z_flat @ N_pre
        island = Z_flat.index_select(dim=-1, index=perm)
        post_island = (Z_flat) @ N_post
        s4_pre_stacks.append(pre_island)
        s4_island_stacks.append(island)
        s4_post_stacks.append(post_island)
    s4_pre = torch.stack(s4_pre_stacks, dim=0)
    s4_island = torch.stack(s4_island_stacks, dim=0)
    s4_post = torch.stack(s4_post_stacks, dim=0)

    per_strategy: dict[str, Any] = {
        "fixed_perm_no_pad": {
            "strategy": "fixed_perm_no_pad",
            "num_trials": num_trials,
            "view": "activation_input_visible",
            **_pairwise_stats(s1_visible),
            "expected_linkability": "highest",
            "interpretation": (
                "Same plaintext + same permutation → identical GPU-visible"
                " tensor every request; pairwise cosine collapses to 1."
            ),
        },
        "fixed_perm_with_linear_boundary_pad": {
            "strategy": "fixed_perm_with_linear_boundary_pad",
            "num_trials": num_trials,
            "view": "dual",
            "boundary_input_visible": _pairwise_stats(s2_boundary),
            "activation_input_visible": _pairwise_stats(s2_activation),
            "expected_linkability": "boundary low / activation high",
            "interpretation": (
                "Fresh pad and mask at the Linear boundary collapse linkability"
                " on (X - T) N_in, but the activation input Z P remains"
                " identical across requests because the permutation is fixed."
            ),
        },
        "fresh_perm_with_linear_boundary_pad": {
            "strategy": "fresh_perm_with_linear_boundary_pad",
            "num_trials": num_trials,
            "view": "dual",
            "boundary_input_visible": _pairwise_stats(s3_boundary),
            "activation_input_visible": _pairwise_stats(s3_activation),
            "expected_linkability": "low on both views",
            "interpretation": (
                "Fresh permutation per trial scrambles the activation input"
                " as well; cosine across trials drops because Z P_i ≠ Z P_j."
            ),
        },
        "dense_to_perm_to_dense_sandwich": {
            "strategy": "dense_to_perm_to_dense_sandwich",
            "num_trials": num_trials,
            "view": "post_island_dense_visible",
            "pre_island_dense_visible": _pairwise_stats(s4_pre),
            "island_visible": _pairwise_stats(s4_island),
            "post_island_dense_visible": _pairwise_stats(s4_post),
            "expected_linkability": "lowest",
            "interpretation": (
                "Fresh dense masks around the permutation island destroy"
                " coordinate-value alignment; the post-island visible tensor"
                " has near-zero pairwise cosine across requests."
            ),
        },
    }

    # Main linkability metric per strategy (used for ranking).
    main_cos = {
        "fixed_perm_no_pad": per_strategy["fixed_perm_no_pad"]["mean_pairwise_cosine"],
        "fixed_perm_with_linear_boundary_pad": per_strategy[
            "fixed_perm_with_linear_boundary_pad"
        ]["activation_input_visible"]["mean_pairwise_cosine"],
        "fresh_perm_with_linear_boundary_pad": per_strategy[
            "fresh_perm_with_linear_boundary_pad"
        ]["activation_input_visible"]["mean_pairwise_cosine"],
        "dense_to_perm_to_dense_sandwich": per_strategy[
            "dense_to_perm_to_dense_sandwich"
        ]["post_island_dense_visible"]["mean_pairwise_cosine"],
    }
    ranking = sorted(main_cos.keys(), key=lambda k: main_cos[k], reverse=True)
    return {
        "per_strategy": per_strategy,
        "main_metric_per_strategy": {
            "metric": "mean_pairwise_cosine over the principal visible view",
            "values": main_cos,
        },
        "linkability_rank_high_to_low": ranking,
        "notes": [
            "Activation island permutation preserves the coordinate-value multiset.",
            "Boundary pad does not protect the activation input when P is fixed.",
            "Dense sandwich is the strongest of the four under naive observers.",
        ],
    }


# ---------------------------------------------------------------------------
# Proxy 3 — Mask family security accounting (static)
# ---------------------------------------------------------------------------


MASK_FAMILY_ACCOUNTING: tuple[dict[str, str], ...] = (
    {
        "mask_family": "dense_invertible",
        "used_for": "Linear / Attention / KV cache boundaries",
        "correctness_role": "right-multiply mask for arbitrary invertible linear ops",
        "preserved_statistics": "rank, dimension, algebraic relations under same mask",
        "gpu_visible_leakage": "channel identity hidden via dense mixing; algebraic structure preserved only under reuse",
        "mitigation": "fresh mask across sessions; pad at Linear boundaries",
        "security_strength_relative_to_dense": "baseline (this IS dense)",
        "notes": "Strongest of the listed families; the reference against which the others are compared.",
    },
    {
        "mask_family": "orthogonal",
        "used_for": "RMSNorm core",
        "correctness_role": "commutes with rms(.) and right-rotates the normalised state",
        "preserved_statistics": "row L2 norm (||X N||_2 = ||X||_2); pairwise dot products if the same mask is reused",
        "gpu_visible_leakage": "row L2 norm preserved by design; coordinate identity hidden under rotation",
        "mitigation": "restrict island lifetime; sandwich with dense masks at Linear boundaries",
        "security_strength_relative_to_dense": "weaker (row L2 norm always preserved)",
        "notes": "Designed to preserve rms; an attacker observing many samples can always read row norms.",
    },
    {
        "mask_family": "mean_preserving_orthogonal",
        "used_for": "LayerNorm core",
        "correctness_role": "commutes with LayerNorm core (mean + centered variance preserved)",
        "preserved_statistics": "row mean (X N · 1 = X · 1); row centered L2 norm",
        "gpu_visible_leakage": "row mean and centered norm preserved by design",
        "mitigation": "sandwich with dense masks at Linear boundaries; avoid reuse across sessions",
        "security_strength_relative_to_dense": "weaker (mean + centered norm always preserved)",
        "notes": "Strict subset of orthogonal masks: also preserves the all-ones direction.",
    },
    {
        "mask_family": "permutation",
        "used_for": "GELU / ReLU / SiLU activation island",
        "correctness_role": "commutes exactly with element-wise activations: f(X P) = f(X) P",
        "preserved_statistics": "coordinate-value multiset; per-token sorted values",
        "gpu_visible_leakage": "channel identity hidden if P is secret; multiset always leaks",
        "mitigation": "fresh permutation per session; permutation pool; dense sandwich at Linear boundaries",
        "security_strength_relative_to_dense": "weaker (multiset and sorted values always preserved)",
        "notes": "Permutation islands hide channel identity but do not hide coordinate-value multisets.",
    },
    {
        "mask_family": "paired_permutation",
        "used_for": "SwiGLU up / gate branches (shared P)",
        "correctness_role": "commutes with SwiGLU when up and gate use the same P",
        "preserved_statistics": "coordinate-value multiset; paired (up, gate) alignment per channel",
        "gpu_visible_leakage": "same as permutation, plus paired alignment exposes that up and gate share P",
        "mitigation": "fresh paired permutation; branch-consistency checks; dense sandwich",
        "security_strength_relative_to_dense": "weaker (paired multiset always preserved)",
        "notes": "Inherits permutation leakage; the paired structure is an additional small leakage channel.",
    },
)


def _proxy_mask_family_accounting() -> dict[str, Any]:
    return {
        "table": list(MASK_FAMILY_ACCOUNTING),
        "summary_notes": [
            "Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.",
            "Permutation islands hide channel identity but do not hide coordinate-value multisets.",
            "Orthogonal masks preserve row norms by design.",
            "Mean-preserving orthogonal masks preserve row means and centered norms by design.",
            "Dense sandwiching and pad at Linear boundaries are required mitigations.",
        ],
    }


# ---------------------------------------------------------------------------
# Fingerprinting (used only for aggregate accounting; no mask content)
# ---------------------------------------------------------------------------


def _fingerprint_count(items: list[torch.Tensor]) -> dict[str, int]:
    seen = set()
    for t in items:
        buf = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
        seen.add(hashlib.sha256(buf).hexdigest())
    return {"num_drawn": len(items), "num_unique_fingerprints": len(seen)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


GLOBAL_LIMITATIONS = (
    "These experiments are security proxies, not formal security proofs.",
    "Compatible mask families are weaker than unrestricted dense masks inside nonlinear islands.",
    "Permutation islands hide channel identity but do not hide coordinate-value multisets.",
    "Orthogonal masks preserve row norms by design.",
    "Mean-preserving orthogonal masks preserve row means and centered norms by design.",
    "Fresh permutation reduces stable statistical recovery but does not provide dense linear mixing.",
    "Dense sandwiching and pad at Linear boundaries are required mitigations.",
    "This stage does not implement adaptive learned inversion attacks.",
    "This stage does not implement real TEE isolation.",
    "This stage does not prove semantic security.",
)


def run_nonlinear_island_security(
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
    island_linkability = _proxy_island_linkability(
        config, dtype, device, generator
    )
    mask_family_accounting = _proxy_mask_family_accounting()

    p = permutation_recovery["per_strategy"]
    l = island_linkability["main_metric_per_strategy"]["values"]
    global_summary = {
        "fixed_perm_recovery_top1": p["fixed_permutation"]["top1_recovery_rate"],
        "fresh_perm_recovery_top1": p["fresh_permutation_per_session"]["top1_recovery_rate"],
        "sandwich_perm_recovery_top1": p["dense_sandwich_reference"]["top1_recovery_rate"],
        "random_chance_top1": permutation_recovery["random_chance_top1"],
        "fixed_perm_no_pad_linkability_cos": l["fixed_perm_no_pad"],
        "fresh_perm_with_pad_linkability_cos": l[
            "fresh_perm_with_linear_boundary_pad"
        ],
        "dense_sandwich_linkability_cos": l["dense_to_perm_to_dense_sandwich"],
        "fixed_permutation_is_more_recoverable_than_fresh": bool(
            p["fixed_permutation"]["top1_recovery_rate"]
            > p["fresh_permutation_per_session"]["top1_recovery_rate"]
        ),
        "fixed_permutation_is_more_recoverable_than_sandwich": bool(
            p["fixed_permutation"]["top1_recovery_rate"]
            > p["dense_sandwich_reference"]["top1_recovery_rate"]
        ),
        "fixed_no_pad_more_linkable_than_fresh_with_pad": bool(
            l["fixed_perm_no_pad"] > l["fresh_perm_with_linear_boundary_pad"]
        ),
    }

    return {
        "config": asdict(config),
        "threat_model": THREAT_MODEL,
        "permutation_recovery": permutation_recovery,
        "island_linkability": island_linkability,
        "mask_family_accounting": mask_family_accounting,
        "global_summary": global_summary,
        "limitations": list(GLOBAL_LIMITATIONS),
    }


__all__ = [
    "GLOBAL_LIMITATIONS",
    "MASK_FAMILY_ACCOUNTING",
    "NonlinearIslandSecurityConfig",
    "THREAT_MODEL",
    "compute_channel_signature",
    "recover_permutation_by_signature",
    "run_nonlinear_island_security",
]
