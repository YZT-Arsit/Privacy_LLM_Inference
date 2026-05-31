"""Stage 5.4 — Adaptive permutation / linkability attackers for nonlinear islands.

Three attackers, each evaluated against six masking strategies:

1. **Learned linear inverter** — ridge least-squares from the GPU-visible
   nonlinear-island tensor ``V`` back to the plaintext ``X``. The attacker
   has access to a labelled training pool ``(V_train, X_train)`` and is
   evaluated on a held-out ``(V_test, X_test)`` pool.

2. **Small MLP inverter** — two-layer ReLU MLP trained with Adam on the
   same labelled pool. Evaluates whether a non-linear learned attacker
   strictly improves over the linear baseline.

3. **Adaptive permutation recovery** — two methods over the per-channel
   ``(mean, std, median, q25, q75, mean_abs)`` signature shared with
   Stage 5.2b. *signature_matching* reproduces the Stage 5.2b naive
   nearest-neighbour proxy on the same data; *soft_assignment* runs a
   Sinkhorn-style log-domain row/column normalisation for a stronger
   matching attack. Compares ``fixed_permutation``,
   ``fresh_permutation_per_session``, ``permutation_pool``, and
   ``dense_sandwich``.

The module also emits a conservative mitigation decision table per
strategy: ``risk_level`` (low / medium / high) and
``default_on_recommendation`` (``unsafe_default_on`` /
``needs_more_evaluation`` / ``acceptable_with_mitigation``). These are
strictly *under the tested adaptive proxy attackers* — they are NOT
formal security claims, NOT side-channel-aware, and NOT adversary
queries against a deployed LLM.

Limitations
-----------
* Adaptive *proxy* attackers, not formal security proofs.
* Synthetic structured channel data with per-channel mean / scale
  profile; not full real-model activation traces.
* No adaptive black-box querying of a deployed LLM.
* No side-channel attacks.
* No real TEE isolation evaluation.
* Default-on recommendations are conditional on the tested threat model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pllo.experiments.nonlinear_island_security import (
    compute_channel_signature,
    recover_permutation_by_signature,
)
from pllo.masks.mask_generator import generate_invertible_matrix
from pllo.masks.pad_generator import generate_pad
from pllo.model_zoo.base import torch_dtype_from_string
from pllo.ops.compatible_masks import generate_permutation


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveIslandAttackConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    hidden_size: int = 64
    num_train_samples: int = 512
    num_test_samples: int = 256
    num_sessions: int = 16
    samples_per_session: int = 32
    permutation_pool_size: int = 4
    pad_scale: float = 1.0
    attacker_steps: int = 200
    attacker_lr: float = 1e-2
    mlp_hidden_size: int = 128
    mlp_batch_size: int = 64
    ridge_lambda: float = 1e-3
    soft_assignment_iters: int = 50
    soft_assignment_temperature: float = 0.05
    dtype: str = "float32"
    device: str = "cpu"


THREAT_MODEL = (
    "Adaptive-proxy adversary: observes the GPU-visible nonlinear-island"
    " tensor across many sessions and has a labelled (visible, plaintext)"
    " training pool. May fit a ridge-regularised linear inverter, a small"
    " MLP inverter, and a Sinkhorn-style soft-assignment over per-channel"
    " signatures. Does NOT see trusted-side tensors, does NOT use side"
    " channels, does NOT query an actual deployed LLM, and does NOT have"
    " formal-security guarantees."
)


STRATEGIES: tuple[str, ...] = (
    "fixed_permutation",
    "fresh_permutation_per_session",
    "permutation_pool",
    "dense_sandwich",
    "boundary_pad_only_boundary_view",
    "boundary_pad_only_activation_view",
    # Stage 5.3e — full mitigation bundle: dense sandwich + fresh
    # permutation + pad at every Linear boundary.
    "fresh_perm_plus_sandwich_plus_pad",
)


PERMUTATION_RECOVERY_STRATEGIES: tuple[str, ...] = (
    "fixed_permutation",
    "fresh_permutation_per_session",
    "permutation_pool",
    "dense_sandwich",
    "fresh_perm_plus_sandwich_plus_pad",
)


# ---------------------------------------------------------------------------
# Synthetic structured channel data
# ---------------------------------------------------------------------------


def _channel_profile(
    hidden: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel ``(mean, scale, skew)``. Attacker has prior knowledge of these."""
    channel_mean = torch.linspace(-2.0, 2.0, hidden, dtype=dtype, device=device)
    channel_scale = torch.linspace(0.5, 3.0, hidden, dtype=dtype, device=device)
    # Light per-channel skew via a smooth profile — keeps signatures
    # distinguishable even after centering.
    channel_skew = 0.5 * torch.cos(
        torch.linspace(0.0, 3.1416, hidden, dtype=dtype, device=device)
    )
    return channel_mean, channel_scale, channel_skew


def generate_structured_channel_data(
    num_samples: int,
    hidden_size: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate ``[num_samples, hidden_size]`` activation samples.

    Channels carry per-column ``(mean, scale, skew)``: ``X[:, j] = mean_j +
    scale_j * (noise + skew_j * noise**2)``. This is more realistic than
    isotropic Gaussian and gives each channel a recoverable signature.
    """
    device_obj = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    channel_mean, channel_scale, channel_skew = _channel_profile(
        hidden_size, dtype, device_obj
    )
    noise = torch.randn(num_samples, hidden_size, dtype=dtype, generator=generator)
    noise = noise.to(device_obj)
    skew_term = channel_skew * (noise * noise - 1.0)  # zero-mean cubic-ish profile
    return channel_mean + channel_scale * (noise + skew_term)


def _sample(
    n: int,
    hidden: int,
    channel_mean: torch.Tensor,
    channel_scale: torch.Tensor,
    channel_skew: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(n, hidden, dtype=dtype, generator=generator)
    noise = noise.to(device)
    skew_term = channel_skew * (noise * noise - 1.0)
    return channel_mean + channel_scale * (noise + skew_term)


# ---------------------------------------------------------------------------
# Strategy-specific (X, V) dataset construction
# ---------------------------------------------------------------------------


def _build_session_chunks(
    total_samples: int, samples_per_session: int
) -> list[int]:
    if total_samples <= 0:
        return []
    n_full = total_samples // samples_per_session
    rem = total_samples - n_full * samples_per_session
    sizes = [samples_per_session] * n_full
    if rem > 0:
        sizes.append(rem)
    return sizes


def _build_dataset(
    strategy: str,
    num_train_samples: int,
    num_test_samples: int,
    samples_per_session: int,
    hidden: int,
    permutation_pool_size: int,
    pad_scale: float,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X_train, V_train, X_test, V_test)`` for one strategy."""
    channel_mean, channel_scale, channel_skew = _channel_profile(hidden, dtype, device)

    def sample(n: int) -> torch.Tensor:
        return _sample(
            n, hidden, channel_mean, channel_scale, channel_skew, dtype, device, generator
        )

    train_sizes = _build_session_chunks(num_train_samples, samples_per_session)
    test_sizes = _build_session_chunks(num_test_samples, samples_per_session)

    def cat_sessions(
        session_sizes: list[int],
        per_session_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs: list[torch.Tensor] = []
        vs: list[torch.Tensor] = []
        for n in session_sizes:
            x_s, v_s = per_session_fn(n)
            xs.append(x_s)
            vs.append(v_s)
        return torch.cat(xs, dim=0), torch.cat(vs, dim=0)

    if strategy == "fixed_permutation":
        # Same permutation across train + test sessions.
        perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]

        def per_session(n):
            x = sample(n)
            v = x.index_select(dim=-1, index=perm)
            return x, v

        X_train, V_train = cat_sessions(train_sizes, per_session)
        X_test, V_test = cat_sessions(test_sizes, per_session)
        return X_train, V_train, X_test, V_test

    if strategy == "fresh_permutation_per_session":
        def per_session(n):
            perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]
            x = sample(n)
            v = x.index_select(dim=-1, index=perm)
            return x, v

        X_train, V_train = cat_sessions(train_sizes, per_session)
        X_test, V_test = cat_sessions(test_sizes, per_session)
        return X_train, V_train, X_test, V_test

    if strategy == "permutation_pool":
        pool_size = max(2, permutation_pool_size)
        pool = [
            generate_permutation(hidden, dtype=dtype, device=device)["perm"]
            for _ in range(pool_size)
        ]
        counter = {"i": 0}

        def per_session(n):
            perm = pool[counter["i"] % pool_size]
            counter["i"] += 1
            x = sample(n)
            v = x.index_select(dim=-1, index=perm)
            return x, v

        X_train, V_train = cat_sessions(train_sizes, per_session)
        X_test, V_test = cat_sessions(test_sizes, per_session)
        return X_train, V_train, X_test, V_test

    if strategy == "dense_sandwich":
        def per_session(n):
            perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]
            N_left, _ = generate_invertible_matrix(hidden, dtype, device)
            N_right, _ = generate_invertible_matrix(hidden, dtype, device)
            x = sample(n)
            v = (x @ N_left).index_select(dim=-1, index=perm) @ N_right
            return x, v

        X_train, V_train = cat_sessions(train_sizes, per_session)
        X_test, V_test = cat_sessions(test_sizes, per_session)
        return X_train, V_train, X_test, V_test

    if strategy == "boundary_pad_only_boundary_view":
        # ``visible = (X - T) @ N`` with fresh T and N per session.
        def per_session(n):
            x = sample(n)
            T = generate_pad(tuple(x.shape), dtype, device, pad_scale)
            N, _ = generate_invertible_matrix(hidden, dtype, device)
            v = (x - T) @ N
            return x, v

        X_train, V_train = cat_sessions(train_sizes, per_session)
        X_test, V_test = cat_sessions(test_sizes, per_session)
        return X_train, V_train, X_test, V_test

    if strategy == "boundary_pad_only_activation_view":
        # Activation view ``Z P`` with fixed P; boundary pad does NOT
        # protect this view — equivalent risk to ``fixed_permutation``.
        perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]

        def per_session(n):
            x = sample(n)
            v = x.index_select(dim=-1, index=perm)
            return x, v

        X_train, V_train = cat_sessions(train_sizes, per_session)
        X_test, V_test = cat_sessions(test_sizes, per_session)
        return X_train, V_train, X_test, V_test

    if strategy == "fresh_perm_plus_sandwich_plus_pad":
        # Full mitigation bundle: per-session pad T, dense N_left, fresh
        # permutation P, dense N_right.  Visible boundary tensor is
        # ``(X - T) @ N_left @ P @ N_right`` — the attacker never sees the
        # internal ``ZP`` view because the dense sandwich rotates it back
        # into a dense-masked coordinate system that changes every session.
        def per_session(n):
            x = sample(n)
            T = generate_pad(tuple(x.shape), dtype, device, pad_scale)
            N_left, _ = generate_invertible_matrix(hidden, dtype, device)
            perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]
            N_right, _ = generate_invertible_matrix(hidden, dtype, device)
            v = ((x - T) @ N_left).index_select(dim=-1, index=perm) @ N_right
            return x, v

        X_train, V_train = cat_sessions(train_sizes, per_session)
        X_test, V_test = cat_sessions(test_sizes, per_session)
        return X_train, V_train, X_test, V_test

    raise ValueError(f"unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Attack 1 — Learned linear inverter (ridge least squares)
# ---------------------------------------------------------------------------


def _fit_linear_inverter(
    V_train: torch.Tensor, X_train: torch.Tensor, ridge_lambda: float
) -> torch.Tensor:
    hidden = V_train.shape[-1]
    A = V_train.T @ V_train + ridge_lambda * torch.eye(
        hidden, dtype=V_train.dtype, device=V_train.device
    )
    B = V_train.T @ X_train
    return torch.linalg.solve(A, B)


def _reconstruction_metrics(
    X_pred: torch.Tensor, X_test: torch.Tensor
) -> dict[str, float]:
    diff = X_pred - X_test
    mse = float((diff * diff).mean().item())
    test_norm = float(X_test.norm().clamp_min(1e-30).item())
    rel_l2 = float((diff.norm() / max(test_norm, 1e-30)).item())
    pred_flat = X_pred.reshape(-1)
    tgt_flat = X_test.reshape(-1)
    denom = pred_flat.norm() * tgt_flat.norm()
    cos = float((pred_flat @ tgt_flat / denom.clamp_min(1e-30)).item())
    return {
        "mse": mse,
        "relative_l2_error": rel_l2,
        "cosine_similarity": cos,
    }


def _run_linear_inverter(
    strategy: str,
    X_train: torch.Tensor,
    V_train: torch.Tensor,
    X_test: torch.Tensor,
    V_test: torch.Tensor,
    config: AdaptiveIslandAttackConfig,
) -> dict[str, Any]:
    W = _fit_linear_inverter(V_train, X_train, config.ridge_lambda)
    X_pred = V_test @ W
    metrics = _reconstruction_metrics(X_pred, X_test)
    return {
        "strategy": strategy,
        "num_train_samples": int(V_train.shape[0]),
        "num_test_samples": int(V_test.shape[0]),
        "ridge_lambda": config.ridge_lambda,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Attack 2 — Small MLP inverter
# ---------------------------------------------------------------------------


class SmallInverter(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def _run_mlp_inverter(
    strategy: str,
    X_train: torch.Tensor,
    V_train: torch.Tensor,
    X_test: torch.Tensor,
    V_test: torch.Tensor,
    config: AdaptiveIslandAttackConfig,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    hidden = V_train.shape[-1]
    model = SmallInverter(hidden, config.mlp_hidden_size).to(
        dtype=V_train.dtype, device=V_train.device
    )
    optim = torch.optim.Adam(model.parameters(), lr=config.attacker_lr)
    batch_size = min(config.mlp_batch_size, V_train.shape[0])
    losses: list[float] = []
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    for _ in range(max(1, config.attacker_steps)):
        idx = torch.randint(
            0, V_train.shape[0], (batch_size,), generator=gen
        ).to(V_train.device)
        v = V_train[idx]
        x = X_train[idx]
        pred = model(v)
        loss = F.mse_loss(pred, x)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        X_pred = model(V_test)
    metrics = _reconstruction_metrics(X_pred, X_test)
    return {
        "strategy": strategy,
        "num_train_samples": int(V_train.shape[0]),
        "num_test_samples": int(V_test.shape[0]),
        "attacker_steps": int(config.attacker_steps),
        "attacker_lr": float(config.attacker_lr),
        "mlp_hidden_size": int(config.mlp_hidden_size),
        "final_train_loss": float(losses[-1]) if losses else None,
        "first_train_loss": float(losses[0]) if losses else None,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Attack 3 — Adaptive permutation recovery
# ---------------------------------------------------------------------------


def _build_permutation_visible(
    strategy: str,
    num_sessions: int,
    samples_per_session: int,
    hidden: int,
    permutation_pool_size: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ref_plain, visible_aligned)`` for permutation-recovery proxies.

    ``visible_aligned[i]`` is the visible signature column that should map to
    plaintext column ``i``. For fresh/pool/sandwich the per-session inverse
    permutation undoes the visible re-ordering before signature aggregation
    (this matches Stage 5.2b's recover_permutation_by_signature convention).
    """
    channel_mean, channel_scale, channel_skew = _channel_profile(hidden, dtype, device)

    def sample(n):
        return _sample(
            n, hidden, channel_mean, channel_scale, channel_skew, dtype, device, generator
        )

    ref_plain = sample(num_sessions * samples_per_session)

    if strategy == "fixed_permutation":
        perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]
        pool = sample(num_sessions * samples_per_session)
        visible = pool.index_select(dim=-1, index=perm)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(hidden, device=perm.device)
        return ref_plain, compute_channel_signature(visible).index_select(dim=0, index=inv)

    if strategy == "fresh_permutation_per_session":
        sig_acc: list[torch.Tensor] = []
        for _ in range(num_sessions):
            perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]
            x = sample(samples_per_session)
            v = x.index_select(dim=-1, index=perm)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(hidden, device=perm.device)
            sig_acc.append(compute_channel_signature(v).index_select(dim=0, index=inv))
        return ref_plain, torch.stack(sig_acc, dim=0).mean(dim=0)

    if strategy == "permutation_pool":
        pool_size = max(2, permutation_pool_size)
        perm_pool = [
            generate_permutation(hidden, dtype=dtype, device=device)["perm"]
            for _ in range(pool_size)
        ]
        sig_acc: list[torch.Tensor] = []
        for s in range(num_sessions):
            perm = perm_pool[s % pool_size]
            x = sample(samples_per_session)
            v = x.index_select(dim=-1, index=perm)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(hidden, device=perm.device)
            sig_acc.append(compute_channel_signature(v).index_select(dim=0, index=inv))
        return ref_plain, torch.stack(sig_acc, dim=0).mean(dim=0)

    if strategy == "dense_sandwich":
        sig_acc: list[torch.Tensor] = []
        for _ in range(num_sessions):
            perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]
            N_left, _ = generate_invertible_matrix(hidden, dtype, device)
            N_right, _ = generate_invertible_matrix(hidden, dtype, device)
            x = sample(samples_per_session)
            v = (x @ N_left).index_select(dim=-1, index=perm) @ N_right
            sig_acc.append(compute_channel_signature(v))
        return ref_plain, torch.stack(sig_acc, dim=0).mean(dim=0)

    if strategy == "fresh_perm_plus_sandwich_plus_pad":
        sig_acc: list[torch.Tensor] = []
        for _ in range(num_sessions):
            perm = generate_permutation(hidden, dtype=dtype, device=device)["perm"]
            N_left, _ = generate_invertible_matrix(hidden, dtype, device)
            N_right, _ = generate_invertible_matrix(hidden, dtype, device)
            x = sample(samples_per_session)
            from pllo.masks.pad_generator import generate_pad as _pad
            T = _pad(tuple(x.shape), dtype, device, 1.0)
            v = ((x - T) @ N_left).index_select(dim=-1, index=perm) @ N_right
            sig_acc.append(compute_channel_signature(v))
        return ref_plain, torch.stack(sig_acc, dim=0).mean(dim=0)

    raise ValueError(f"unknown permutation-recovery strategy: {strategy}")


def _soft_assignment_top1(
    ref_signature: torch.Tensor,
    visible_signature: torch.Tensor,
    iters: int,
    temperature: float,
) -> dict[str, float]:
    """Sinkhorn-style soft assignment over per-channel signatures.

    The attacker has aligned ``visible_signature[i]`` to plaintext column ``i``
    (the recovery target is the identity), then runs row/column-normalised
    log-domain iterations on the similarity matrix and reads top-1 from the
    diagonal of the final permutation matrix.
    """
    ref = ref_signature.to(torch.float64)
    vis = visible_signature.to(torch.float64)
    ref_n = ref / ref.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    vis_n = vis / vis.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    sim = vis_n @ ref_n.T  # [H, H]
    log_k = sim / max(temperature, 1e-12)
    for _ in range(max(1, iters)):
        log_k = log_k - log_k.logsumexp(dim=-1, keepdim=True)
        log_k = log_k - log_k.logsumexp(dim=0, keepdim=True)
    assignment = log_k.exp()
    H = ref.shape[0]
    targets = torch.arange(H)
    top1_idx = assignment.argmax(dim=-1)
    top1 = float((top1_idx == targets).to(torch.float64).mean().item())
    k = min(5, H)
    topk_idx = assignment.topk(k, dim=-1).indices
    top5 = float(
        (topk_idx == targets.unsqueeze(-1))
        .any(dim=-1)
        .to(torch.float64)
        .mean()
        .item()
    )
    ranks = assignment.argsort(dim=-1, descending=True)
    rank_pos = (ranks == targets.unsqueeze(-1)).int().argmax(dim=-1).to(torch.float64)
    return {
        "top1_recovery_rate": top1,
        "top5_recovery_rate": top5,
        "mean_correct_rank": float(rank_pos.mean().item()),
        "iterations": int(iters),
        "temperature": float(temperature),
    }


def _run_permutation_recovery(
    config: AdaptiveIslandAttackConfig,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    signature_results: dict[str, dict[str, float]] = {}
    soft_results: dict[str, dict[str, float]] = {}
    for strategy in PERMUTATION_RECOVERY_STRATEGIES:
        ref_plain, visible_aligned = _build_permutation_visible(
            strategy,
            config.num_sessions,
            config.samples_per_session,
            config.hidden_size,
            config.permutation_pool_size,
            dtype,
            device,
            generator,
        )
        ref_sig = compute_channel_signature(ref_plain)
        signature_results[strategy] = recover_permutation_by_signature(
            ref_sig, visible_aligned
        )
        soft_results[strategy] = _soft_assignment_top1(
            ref_sig,
            visible_aligned,
            config.soft_assignment_iters,
            config.soft_assignment_temperature,
        )
    random_chance = 1.0 / config.hidden_size
    return {
        "random_chance_top1": random_chance,
        "signature_matching": signature_results,
        "soft_assignment": soft_results,
        "ranking_by_soft_assignment_top1_descending": sorted(
            soft_results.keys(),
            key=lambda k: soft_results[k]["top1_recovery_rate"],
            reverse=True,
        ),
    }


# ---------------------------------------------------------------------------
# Mitigation decision table
# ---------------------------------------------------------------------------


_DEFAULT_BUDGETS = {
    "linear_recovery_high_threshold": 0.20,    # rel_l2 < this ⇒ high risk
    "linear_recovery_medium_threshold": 0.60,  # rel_l2 < this ⇒ medium
    "perm_high_threshold": 0.50,               # best perm top1 > this ⇒ high
    "perm_medium_factor": 4.0,                 # best perm top1 > medium_factor × random ⇒ medium
}


_STRATEGY_REQUIRED_MITIGATIONS: dict[str, list[str]] = {
    "fixed_permutation": [
        "do not deploy without per-session fresh permutation",
        "must add a dense sandwich at Linear boundaries",
        "must pad at Linear boundaries",
    ],
    "fresh_permutation_per_session": [
        "must combine with pad at Linear boundaries",
        "should add a dense sandwich on at least one side",
        "short island lifetime + per-session rotation",
    ],
    "permutation_pool": [
        "pool size must be large; treat as fixed_permutation for small pools",
        "rotate the pool frequently",
        "must combine with pad and dense sandwich",
    ],
    "dense_sandwich": [
        "still requires fresh permutation per session",
        "still requires pad at Linear boundaries",
        "not formally secure; default-on only under tested adaptive proxy",
    ],
    "boundary_pad_only_boundary_view": [
        "pad protects the boundary view only",
        "must combine with dense sandwich or fresh permutation for activation view",
    ],
    "boundary_pad_only_activation_view": [
        "boundary pad does NOT protect this view",
        "must replace with fresh permutation + dense sandwich",
    ],
    "fresh_perm_plus_sandwich_plus_pad": [
        "fresh permutation per session is mandatory",
        "dense sandwich on both sides of the permutation island is mandatory",
        "pad must remain at Linear boundaries only — never pushed through the activation",
        "remains gated behind the ``nonlinear_mode`` feature flag",
    ],
}


def _classify_risk(
    linear_rel_l2: float,
    mlp_rel_l2: float,
    perm_soft_top1: float | None,
    random_chance: float,
    budgets: dict[str, float],
) -> tuple[str, str]:
    """Return ``(risk_level, default_on_recommendation)``."""
    best_rel_l2 = min(linear_rel_l2, mlp_rel_l2)
    if (
        best_rel_l2 < budgets["linear_recovery_high_threshold"]
        or (perm_soft_top1 is not None and perm_soft_top1 > budgets["perm_high_threshold"])
    ):
        return "high", "unsafe_default_on"
    if best_rel_l2 < budgets["linear_recovery_medium_threshold"] or (
        perm_soft_top1 is not None
        and perm_soft_top1 > budgets["perm_medium_factor"] * random_chance
    ):
        return "medium", "needs_more_evaluation"
    return "low", "acceptable_with_mitigation"


def _build_mitigation_summary(
    linear_results: dict[str, dict[str, Any]],
    mlp_results: dict[str, dict[str, Any]],
    permutation_results: dict[str, Any],
    config: AdaptiveIslandAttackConfig,
) -> dict[str, Any]:
    random_chance = 1.0 / config.hidden_size
    budgets = dict(_DEFAULT_BUDGETS)
    soft = permutation_results["soft_assignment"]
    sig = permutation_results["signature_matching"]

    def _best_perm_top1(key: str | None) -> float | None:
        if key is None:
            return None
        s_top1 = soft.get(key, {}).get("top1_recovery_rate")
        n_top1 = sig.get(key, {}).get("top1_recovery_rate")
        if s_top1 is None and n_top1 is None:
            return None
        return max(s_top1 or 0.0, n_top1 or 0.0)

    per_strategy: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        linear = linear_results[strategy]
        mlp = mlp_results[strategy]
        # Map to the permutation-recovery proxy strategy used for the
        # decision. Activation view is equivalent to fixed_permutation;
        # boundary view has no fixed permutation to recover ⇒ omit top1.
        if strategy == "boundary_pad_only_activation_view":
            perm_key = "fixed_permutation"
        elif strategy == "boundary_pad_only_boundary_view":
            perm_key = None
        else:
            perm_key = strategy
        perm_top1 = _best_perm_top1(perm_key)
        is_recommended_bundle = (
            strategy == "fresh_perm_plus_sandwich_plus_pad"
        )
        risk, recommendation = _classify_risk(
            linear["relative_l2_error"],
            mlp["relative_l2_error"],
            perm_top1,
            random_chance,
            budgets,
        )
        per_strategy.append(
            {
                "strategy": strategy,
                "best_linear_relative_l2_error": float(linear["relative_l2_error"]),
                "best_linear_cosine_similarity": float(linear["cosine_similarity"]),
                "best_mlp_relative_l2_error": float(mlp["relative_l2_error"]),
                "best_mlp_cosine_similarity": float(mlp["cosine_similarity"]),
                "best_soft_assignment_top1": (
                    float(soft.get(perm_key, {}).get("top1_recovery_rate"))
                    if perm_key is not None and perm_key in soft
                    else None
                ),
                "best_signature_matching_top1": (
                    float(sig.get(perm_key, {}).get("top1_recovery_rate"))
                    if perm_key is not None and perm_key in sig
                    else None
                ),
                "best_permutation_recovery_top1": (
                    float(perm_top1) if perm_top1 is not None else None
                ),
                "risk_level": risk,
                "default_on_recommendation": recommendation,
                "is_recommended_default_on_bundle": is_recommended_bundle,
                "required_mitigations": list(
                    _STRATEGY_REQUIRED_MITIGATIONS.get(strategy, [])
                ),
            }
        )
    full_bundle_row = next(
        (
            r for r in per_strategy
            if r["strategy"] == "fresh_perm_plus_sandwich_plus_pad"
        ),
        None,
    )
    return {
        "budgets": budgets,
        "random_chance_top1": random_chance,
        "per_strategy": per_strategy,
        "recommended_default_on_bundle": "fresh_perm_plus_sandwich_plus_pad",
        "recommended_default_on_bundle_status": (
            full_bundle_row["default_on_recommendation"]
            if full_bundle_row is not None
            else None
        ),
        "recommended_default_on_bundle_risk_level": (
            full_bundle_row["risk_level"] if full_bundle_row is not None else None
        ),
        "recommended_default_on_candidate": (
            "fresh_permutation + dense_sandwich + pad at Linear boundaries"
        ),
        "default_on_caveat": (
            "Safe-to-default-on only means \"within the tested adaptive proxy"
            " attackers (ridge linear inverter, small MLP inverter,"
            " Sinkhorn-style permutation recovery)\". This is NOT a formal"
            " security claim and NOT a TEE measurement."
        ),
    }


# ---------------------------------------------------------------------------
# Comparison with Stage 5.2b naive proxy (read from this run's data)
# ---------------------------------------------------------------------------


def _comparison_with_naive_proxy(
    permutation_results: dict[str, Any],
) -> dict[str, Any]:
    """Compare same-strategy ``soft_assignment`` vs. ``signature_matching`` top1."""
    sig = permutation_results["signature_matching"]
    soft = permutation_results["soft_assignment"]
    per_strategy: dict[str, dict[str, float]] = {}
    for strategy in PERMUTATION_RECOVERY_STRATEGIES:
        s_top1 = sig.get(strategy, {}).get("top1_recovery_rate", 0.0)
        a_top1 = soft.get(strategy, {}).get("top1_recovery_rate", 0.0)
        per_strategy[strategy] = {
            "naive_signature_matching_top1": float(s_top1),
            "adaptive_soft_assignment_top1": float(a_top1),
            "absolute_uplift": float(a_top1 - s_top1),
        }
    return {
        "per_strategy": per_strategy,
        "note": (
            "Stage 5.4 reproduces Stage 5.2b's signature-matching proxy on the"
            " same data and compares it against the Sinkhorn-style"
            " soft-assignment adaptive attacker. Larger uplift means the"
            " adaptive attacker is strictly stronger on that strategy."
        ),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_adaptive_island_attacks(
    config: AdaptiveIslandAttackConfig,
) -> dict[str, Any]:
    """Run all three adaptive attackers across all strategies and return a report."""
    torch.manual_seed(config.seed)
    dtype = torch_dtype_from_string(config.dtype, config.device)
    device = torch.device(config.device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    linear_results: dict[str, dict[str, Any]] = {}
    mlp_results: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        X_train, V_train, X_test, V_test = _build_dataset(
            strategy,
            config.num_train_samples,
            config.num_test_samples,
            config.samples_per_session,
            config.hidden_size,
            config.permutation_pool_size,
            config.pad_scale,
            dtype,
            device,
            generator,
        )
        linear_results[strategy] = _run_linear_inverter(
            strategy, X_train, V_train, X_test, V_test, config
        )
        mlp_results[strategy] = _run_mlp_inverter(
            strategy,
            X_train,
            V_train,
            X_test,
            V_test,
            config,
            seed=config.seed + abs(hash(strategy)) % (2**16),
        )

    permutation_results = _run_permutation_recovery(config, dtype, device, generator)

    # ``improvement_over_linear`` per strategy.
    improvement: dict[str, dict[str, float]] = {}
    for strategy in STRATEGIES:
        l_l2 = linear_results[strategy]["relative_l2_error"]
        m_l2 = mlp_results[strategy]["relative_l2_error"]
        improvement[strategy] = {
            "linear_relative_l2_error": float(l_l2),
            "mlp_relative_l2_error": float(m_l2),
            # Positive ⇒ MLP strictly improves (lowers reconstruction error).
            "mlp_minus_linear_relative_l2_error": float(m_l2 - l_l2),
            "mlp_improves_over_linear": bool(m_l2 < l_l2),
        }

    mitigation = _build_mitigation_summary(
        linear_results, mlp_results, permutation_results, config
    )
    comparison = _comparison_with_naive_proxy(permutation_results)

    linear_ranking = sorted(
        STRATEGIES,
        key=lambda s: linear_results[s]["relative_l2_error"],
    )
    mlp_ranking = sorted(
        STRATEGIES,
        key=lambda s: mlp_results[s]["relative_l2_error"],
    )

    return {
        "config": asdict(config),
        "threat_model": THREAT_MODEL,
        "structured_data": {
            "num_train_samples": int(config.num_train_samples),
            "num_test_samples": int(config.num_test_samples),
            "hidden_size": int(config.hidden_size),
            "samples_per_session": int(config.samples_per_session),
            "channel_mean_range": [-2.0, 2.0],
            "channel_scale_range": [0.5, 3.0],
            "channel_skew_profile": "0.5 * cos(linspace(0, pi, hidden))",
            "distribution_summary": (
                "X[:, j] = mean_j + scale_j * (noise + skew_j * (noise**2 - 1))"
            ),
        },
        "linear_inverter": {
            "strategies": linear_results,
            "ranking_by_relative_l2_error_ascending": linear_ranking,
            "weakest_mitigation": linear_ranking[0],
        },
        "mlp_inverter": {
            "strategies": mlp_results,
            "improvement_over_linear": improvement,
            "ranking_by_relative_l2_error_ascending": mlp_ranking,
            "weakest_mitigation": mlp_ranking[0],
        },
        "permutation_recovery": permutation_results,
        "mitigation_summary": mitigation,
        "comparison_with_naive_proxy": comparison,
        "limitations": [
            "These are adaptive/proxy attacks, not formal security proofs.",
            "The attacks use synthetic structured channel data, not full"
            " real-model activation traces.",
            "No adaptive black-box querying of a deployed LLM is implemented.",
            "No side-channel attack is implemented.",
            "No real TEE isolation is evaluated.",
            "Dense sandwiching reduces tested recovery but does not imply"
            " semantic security.",
            "Default-on recommendations are conditional on the tested threat"
            " model only.",
        ],
    }


__all__ = [
    "AdaptiveIslandAttackConfig",
    "STRATEGIES",
    "PERMUTATION_RECOVERY_STRATEGIES",
    "THREAT_MODEL",
    "generate_structured_channel_data",
    "run_adaptive_island_attacks",
]
