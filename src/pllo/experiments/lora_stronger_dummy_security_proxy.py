"""Stage 7.4 — stronger-dummy LoRA security proxy.

Evaluates whether Stage 7.4 stronger dummy distributions actually reduce
the spectral / gradient rank-inference risk reported in Stage 7.2 / 7.3,
and whether a dummy-strategy classifier can recover which strategy the
trusted side used from the GPU-visible padded transcript. Four sub-attacks:

1. **Spectral rank inference** — ensemble of attackers (SVD-cliff,
   energy-ratio, elbow) applied to ``A_pad_tilde`` / ``B_pad_tilde``.
   Reports inferred rank accuracy + risk per strategy.

2. **Gradient rank inference** — same ensemble applied to
   ``grad_A_pad_tilde`` / ``grad_B_pad_tilde`` derived from synthetic
   upstream gradients.

3. **Dummy strategy classification** — given the visible spectrum of
   ``A_pad_tilde`` / ``B_pad_tilde``, can a nearest-bucket-mean
   classifier predict which dummy strategy was used? Higher accuracy
   ⇒ higher leakage of the trusted-side secret choice.

4. **Cross-layer linkage** — same Stage 7.3 fingerprint AUC across
   multiple LoRA-augmented modules, evaluated per dummy strategy.

All metrics are PROXY hardenings. The attacker model is generous (sees
all visible transcripts + uses bucket-mean labels for classification).
Reports are honest about residual risk; conservative verdicts per
constraint 12 (no overclaim that true_rank is hidden).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from pllo.ops.lora import (
    LoRAConfig,
    LoRAState,
    MaskedLoRAForwardConfig,
    create_masked_lora_state,
    init_lora_adapters,
)
from pllo.ops.lora_backward import (
    masked_lora_backward,
    transform_upstream_gradient,
)
from pllo.ops.lora_dummy_strategies import (
    StrongDummyConfig,
    VALID_STRONG_DUMMY_STRATEGIES,
    create_stronger_rank_padded_lora_adapters,
    validate_strong_dummy_config,
)
from pllo.ops.lora_rank_padding import _effective_alpha


@dataclass
class StrongerDummySecurityProxyConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    d_in: int = 32
    d_out: int = 16
    true_ranks: tuple[int, ...] = (2, 4, 8)
    padded_rank: int = 16
    alpha: float = 1.0
    num_trials: int = 32
    num_lora_modules_for_linkage: int = 4
    use_pad: bool = True
    pad_scale: float = 1.0
    dummy_scale: float = 1.0
    noise_scale: float = 1e-3
    spectrum_match_strength: float = 1.0
    dummy_strategies: tuple[str, ...] = field(
        default_factory=lambda: tuple(VALID_STRONG_DUMMY_STRATEGIES)
    )
    dtype: str = "float64"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Spectral inference helpers
# ---------------------------------------------------------------------------


def _svd_cliff_index(t: torch.Tensor) -> tuple[int, float]:
    if t.numel() == 0:
        return 0, 0.0
    sv = torch.linalg.svdvals(t)
    if sv.numel() < 2:
        return int(sv.numel()), 1.0
    ratios = (sv[1:] / sv[:-1].clamp_min(1e-30)).cpu()
    drop_idx = int(torch.argmin(ratios).item())
    return drop_idx + 1, float(ratios[drop_idx].item())


def _energy_ratio_index(t: torch.Tensor, threshold: float = 0.95) -> int:
    if t.numel() == 0:
        return 0
    sv = torch.linalg.svdvals(t)
    if sv.numel() == 0:
        return 0
    total = (sv * sv).sum().clamp_min(1e-30).item()
    cum = 0.0
    for i, s in enumerate(sv.tolist()):
        cum += float(s) * float(s)
        if cum / total >= threshold:
            return i + 1
    return int(sv.numel())


def _elbow_index(t: torch.Tensor) -> int:
    """Elbow detector via second-difference of log singular values."""
    if t.numel() == 0:
        return 0
    sv = torch.linalg.svdvals(t)
    if sv.numel() < 3:
        return int(sv.numel())
    log_sv = sv.clamp_min(1e-30).log().cpu()
    second_diff = log_sv[2:] - 2.0 * log_sv[1:-1] + log_sv[:-2]
    # Most negative second diff → elbow.
    idx = int(torch.argmin(second_diff).item())
    return idx + 2


def _build_state(
    d_in: int, d_out: int, padded_rank: int,
    dtype: torch.dtype, device: torch.device,
    *, use_pad: bool, pad_scale: float, seq_len: int,
    generator: torch.Generator,
) -> LoRAState:
    inner = LoRAConfig(
        d_in=d_in, d_out=d_out, rank=padded_rank, alpha=1.0,
        dtype="float64" if dtype == torch.float64 else "float32",
        device=device.type,
    )
    fcfg = MaskedLoRAForwardConfig(
        use_pad=use_pad, fresh_u_per_call=True,
        fresh_masks_per_call=True, pad_scale=pad_scale,
        dtype=inner.dtype, device=inner.device,
    )
    return create_masked_lora_state(
        inner, fcfg, seq_len=seq_len, state=None, generator=generator,
    )


def _mask_padded(
    a_pad: torch.Tensor, b_pad: torch.Tensor, state: LoRAState,
) -> tuple[torch.Tensor, torch.Tensor]:
    return state.n_in_inv @ a_pad @ state.u, state.u_inv @ b_pad @ state.n_out


def _spectrum_features(
    a_tilde: torch.Tensor, b_tilde: torch.Tensor, k: int = 4,
) -> list[float]:
    """Return the top-``k`` normalised singular values of A_tilde + B_tilde
    as a feature vector for the dummy-strategy classifier.
    """
    sv_a = torch.linalg.svdvals(a_tilde)
    sv_b = torch.linalg.svdvals(b_tilde)
    sv_a = sv_a / sv_a.max().clamp_min(1e-30)
    sv_b = sv_b / sv_b.max().clamp_min(1e-30)
    out: list[float] = []
    for sv in (sv_a, sv_b):
        for i in range(k):
            if i < sv.numel():
                out.append(float(sv[i].item()))
            else:
                out.append(0.0)
    return out


# ---------------------------------------------------------------------------
# Proxy 1 — spectral rank inference per strategy
# ---------------------------------------------------------------------------


def _spectral_rank_inference(
    config: StrongerDummySecurityProxyConfig,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    seq_len = max(4, config.d_in // 4)
    for strategy in config.dummy_strategies:
        for true_rank in config.true_ranks:
            gen = torch.Generator(device="cpu").manual_seed(
                config.seed * 311 + true_rank * 13 + hash(strategy) % 9999
            )
            real_cfg = LoRAConfig(
                d_in=config.d_in, d_out=config.d_out, rank=true_rank,
                alpha=config.alpha,
                dtype=config.dtype, device=config.device,
            )
            dummy_cfg = StrongDummyConfig(
                true_rank=true_rank, padded_rank=config.padded_rank,
                dummy_strategy=strategy,
                dummy_scale=config.dummy_scale,
                noise_scale=config.noise_scale,
                spectrum_match_strength=config.spectrum_match_strength,
                fresh_dummy_per_step=True,
                dtype=config.dtype, device=config.device,
            )
            validate_strong_dummy_config(dummy_cfg)

            cliff_a: list[int] = []
            cliff_b: list[int] = []
            energy_b: list[int] = []
            elbow_b: list[int] = []
            ensemble_correct = 0
            ensemble_total = 0
            cliff_ratios: list[float] = []
            for _ in range(config.num_trials):
                a_real, b_real = init_lora_adapters(real_cfg, generator=gen)
                b_real = b_real + 0.1 * torch.randn(
                    true_rank, config.d_out, generator=gen,
                    dtype=dtype, device=device,
                )
                pack = create_stronger_rank_padded_lora_adapters(
                    a_real, b_real, dummy_cfg, generator=gen,
                )
                state = _build_state(
                    config.d_in, config.d_out, config.padded_rank,
                    dtype, device,
                    use_pad=config.use_pad, pad_scale=config.pad_scale,
                    seq_len=seq_len, generator=gen,
                )
                a_tilde, b_tilde = _mask_padded(
                    pack["a_pad"], pack["b_pad"], state,
                )
                inferred_a, _ = _svd_cliff_index(a_tilde)
                inferred_b, cliff_ratio_b = _svd_cliff_index(b_tilde)
                inferred_energy = _energy_ratio_index(b_tilde, 0.99)
                inferred_elbow = _elbow_index(b_tilde)
                cliff_a.append(inferred_a)
                cliff_b.append(inferred_b)
                cliff_ratios.append(cliff_ratio_b)
                energy_b.append(inferred_energy)
                elbow_b.append(inferred_elbow)
                # Ensemble vote — majority of the three detectors on B_tilde.
                votes = [inferred_b, inferred_energy, inferred_elbow]
                # Most-common vote.
                vote_value = max(set(votes), key=votes.count)
                if vote_value == true_rank:
                    ensemble_correct += 1
                ensemble_total += 1

            def _mean_int(xs):
                return float(sum(xs) / max(1, len(xs)))

            def _mean(xs):
                return float(sum(xs) / max(1, len(xs)))

            acc_cliff = sum(1 for v in cliff_b if v == true_rank) / max(
                1, len(cliff_b)
            )
            acc_energy = sum(1 for v in energy_b if v == true_rank) / max(
                1, len(energy_b)
            )
            acc_elbow = sum(1 for v in elbow_b if v == true_rank) / max(
                1, len(elbow_b)
            )
            acc_ensemble = float(
                ensemble_correct / max(1, ensemble_total)
            )

            # Conservative risk verdict (constraint 12).
            risk: str
            if strategy == "zero_dummy":
                risk = "high"
            elif acc_ensemble >= 0.5 or acc_cliff >= 0.5:
                risk = "high"
            elif acc_ensemble >= 0.2 or acc_cliff >= 0.2:
                risk = "medium"
            else:
                risk = "needs_more_evaluation"

            rows.append({
                "dummy_strategy": strategy,
                "true_rank": true_rank,
                "inferred_rank_cliff_b_mean": _mean_int(cliff_b),
                "inferred_rank_energy_b_mean": _mean_int(energy_b),
                "inferred_rank_elbow_b_mean": _mean_int(elbow_b),
                "cliff_inference_accuracy": float(acc_cliff),
                "energy_inference_accuracy": float(acc_energy),
                "elbow_inference_accuracy": float(acc_elbow),
                "ensemble_inference_accuracy": acc_ensemble,
                "confidence_gap_b": _mean(cliff_ratios),
                "risk_level": risk,
            })
    return {
        "rows": rows,
        "interpretation": (
            "Spectral rank inference with three detectors (SVD-cliff,"
            " 99%-energy, log-elbow) and a majority ensemble. Higher"
            " accuracy ⇒ true_rank more readable from the visible spectrum."
            " Conservative verdicts per requirement 12."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 2 — gradient rank inference per strategy
# ---------------------------------------------------------------------------


def _gradient_rank_inference(
    config: StrongerDummySecurityProxyConfig,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    seq_len = max(4, config.d_in // 4)
    for strategy in config.dummy_strategies:
        for true_rank in config.true_ranks:
            gen = torch.Generator(device="cpu").manual_seed(
                config.seed * 757 + true_rank * 11 + hash(strategy) % 6661
            )
            real_cfg = LoRAConfig(
                d_in=config.d_in, d_out=config.d_out, rank=true_rank,
                alpha=config.alpha,
                dtype=config.dtype, device=config.device,
            )
            dummy_cfg = StrongDummyConfig(
                true_rank=true_rank, padded_rank=config.padded_rank,
                dummy_strategy=strategy,
                dummy_scale=config.dummy_scale,
                noise_scale=config.noise_scale,
                spectrum_match_strength=config.spectrum_match_strength,
                fresh_dummy_per_step=True,
                dtype=config.dtype, device=config.device,
            )
            inferred_grad_a: list[int] = []
            inferred_grad_b: list[int] = []
            ensemble_correct = 0
            ensemble_total = 0
            for _ in range(config.num_trials):
                a_real, b_real = init_lora_adapters(real_cfg, generator=gen)
                b_real = b_real + 0.1 * torch.randn(
                    true_rank, config.d_out, generator=gen,
                    dtype=dtype, device=device,
                )
                pack = create_stronger_rank_padded_lora_adapters(
                    a_real, b_real, dummy_cfg, generator=gen,
                )
                state = _build_state(
                    config.d_in, config.d_out, config.padded_rank,
                    dtype, device,
                    use_pad=config.use_pad, pad_scale=config.pad_scale,
                    seq_len=seq_len, generator=gen,
                )
                a_tilde, b_tilde = _mask_padded(
                    pack["a_pad"], pack["b_pad"], state,
                )
                x = torch.randn(
                    seq_len, config.d_in, generator=gen,
                    dtype=dtype, device=device,
                )
                g = torch.randn(
                    seq_len, config.d_out, generator=gen,
                    dtype=dtype, device=device,
                )
                if state.pad is None:
                    x_tilde = x @ state.n_in
                else:
                    x_tilde = (x - state.pad) @ state.n_in
                grad_y_tilde = transform_upstream_gradient(g, state.n_out)
                masked = masked_lora_backward(
                    x_tilde, a_tilde, b_tilde, grad_y_tilde,
                    alpha=_effective_alpha(
                        config.alpha, true_rank, config.padded_rank,
                    ),
                    recover_grad_x=False,
                )
                inf_ga, _ = _svd_cliff_index(masked["grad_a_tilde"])
                inf_gb, _ = _svd_cliff_index(masked["grad_b_tilde"])
                inferred_grad_a.append(inf_ga)
                inferred_grad_b.append(inf_gb)
                votes = [inf_ga, inf_gb,
                         _energy_ratio_index(masked["grad_b_tilde"], 0.99)]
                vote_value = max(set(votes), key=votes.count)
                if vote_value == true_rank:
                    ensemble_correct += 1
                ensemble_total += 1

            def _mean_int(xs):
                return float(sum(xs) / max(1, len(xs)))

            acc_grad_a = sum(
                1 for v in inferred_grad_a if v == true_rank
            ) / max(1, len(inferred_grad_a))
            acc_grad_b = sum(
                1 for v in inferred_grad_b if v == true_rank
            ) / max(1, len(inferred_grad_b))
            acc_ensemble = float(
                ensemble_correct / max(1, ensemble_total)
            )
            if strategy == "zero_dummy":
                risk = "high"
            elif acc_ensemble >= 0.5 or max(acc_grad_a, acc_grad_b) >= 0.5:
                risk = "high"
            elif acc_ensemble >= 0.2:
                risk = "medium"
            else:
                risk = "needs_more_evaluation"
            rows.append({
                "dummy_strategy": strategy,
                "true_rank": true_rank,
                "inferred_rank_grad_a_cliff_mean": _mean_int(inferred_grad_a),
                "inferred_rank_grad_b_cliff_mean": _mean_int(inferred_grad_b),
                "grad_a_cliff_accuracy": float(acc_grad_a),
                "grad_b_cliff_accuracy": float(acc_grad_b),
                "gradient_rank_inference_accuracy": acc_ensemble,
                "risk_level": risk,
            })
    return {
        "rows": rows,
        "interpretation": (
            "Gradient-side spectral rank inference. Gradients depend on"
            " (X, A, B, G); their spectrum may differ from the static"
            " adapter spectrum. Conservative verdicts per requirement 12."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 3 — dummy strategy classification
# ---------------------------------------------------------------------------


def _dummy_strategy_classification(
    config: StrongerDummySecurityProxyConfig,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    true_rank = config.true_ranks[0]
    seq_len = max(4, config.d_in // 4)
    samples: list[tuple[str, list[float]]] = []
    for strategy in config.dummy_strategies:
        gen = torch.Generator(device="cpu").manual_seed(
            config.seed * 991 + hash(strategy) % 7919
        )
        real_cfg = LoRAConfig(
            d_in=config.d_in, d_out=config.d_out, rank=true_rank,
            alpha=config.alpha,
            dtype=config.dtype, device=config.device,
        )
        dummy_cfg = StrongDummyConfig(
            true_rank=true_rank, padded_rank=config.padded_rank,
            dummy_strategy=strategy,
            dummy_scale=config.dummy_scale,
            noise_scale=config.noise_scale,
            spectrum_match_strength=config.spectrum_match_strength,
            fresh_dummy_per_step=True,
            dtype=config.dtype, device=config.device,
        )
        for _ in range(config.num_trials):
            a_real, b_real = init_lora_adapters(real_cfg, generator=gen)
            b_real = b_real + 0.1 * torch.randn(
                true_rank, config.d_out, generator=gen,
                dtype=dtype, device=device,
            )
            pack = create_stronger_rank_padded_lora_adapters(
                a_real, b_real, dummy_cfg, generator=gen,
            )
            state = _build_state(
                config.d_in, config.d_out, config.padded_rank,
                dtype, device,
                use_pad=config.use_pad, pad_scale=config.pad_scale,
                seq_len=seq_len, generator=gen,
            )
            a_tilde, b_tilde = _mask_padded(
                pack["a_pad"], pack["b_pad"], state,
            )
            samples.append((strategy, _spectrum_features(a_tilde, b_tilde)))

    # Nearest-bucket-mean classifier.
    bucket_to_features: dict[str, list[list[float]]] = {
        s: [] for s in config.dummy_strategies
    }
    for label, feat in samples:
        bucket_to_features[label].append(feat)
    bucket_means: dict[str, list[float]] = {}
    for label, feats in bucket_to_features.items():
        if not feats:
            continue
        dim = len(feats[0])
        means = [sum(f[i] for f in feats) / len(feats) for i in range(dim)]
        bucket_means[label] = means

    def _dist(a: list[float], b: list[float]) -> float:
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    correct = 0
    total = 0
    confusion: dict[str, dict[str, int]] = {
        s: {t: 0 for t in config.dummy_strategies}
        for s in config.dummy_strategies
    }
    for label, feat in samples:
        best = None
        best_dist = float("inf")
        for candidate, mean in bucket_means.items():
            d = _dist(feat, mean)
            if d < best_dist:
                best_dist = d
                best = candidate
        if best is not None:
            confusion[label][best] += 1
            if best == label:
                correct += 1
            total += 1
    accuracy = float(correct / max(1, total))
    chance = 1.0 / max(1, len(config.dummy_strategies))
    if accuracy >= chance + 0.35:
        risk = "high"
    elif accuracy >= chance + 0.15:
        risk = "medium"
    elif accuracy >= chance + 0.05:
        risk = "low"
    else:
        risk = "low"
    return {
        "strategy_classification_accuracy": accuracy,
        "random_chance_baseline": chance,
        "risk_level": risk,
        "confusion_counts": confusion,
        "interpretation": (
            "Nearest-bucket-mean classifier on top-k normalised singular"
            " values of A_pad_tilde / B_pad_tilde. Higher accuracy ⇒ the"
            " visible spectrum carries enough signal to discriminate the"
            " trusted-side dummy strategy choice."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 4 — cross-layer linkage per strategy
# ---------------------------------------------------------------------------


_LORA_TARGETS_ALL = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _module_io_shapes(
    module: str, hidden: int, inter: int,
) -> tuple[int, int]:
    if module in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return hidden, hidden
    if module in ("gate_proj", "up_proj"):
        return hidden, inter
    if module == "down_proj":
        return inter, hidden
    raise ValueError(f"unknown module {module!r}")


def _cross_layer_linkage(
    config: StrongerDummySecurityProxyConfig,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    true_rank = config.true_ranks[0]
    hidden = config.d_in
    inter = max(config.d_in, config.d_out)
    seq_len = max(4, hidden // 4)
    rows: list[dict[str, Any]] = []
    for strategy in config.dummy_strategies:
        gen = torch.Generator(device="cpu").manual_seed(
            config.seed * 4001 + hash(strategy) % 7349
        )
        dummy_cfg = StrongDummyConfig(
            true_rank=true_rank, padded_rank=config.padded_rank,
            dummy_strategy=strategy,
            dummy_scale=config.dummy_scale,
            noise_scale=config.noise_scale,
            spectrum_match_strength=config.spectrum_match_strength,
            fresh_dummy_per_step=True,
            dtype=config.dtype, device=config.device,
        )
        same_module_distances: list[float] = []
        different_module_distances: list[float] = []
        retrieval_hits = 0
        retrieval_total = 0
        modules = _LORA_TARGETS_ALL[: config.num_lora_modules_for_linkage]
        for _ in range(config.num_trials):
            visible_a_norms: dict[tuple[int, str], float] = {}
            visible_b_norms: dict[tuple[int, str], float] = {}
            for layer in range(2):
                for mod in modules:
                    d_in, d_out = _module_io_shapes(mod, hidden, inter)
                    real_cfg = LoRAConfig(
                        d_in=d_in, d_out=d_out, rank=true_rank,
                        alpha=config.alpha,
                        dtype=config.dtype, device=config.device,
                    )
                    a, b = init_lora_adapters(real_cfg, generator=gen)
                    b = b + 0.1 * torch.randn(
                        true_rank, d_out, generator=gen,
                        dtype=dtype, device=device,
                    )
                    pack = create_stronger_rank_padded_lora_adapters(
                        a, b, dummy_cfg, generator=gen,
                    )
                    state = _build_state(
                        d_in, d_out, config.padded_rank,
                        dtype, device,
                        use_pad=config.use_pad, pad_scale=config.pad_scale,
                        seq_len=seq_len, generator=gen,
                    )
                    a_tilde, b_tilde = _mask_padded(
                        pack["a_pad"], pack["b_pad"], state,
                    )
                    visible_a_norms[(layer, mod)] = float(
                        a_tilde.norm().item()
                    )
                    visible_b_norms[(layer, mod)] = float(
                        b_tilde.norm().item()
                    )

            # Same / different module distance.
            ordered = list(visible_a_norms.items())
            for i, ((layer_i, mod_i), fp_a_i) in enumerate(ordered):
                fp_b_i = visible_b_norms[(layer_i, mod_i)]
                best_j = -1
                best_score = float("inf")
                truth_j = -1
                for j, ((layer_j, mod_j), fp_a_j) in enumerate(ordered):
                    if i == j:
                        continue
                    fp_b_j = visible_b_norms[(layer_j, mod_j)]
                    score = abs(fp_a_i - fp_a_j) + abs(fp_b_i - fp_b_j)
                    if score < best_score:
                        best_score = score
                        best_j = j
                    if mod_j == mod_i and layer_j != layer_i:
                        if truth_j == -1:
                            truth_j = j
                if truth_j != -1:
                    retrieval_total += 1
                    if best_j == truth_j:
                        retrieval_hits += 1

            module_fps_a: dict[str, list[float]] = {}
            for (_, mod), fp in visible_a_norms.items():
                module_fps_a.setdefault(mod, []).append(fp)
            for mod, fps in module_fps_a.items():
                for ii in range(len(fps)):
                    for jj in range(ii + 1, len(fps)):
                        same_module_distances.append(abs(fps[ii] - fps[jj]))
            module_names = list(module_fps_a.keys())
            for ii in range(len(module_names)):
                for jj in range(ii + 1, len(module_names)):
                    for fp_a in module_fps_a[module_names[ii]]:
                        for fp_b in module_fps_a[module_names[jj]]:
                            different_module_distances.append(
                                abs(fp_a - fp_b)
                            )

        wins = 0
        total = 0
        for s in same_module_distances:
            for d in different_module_distances:
                if s < d:
                    wins += 1
                elif s == d:
                    wins += 0.5
                total += 1
        auc = float(wins / max(1, total))
        retrieval_top1 = float(retrieval_hits / max(1, retrieval_total))
        same_mean = (
            float(sum(same_module_distances) / len(same_module_distances))
            if same_module_distances else 0.0
        )
        diff_mean = (
            float(sum(different_module_distances) / len(different_module_distances))
            if different_module_distances else 0.0
        )
        if auc > 0.85 or retrieval_top1 > 0.6:
            risk = "high"
        elif auc > 0.65 or retrieval_top1 > 0.3:
            risk = "medium"
        else:
            risk = "low"
        rows.append({
            "dummy_strategy": strategy,
            "layer_linkability_auc": auc,
            "module_identity_retrieval_top1": retrieval_top1,
            "same_module_similarity": same_mean,
            "different_module_similarity": diff_mean,
            "rank_group_retrieval_top1": retrieval_top1,
            "risk_level": risk,
        })
    return {
        "rows": rows,
        "interpretation": (
            "Cross-layer linkage proxy under fresh masks per module. AUC"
            " ≈ 0.5 ⇒ same-module / different-module distance distributions"
            " overlap; retrieval_top1 near 1/(num_modules - 1) ⇒ no"
            " systematic linkage. Stronger dummies aim to bring both"
            " statistics closer to the random baseline."
        ),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "These are stronger-dummy proxy attacks, not formal security proofs.",
    "Padded rank r_pad remains visible from tensor shape — only true_rank is hidden.",
    "Spectral hardening does not imply cryptographic hiding.",
    "Dummy strategy classification uses a generous bucket-mean attacker model.",
    "No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.",
    "Optimizer state remains trusted-only and is sized to true_rank.",
    "Hardware side-channels (cache / power / EM) are NOT evaluated.",
    "No full Qwen / TinyLlama / LLaMA LoRA fine-tuning is evaluated; this is a single-linear + cross-layer proxy.",
    "Adapter is NEVER merged into the public base weight W.",
]


def run_lora_stronger_dummy_security_proxy(
    config: StrongerDummySecurityProxyConfig,
) -> dict[str, Any]:
    for strategy in config.dummy_strategies:
        if strategy not in VALID_STRONG_DUMMY_STRATEGIES:
            raise ValueError(
                f"unknown dummy_strategy {strategy!r}; expected one of"
                f" {VALID_STRONG_DUMMY_STRATEGIES}"
            )
    for r in config.true_ranks:
        if r <= 0 or r > config.padded_rank:
            raise ValueError(
                f"each true_rank must satisfy 0 < r <= padded_rank"
                f" ({config.padded_rank}), got {r}"
            )

    spectral = _spectral_rank_inference(config)
    gradient = _gradient_rank_inference(config)
    classification = _dummy_strategy_classification(config)
    linkage = _cross_layer_linkage(config)

    # Aggregate verdicts.
    spectral_risks = [r["risk_level"] for r in spectral["rows"]]
    gradient_risks = [r["risk_level"] for r in gradient["rows"]]
    linkage_risks = [r["risk_level"] for r in linkage["rows"]]

    def _worst(risks: list[str]) -> str:
        for level in ("high", "medium", "needs_more_evaluation", "low"):
            if level in risks:
                return level
        return "low"

    return {
        "config": asdict(config),
        "scope": (
            "stronger-dummy LoRA security proxy: spectral inference,"
            " gradient inference, dummy strategy classification,"
            " cross-layer linkage"
        ),
        "spectral_rank_inference": spectral,
        "gradient_rank_inference": gradient,
        "dummy_strategy_classification": classification,
        "cross_layer_linkage": linkage,
        "interpretation": {
            "spectral_summary": (
                f"Worst spectral rank inference risk across strategies:"
                f" **{_worst(spectral_risks)}**."
            ),
            "gradient_summary": (
                f"Worst gradient rank inference risk across strategies:"
                f" **{_worst(gradient_risks)}**."
            ),
            "dummy_strategy_classification_summary": (
                f"Dummy strategy classifier accuracy:"
                f" {classification['strategy_classification_accuracy']:.3f}"
                f" (chance {classification['random_chance_baseline']:.3f}),"
                f" risk **{classification['risk_level']}**."
            ),
            "cross_layer_linkage_summary": (
                f"Worst cross-layer linkage risk across strategies:"
                f" **{_worst(linkage_risks)}**."
            ),
            "true_rank_shape_hidden_when_padded": True,
            "padded_rank_visibility_note": (
                "padded_rank itself remains visible from tensor shape;"
                " hiding it is out of Stage 7.4 scope."
            ),
            "merge_adapter_into_w": False,
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_dummy_hardening": (
            "spectral-rank-hardening-proxy-evaluated, not formal"
        ),
        "lora_stronger_dummy_security_status": "implemented",
        "lora_spectral_rank_hardening_status": "proxy-evaluated",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.5 — paper artifact consolidation + projected vs measured runtime alignment.",
            "Stage 7.x — heterogeneous padded_rank across modules / layers to hide padded_rank itself.",
        ],
    }


def stronger_dummy_security_csv_rows(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cfg = report["config"]
    for k, v in cfg.items():
        if isinstance(v, (tuple, list)):
            v = "|".join(str(x) for x in v)
        rows.append({
            "section": "config",
            "attack": "n/a",
            "strategy": "n/a",
            "metric": k,
            "value": v,
            "notes": "",
        })
    for r in report["spectral_rank_inference"]["rows"]:
        for k, v in r.items():
            if k in ("dummy_strategy", "true_rank"):
                continue
            rows.append({
                "section": "spectral_rank_inference",
                "attack": "spectral",
                "strategy": (
                    f"{r['dummy_strategy']}.true_rank_{r['true_rank']}"
                ),
                "metric": k,
                "value": v,
                "notes": "",
            })
    for r in report["gradient_rank_inference"]["rows"]:
        for k, v in r.items():
            if k in ("dummy_strategy", "true_rank"):
                continue
            rows.append({
                "section": "gradient_rank_inference",
                "attack": "gradient",
                "strategy": (
                    f"{r['dummy_strategy']}.true_rank_{r['true_rank']}"
                ),
                "metric": k,
                "value": v,
                "notes": "",
            })
    cls = report["dummy_strategy_classification"]
    for k in (
        "strategy_classification_accuracy",
        "random_chance_baseline",
        "risk_level",
    ):
        rows.append({
            "section": "dummy_strategy_classification",
            "attack": "classification",
            "strategy": "aggregate",
            "metric": k,
            "value": cls[k],
            "notes": "",
        })
    for r in report["cross_layer_linkage"]["rows"]:
        for k, v in r.items():
            if k == "dummy_strategy":
                continue
            rows.append({
                "section": "cross_layer_linkage",
                "attack": "linkage",
                "strategy": r["dummy_strategy"],
                "metric": k,
                "value": v,
                "notes": "",
            })
    for k, v in report["interpretation"].items():
        rows.append({
            "section": "interpretation",
            "attack": "summary",
            "strategy": "summary",
            "metric": k,
            "value": v,
            "notes": "",
        })
    return rows


__all__ = [
    "StrongerDummySecurityProxyConfig",
    "run_lora_stronger_dummy_security_proxy",
    "stronger_dummy_security_csv_rows",
]
