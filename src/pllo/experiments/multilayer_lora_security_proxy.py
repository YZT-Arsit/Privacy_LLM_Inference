"""Stage 7.3 — multi-layer LoRA cross-layer security proxy.

Three sub-attacks evaluate residual leakage when Stage 7.0 / 7.1 / 7.2
are layered together across multiple LoRA-augmented linears in a tiny
Transformer-style block stack:

1. **Cross-layer adapter linkage** — an attacker sees ``A_pad_tilde`` /
   ``B_pad_tilde`` from every module across multiple layers. Does the
   per-module mask strategy bind two different modules' transcripts
   together (e.g., shared ``U`` across layers, shared ``N_in`` / ``N_out``
   per session)? Per requirement 12, we only report shape-level + masked
   tensor similarity; we do NOT publish the underlying tensors.

2. **Heterogeneous true rank with shared padded rank** — set q/k/v/o
   true_rank=r_a and gate/up/down true_rank=r_b with a single padded_rank
   across all modules. Verify the shape-level rank is invariant per
   requirement 12, and report the residual spectral / gradient rank
   inference per module.

3. **Multi-step membership linkability** — across multiple optimization
   steps, can the attacker distinguish the same private sample being
   replayed from a different sample? Reuses the Stage 7.2 single-linear
   methodology but aggregates per-module distances across layers.

All metrics are proxy. The masking + padding equations do not bound a
real attacker; this is a ranking-strategy proxy. JSON / CSV / Markdown
outputs publish only summary metrics + fingerprints — no plaintext
adapters, gradients, masks, or private samples.
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
from pllo.ops.lora_rank_padding import (
    RankPaddingConfig,
    VALID_DUMMY_STRATEGIES,
    _effective_alpha,
    create_rank_padded_lora_adapters,
    validate_rank_padding_config,
)


VALID_LINKAGE_STRATEGIES: tuple[str, ...] = (
    "fixed_masks_shared_u",
    "independent_u_per_layer",
    "fresh_masks_independent_u",
    "rank_padding_full_bundle",
)


@dataclass
class MultiLayerLoRASecurityProxyConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    num_layers: int = 2
    hidden_size: int = 32
    intermediate_size: int = 64
    true_ranks: tuple[int, ...] = (2, 4)
    padded_rank: int = 8
    alpha: float = 1.0
    num_trials: int = 32
    use_pad: bool = True
    dummy_strategy: str = "paired_cancellation_dummy"
    pad_scale: float = 1.0
    dummy_scale: float = 1.0
    membership_trials_per_sample: int = 6
    membership_num_steps: int = 3
    dtype: str = "float64"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Helpers (mirror the Stage 7.2 single-linear proxy where it makes sense)
# ---------------------------------------------------------------------------


def _largest_sv_gap_index(t: torch.Tensor) -> tuple[int, float]:
    if t.numel() == 0:
        return 0, 0.0
    sv = torch.linalg.svdvals(t)
    if sv.numel() < 2:
        return int(sv.numel()), 1.0
    ratios = (sv[1:] / sv[:-1].clamp_min(1e-30)).cpu()
    drop_idx = int(torch.argmin(ratios).item())
    return drop_idx + 1, float(ratios[drop_idx].item())


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


def _build_state_for_dims(
    d_in: int, d_out: int, rank: int, *,
    dtype: torch.dtype, device: torch.device,
    use_pad: bool, pad_scale: float,
    fresh_masks_per_call: bool, fresh_u_per_call: bool,
    seq_len: int, generator: torch.Generator,
    base: LoRAState | None = None,
) -> LoRAState:
    inner = LoRAConfig(
        d_in=d_in, d_out=d_out, rank=rank, alpha=1.0,
        dtype="float64" if dtype == torch.float64 else "float32",
        device=device.type,
    )
    fcfg = MaskedLoRAForwardConfig(
        use_pad=use_pad, fresh_u_per_call=fresh_u_per_call,
        fresh_masks_per_call=fresh_masks_per_call,
        pad_scale=pad_scale,
        dtype=inner.dtype, device=inner.device,
    )
    return create_masked_lora_state(
        inner, fcfg, seq_len=seq_len, state=base, generator=generator,
    )


def _mask_padded_adapter(
    a_pad: torch.Tensor, b_pad: torch.Tensor, state: LoRAState,
) -> tuple[torch.Tensor, torch.Tensor]:
    return state.n_in_inv @ a_pad @ state.u, state.u_inv @ b_pad @ state.n_out


# ---------------------------------------------------------------------------
# Proxy 1 — cross-layer adapter linkage
# ---------------------------------------------------------------------------


_LORA_TARGETS_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_LORA_TARGETS_MLP = ("gate_proj", "up_proj", "down_proj")
_LORA_TARGETS_ALL = _LORA_TARGETS_ATTN + _LORA_TARGETS_MLP


def _cross_layer_adapter_linkage(
    config: MultiLayerLoRASecurityProxyConfig,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    # Use the first true_rank in the list for the linkage proxy — the proxy
    # is about cross-layer pattern stability, not rank inference.
    true_rank = config.true_ranks[0]
    rpc = RankPaddingConfig(
        true_rank=true_rank, padded_rank=config.padded_rank,
        dummy_scale=config.dummy_scale,
        dummy_strategy=config.dummy_strategy,
        fresh_dummy_per_step=True,
        dtype=config.dtype, device=config.device,
    )
    validate_rank_padding_config(rpc)
    hidden = config.hidden_size
    inter = config.intermediate_size
    seq_len = max(4, hidden // 4)

    for strategy in VALID_LINKAGE_STRATEGIES:
        fresh_masks = strategy in (
            "fresh_masks_independent_u", "rank_padding_full_bundle",
        )
        independent_u = strategy in (
            "independent_u_per_layer",
            "fresh_masks_independent_u",
            "rank_padding_full_bundle",
        )
        rank_padding_on = strategy == "rank_padding_full_bundle"

        # For each strategy collect masked-adapter fingerprints per module.
        # We compute pairwise Frobenius distance between (same module,
        # different layer) and (different module, same layer) — both
        # should be uncorrelated to the underlying plaintext under fresh
        # masks.
        same_module_distances: list[float] = []
        different_module_distances: list[float] = []
        retrieval_top1_hits = 0
        retrieval_total = 0

        for trial in range(config.num_trials):
            gen = torch.Generator(device="cpu").manual_seed(
                config.seed * 101 + trial * 7 + len(strategy)
            )
            # Sample plain adapters per (layer, module).
            adapters: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
            for layer in range(config.num_layers):
                for module in _LORA_TARGETS_ALL:
                    d_in, d_out = _module_io_shapes(module, hidden, inter)
                    cfg_a = LoRAConfig(
                        d_in=d_in, d_out=d_out,
                        rank=true_rank, alpha=config.alpha,
                        dtype=config.dtype, device=config.device,
                    )
                    a, b = init_lora_adapters(cfg_a, generator=gen)
                    b = b + 0.1 * torch.randn(
                        true_rank, d_out, generator=gen,
                        dtype=dtype, device=device,
                    )
                    adapters[(layer, module)] = (a, b)

            # For non-rank-padding strategies, use the plain adapter as
            # `a_pad` (i.e. rank dim = true_rank). For the full bundle, pad
            # to padded_rank.
            visible_modules: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
            mask_states_per_layer: dict[int, dict[str, LoRAState]] = {
                l: {} for l in range(config.num_layers)
            }
            shared_u_per_d_in_d_out: dict[tuple[int, int, int], LoRAState] = {}
            for (layer, module), (a, b) in adapters.items():
                d_in, d_out = _module_io_shapes(module, hidden, inter)
                if rank_padding_on:
                    pack = create_rank_padded_lora_adapters(
                        a, b, rpc, generator=gen,
                    )
                    a_pad = pack["a_pad"]
                    b_pad = pack["b_pad"]
                    rank_dim = config.padded_rank
                else:
                    a_pad = a
                    b_pad = b
                    rank_dim = true_rank

                # Decide mask reuse based on strategy.
                key_for_shared = (d_in, d_out, rank_dim)
                base_state: LoRAState | None = None
                if (
                    strategy == "fixed_masks_shared_u"
                    and key_for_shared in shared_u_per_d_in_d_out
                ):
                    base_state = shared_u_per_d_in_d_out[key_for_shared]
                fresh_u = independent_u or base_state is None
                fresh_masks_call = fresh_masks or base_state is None
                state = _build_state_for_dims(
                    d_in, d_out, rank_dim,
                    dtype=dtype, device=device,
                    use_pad=config.use_pad, pad_scale=config.pad_scale,
                    fresh_masks_per_call=fresh_masks_call,
                    fresh_u_per_call=fresh_u,
                    seq_len=seq_len, generator=gen,
                    base=base_state,
                )
                if (
                    strategy == "fixed_masks_shared_u"
                    and key_for_shared not in shared_u_per_d_in_d_out
                ):
                    shared_u_per_d_in_d_out[key_for_shared] = state
                mask_states_per_layer[layer][module] = state
                a_tilde, b_tilde = _mask_padded_adapter(a_pad, b_pad, state)
                visible_modules[(layer, module)] = (a_tilde, b_tilde)

            # Compute Frobenius-norm fingerprints — never publish raw tensors.
            def _fp(t: torch.Tensor) -> float:
                return float(t.norm().item())

            # Pairwise distance: same module across layers ↔ different
            # module across layers, when shapes align.
            module_layer_a_fps: dict[str, list[tuple[int, float]]] = {}
            for (layer, module), (a_t, b_t) in visible_modules.items():
                module_layer_a_fps.setdefault(module, []).append(
                    (layer, _fp(a_t))
                )

            # Retrieval test: for each (layer, module) try to identify
            # *another* layer that hosts the same module from its
            # transcript fingerprint alone (closest |fp| among modules
            # that match shape).
            ordered = list(visible_modules.items())
            for i, ((layer_i, mod_i), (a_i, b_i)) in enumerate(ordered):
                d_i = a_i.shape
                fp_i_a, fp_i_b = _fp(a_i), _fp(b_i)
                best_j = -1
                best_score = float("inf")
                truth_j = -1
                for j, ((layer_j, mod_j), (a_j, b_j)) in enumerate(ordered):
                    if i == j:
                        continue
                    if a_j.shape != d_i:
                        continue
                    score = abs(fp_i_a - _fp(a_j)) + abs(fp_i_b - _fp(b_j))
                    if score < best_score:
                        best_score = score
                        best_j = j
                    if mod_j == mod_i and layer_j != layer_i:
                        if truth_j == -1:
                            truth_j = j
                if truth_j == -1:
                    continue
                retrieval_total += 1
                if best_j == truth_j:
                    retrieval_top1_hits += 1

            # Same-module / different-module distance distribution.
            for module, layer_fps in module_layer_a_fps.items():
                fps = [fp for _, fp in layer_fps]
                for ii in range(len(fps)):
                    for jj in range(ii + 1, len(fps)):
                        same_module_distances.append(
                            abs(fps[ii] - fps[jj])
                        )
            module_names = list(module_layer_a_fps.keys())
            for ii in range(len(module_names)):
                for jj in range(ii + 1, len(module_names)):
                    m1 = module_names[ii]
                    m2 = module_names[jj]
                    for _, fp_a in module_layer_a_fps[m1]:
                        for _, fp_b in module_layer_a_fps[m2]:
                            different_module_distances.append(
                                abs(fp_a - fp_b)
                            )

        # Linkability AUC proxy: same-module vs different-module
        # distance separation.
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
        # AUC > 0.5 → same-module distances are smaller (more linkable)
        same_mean = (
            float(sum(same_module_distances) / len(same_module_distances))
            if same_module_distances else 0.0
        )
        diff_mean = (
            float(sum(different_module_distances) / len(different_module_distances))
            if different_module_distances else 0.0
        )
        retrieval_top1 = (
            float(retrieval_top1_hits / retrieval_total)
            if retrieval_total > 0 else 0.0
        )
        if strategy == "fixed_masks_shared_u":
            risk = "high"
        elif auc > 0.80 or retrieval_top1 > 0.50:
            risk = "high"
        elif auc > 0.65 or retrieval_top1 > 0.25:
            risk = "medium"
        else:
            risk = "low"

        rows.append({
            "strategy": strategy,
            "layer_linkability_auc": auc,
            "module_identity_retrieval_top1": retrieval_top1,
            "same_module_similarity": same_mean,
            "different_module_similarity": diff_mean,
            "risk_level": risk,
        })
    return {
        "rows": rows,
        "interpretation": (
            "Cross-layer adapter linkage proxy. AUC > 0.5 means the"
            " attacker's same-module distance distribution is smaller than"
            " the different-module distribution, so transcript fingerprints"
            " correlate with module identity across layers. fixed_masks_shared_u"
            " is the worst case; rank_padding_full_bundle is the most"
            " defensive setting tested here, but no result below is"
            " formally secure."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 2 — heterogeneous true_rank with shared padded_rank
# ---------------------------------------------------------------------------


def _heterogeneous_true_rank(
    config: MultiLayerLoRASecurityProxyConfig,
) -> dict[str, Any]:
    if len(config.true_ranks) < 1:
        raise ValueError("true_ranks must be non-empty")
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    rank_a = config.true_ranks[0]
    rank_b = (
        config.true_ranks[1] if len(config.true_ranks) > 1 else config.true_ranks[0]
    )
    seq_len = max(4, config.hidden_size // 4)
    rows: list[dict[str, Any]] = []
    for layer in range(config.num_layers):
        for module in _LORA_TARGETS_ALL:
            true_rank = rank_a if module in _LORA_TARGETS_ATTN else rank_b
            rpc = RankPaddingConfig(
                true_rank=true_rank, padded_rank=config.padded_rank,
                dummy_scale=config.dummy_scale,
                dummy_strategy=config.dummy_strategy,
                fresh_dummy_per_step=True,
                dtype=config.dtype, device=config.device,
            )
            validate_rank_padding_config(rpc)
            d_in, d_out = _module_io_shapes(
                module, config.hidden_size, config.intermediate_size,
            )
            cfg_a = LoRAConfig(
                d_in=d_in, d_out=d_out,
                rank=true_rank, alpha=config.alpha,
                dtype=config.dtype, device=config.device,
            )
            gen = torch.Generator(device="cpu").manual_seed(
                config.seed * 211 + layer * 17 + hash(module) % 1009
            )
            spectral_acc = 0
            gradient_acc = 0
            shape_hidden = 0
            for _ in range(config.num_trials):
                a, b = init_lora_adapters(cfg_a, generator=gen)
                b = b + 0.1 * torch.randn(
                    true_rank, d_out, generator=gen,
                    dtype=dtype, device=device,
                )
                pack = create_rank_padded_lora_adapters(
                    a, b, rpc, generator=gen,
                )
                a_pad = pack["a_pad"]
                b_pad = pack["b_pad"]
                # Shape-level rank stays at padded_rank regardless of
                # module true_rank — that's the headline of requirement 12.
                if a_pad.shape[1] == config.padded_rank:
                    shape_hidden += 1
                state = _build_state_for_dims(
                    d_in, d_out, config.padded_rank,
                    dtype=dtype, device=device,
                    use_pad=config.use_pad, pad_scale=config.pad_scale,
                    fresh_masks_per_call=True, fresh_u_per_call=True,
                    seq_len=seq_len, generator=gen,
                )
                a_tilde, b_tilde = _mask_padded_adapter(a_pad, b_pad, state)
                inferred_a, _ = _largest_sv_gap_index(a_tilde)
                inferred_b, _ = _largest_sv_gap_index(b_tilde)
                if min(inferred_a, inferred_b) == true_rank:
                    spectral_acc += 1

                # Gradient-side inference.
                x = torch.randn(
                    seq_len, d_in, generator=gen, dtype=dtype, device=device,
                )
                g = torch.randn(
                    seq_len, d_out, generator=gen, dtype=dtype, device=device,
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
                inf_ga, _ = _largest_sv_gap_index(masked["grad_a_tilde"])
                inf_gb, _ = _largest_sv_gap_index(masked["grad_b_tilde"])
                if min(inf_ga, inf_gb) == true_rank:
                    gradient_acc += 1
            trials = max(1, config.num_trials)
            spectral_rate = spectral_acc / trials
            gradient_rate = gradient_acc / trials
            shape_hidden_rate = shape_hidden / trials
            if config.dummy_strategy == "zero_dummy":
                risk = "high"
            elif spectral_rate >= 0.5 or gradient_rate >= 0.5:
                risk = "high"
            elif spectral_rate >= 0.2 or gradient_rate >= 0.2:
                risk = "medium"
            else:
                risk = "needs_more_evaluation"
            rows.append({
                "layer_index": layer,
                "module_name": module,
                "true_rank": true_rank,
                "padded_rank": config.padded_rank,
                "visible_rank_from_shape": config.padded_rank,
                "true_rank_shape_hidden_rate": float(shape_hidden_rate),
                "spectral_rank_inference_accuracy": float(spectral_rate),
                "gradient_rank_inference_accuracy": float(gradient_rate),
                "risk_level": risk,
            })
    return {
        "rows": rows,
        "interpretation": (
            "Heterogeneous true_rank with a shared padded_rank. Shape-level"
            " leakage is fully closed in all rows when padded_rank is the"
            " same across modules (true_rank_shape_hidden_rate == 1.0)."
            " Spectral / gradient inference is reported per module under"
            f" dummy_strategy={config.dummy_strategy!r}; paired_cancellation"
            " yields needs_more_evaluation, zero_dummy is high."
        ),
    }


# ---------------------------------------------------------------------------
# Proxy 3 — multi-step membership linkability
# ---------------------------------------------------------------------------


def _multi_step_membership(
    config: MultiLayerLoRASecurityProxyConfig,
) -> dict[str, Any]:
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(config.device)
    rows: list[dict[str, Any]] = []
    true_rank = config.true_ranks[0]
    rpc = RankPaddingConfig(
        true_rank=true_rank, padded_rank=config.padded_rank,
        dummy_scale=config.dummy_scale,
        dummy_strategy=config.dummy_strategy,
        fresh_dummy_per_step=True,
        dtype=config.dtype, device=config.device,
    )
    validate_rank_padding_config(rpc)
    hidden = config.hidden_size
    inter = config.intermediate_size
    seq_len = max(4, hidden // 4)

    for module in _LORA_TARGETS_ALL:
        gen = torch.Generator(device="cpu").manual_seed(
            config.seed * 313 + hash(module) % 7919
        )
        d_in, d_out = _module_io_shapes(module, hidden, inter)
        cfg_a = LoRAConfig(
            d_in=d_in, d_out=d_out,
            rank=true_rank, alpha=config.alpha,
            dtype=config.dtype, device=config.device,
        )
        a, b = init_lora_adapters(cfg_a, generator=gen)
        b = b + 0.1 * torch.randn(
            true_rank, d_out, generator=gen, dtype=dtype, device=device,
        )
        x1 = torch.randn(seq_len, d_in, generator=gen, dtype=dtype, device=device)
        x2 = torch.randn(seq_len, d_in, generator=gen, dtype=dtype, device=device)
        g1 = torch.randn(seq_len, d_out, generator=gen, dtype=dtype, device=device)
        g2 = torch.randn(seq_len, d_out, generator=gen, dtype=dtype, device=device)

        def _trace(x: torch.Tensor, g: torch.Tensor):
            pack = create_rank_padded_lora_adapters(
                a, b, rpc, generator=gen,
            )
            state = _build_state_for_dims(
                d_in, d_out, config.padded_rank,
                dtype=dtype, device=device,
                use_pad=config.use_pad, pad_scale=config.pad_scale,
                fresh_masks_per_call=True, fresh_u_per_call=True,
                seq_len=seq_len, generator=gen,
            )
            a_tilde, b_tilde = _mask_padded_adapter(
                pack["a_pad"], pack["b_pad"], state,
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
            return (
                a_tilde, b_tilde,
                masked["grad_a_tilde"], masked["grad_b_tilde"],
            )

        def _dist(p1, p2) -> float:
            return float(
                sum(
                    ((p1[k] - p2[k]) ** 2).sum().item() for k in range(4)
                ) ** 0.5
            )

        # Replay each sample across membership_num_steps × trials_per_sample.
        trials_per_sample = max(1, config.membership_trials_per_sample)
        num_steps = max(1, config.membership_num_steps)
        same1 = [_trace(x1, g1) for _ in range(trials_per_sample * num_steps)]
        same2 = [_trace(x2, g2) for _ in range(trials_per_sample * num_steps)]
        same: list[float] = []
        diff: list[float] = []
        for i in range(len(same1)):
            for j in range(i + 1, len(same1)):
                same.append(_dist(same1[i], same1[j]))
                same.append(_dist(same2[i], same2[j]))
        for i in range(len(same1)):
            for j in range(len(same2)):
                diff.append(_dist(same1[i], same2[j]))
        wins = 0
        total = 0
        for s in same:
            for d in diff:
                if s < d:
                    wins += 1
                elif s == d:
                    wins += 0.5
                total += 1
        auc = float(wins / max(1, total))
        if auc > 0.85:
            risk = "high"
        elif auc > 0.65:
            risk = "medium"
        else:
            risk = "low"
        rows.append({
            "module_name": module,
            "same_sample_distance_mean": float(
                sum(same) / max(1, len(same))
            ),
            "different_sample_distance_mean": float(
                sum(diff) / max(1, len(diff))
            ),
            "membership_auc_proxy": auc,
            "linkability_rank": 2.0 * abs(auc - 0.5),
            "risk_level": risk,
        })
    # Aggregate gradient-update linkability is the average of per-module
    # AUC values for downstream consumers.
    mean_auc = float(
        sum(r["membership_auc_proxy"] for r in rows) / max(1, len(rows))
    )
    return {
        "rows": rows,
        "aggregate": {
            "mean_membership_auc_proxy": mean_auc,
            "adapter_update_linkability": float(2.0 * abs(mean_auc - 0.5)),
        },
        "interpretation": (
            "Per-module multi-step membership linkability proxy. Under"
            " fresh masks per call + paired_cancellation_dummy, the"
            " transcript distance distribution should be roughly the same"
            " for same-sample replays and different-sample comparisons,"
            " so AUC ≈ 0.5 and linkability_rank ≈ 0."
        ),
    }


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "These are cross-layer leakage proxy attacks, not formal security proofs.",
    "True rank is hidden from shape-level leakage only when padded_rank is shared across modules.",
    "padded_rank itself remains visible from tensor shape.",
    "Spectral / gradient rank inference may still narrow the attacker's range, especially under zero_dummy.",
    "Cross-layer linkability under fixed_masks_shared_u is reported high by construction; the no-mitigation baseline IS leaky.",
    "No real TEE isolation is evaluated; security_profile stays 'proxy-evaluated, not formal'.",
    "Optimizer state remains trusted-only and is sized to true_rank for every module.",
    "Hardware side-channels (cache / power / EM) are NOT evaluated.",
    "No full Qwen / TinyLlama / LLaMA fine-tuning is evaluated; this is a multi-linear, tiny-dimension proxy.",
    "Adapter is NEVER merged into the public base weight W.",
]


def run_multilayer_lora_security_proxy(
    config: MultiLayerLoRASecurityProxyConfig,
) -> dict[str, Any]:
    if config.dummy_strategy not in VALID_DUMMY_STRATEGIES:
        raise ValueError(
            f"unknown dummy_strategy {config.dummy_strategy!r};"
            f" expected one of {VALID_DUMMY_STRATEGIES}"
        )
    for r in config.true_ranks:
        if r <= 0 or r > config.padded_rank:
            raise ValueError(
                f"each true_rank must satisfy 0 < r <= padded_rank"
                f" ({config.padded_rank}), got {r}"
            )

    linkage = _cross_layer_adapter_linkage(config)
    heterogeneous = _heterogeneous_true_rank(config)
    membership = _multi_step_membership(config)

    # Aggregate verdicts.
    linkage_risks = [r["risk_level"] for r in linkage["rows"]]
    heterogeneous_risks = [r["risk_level"] for r in heterogeneous["rows"]]
    overall_linkage_risk = (
        "high" if "high" in linkage_risks
        else "medium" if "medium" in linkage_risks
        else "low"
    )
    overall_rank_risk = (
        "high" if "high" in heterogeneous_risks
        else "medium" if "medium" in heterogeneous_risks
        else "needs_more_evaluation" if "needs_more_evaluation" in heterogeneous_risks
        else "low"
    )

    true_rank_shape_hidden_rate = (
        sum(r["true_rank_shape_hidden_rate"] for r in heterogeneous["rows"])
        / max(1, len(heterogeneous["rows"]))
    )

    return {
        "config": asdict(config),
        "scope": (
            "multi-layer tiny LoRA model (q/k/v/o/gate/up/down per layer),"
            " synthetic adapters + synthetic upstream gradients, rank"
            " padding optional per strategy"
        ),
        "cross_layer_adapter_linkage": linkage,
        "heterogeneous_true_rank_with_shared_padded_rank": heterogeneous,
        "multi_step_membership_linkability": membership,
        "interpretation": {
            "shape_level_summary": (
                "true_rank is hidden from tensor shape across all modules"
                " when padded_rank is shared; padded_rank itself remains"
                " visible."
            ),
            "cross_layer_linkage_summary": (
                f"Cross-layer linkage risk under multi-layer strategies is"
                f" **{overall_linkage_risk}** (worst-case)."
            ),
            "heterogeneous_rank_summary": (
                f"Heterogeneous true_rank with shared padded_rank"
                f" inference risk under dummy_strategy={config.dummy_strategy!r}"
                f" is **{overall_rank_risk}**."
            ),
            "true_rank_shape_hidden_rate": float(true_rank_shape_hidden_rate),
            "padded_rank_visibility_note": (
                "padded_rank is still visible across modules; hiding it"
                " is out of Stage 7.3 scope."
            ),
            "merge_adapter_into_w": False,
        },
        "security_profile": "proxy-evaluated, not formal",
        "security_profile_detail_with_lora_multilayer": (
            "multi-layer-lora-proxy-evaluated, not formal"
        ),
        "lora_multilayer_security_proxy_status": "implemented",
        "limitations": list(_LIMITATIONS),
        "next_stage_plan": [
            "Stage 7.4 — stronger dummy distributions / spectral-rank hardening.",
            "Stage 7.x — heterogeneous padded_rank across modules to hide r_pad itself.",
            "Stage 7.x — gradient-side noise to weaken cross-layer linkability under shared-U setups.",
        ],
    }


def multilayer_security_csv_rows(
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
    for r in report["cross_layer_adapter_linkage"]["rows"]:
        for k, v in r.items():
            if k == "strategy":
                continue
            rows.append({
                "section": "cross_layer_adapter_linkage",
                "attack": "cross_layer_linkage",
                "strategy": r["strategy"],
                "metric": k,
                "value": v,
                "notes": "",
            })
    for r in report["heterogeneous_true_rank_with_shared_padded_rank"]["rows"]:
        for k, v in r.items():
            if k in ("layer_index", "module_name"):
                continue
            rows.append({
                "section": "heterogeneous_true_rank",
                "attack": "rank_inference",
                "strategy": (
                    f"layer_{r['layer_index']}.{r['module_name']}"
                ),
                "metric": k,
                "value": v,
                "notes": "",
            })
    for r in report["multi_step_membership_linkability"]["rows"]:
        for k, v in r.items():
            if k == "module_name":
                continue
            rows.append({
                "section": "multi_step_membership",
                "attack": "membership_linkability",
                "strategy": r["module_name"],
                "metric": k,
                "value": v,
                "notes": "",
            })
    agg = report["multi_step_membership_linkability"]["aggregate"]
    for k, v in agg.items():
        rows.append({
            "section": "multi_step_membership",
            "attack": "membership_linkability",
            "strategy": "aggregate",
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
    "MultiLayerLoRASecurityProxyConfig",
    "VALID_LINKAGE_STRATEGIES",
    "multilayer_security_csv_rows",
    "run_multilayer_lora_security_proxy",
]
