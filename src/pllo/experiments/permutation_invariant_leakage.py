"""Stage 5.7 — Permutation-Invariant Leakage Audit.

Reuses the Stage 5.5b real-token trace collector to obtain ``(plain,
visible)`` activation pairs at the existing nonlinear-island
boundaries (``gate``, ``up``, ``swiglu_intermediate``, ``post_island``,
``q``, ``k``, ``v``, plus ``boundary_input`` / ``final`` for scope
context), under both selectable mitigation bundles
(``fresh_perm_only`` and ``fresh_perm_plus_sandwich_plus_pad``).

It then computes four families of *permutation-invariant* statistics on
the per-row visible vs plain tensors:

* row-wise norm preservation (L1, L2, Linf correlations and abs-diffs);
* row-wise extrema preservation (per-row max / min correlations);
* sorted multiset preservation (per-row sorted-MSE);
* quantile preservation (5-quantile per-row MSE).

A statistics-only classifier (nearest-centroid on visible-only per-row
features) then probes three proxy adversarial tasks:

* scope classification (prefill vs decode);
* prompt-id linkability (1-NN retrieval by aggregated row signature);
* position-bucket classification (low / mid / high).

A fixed-permutation synthetic ablation demonstrates that the
permutation-invariant statistics are preserved by *any* permutation
(fresh or fixed), and that the freshness contract delivered by the
mitigation bundles changes the linkability / temporal posture, not the
single-shot value-level multiset leakage.

This is a *proxy* audit. No formal, cryptographic, or semantic
security is claimed. No raw tensors, masks, permutations, adapters,
gradients, or private data are exported. JSON / CSV / Markdown carry
only summary scalars, shapes, and short fingerprints.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch

from pllo.experiments.real_token_trace import (
    DEFAULT_TARGET_TENSORS,
    RealTokenTraceConfig,
    collect_real_token_traces,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_DEFAULT_TARGETS: tuple[str, ...] = (
    "gate", "up", "swiglu_intermediate", "post_island", "q", "k", "v",
    "boundary_input", "final",
)


_REQUIRED_HONESTY_PHRASES: tuple[str, ...] = (
    "Permutation-only nonlinear views provide channel-index hiding, "
    "not value hiding.",
    "This is a proxy leakage audit, not a formal security proof.",
    "Dense sandwiching and boundary pads mitigate temporal and "
    "boundary exposure but do not remove single-shot "
    "permutation-invariant statistics inside the activation core.",
    "No real TEE isolation or hardware side-channel resistance is "
    "evaluated.",
    "Raw tensors, masks, permutations, adapters, gradients, and "
    "private data are not exported.",
)


@dataclass(frozen=True)
class PermutationInvariantLeakageConfig:
    num_prompts: int = 8
    prompt_max_length: int = 8
    max_new_tokens: int = 3
    max_layers: int = 2
    bundles: tuple[str, ...] = (
        "fresh_perm_only",
        "fresh_perm_plus_sandwich_plus_pad",
    )
    target_tensors: tuple[str, ...] = _DEFAULT_TARGETS
    use_pad: bool = True
    include_fixed_debug: bool = True
    attempt_tokenizer_load: bool = False
    attempt_real_model_load: bool = False
    model_id: Optional[str] = None
    seed: int = 0
    output_dir: str = "outputs"
    # Synthetic-fallback shape (forwarded to real_token_trace).
    synthetic_vocab_size: int = 256
    synthetic_hidden_size: int = 32
    synthetic_intermediate_size: int = 64
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 8


# ---------------------------------------------------------------------------
# Helpers — permutation-invariant statistics
# ---------------------------------------------------------------------------


_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def _safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors, robust to zero std."""
    a = a.detach().to(torch.float32).flatten()
    b = b.detach().to(torch.float32).flatten()
    if a.numel() == 0:
        return 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = float(
        torch.sqrt(a_centered.pow(2).sum() * b_centered.pow(2).sum()).item()
    )
    if denom == 0.0:
        # Both constant -> perfect agreement.
        return 1.0
    return float((a_centered * b_centered).sum().item() / denom)


def _row_norms(t: torch.Tensor) -> dict[str, torch.Tensor]:
    t = t.detach().to(torch.float32)
    return {
        "l1": t.abs().sum(dim=-1),
        "l2": t.pow(2).sum(dim=-1).clamp_min(0).sqrt(),
        "linf": t.abs().amax(dim=-1),
    }


def _row_extrema(t: torch.Tensor) -> dict[str, torch.Tensor]:
    t = t.detach().to(torch.float32)
    return {
        "max": t.amax(dim=-1),
        "min": t.amin(dim=-1),
    }


def _row_sorted(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).sort(dim=-1).values


def _row_quantiles(t: torch.Tensor) -> torch.Tensor:
    """Per-row quantile vector ``[N, len(_QUANTILES)]``."""
    t = t.detach().to(torch.float32)
    q = torch.tensor(_QUANTILES, dtype=torch.float32)
    # torch.quantile expects a [Q] tensor as second arg.
    out = torch.quantile(t, q, dim=-1)  # [Q, N]
    return out.transpose(0, 1).contiguous()  # [N, Q]


def _row_stat_features(t: torch.Tensor) -> torch.Tensor:
    """Per-row visible-only statistical feature vector.

    Columns: ``l1, l2, linf, max, min, mean, std, q10, q25, q50, q75,
    q90, positive_ratio``.
    """
    t = t.detach().to(torch.float32)
    norms = _row_norms(t)
    extrema = _row_extrema(t)
    quantiles = _row_quantiles(t)
    mean = t.mean(dim=-1, keepdim=True)
    std = t.std(dim=-1, unbiased=False, keepdim=True)
    positive_ratio = (t > 0).float().mean(dim=-1, keepdim=True)
    feats = torch.cat([
        norms["l1"].unsqueeze(-1),
        norms["l2"].unsqueeze(-1),
        norms["linf"].unsqueeze(-1),
        extrema["max"].unsqueeze(-1),
        extrema["min"].unsqueeze(-1),
        mean, std,
        quantiles,
        positive_ratio,
    ], dim=-1)
    # Replace NaN/Inf with 0 so the classifier is stable on edge cases.
    feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def _pair_metrics(
    plain: torch.Tensor, visible: torch.Tensor,
) -> dict[str, float]:
    """Compute (A) norm, (B) extrema, (C) sorted, (D) quantile metrics."""
    p_norms = _row_norms(plain)
    v_norms = _row_norms(visible)
    p_extrema = _row_extrema(plain)
    v_extrema = _row_extrema(visible)
    p_sorted = _row_sorted(plain)
    v_sorted = _row_sorted(visible)
    p_quant = _row_quantiles(plain)
    v_quant = _row_quantiles(visible)

    sorted_se = (p_sorted - v_sorted).pow(2)
    sorted_l2 = sorted_se.sum(dim=-1).sqrt()
    p_sorted_l2 = p_sorted.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
    sorted_l2_rel = sorted_l2 / p_sorted_l2

    quant_se = (p_quant - v_quant).pow(2)

    return {
        # A. Norm preservation.
        "l1_corr": _safe_corr(p_norms["l1"], v_norms["l1"]),
        "l2_corr": _safe_corr(p_norms["l2"], v_norms["l2"]),
        "linf_corr": _safe_corr(p_norms["linf"], v_norms["linf"]),
        "l2_max_abs_diff": float(
            (p_norms["l2"] - v_norms["l2"]).abs().max().item()
        ),
        "l2_mean_abs_diff": float(
            (p_norms["l2"] - v_norms["l2"]).abs().mean().item()
        ),
        # B. Extrema preservation.
        "max_corr": _safe_corr(p_extrema["max"], v_extrema["max"]),
        "min_corr": _safe_corr(p_extrema["min"], v_extrema["min"]),
        "max_abs_diff": float(
            (p_extrema["max"] - v_extrema["max"]).abs().max().item()
        ),
        "min_abs_diff": float(
            (p_extrema["min"] - v_extrema["min"]).abs().max().item()
        ),
        # C. Sorted multiset preservation.
        "sorted_mse_mean": float(sorted_se.mean().item()),
        "sorted_mse_max": float(sorted_se.mean(dim=-1).max().item()),
        "sorted_l2_rel_mean": float(sorted_l2_rel.mean().item()),
        # D. Quantile preservation.
        "quantile_mse_mean": float(quant_se.mean().item()),
        "quantile_p10_abs_err_mean": float(
            (p_quant[:, 0] - v_quant[:, 0]).abs().mean().item()
        ),
        "quantile_p25_abs_err_mean": float(
            (p_quant[:, 1] - v_quant[:, 1]).abs().mean().item()
        ),
        "quantile_p50_abs_err_mean": float(
            (p_quant[:, 2] - v_quant[:, 2]).abs().mean().item()
        ),
        "quantile_p75_abs_err_mean": float(
            (p_quant[:, 3] - v_quant[:, 3]).abs().mean().item()
        ),
        "quantile_p90_abs_err_mean": float(
            (p_quant[:, 4] - v_quant[:, 4]).abs().mean().item()
        ),
    }


# ---------------------------------------------------------------------------
# Helpers — risk labels
# ---------------------------------------------------------------------------


def _statistical_leakage_label(metrics: dict[str, float]) -> str:
    """Conservative label: ``high`` if norm/sorted preservation is near
    exact; ``medium`` if preserved but not exact; ``low`` if disrupted.
    """
    l2 = metrics.get("l2_corr", 0.0)
    sorted_mean = metrics.get("sorted_mse_mean", float("inf"))
    sorted_l2_rel = metrics.get("sorted_l2_rel_mean", float("inf"))
    if l2 > 0.999 and sorted_mean < 1e-6 and sorted_l2_rel < 1e-3:
        return "statistical_leakage_detected_high"
    if l2 > 0.9 and sorted_l2_rel < 0.1:
        return "statistical_leakage_detected_medium"
    if l2 < 0.5 or sorted_l2_rel > 0.5:
        return "statistical_leakage_low"
    return "statistical_leakage_borderline"


def _proxy_attack_label(
    acc: float, baseline: float, num_classes: int,
) -> str:
    """Risk label for classifier accuracy vs random chance."""
    chance = 1.0 / max(1, num_classes)
    if num_classes <= 1:
        return "proxy_attack_skipped_single_class"
    if acc - chance >= 0.5:
        return "proxy_attack_high"
    if acc - chance >= 0.2:
        return "proxy_attack_medium"
    if acc - chance >= 0.05:
        return "proxy_attack_low"
    return "proxy_attack_chance_level"


# ---------------------------------------------------------------------------
# Helpers — nearest-centroid classifier
# ---------------------------------------------------------------------------


def _nearest_centroid(
    feats: torch.Tensor, labels: torch.Tensor,
) -> dict[str, Any]:
    """Stable nearest-centroid classifier on feature rows.

    Splits 50/50 by label-stratified deterministic indices, fits
    centroids on train, evaluates on test. Returns accuracy + chance.
    """
    feats = feats.detach().to(torch.float32)
    labels = labels.detach().to(torch.long)
    classes = torch.unique(labels).tolist()
    if len(classes) <= 1:
        return {
            "status": "skipped_single_class",
            "accuracy": None,
            "chance_level": None,
            "num_classes": len(classes),
            "num_samples": int(feats.shape[0]),
        }
    if feats.shape[0] < 2 * len(classes):
        return {
            "status": "skipped_too_few_samples",
            "accuracy": None,
            "chance_level": 1.0 / max(1, len(classes)),
            "num_classes": len(classes),
            "num_samples": int(feats.shape[0]),
        }
    # Stratified deterministic split: take every other index per class.
    train_idx_list: list[int] = []
    test_idx_list: list[int] = []
    for c in classes:
        mask = (labels == c).nonzero(as_tuple=False).flatten().tolist()
        for j, idx in enumerate(mask):
            (train_idx_list if j % 2 == 0 else test_idx_list).append(idx)
    if not train_idx_list or not test_idx_list:
        return {
            "status": "skipped_no_split",
            "accuracy": None,
            "chance_level": 1.0 / max(1, len(classes)),
            "num_classes": len(classes),
            "num_samples": int(feats.shape[0]),
        }
    train_idx = torch.tensor(train_idx_list, dtype=torch.long)
    test_idx = torch.tensor(test_idx_list, dtype=torch.long)
    train_feats, train_labels = feats[train_idx], labels[train_idx]
    test_feats, test_labels = feats[test_idx], labels[test_idx]
    # Per-feature standardisation using train stats so the classifier
    # cannot trivially win via scale.
    mean = train_feats.mean(dim=0, keepdim=True)
    std = train_feats.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    train_feats = (train_feats - mean) / std
    test_feats = (test_feats - mean) / std
    centroids = torch.stack([
        train_feats[train_labels == c].mean(dim=0) for c in classes
    ], dim=0)  # [C, F]
    dists = torch.cdist(
        test_feats.unsqueeze(0), centroids.unsqueeze(0)
    ).squeeze(0)  # [N_test, C]
    pred = dists.argmin(dim=-1)
    pred_labels = torch.tensor([classes[int(i)] for i in pred.tolist()],
                                dtype=torch.long)
    acc = float((pred_labels == test_labels).float().mean().item())
    return {
        "status": "ok",
        "accuracy": acc,
        "chance_level": 1.0 / len(classes),
        "num_classes": len(classes),
        "num_samples": int(feats.shape[0]),
        "num_train": int(train_feats.shape[0]),
        "num_test": int(test_feats.shape[0]),
    }


# ---------------------------------------------------------------------------
# Per-tensor audit
# ---------------------------------------------------------------------------


def _audit_tensor(
    plain: torch.Tensor,
    visible: torch.Tensor,
    *,
    scope: str,
    tensor_name: str,
    bundle: str,
) -> dict[str, Any]:
    metrics = _pair_metrics(plain, visible)
    leakage_label = _statistical_leakage_label(metrics)
    return {
        "scope": scope,
        "tensor": tensor_name,
        "bundle": bundle,
        "num_rows": int(plain.shape[0]),
        "feature_dim": int(plain.shape[-1]),
        "metrics": metrics,
        "statistical_leakage_label": leakage_label,
    }


# ---------------------------------------------------------------------------
# Classifier proxy tasks
# ---------------------------------------------------------------------------


def _classifier_tasks(
    traces: dict[str, dict[str, dict[str, torch.Tensor]]],
    tensor_name: str,
) -> dict[str, dict[str, Any]]:
    """Build proxy classifier tasks for one tensor across scopes."""
    out: dict[str, dict[str, Any]] = {}

    # Scope classification: visible-only features, label = scope id.
    feat_chunks: list[torch.Tensor] = []
    scope_labels: list[int] = []
    scope_id = {"prefill": 0, "decode": 1}
    for scope in ("prefill", "decode"):
        pair = traces.get(scope, {}).get(tensor_name)
        if pair is None:
            continue
        v = pair["visible"]
        feats = _row_stat_features(v)
        feat_chunks.append(feats)
        scope_labels.extend([scope_id[scope]] * feats.shape[0])
    if len(feat_chunks) >= 2:
        feats_all = torch.cat(feat_chunks, dim=0)
        labels_all = torch.tensor(scope_labels, dtype=torch.long)
        res = _nearest_centroid(feats_all, labels_all)
        if res["status"] == "ok":
            res["task"] = "scope_classification"
            res["proxy_attack_label"] = _proxy_attack_label(
                res["accuracy"], res["chance_level"], res["num_classes"],
            )
        else:
            res["task"] = "scope_classification"
            res["proxy_attack_label"] = "proxy_attack_skipped"
        out["scope_classification"] = res
    else:
        out["scope_classification"] = {
            "task": "scope_classification",
            "status": "skipped_missing_scope",
            "proxy_attack_label": "proxy_attack_skipped",
        }

    # Prompt-id linkability: 1-NN retrieval. We do not have per-row prompt
    # labels in the stitched [N, D] view (rows are flattened across
    # prompts/tokens). Mark as skipped to avoid an unfounded claim; this
    # keeps the audit conservative.
    out["prompt_id_linkability"] = {
        "task": "prompt_id_linkability",
        "status": "skipped_no_per_row_prompt_label_in_stitched_view",
        "proxy_attack_label": "proxy_attack_skipped",
    }
    # Position-bucket classification: similar reason -- per-row position
    # metadata is not exposed through the stitched [N, D] view. Skip
    # rather than fabricate labels.
    out["position_bucket_classification"] = {
        "task": "position_bucket_classification",
        "status": "skipped_no_per_row_position_label_in_stitched_view",
        "proxy_attack_label": "proxy_attack_skipped",
    }
    return out


# ---------------------------------------------------------------------------
# Fixed-permutation synthetic ablation
# ---------------------------------------------------------------------------


def _fixed_permutation_ablation(
    *, seed: int, hidden_dim: int, num_rows: int,
) -> dict[str, Any]:
    """Synthetic ablation comparing a *fixed* permutation across rows
    against a *fresh* (per-row) permutation.

    Demonstrates explicitly that the permutation-invariant statistics
    (norm, sorted-multiset, quantile) are preserved by *both* fixed and
    fresh permutations -- the freshness contract changes
    linkability / temporal posture, not single-shot value-level
    multiset visibility inside the activation core.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(num_rows, hidden_dim, dtype=torch.float32, generator=g)
    # Fixed permutation: same permutation reused for every row.
    fixed_perm = torch.randperm(hidden_dim, generator=g)
    fixed_visible = X.index_select(dim=-1, index=fixed_perm)
    # Fresh permutation: independent permutation per row.
    fresh_visible = torch.empty_like(X)
    for i in range(num_rows):
        p = torch.randperm(hidden_dim, generator=g)
        fresh_visible[i] = X[i].index_select(dim=-1, index=p)
    fixed_metrics = _pair_metrics(X, fixed_visible)
    fresh_metrics = _pair_metrics(X, fresh_visible)

    # Linkability proxy: under fixed perm, visible[..., j] is a
    # deterministic function of plain[..., perm^{-1}[j]] -- a per-row
    # exact 1-NN match. Under fresh perm, the per-row permutation
    # destroys index linkability. We measure this with a row-fingerprint
    # match rate: take a small per-row sorted+statistic signature on
    # visible; check the nearest plain row by L2 of the same signature.
    def _signature(t: torch.Tensor) -> torch.Tensor:
        return _row_stat_features(t)

    plain_sig = _signature(X)
    fixed_sig = _signature(fixed_visible)
    fresh_sig = _signature(fresh_visible)
    dist_fixed = torch.cdist(fixed_sig.unsqueeze(0), plain_sig.unsqueeze(0)).squeeze(0)
    dist_fresh = torch.cdist(fresh_sig.unsqueeze(0), plain_sig.unsqueeze(0)).squeeze(0)
    pred_fixed = dist_fixed.argmin(dim=-1)
    pred_fresh = dist_fresh.argmin(dim=-1)
    target = torch.arange(num_rows)
    link_fixed = float((pred_fixed == target).float().mean().item())
    link_fresh = float((pred_fresh == target).float().mean().item())

    return {
        "status": "ok",
        "hidden_dim": hidden_dim,
        "num_rows": num_rows,
        "fixed_permutation_metrics": fixed_metrics,
        "fresh_permutation_metrics": fresh_metrics,
        "fixed_perm_linkability_accuracy": link_fixed,
        "fresh_perm_linkability_accuracy": link_fresh,
        "linkability_chance_level": 1.0 / max(1, num_rows),
        "interpretation": (
            "Both fixed and fresh permutation views preserve "
            "per-row norms, sorted multisets, and quantiles by "
            "construction. The freshness contract changes the "
            "row-signature linkability (fixed perm preserves it "
            "exactly via deterministic channel re-mapping; fresh perm "
            "does not), but does not remove single-shot "
            "permutation-invariant statistics inside the activation "
            "core."
        ),
    }


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def _collect_one_bundle(
    cfg: PermutationInvariantLeakageConfig, bundle: str,
) -> dict[str, Any]:
    rt_cfg = RealTokenTraceConfig(
        output_dir=cfg.output_dir,
        seed=cfg.seed,
        model_id=cfg.model_id,
        attempt_real_model_load=cfg.attempt_real_model_load,
        attempt_tokenizer_load=cfg.attempt_tokenizer_load,
        max_layers=cfg.max_layers,
        max_new_tokens=cfg.max_new_tokens,
        prompt_max_length=cfg.prompt_max_length,
        num_prompts=cfg.num_prompts,
        use_pad=cfg.use_pad,
        nonlinear_mode="compatible_islands",
        mitigation_bundle=bundle,
        inter_block_mask_mode="plain_boundary",
        target_tensors=cfg.target_tensors,
        synthetic_vocab_size=cfg.synthetic_vocab_size,
        synthetic_hidden_size=cfg.synthetic_hidden_size,
        synthetic_intermediate_size=cfg.synthetic_intermediate_size,
        synthetic_num_attention_heads=cfg.synthetic_num_attention_heads,
        synthetic_num_key_value_heads=cfg.synthetic_num_key_value_heads,
        synthetic_head_dim=cfg.synthetic_head_dim,
    )
    return collect_real_token_traces(rt_cfg)


def run_permutation_invariant_leakage(
    cfg: PermutationInvariantLeakageConfig,
) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    scopes = ("prefill", "decode")
    per_bundle: dict[str, Any] = {}
    tensor_inventory: dict[str, dict[str, bool]] = {}

    for bundle in cfg.bundles:
        pkg = _collect_one_bundle(cfg, bundle)
        traces: dict[str, dict[str, dict[str, torch.Tensor]]] = pkg.get(
            "traces", {}
        )
        per_tensor: dict[str, dict[str, dict[str, Any]]] = {
            tn: {} for tn in cfg.target_tensors
        }
        skipped: list[dict[str, str]] = []
        for tn in cfg.target_tensors:
            tensor_inventory.setdefault(tn, {})
            for scope in scopes:
                pair = traces.get(scope, {}).get(tn)
                if pair is None:
                    per_tensor[tn][scope] = {
                        "skipped_with_reason":
                            f"tensor_{tn}_not_in_scope_{scope}",
                    }
                    tensor_inventory[tn][f"{scope}_present"] = False
                    skipped.append({
                        "tensor": tn, "scope": scope, "bundle": bundle,
                        "reason": "tensor_not_in_trace_scope",
                    })
                    continue
                audit = _audit_tensor(
                    pair["plain"], pair["visible"],
                    scope=scope, tensor_name=tn, bundle=bundle,
                )
                per_tensor[tn][scope] = audit
                tensor_inventory[tn][f"{scope}_present"] = True
        # Classifier proxy tasks per tensor (cross-scope).
        classifier_results: dict[str, dict[str, dict[str, Any]]] = {}
        for tn in cfg.target_tensors:
            classifier_results[tn] = _classifier_tasks(traces, tn)

        per_bundle[bundle] = {
            "bundle": bundle,
            "trace_source": pkg.get("source"),
            "model_loading_status": pkg.get("model_loading", {}).get(
                "model_loading_status",
            ) if isinstance(pkg.get("model_loading"), dict) else None,
            "tokenizer_status": pkg.get("tokenizer_loading", {}).get(
                "tokenizer_status",
            ) if isinstance(pkg.get("tokenizer_loading"), dict) else None,
            "per_tensor": per_tensor,
            "classifier_proxy_tasks": classifier_results,
            "skipped_target_tensors": skipped,
            "generation_token_match_rate": (
                pkg.get("generation_summary", {})
                .get("mean_token_match_rate", None)
            ),
        }

    fixed_debug: Optional[dict[str, Any]] = None
    if cfg.include_fixed_debug:
        fixed_debug = _fixed_permutation_ablation(
            seed=cfg.seed + 1009,
            hidden_dim=cfg.synthetic_hidden_size,
            num_rows=max(8, cfg.num_prompts),
        )

    report: dict[str, Any] = {
        "status": "ok",
        "stage": "5.7",
        "main_mode": "permutation_invariant_leakage_audit",
        "device": "cpu",
        "config": asdict(cfg),
        "target_tensor_inventory": tensor_inventory,
        "per_bundle": per_bundle,
        "fixed_permutation_debug": fixed_debug,
        "formal_security_claim": False,
        "honesty_phrases": list(_REQUIRED_HONESTY_PHRASES),
        "limitations": [
            "Permutation-only nonlinear views provide channel-index "
            "hiding, not value hiding.",
            "Dense sandwiching and boundary pads mitigate temporal and "
            "boundary exposure but do not remove single-shot "
            "permutation-invariant statistics inside the activation "
            "core.",
            "This is a proxy leakage audit, not a formal security "
            "proof.",
            "No real TEE isolation or hardware side-channel "
            "resistance is evaluated.",
            "Raw tensors, masks, permutations, adapters, gradients, "
            "and private data are not exported.",
            "Per-row prompt-id and position-bucket labels are not "
            "available in the stitched [N, D] view; the corresponding "
            "classifier tasks are skipped rather than fabricated.",
        ],
        "paper_safe_wording": (
            "We report a permutation-invariant leakage audit of the "
            "existing ZP-style compatible nonlinear islands. Row-wise "
            "norms, extrema, sorted multisets, and quantiles are "
            "preserved across both selectable mitigation bundles; "
            "this is a *channel-index hiding* property, not value "
            "hiding. The full bundle changes the freshness / "
            "temporal contract but does not remove single-shot "
            "permutation-invariant statistics inside the activation "
            "core."
        ),
        "unsafe_wording_to_avoid": [
            "Permutation-only views give value-level privacy.",
            "Fresh permutation hides per-row magnitudes.",
            "Dense sandwiching provides cryptographic security.",
            "This is a formal security proof.",
            "Real TEE / GPU evaluated.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# JSON / CSV / Markdown writers
# ---------------------------------------------------------------------------


def _safe_round(x: Any, digits: int = 6) -> Any:
    if isinstance(x, float):
        if x != x:  # NaN
            return "NaN"
        return round(x, digits)
    return x


def _write_json(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)


def _flatten_for_csv(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle, bundle_data in report["per_bundle"].items():
        for tensor, scope_map in bundle_data["per_tensor"].items():
            for scope, audit in scope_map.items():
                if "skipped_with_reason" in audit:
                    rows.append({
                        "bundle": bundle, "tensor": tensor, "scope": scope,
                        "skipped_with_reason": audit["skipped_with_reason"],
                    })
                    continue
                row = {
                    "bundle": bundle, "tensor": tensor, "scope": scope,
                    "num_rows": audit.get("num_rows"),
                    "feature_dim": audit.get("feature_dim"),
                    "statistical_leakage_label": audit.get(
                        "statistical_leakage_label",
                    ),
                }
                for k, v in audit.get("metrics", {}).items():
                    row[k] = _safe_round(v, 8)
                rows.append(row)
    return rows


def _write_csv(report: dict[str, Any], path: str) -> None:
    rows = _flatten_for_csv(report)
    if not rows:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "---|" * len(cols))
    for r in rows:
        out.append(
            "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        )
    return "\n".join(out)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Stage 5.7 — Permutation-Invariant Leakage Audit")
    w()
    w("## 1. Experiment Scope")
    w()
    w(
        "We audit single-shot permutation-invariant leakage of the "
        "existing ZP-style compatible nonlinear islands under both "
        "selectable mitigation bundles "
        "(`fresh_perm_only` and "
        "`fresh_perm_plus_sandwich_plus_pad`). Trace pairs are "
        "collected with the Stage 5.5b real-token trace collector at "
        "the model-level wrapper boundaries. No protocol math, "
        "wrappers, defaults, or existing tests are modified."
    )
    w()
    w("## 2. Threat Model")
    w()
    w(
        "Honest-but-curious accelerator that observes the visible "
        "(ZP-permuted) activations at the wrapper-side boundaries. "
        "No real TEE isolation or hardware side-channel resistance "
        "is evaluated. Raw tensors, masks, permutations, adapters, "
        "gradients, and private data are not exported."
    )
    w()
    w("## 3. Theoretical Claim Tested")
    w()
    w(
        "Permutation-only nonlinear views provide channel-index "
        "hiding, not value hiding. Per-row L1 / L2 / Linf norms, "
        "extrema, sorted multisets, and quantiles are preserved by "
        "construction. The full mitigation bundle changes the "
        "freshness / temporal contract but does not remove "
        "single-shot permutation-invariant statistics inside the "
        "activation core."
    )
    w()
    w("## 4. Target Tensor Inventory")
    w()
    inv = report["target_tensor_inventory"]
    inv_rows = [{
        "tensor": tn,
        "prefill_present": v.get("prefill_present", False),
        "decode_present": v.get("decode_present", False),
    } for tn, v in inv.items()]
    w(_md_table(inv_rows, ["tensor", "prefill_present", "decode_present"]))
    w()

    bundles = list(report["per_bundle"].keys())
    w("## 5. Norm Preservation")
    w()
    rows: list[dict[str, Any]] = []
    for b in bundles:
        per_tensor = report["per_bundle"][b]["per_tensor"]
        for tn, scope_map in per_tensor.items():
            for scope, audit in scope_map.items():
                if "metrics" not in audit:
                    continue
                m = audit["metrics"]
                rows.append({
                    "bundle": b, "tensor": tn, "scope": scope,
                    "l1_corr": _safe_round(m["l1_corr"], 4),
                    "l2_corr": _safe_round(m["l2_corr"], 4),
                    "linf_corr": _safe_round(m["linf_corr"], 4),
                    "l2_max_abs_diff": _safe_round(m["l2_max_abs_diff"], 4),
                    "l2_mean_abs_diff": _safe_round(m["l2_mean_abs_diff"], 4),
                    "label": audit.get("statistical_leakage_label"),
                })
    if rows:
        w(_md_table(rows, [
            "bundle", "tensor", "scope",
            "l1_corr", "l2_corr", "linf_corr",
            "l2_max_abs_diff", "l2_mean_abs_diff", "label",
        ]))
    else:
        w("_(no measured tensors in either scope under either bundle)_")
    w()

    w("## 6. Sorted Multiset Preservation")
    w()
    rows = []
    for b in bundles:
        per_tensor = report["per_bundle"][b]["per_tensor"]
        for tn, scope_map in per_tensor.items():
            for scope, audit in scope_map.items():
                if "metrics" not in audit:
                    continue
                m = audit["metrics"]
                rows.append({
                    "bundle": b, "tensor": tn, "scope": scope,
                    "sorted_mse_mean": _safe_round(m["sorted_mse_mean"], 8),
                    "sorted_mse_max": _safe_round(m["sorted_mse_max"], 8),
                    "sorted_l2_rel_mean": _safe_round(
                        m["sorted_l2_rel_mean"], 6,
                    ),
                })
    if rows:
        w(_md_table(rows, [
            "bundle", "tensor", "scope",
            "sorted_mse_mean", "sorted_mse_max", "sorted_l2_rel_mean",
        ]))
    else:
        w("_(no measured tensors)_")
    w()

    w("## 7. Quantile / Extrema Preservation")
    w()
    rows = []
    for b in bundles:
        per_tensor = report["per_bundle"][b]["per_tensor"]
        for tn, scope_map in per_tensor.items():
            for scope, audit in scope_map.items():
                if "metrics" not in audit:
                    continue
                m = audit["metrics"]
                rows.append({
                    "bundle": b, "tensor": tn, "scope": scope,
                    "max_corr": _safe_round(m["max_corr"], 4),
                    "min_corr": _safe_round(m["min_corr"], 4),
                    "quantile_mse_mean": _safe_round(
                        m["quantile_mse_mean"], 8,
                    ),
                })
    if rows:
        w(_md_table(rows, [
            "bundle", "tensor", "scope",
            "max_corr", "min_corr", "quantile_mse_mean",
        ]))
    else:
        w("_(no measured tensors)_")
    w()

    w("## 8. Statistics-Only Classifier")
    w()
    rows = []
    for b in bundles:
        ctasks = report["per_bundle"][b]["classifier_proxy_tasks"]
        for tn, task_map in ctasks.items():
            for task_name, res in task_map.items():
                rows.append({
                    "bundle": b, "tensor": tn, "task": task_name,
                    "status": res.get("status"),
                    "accuracy": _safe_round(res.get("accuracy"), 4),
                    "chance_level": _safe_round(res.get("chance_level"), 4),
                    "label": res.get("proxy_attack_label"),
                })
    if rows:
        w(_md_table(rows, [
            "bundle", "tensor", "task", "status",
            "accuracy", "chance_level", "label",
        ]))
    else:
        w("_(no classifier tasks were runnable)_")
    w()

    w("## 9. Freshness Ablation")
    w()
    if report.get("fixed_permutation_debug") is None:
        w("_(fixed-permutation debug disabled)_")
    else:
        fd = report["fixed_permutation_debug"]
        w("Fixed-permutation vs fresh-permutation synthetic ablation:")
        w()
        w(f"- hidden_dim: `{fd['hidden_dim']}`")
        w(f"- num_rows: `{fd['num_rows']}`")
        w(
            f"- fixed_perm linkability accuracy: "
            f"`{_safe_round(fd['fixed_perm_linkability_accuracy'], 4)}` "
            f"(chance `{_safe_round(fd['linkability_chance_level'], 4)}`)"
        )
        w(
            f"- fresh_perm linkability accuracy: "
            f"`{_safe_round(fd['fresh_perm_linkability_accuracy'], 4)}` "
            f"(chance `{_safe_round(fd['linkability_chance_level'], 4)}`)"
        )
        fixed_m = fd["fixed_permutation_metrics"]
        fresh_m = fd["fresh_permutation_metrics"]
        w(
            f"- fixed_perm `sorted_l2_rel_mean = "
            f"{_safe_round(fixed_m['sorted_l2_rel_mean'], 6)}`, "
            f"`l2_corr = {_safe_round(fixed_m['l2_corr'], 4)}`"
        )
        w(
            f"- fresh_perm `sorted_l2_rel_mean = "
            f"{_safe_round(fresh_m['sorted_l2_rel_mean'], 6)}`, "
            f"`l2_corr = {_safe_round(fresh_m['l2_corr'], 4)}`"
        )
        w()
        w(fd["interpretation"])
    w()

    w("## 10. Interpretation")
    w()
    w(
        "For raw permutation-island views (`gate`, `up`, "
        "`swiglu_intermediate`), the per-row norm, sorted-multiset, "
        "and quantile metrics are preserved at or near machine "
        "precision. The corresponding rows are labelled "
        "`statistical_leakage_detected_*`. This is the *channel-"
        "index hiding, not value hiding* property of ZP-style "
        "permutation masks. For dense post-island views (`post_island`, "
        "`boundary_input`, `final`), the same metrics generally do "
        "NOT match, because the dense right-mask plus boundary pad "
        "no longer share the multiset of the plain activations."
    )
    w()
    w(
        "The two mitigation bundles "
        "(`fresh_perm_only` vs `fresh_perm_plus_sandwich_plus_pad`) "
        "are equivalent at the single-shot activation-core multiset "
        "level. The full bundle's added value is in temporal / "
        "boundary / freshness posture, not in single-shot "
        "permutation-invariant statistics."
    )
    w()

    w("## 11. Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()

    w("## 12. Next Stage Plan")
    w()
    w(
        "Future work: add an enhanced nonlinear protection that breaks "
        "row-wise permutation invariance (for example, masked dense "
        "expansion + paired-permutation absorption that mixes channels "
        "across the islands), and a side-channel-aware proxy. Both "
        "are out of scope for Stage 5.7; they would lift the current "
        "permutation-invariant leakage from *channel-index hidden* to "
        "*value-multiset hidden* on these boundaries. As of this "
        "stage, no real TEE / GPU isolation or hardware side-channel "
        "resistance is claimed."
    )
    w()
    w(f"`formal_security_claim`: `{report['formal_security_claim']}`")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any], *,
    outputs_dir: str = "outputs",
    json_filename: str = "permutation_invariant_leakage.json",
    csv_filename: str = "permutation_invariant_leakage.csv",
    md_filename: str = "permutation_invariant_leakage.md",
) -> tuple[str, str, str]:
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, json_filename)
    csv_path = os.path.join(outputs_dir, csv_filename)
    md_path = os.path.join(outputs_dir, md_filename)
    _write_json(report, json_path)
    _write_csv(report, csv_path)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return json_path, csv_path, md_path


__all__ = [
    "PermutationInvariantLeakageConfig",
    "render_markdown",
    "run_permutation_invariant_leakage",
    "write_reports",
]
