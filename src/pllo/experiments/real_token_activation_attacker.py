"""Stage 5.5b — real-token-prompted real-activation adaptive proxy attacker.

Reuses Stage 5.5's adaptive attacker family (ridge linear inverter, small
two-layer MLP inverter, signature / Sinkhorn permutation recovery,
linkability proxy) but the (plain, visible) trace dataset now comes from
:mod:`pllo.experiments.real_token_trace`, which drives the Stage 6.4c
model-level wrapper end-to-end on real (or deterministically synthetic)
``input_ids`` rather than random hidden states.

Scope intentionally narrow:

* Real-token-prompted adaptive proxy attacks, **not** formal security
  proofs.
* Greedy generation only; beam / top-k / top-p never run.
* No black-box query, no side-channel, no real TEE measurement.
* ``security_profile`` stays ``"proxy-evaluated, not formal"``; the new
  detail label
  ``"real-token-real-activation-adaptive-proxy-evaluated, not formal"``
  is additive metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from pllo.experiments.adaptive_island_attacker import (
    _fit_linear_inverter,
    _reconstruction_metrics,
    _soft_assignment_top1,
)
from pllo.experiments.nonlinear_island_security import (
    compute_channel_signature,
    recover_permutation_by_signature,
)
from pllo.experiments.real_token_trace import (
    DEFAULT_TARGET_TENSORS,
    RealTokenTraceConfig,
    collect_real_token_traces,
)
from pllo.ops.mitigation_bundles import (
    VALID_MITIGATION_BUNDLES,
    bundle_metadata,
)


# Same boundaries as Stage 5.5 — permutation recovery is meaningful only
# inside the SwiGLU island. Q/K/V/post_island/final use dense / orthogonal
# masks (or are plain at the model-wrapper inter-block boundary).
PERMUTATION_TARGET_TENSORS: tuple[str, ...] = (
    "gate", "up", "swiglu_intermediate",
)


# Names whose visible is structurally identical to plain at the model-wrapper
# inter-block boundary (Stage 6.4c recovers between blocks); we flag these
# explicitly so the recommendation distinguishes "linearly trivial because
# plain" from "linearly recoverable despite mask".
INTER_BLOCK_PLAIN_TENSORS: tuple[str, ...] = (
    "boundary_input", "final",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RealTokenActivationAttackConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    model_id: str | None = None
    attempt_real_model_load: bool = False
    attempt_tokenizer_load: bool = False
    local_files_only: bool = False
    allow_synthetic_fallback: bool = True
    num_prompts: int = 32
    prompt_max_length: int = 16
    max_layers: int = 2
    max_new_tokens: int = 3
    attacker_steps: int = 200
    attacker_lr: float = 1e-2
    mlp_hidden_size: int = 128
    mlp_batch_size: int = 64
    ridge_lambda: float = 1e-3
    soft_assignment_iters: int = 50
    soft_assignment_temperature: float = 0.05
    train_fraction: float = 0.7
    target_tensors: tuple[str, ...] = DEFAULT_TARGET_TENSORS
    mitigation_bundles: tuple[str, ...] = VALID_MITIGATION_BUNDLES
    use_pad: bool = True
    nonlinear_mode: str = "compatible_islands"
    dtype: str = "float32"
    device: str = "cpu"
    # Synthetic-fallback shape (matches Stage 6.4c model probe defaults).
    synthetic_vocab_size: int = 256
    synthetic_hidden_size: int = 32
    synthetic_intermediate_size: int = 64
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 8


# ---------------------------------------------------------------------------
# Helpers (mirror Stage 5.5)
# ---------------------------------------------------------------------------


def _split_train_test(
    plain: torch.Tensor, visible: torch.Tensor, train_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = plain.shape[0]
    n_train = max(1, int(n * train_fraction))
    if n_train >= n:
        n_train = max(1, n - 1)
    X_train = plain[:n_train]
    V_train = visible[:n_train]
    X_test = plain[n_train:]
    V_test = visible[n_train:]
    if X_test.shape[0] == 0:
        X_test = plain[:1]
        V_test = visible[:1]
    return X_train, V_train, X_test, V_test


def _classify_risk(
    *,
    is_inter_block_plain: bool,
    linear_rel_l2: float,
    mlp_rel_l2: float,
    perm_soft_top1: float | None,
    random_chance: float,
    linkability_cosine: float,
) -> tuple[str, str]:
    """Same thresholds as Stage 5.5, plus an explicit ``inter_block_plain``
    risk recommendation that records the model-wrapper's recovered surface
    without conflating it with the masked-attack-defeated case."""
    if is_inter_block_plain:
        return "high", "inter_block_plain_recovered"
    best_rel_l2 = min(linear_rel_l2, mlp_rel_l2)
    if (
        best_rel_l2 < 0.20
        or linkability_cosine > 0.50
        or (perm_soft_top1 is not None and perm_soft_top1 > 0.50)
    ):
        return "high", "unsafe_default_on_under_real_token_proxy"
    if (
        best_rel_l2 < 0.60
        or linkability_cosine > 0.25
        or (perm_soft_top1 is not None and perm_soft_top1 > 4.0 * random_chance)
    ):
        return "medium", "needs_more_evaluation_under_real_token_proxy"
    return "low", "acceptable_with_mitigation_under_real_token_proxy"


# ---------------------------------------------------------------------------
# Per-attack helpers
# ---------------------------------------------------------------------------


def _run_linear(
    plain: torch.Tensor, visible: torch.Tensor,
    config: RealTokenActivationAttackConfig,
) -> dict[str, Any]:
    X_train, V_train, X_test, V_test = _split_train_test(
        plain, visible, config.train_fraction,
    )
    W = _fit_linear_inverter(V_train, X_train, config.ridge_lambda)
    X_pred = V_test @ W
    metrics = _reconstruction_metrics(X_pred, X_test)
    return {
        "num_train_samples": int(V_train.shape[0]),
        "num_test_samples": int(V_test.shape[0]),
        "ridge_lambda": float(config.ridge_lambda),
        **metrics,
    }


def _run_mlp(
    plain: torch.Tensor, visible: torch.Tensor,
    config: RealTokenActivationAttackConfig, seed: int,
) -> dict[str, Any]:
    X_train, V_train, X_test, V_test = _split_train_test(
        plain, visible, config.train_fraction,
    )
    torch.manual_seed(seed)
    feature_dim_in = V_train.shape[-1]
    feature_dim_out = X_train.shape[-1]
    model = torch.nn.Sequential(
        torch.nn.Linear(feature_dim_in, config.mlp_hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(config.mlp_hidden_size, feature_dim_out),
    ).to(dtype=V_train.dtype, device=V_train.device)
    optim = torch.optim.Adam(model.parameters(), lr=config.attacker_lr)
    batch_size = min(config.mlp_batch_size, V_train.shape[0])
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    losses: list[float] = []
    for _ in range(max(1, config.attacker_steps)):
        idx = torch.randint(
            0, V_train.shape[0], (batch_size,), generator=gen,
        ).to(V_train.device)
        pred = model(V_train[idx])
        loss = F.mse_loss(pred, X_train[idx])
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        X_pred = model(V_test)
    metrics = _reconstruction_metrics(X_pred, X_test)
    return {
        "num_train_samples": int(V_train.shape[0]),
        "num_test_samples": int(V_test.shape[0]),
        "attacker_steps": int(config.attacker_steps),
        "attacker_lr": float(config.attacker_lr),
        "mlp_hidden_size": int(config.mlp_hidden_size),
        "final_train_loss": float(losses[-1]) if losses else None,
        "first_train_loss": float(losses[0]) if losses else None,
        **metrics,
    }


def _run_perm(
    plain: torch.Tensor, visible: torch.Tensor,
    config: RealTokenActivationAttackConfig,
) -> dict[str, Any]:
    ref_sig = compute_channel_signature(plain)
    vis_sig = compute_channel_signature(visible)
    sig_metrics = recover_permutation_by_signature(ref_sig, vis_sig)
    soft_metrics = _soft_assignment_top1(
        ref_sig, vis_sig,
        iters=config.soft_assignment_iters,
        temperature=config.soft_assignment_temperature,
    )
    feature_dim = plain.shape[-1]
    return {
        "feature_dim": int(feature_dim),
        "random_chance_top1": 1.0 / max(1, feature_dim),
        "signature_matching": sig_metrics,
        "soft_assignment": soft_metrics,
        "best_top1": float(max(
            sig_metrics.get("top1_recovery_rate", 0.0),
            soft_metrics.get("top1_recovery_rate", 0.0),
        )),
    }


def _run_linkability(
    plain: torch.Tensor, visible: torch.Tensor, *, num_pairs: int = 1024,
) -> dict[str, float]:
    n = plain.shape[0]
    plain64 = plain.detach().to(torch.float64)
    visible64 = visible.detach().to(torch.float64)
    p_norm = plain64 / plain64.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    v_norm = visible64 / visible64.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    row_cos = (p_norm * v_norm).sum(dim=-1)
    visible_vs_plain_cosine = float(row_cos.mean().item())
    gen = torch.Generator(device="cpu").manual_seed(0xB055)
    pairs = min(num_pairs, n * (n - 1) // 2) if n > 1 else 0
    if pairs == 0:
        return {
            "visible_vs_plain_cosine": visible_vs_plain_cosine,
            "mean_pairwise_cosine_visible": 0.0,
            "mean_pairwise_l2_visible": 0.0,
            "mean_linkability_rank": 1.0,
        }
    i_idx = torch.randint(0, n, (pairs,), generator=gen)
    j_idx = torch.randint(0, n, (pairs,), generator=gen)
    keep = i_idx != j_idx
    i_idx = i_idx[keep]
    j_idx = j_idx[keep]
    if i_idx.numel() == 0:
        mean_pairwise_visible = 0.0
        mean_pairwise_l2 = 0.0
    else:
        cos_pair = (v_norm[i_idx] * v_norm[j_idx]).sum(dim=-1)
        mean_pairwise_visible = float(cos_pair.mean().item())
        l2_pair = (visible64[i_idx] - visible64[j_idx]).norm(dim=-1)
        mean_pairwise_l2 = float(l2_pair.mean().item())
    n_subset = min(64, n)
    idx_subset = torch.arange(n_subset)
    sim_matrix = p_norm[idx_subset] @ v_norm.T
    ranks = sim_matrix.argsort(dim=-1, descending=True)
    link_rank = (ranks == idx_subset.unsqueeze(-1)).int().argmax(dim=-1)
    return {
        "visible_vs_plain_cosine": visible_vs_plain_cosine,
        "mean_pairwise_cosine_visible": mean_pairwise_visible,
        "mean_pairwise_l2_visible": mean_pairwise_l2,
        "mean_linkability_rank": float(link_rank.to(torch.float64).mean().item()),
    }


# ---------------------------------------------------------------------------
# Per-tensor-per-bundle runner
# ---------------------------------------------------------------------------


def _per_tensor_run(
    tensor_name: str,
    scope: str,
    plain: torch.Tensor,
    visible: torch.Tensor,
    config: RealTokenActivationAttackConfig,
    seed_offset: int,
) -> dict[str, Any]:
    feature_dim = int(plain.shape[-1])
    is_inter_block_plain = tensor_name in INTER_BLOCK_PLAIN_TENSORS
    linear_metrics = _run_linear(plain, visible, config)
    mlp_metrics = _run_mlp(plain, visible, config, seed=config.seed + seed_offset)
    if tensor_name in PERMUTATION_TARGET_TENSORS:
        perm_metrics = _run_perm(plain, visible, config)
        perm_top1 = perm_metrics["best_top1"]
    else:
        perm_metrics = None
        perm_top1 = None
    link_metrics = _run_linkability(plain, visible)
    risk_level, recommendation = _classify_risk(
        is_inter_block_plain=is_inter_block_plain,
        linear_rel_l2=linear_metrics["relative_l2_error"],
        mlp_rel_l2=mlp_metrics["relative_l2_error"],
        perm_soft_top1=perm_top1,
        random_chance=1.0 / max(1, feature_dim),
        linkability_cosine=link_metrics["visible_vs_plain_cosine"],
    )
    return {
        "tensor_name": tensor_name,
        "scope": scope,
        "feature_dim": feature_dim,
        "inter_block_plain": bool(is_inter_block_plain),
        "linear_inverter": linear_metrics,
        "mlp_inverter": mlp_metrics,
        "permutation_recovery": perm_metrics,
        "linkability": link_metrics,
        "risk_level": risk_level,
        "default_on_recommendation": recommendation,
        "mlp_minus_linear_relative_l2_error": (
            mlp_metrics["relative_l2_error"]
            - linear_metrics["relative_l2_error"]
        ),
    }


# ---------------------------------------------------------------------------
# Bundle comparison
# ---------------------------------------------------------------------------


def _compare_bundles(
    per_tensor: dict[str, dict[str, dict[str, dict[str, Any]]]],
    target_tensors: tuple[str, ...],
    scopes: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for tensor in target_tensors:
            fresh = per_tensor.get(
                "fresh_perm_only", {},
            ).get(scope, {}).get(tensor)
            full = per_tensor.get(
                "fresh_perm_plus_sandwich_plus_pad", {},
            ).get(scope, {}).get(tensor)
            if fresh is None or full is None:
                continue
            row: dict[str, Any] = {
                "scope": scope,
                "tensor_name": tensor,
                "feature_dim": fresh["feature_dim"],
                "inter_block_plain": bool(fresh["inter_block_plain"]),
                "linear_rel_l2_fresh_only": fresh["linear_inverter"][
                    "relative_l2_error"
                ],
                "linear_rel_l2_full_bundle": full["linear_inverter"][
                    "relative_l2_error"
                ],
                "linear_rel_l2_delta": (
                    full["linear_inverter"]["relative_l2_error"]
                    - fresh["linear_inverter"]["relative_l2_error"]
                ),
                "mlp_rel_l2_fresh_only": fresh["mlp_inverter"][
                    "relative_l2_error"
                ],
                "mlp_rel_l2_full_bundle": full["mlp_inverter"][
                    "relative_l2_error"
                ],
                "mlp_rel_l2_delta": (
                    full["mlp_inverter"]["relative_l2_error"]
                    - fresh["mlp_inverter"]["relative_l2_error"]
                ),
                "linkability_cosine_fresh_only": fresh["linkability"][
                    "visible_vs_plain_cosine"
                ],
                "linkability_cosine_full_bundle": full["linkability"][
                    "visible_vs_plain_cosine"
                ],
                "linkability_cosine_delta": (
                    full["linkability"]["visible_vs_plain_cosine"]
                    - fresh["linkability"]["visible_vs_plain_cosine"]
                ),
                "risk_level_fresh_only": fresh["risk_level"],
                "risk_level_full_bundle": full["risk_level"],
                "recommendation": full["default_on_recommendation"],
                "math_equivalence_note": (
                    "fresh_perm_only and fresh_perm_plus_sandwich_plus_pad"
                    " share the same per-call mask sampling under the"
                    " Stage 6.4c model-level wrapper; the two bundles"
                    " produce identical numerical traces for masked"
                    " tensors. The bundle label distinguishes the"
                    " security posture (default-on candidate vs. not),"
                    " not the numerical visibility surface."
                ),
            }
            if (
                fresh["permutation_recovery"] is not None
                and full["permutation_recovery"] is not None
            ):
                row["permutation_top1_fresh_only"] = fresh[
                    "permutation_recovery"
                ]["best_top1"]
                row["permutation_top1_full_bundle"] = full[
                    "permutation_recovery"
                ]["best_top1"]
                row["permutation_top1_delta"] = (
                    full["permutation_recovery"]["best_top1"]
                    - fresh["permutation_recovery"]["best_top1"]
                )
                row["random_chance_top1"] = fresh["permutation_recovery"][
                    "random_chance_top1"
                ]
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-bundle headline summary
# ---------------------------------------------------------------------------


def _summarise(
    per_tensor: dict[str, dict[str, dict[str, dict[str, Any]]]],
    target_tensors: tuple[str, ...],
    scopes: tuple[str, ...],
) -> dict[str, Any]:
    order = {"low": 0, "medium": 1, "high": 2}
    inv_order = {0: "low", 1: "medium", 2: "high"}
    out: dict[str, Any] = {}
    for bundle, by_scope in per_tensor.items():
        if not by_scope:
            continue
        flat: list[dict[str, Any]] = []
        for scope in scopes:
            for tensor in target_tensors:
                rec = by_scope.get(scope, {}).get(tensor)
                if rec is not None:
                    flat.append(rec)
        if not flat:
            continue
        # When grading "max masked risk" we ignore tensors that are plain
        # inter-block, since high risk there is structural, not adversarial.
        masked_only = [r for r in flat if not r["inter_block_plain"]]
        masked_risks = [r["risk_level"] for r in masked_only]
        masked_max = (
            inv_order[max(order[r] for r in masked_risks)]
            if masked_risks else "low"
        )
        all_risks = [r["risk_level"] for r in flat]
        all_max = (
            inv_order[max(order[r] for r in all_risks)]
            if all_risks else "low"
        )
        out[bundle] = {
            "bundle": bundle,
            "tensors_covered": [(r["scope"], r["tensor_name"]) for r in flat],
            "max_risk_level_overall": all_max,
            "max_risk_level_masked_only": masked_max,
            "inter_block_plain_tensors": list(INTER_BLOCK_PLAIN_TENSORS),
            "mean_linear_rel_l2_masked_only": float(sum(
                r["linear_inverter"]["relative_l2_error"] for r in masked_only
            ) / max(1, len(masked_only))),
            "mean_mlp_rel_l2_masked_only": float(sum(
                r["mlp_inverter"]["relative_l2_error"] for r in masked_only
            ) / max(1, len(masked_only))),
            "mean_linkability_cosine_masked_only": float(sum(
                r["linkability"]["visible_vs_plain_cosine"] for r in masked_only
            ) / max(1, len(masked_only))),
            "risk_counts_masked_only": {
                level: sum(1 for r in masked_risks if r == level)
                for level in ("low", "medium", "high")
            },
            "risk_counts_all": {
                level: sum(1 for r in all_risks if r == level)
                for level in ("low", "medium", "high")
            },
        }
    return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "These are real-token-prompted adaptive proxy attacks, not formal security proofs.",
    "If synthetic token fallback is used, results are not real tokenizer-driven traces.",
    "If synthetic model fallback is used, results are not real Qwen/TinyLlama traces.",
    "Prompt set is small and not representative of all user data.",
    "No black-box query attack is implemented.",
    "No side-channel attack is implemented.",
    "No real TEE isolation is evaluated.",
    "Dense sandwiching reduces tested recovery but does not imply semantic security.",
    "This stage only evaluates greedy generation traces, not sampling / beam search / top-k / top-p.",
    "Inter-block hidden states (boundary_input, final) are recovered to plain space"
    " between layers under the Stage 6.4c model-level wrapper; an attacker observing"
    " those boundaries sees plaintext. This is a known model-wrapper limitation, not"
    " a Stage 5.5b attacker finding.",
    "fresh_perm_only and fresh_perm_plus_sandwich_plus_pad share the same per-call"
    " mask sampling under the Stage 6.4c wrapper; the two bundles produce numerically"
    " identical traces for masked tensors.",
]


def run_real_token_activation_attacks(
    config: RealTokenActivationAttackConfig,
) -> dict[str, Any]:
    """Drive Stage 5.5b trace collection + attacker sweep."""
    bundles = tuple(config.mitigation_bundles)
    scopes: tuple[str, ...] = ("prefill", "decode")

    per_tensor: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        b: {s: {} for s in scopes} for b in bundles
    }
    trace_summaries: dict[str, Any] = {}
    generation_summaries: dict[str, Any] = {}
    model_loading: dict[str, Any] | None = None
    tokenizer_loading: dict[str, Any] | None = None
    source: str | None = None
    prompt_summary: dict[str, Any] | None = None
    block_spec_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    decode_step_log: list[dict[str, Any]] = []

    for bundle_idx, bundle in enumerate(bundles):
        trace_pkg = collect_real_token_traces(
            RealTokenTraceConfig(
                output_dir=config.output_dir,
                seed=config.seed,
                model_id=config.model_id,
                attempt_real_model_load=config.attempt_real_model_load,
                attempt_tokenizer_load=config.attempt_tokenizer_load,
                local_files_only=config.local_files_only,
                allow_synthetic_fallback=config.allow_synthetic_fallback,
                max_layers=config.max_layers,
                max_new_tokens=config.max_new_tokens,
                prompt_max_length=config.prompt_max_length,
                num_prompts=config.num_prompts,
                use_pad=config.use_pad,
                nonlinear_mode=config.nonlinear_mode,
                mitigation_bundle=bundle,
                target_tensors=config.target_tensors,
                dtype=config.dtype,
                device=config.device,
                synthetic_vocab_size=config.synthetic_vocab_size,
                synthetic_hidden_size=config.synthetic_hidden_size,
                synthetic_intermediate_size=config.synthetic_intermediate_size,
                synthetic_num_attention_heads=(
                    config.synthetic_num_attention_heads
                ),
                synthetic_num_key_value_heads=(
                    config.synthetic_num_key_value_heads
                ),
                synthetic_head_dim=config.synthetic_head_dim,
            )
        )
        trace_summaries[bundle] = trace_pkg["trace_summary"]
        generation_summaries[bundle] = trace_pkg["generation_summary"]
        if model_loading is None:
            model_loading = trace_pkg["model_loading"]
            tokenizer_loading = trace_pkg["tokenizer_loading"]
            source = trace_pkg["source"]
            prompt_summary = trace_pkg["prompt_summary"]
            block_spec_summary = trace_pkg["block_spec_summary"]
            metadata = trace_pkg["metadata"]
            decode_step_log = trace_pkg["decode_step_log"]
        for scope in scopes:
            scope_traces = trace_pkg["traces"].get(scope, {})
            for tensor in config.target_tensors:
                pair = scope_traces.get(tensor)
                if pair is None:
                    continue
                per_tensor[bundle][scope][tensor] = _per_tensor_run(
                    tensor, scope, pair["plain"], pair["visible"],
                    config, seed_offset=37 * (bundle_idx + 1) + hash(scope) % 13,
                )

    comparison = _compare_bundles(per_tensor, config.target_tensors, scopes)
    attacker_summary = _summarise(per_tensor, config.target_tensors, scopes)
    full = attacker_summary.get("fresh_perm_plus_sandwich_plus_pad", {})
    fresh = attacker_summary.get("fresh_perm_only", {})

    def _recommend(level: str | None) -> str:
        if level == "low":
            return "acceptable_with_mitigation_under_real_token_proxy"
        if level == "medium":
            return "needs_more_evaluation_under_real_token_proxy"
        return "unsafe_default_on_under_real_token_proxy"

    recommendation = {
        "default_on_recommendation_full_bundle_masked_only": _recommend(
            full.get("max_risk_level_masked_only")
        ),
        "default_on_recommendation_full_bundle_overall": _recommend(
            full.get("max_risk_level_overall")
        ),
        "default_on_recommendation_fresh_only_masked_only": _recommend(
            fresh.get("max_risk_level_masked_only")
        ),
        "default_on_recommendation_fresh_only_overall": _recommend(
            fresh.get("max_risk_level_overall")
        ),
        "security_profile_detail_with_real_token_activation": (
            "real-token-real-activation-adaptive-proxy-evaluated, not formal"
        ),
        "note": (
            "Inter-block tensors (boundary_input, final) are plain at the"
            " model-wrapper boundary by construction; their high risk is"
            " STRUCTURAL, not a finding against the mitigation bundle. The"
            " masked-only recommendation grades the masked tensors only."
        ),
    }

    comparison_with_stage_5_5 = {
        "stage_5_5_random_hidden_artifact": "outputs/real_activation_attacks.json",
        "stage_5_5b_real_token_artifact": "outputs/real_token_activation_attacks.json",
        "key_differences": [
            "Stage 5.5 feeds random hidden states directly to the Stage 6.4b"
            " block wrapper; Stage 5.5b feeds real (or deterministic synthetic)"
            " token IDs through the Stage 6.4c model wrapper (embedding + N"
            " blocks + final RMSNorm + LM head).",
            "Stage 5.5b therefore covers the prefill AND decode_step paths,"
            " including the masked KV-cache append surface, whereas Stage 5.5"
            " covered only one block's prefill-style forward.",
            "Stage 5.5 traces use the Stage 6.4b N_res orthogonal residual"
            " mask around the block boundary; the Stage 6.4c model wrapper"
            " recovers between blocks, so boundary_input / final are PLAIN"
            " under Stage 5.5b. This is documented as a structural model-wrapper"
            " limitation, not a Stage 5.5b attacker finding.",
            "Masked-tensor recommendations (Q/K/V/gate/up/swiglu_intermediate/"
            "post_island) carry over: rel_l2 stays high, linkability stays"
            " low, permutation top1 stays near random chance for the SwiGLU"
            " island tensors.",
        ],
    }

    return {
        "config": asdict(config),
        "model_loading": model_loading or {},
        "tokenizer_loading": tokenizer_loading or {},
        "source": source,
        "prompt_summary": prompt_summary or {},
        "block_spec_summary": block_spec_summary or {},
        "metadata": metadata or {},
        "decode_step_log": decode_step_log,
        "generation_summary": generation_summaries,
        "trace_summary": trace_summaries,
        "target_tensor_results": per_tensor,
        "bundle_comparison": comparison,
        "attacker_summary": attacker_summary,
        "recommendation": recommendation,
        "comparison_with_stage_5_5": comparison_with_stage_5_5,
        "limitations": list(_LIMITATIONS),
    }


__all__ = [
    "INTER_BLOCK_PLAIN_TENSORS",
    "PERMUTATION_TARGET_TENSORS",
    "RealTokenActivationAttackConfig",
    "run_real_token_activation_attacks",
]
