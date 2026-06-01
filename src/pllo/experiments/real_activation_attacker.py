"""Stage 5.5 — real-activation adaptive proxy attacker.

Drives :mod:`pllo.experiments.real_activation_trace` to collect
(plain, visible) tensor pairs from the Stage 6.4b modern decoder block
wrapper, then re-runs the Stage 5.4 attackers (ridge linear inverter,
small two-layer MLP inverter, signature / Sinkhorn permutation recovery,
linkability proxy) against those real-activation traces.

Compares ``fresh_perm_only`` (the safe default) against
``fresh_perm_plus_sandwich_plus_pad`` (the Stage 5.3e default-on
candidate). Both bundles use the SAME freshly-sampled
``N_in / perm / N_out`` per call — so numerically the traces match
byte-for-byte (the bundle is metadata over the same math).
``fixed_permutation_debug`` is an optional baseline that pins the
per-session masks via a fixed RNG seed; it is offered only as a
sanity reference and is NEVER recommended for deployment.

Scope is intentionally narrow: this is a real-activation adaptive
PROXY attacker. It is **not** a formal security proof, **not** a
black-box query attack, **not** a side-channel attack, and **not** a
real TEE measurement. ``security_profile`` stays
``"proxy-evaluated, not formal"``; the new detail label
``"real-activation-adaptive-proxy-evaluated, not formal"`` is additive
metadata in the Stage 5.5 report.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from pllo.experiments.adaptive_island_attacker import (
    AdaptiveIslandAttackConfig,
    SmallInverter,
    _fit_linear_inverter,
    _reconstruction_metrics,
    _soft_assignment_top1,
)
from pllo.experiments.nonlinear_island_security import (
    compute_channel_signature,
    recover_permutation_by_signature,
)
from pllo.experiments.real_activation_trace import (
    DEFAULT_TARGET_TENSORS,
    RealActivationTraceConfig,
    collect_real_activation_traces,
)
from pllo.ops.mitigation_bundles import (
    VALID_MITIGATION_BUNDLES,
    bundle_metadata,
)


# Permutation recovery is meaningful only at column-permutation boundaries
# (Stage 5.2a SwiGLU island). q / k / v / boundary_input / post_island /
# final use dense / orthogonal masks instead.
PERMUTATION_TARGET_TENSORS: tuple[str, ...] = (
    "gate", "up", "swiglu_intermediate",
)

# Stage 5.5 reference baselines. ``fixed_permutation_debug`` is opt-in;
# the actual deployment default is ``fresh_perm_only``.
EXTENDED_BUNDLES: tuple[str, ...] = (
    *VALID_MITIGATION_BUNDLES,
    "fixed_permutation_debug",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RealActivationAttackConfig:
    output_dir: str = "outputs"
    seed: int = 2026
    attempt_real_model_load: bool = False
    model_id: str | None = None
    local_files_only: bool = False
    allow_synthetic_fallback: bool = True
    num_samples: int = 512
    train_fraction: float = 0.7
    attacker_steps: int = 200
    attacker_lr: float = 1e-2
    mlp_hidden_size: int = 128
    mlp_batch_size: int = 64
    ridge_lambda: float = 1e-3
    soft_assignment_iters: int = 50
    soft_assignment_temperature: float = 0.05
    target_tensors: tuple[str, ...] = DEFAULT_TARGET_TENSORS
    mitigation_bundles: tuple[str, ...] = VALID_MITIGATION_BUNDLES
    fixed_permutation_seed: int = 7777  # for the debug baseline only
    batch_size: int = 2
    seq_len: int = 8
    synthetic_hidden_size: int = 64
    synthetic_intermediate_size: int = 128
    synthetic_num_attention_heads: int = 4
    synthetic_num_key_value_heads: int = 2
    synthetic_head_dim: int = 16
    use_pad: bool = True
    nonlinear_mode: str = "compatible_islands"
    dtype: str = "float32"
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Helpers — sample split, decision rules
# ---------------------------------------------------------------------------


def _split_train_test(
    plain: torch.Tensor, visible: torch.Tensor, train_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = plain.shape[0]
    n_train = max(1, int(n * train_fraction))
    X_train = plain[:n_train]
    V_train = visible[:n_train]
    X_test = plain[n_train:] if n_train < n else plain[:1]
    V_test = visible[n_train:] if n_train < n else visible[:1]
    return X_train, V_train, X_test, V_test


def _classify_risk(
    linear_rel_l2: float,
    mlp_rel_l2: float,
    perm_soft_top1: float | None,
    random_chance: float,
    linkability_cosine: float,
) -> tuple[str, str]:
    """Risk classification for a single tensor under a single bundle."""
    best_rel_l2 = min(linear_rel_l2, mlp_rel_l2)
    # High risk: linear inverter recovers (rel_l2 < 0.2) OR linkability
    # cosine is very high (>0.5) OR permutation top1 is high (>0.5).
    if (
        best_rel_l2 < 0.20
        or linkability_cosine > 0.50
        or (perm_soft_top1 is not None and perm_soft_top1 > 0.50)
    ):
        return "high", "unsafe_default_on"
    # Medium risk: linear inverter partial (rel_l2 < 0.6) OR perm top1
    # significantly above random chance (>4× random).
    if (
        best_rel_l2 < 0.60
        or linkability_cosine > 0.25
        or (perm_soft_top1 is not None and perm_soft_top1 > 4.0 * random_chance)
    ):
        return "medium", "needs_more_evaluation"
    return "low", "acceptable_with_mitigation"


# ---------------------------------------------------------------------------
# Linear inverter
# ---------------------------------------------------------------------------


def _run_linear_inverter_real(
    plain: torch.Tensor, visible: torch.Tensor, config: RealActivationAttackConfig,
) -> dict[str, Any]:
    X_train, V_train, X_test, V_test = _split_train_test(
        plain, visible, config.train_fraction
    )
    W = _fit_linear_inverter(V_train, X_train, config.ridge_lambda)
    X_pred = V_test @ W
    metrics = _reconstruction_metrics(X_pred, X_test)
    return {
        "num_train_samples": int(V_train.shape[0]),
        "num_test_samples": int(V_test.shape[0]),
        "ridge_lambda": config.ridge_lambda,
        **metrics,
    }


# ---------------------------------------------------------------------------
# MLP inverter
# ---------------------------------------------------------------------------


def _run_mlp_inverter_real(
    plain: torch.Tensor,
    visible: torch.Tensor,
    config: RealActivationAttackConfig,
    seed: int,
) -> dict[str, Any]:
    X_train, V_train, X_test, V_test = _split_train_test(
        plain, visible, config.train_fraction
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
# Permutation recovery
# ---------------------------------------------------------------------------


def _run_permutation_recovery_real(
    plain: torch.Tensor,
    visible: torch.Tensor,
    config: RealActivationAttackConfig,
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
        "best_top1": float(
            max(
                sig_metrics.get("top1_recovery_rate", 0.0),
                soft_metrics.get("top1_recovery_rate", 0.0),
            )
        ),
    }


# ---------------------------------------------------------------------------
# Linkability proxy
# ---------------------------------------------------------------------------


def _run_linkability_real(
    plain: torch.Tensor, visible: torch.Tensor, num_pairs: int = 1024,
) -> dict[str, float]:
    """Cosine-based linkability proxy.

    * ``visible_vs_plain_cosine`` — mean of ``cos(V[i], X[i])`` across
      rows. If close to 1.0, the visible row is almost the plain row.
    * ``mean_pairwise_cosine_visible`` — mean of ``cos(V[i], V[j])`` over
      random sample pairs. A useful baseline for "do all visible rows
      look alike" (which would indicate the mask collapses information).
    * ``linkability_rank`` — for each row ``i``, rank V[i] among all
      visible rows by similarity to X[i]; report mean reciprocal rank.
      A rank of 1 means the attacker can match X[i] to V[i] perfectly.
    """
    n = plain.shape[0]
    plain64 = plain.detach().to(torch.float64)
    visible64 = visible.detach().to(torch.float64)
    p_norm = plain64 / plain64.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    v_norm = visible64 / visible64.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    row_cos = (p_norm * v_norm).sum(dim=-1)            # [N]
    visible_vs_plain_cosine = float(row_cos.mean().item())
    # Pairwise visible cosine.
    gen = torch.Generator(device="cpu").manual_seed(0xC05E)
    pairs = min(num_pairs, n * (n - 1) // 2) if n > 1 else 0
    if pairs == 0:
        mean_pairwise_visible = 0.0
        mean_pairwise_l2 = 0.0
        link_rank_mean = 1.0
    else:
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
        # Linkability rank: for each row i, rank V[i] among all visible
        # rows by similarity to X[i]; lower mean rank → better linkage.
        n_subset = min(64, n)
        idx_subset = torch.arange(n_subset)
        sim_matrix = p_norm[idx_subset] @ v_norm.T     # [n_subset, N]
        ranks = sim_matrix.argsort(dim=-1, descending=True)
        link_rank = (ranks == idx_subset.unsqueeze(-1)).int().argmax(dim=-1)
        link_rank_mean = float(link_rank.to(torch.float64).mean().item())
    return {
        "visible_vs_plain_cosine": visible_vs_plain_cosine,
        "mean_pairwise_cosine_visible": mean_pairwise_visible,
        "mean_pairwise_l2_visible": mean_pairwise_l2,
        "mean_linkability_rank": link_rank_mean,
    }


# ---------------------------------------------------------------------------
# Trace acquisition with the fixed-permutation debug baseline
# ---------------------------------------------------------------------------


def _maybe_fixed_seed_traces(
    base_config: RealActivationAttackConfig,
    bundle: str,
) -> dict[str, Any]:
    """Run trace collection with optional fixed-permutation seeding."""
    if bundle == "fixed_permutation_debug":
        # Pin masks/permutations by re-seeding before each wrapper call.
        # We achieve this by setting the trace collector seed and then
        # wrapping the wrapper.forward via monkey-patch is undesirable;
        # instead we run a small inline version that re-seeds globally
        # before each call. To keep the implementation simple we run a
        # patched collector by patching torch.manual_seed via
        # contextlib.
        return _collect_fixed_permutation_traces(base_config)
    return collect_real_activation_traces(
        RealActivationTraceConfig(
            model_id=base_config.model_id,
            attempt_real_model_load=base_config.attempt_real_model_load,
            allow_synthetic_fallback=base_config.allow_synthetic_fallback,
            local_files_only=base_config.local_files_only,
            output_dir=base_config.output_dir,
            seed=base_config.seed,
            num_samples=base_config.num_samples,
            batch_size=base_config.batch_size,
            seq_len=base_config.seq_len,
            synthetic_hidden_size=base_config.synthetic_hidden_size,
            synthetic_intermediate_size=base_config.synthetic_intermediate_size,
            synthetic_num_attention_heads=base_config.synthetic_num_attention_heads,
            synthetic_num_key_value_heads=base_config.synthetic_num_key_value_heads,
            synthetic_head_dim=base_config.synthetic_head_dim,
            use_pad=base_config.use_pad,
            nonlinear_mode=base_config.nonlinear_mode,
            mitigation_bundle=bundle,
            target_tensors=base_config.target_tensors,
            dtype=base_config.dtype,
            device=base_config.device,
        )
    )


def _collect_fixed_permutation_traces(
    base_config: RealActivationAttackConfig,
) -> dict[str, Any]:
    """Trace collection with a single pinned RNG seed across all sessions.

    Re-seeds the global torch RNG to the SAME value before each wrapper
    call so the wrapper samples identical ``N_res / N_in_island / perm /
    N_out_island / N_Q / N_K / N_V`` every call. Caller input X is drawn
    from a separate generator so the input distribution still varies.
    """
    from pllo.experiments.real_activation_trace import (
        _flatten_pair, _resolve_block, _tensor_statistics,
    )
    from pllo.hf_wrappers.modern_decoder_block_wrapper import (
        ObfuscatedModernDecoderBlockWrapper,
    )
    from pllo.model_zoo.modern_decoder_spec import spec_to_dict
    from pllo.ops.mitigation_bundles import normalize_mitigation_bundle

    cfg = RealActivationTraceConfig(
        model_id=base_config.model_id,
        attempt_real_model_load=base_config.attempt_real_model_load,
        allow_synthetic_fallback=base_config.allow_synthetic_fallback,
        local_files_only=base_config.local_files_only,
        seed=base_config.seed,
        num_samples=base_config.num_samples,
        batch_size=base_config.batch_size,
        seq_len=base_config.seq_len,
        synthetic_hidden_size=base_config.synthetic_hidden_size,
        synthetic_intermediate_size=base_config.synthetic_intermediate_size,
        synthetic_num_attention_heads=base_config.synthetic_num_attention_heads,
        synthetic_num_key_value_heads=base_config.synthetic_num_key_value_heads,
        synthetic_head_dim=base_config.synthetic_head_dim,
        use_pad=base_config.use_pad,
        nonlinear_mode=base_config.nonlinear_mode,
        mitigation_bundle="fresh_perm_plus_sandwich_plus_pad",  # mask family is the same
        target_tensors=base_config.target_tensors,
        dtype=base_config.dtype,
        device=base_config.device,
    )
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    device = torch.device(cfg.device)
    spec, weights, load_record, source = _resolve_block(cfg, dtype, device)
    bundle = "fresh_perm_plus_sandwich_plus_pad"   # internal: same math, label below
    wrapper = ObfuscatedModernDecoderBlockWrapper(
        weights, dtype=dtype, device=device,
        use_pad=cfg.use_pad, nonlinear_mode=cfg.nonlinear_mode,
        mitigation_bundle=bundle,
    )

    tokens_per_session = cfg.batch_size * cfg.seq_len
    num_sessions = max(1, (cfg.num_samples + tokens_per_session - 1) // tokens_per_session)
    H = spec.hidden_size
    input_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)

    accum: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"plain": [], "visible": []} for name in cfg.target_tensors
    }
    final_allclose: list[bool] = []
    for _ in range(num_sessions):
        x = torch.randn(
            cfg.batch_size, cfg.seq_len, H,
            generator=input_gen, dtype=torch.float32,
        ).to(dtype=dtype, device=device)
        # *** Pin the wrapper's internal RNG to a fixed seed ***
        torch.manual_seed(base_config.fixed_permutation_seed)
        _, report, traces = wrapper.forward_with_traces(x)
        final_allclose.append(bool(report["allclose"]))
        for name in cfg.target_tensors:
            pk, vk = f"{name}_plain", f"{name}_visible"
            if pk not in traces or vk not in traces:
                continue
            pair = _flatten_pair(name, traces[pk], traces[vk])
            accum[name]["plain"].append(pair["plain"])
            accum[name]["visible"].append(pair["visible"])

    stitched: dict[str, dict[str, torch.Tensor]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for name in cfg.target_tensors:
        if not accum[name]["plain"]:
            continue
        plain_t = torch.cat(accum[name]["plain"], dim=0)
        visible_t = torch.cat(accum[name]["visible"], dim=0)
        n_keep = min(plain_t.shape[0], max(cfg.num_samples, 16))
        plain_t = plain_t[:n_keep]
        visible_t = visible_t[:n_keep]
        stitched[name] = {"plain": plain_t, "visible": visible_t}
        summary[name] = {
            "tensor_name": name,
            "num_samples": int(plain_t.shape[0]),
            "feature_dim": int(plain_t.shape[-1]),
            "plain_shape": list(plain_t.shape),
            "visible_shape": list(visible_t.shape),
            "source": source,
            "mitigation_bundle": "fixed_permutation_debug",
            "use_pad": bool(cfg.use_pad),
            "plain_statistics": _tensor_statistics(plain_t),
            "visible_statistics": _tensor_statistics(visible_t),
        }
    return {
        "config": asdict(cfg),
        "model_loading": load_record,
        "source": source,
        "block_spec": spec_to_dict(spec),
        "traces": stitched,
        "trace_summary": summary,
        "metadata": {
            "mitigation_bundle": "fixed_permutation_debug",
            "use_pad": bool(cfg.use_pad),
            "nonlinear_mode": cfg.nonlinear_mode,
            "num_sessions": int(num_sessions),
            "tokens_per_session": int(tokens_per_session),
            "all_sessions_allclose": bool(all(final_allclose)),
            "note": (
                "RNG pinned to fixed seed before every wrapper call; masks /"
                " permutations are identical across sessions. Debug baseline"
                " ONLY — never recommended for deployment."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Bundle comparison + per-tensor decision
# ---------------------------------------------------------------------------


def _per_tensor_attacker_run(
    tensor_name: str,
    plain: torch.Tensor,
    visible: torch.Tensor,
    config: RealActivationAttackConfig,
    seed_offset: int,
) -> dict[str, Any]:
    feature_dim = int(plain.shape[-1])
    linear_metrics = _run_linear_inverter_real(plain, visible, config)
    mlp_metrics = _run_mlp_inverter_real(
        plain, visible, config, seed=config.seed + seed_offset
    )
    if tensor_name in PERMUTATION_TARGET_TENSORS:
        perm_metrics = _run_permutation_recovery_real(plain, visible, config)
        perm_top1 = perm_metrics["best_top1"]
    else:
        perm_metrics = None
        perm_top1 = None
    linkability_metrics = _run_linkability_real(plain, visible)
    risk_level, recommendation = _classify_risk(
        linear_rel_l2=linear_metrics["relative_l2_error"],
        mlp_rel_l2=mlp_metrics["relative_l2_error"],
        perm_soft_top1=perm_top1,
        random_chance=1.0 / max(1, feature_dim),
        linkability_cosine=linkability_metrics["visible_vs_plain_cosine"],
    )
    return {
        "tensor_name": tensor_name,
        "feature_dim": feature_dim,
        "linear_inverter": linear_metrics,
        "mlp_inverter": mlp_metrics,
        "permutation_recovery": perm_metrics,
        "linkability": linkability_metrics,
        "risk_level": risk_level,
        "default_on_recommendation": recommendation,
        "mlp_minus_linear_relative_l2_error": (
            mlp_metrics["relative_l2_error"]
            - linear_metrics["relative_l2_error"]
        ),
    }


def _compare_bundles(
    per_tensor: dict[str, dict[str, dict[str, Any]]],
    target_tensors: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Build the bundle_comparison table.

    Each row reports ``linear_rel_l2_delta``, ``mlp_rel_l2_delta``,
    ``permutation_top1_delta``, ``linkability_cosine_delta`` and the
    per-bundle risk levels for one tensor. Deltas are
    ``full_bundle − fresh_only``: positive ⇒ the full bundle makes
    recovery *harder* (safer).
    """
    rows: list[dict[str, Any]] = []
    for tensor in target_tensors:
        fresh = per_tensor.get("fresh_perm_only", {}).get(tensor)
        full = per_tensor.get("fresh_perm_plus_sandwich_plus_pad", {}).get(tensor)
        fixed = per_tensor.get("fixed_permutation_debug", {}).get(tensor)
        if fresh is None or full is None:
            continue
        row: dict[str, Any] = {
            "tensor_name": tensor,
            "feature_dim": fresh["feature_dim"],
            "linear_rel_l2_fresh_only": fresh["linear_inverter"]["relative_l2_error"],
            "linear_rel_l2_full_bundle": full["linear_inverter"]["relative_l2_error"],
            "linear_rel_l2_delta": (
                full["linear_inverter"]["relative_l2_error"]
                - fresh["linear_inverter"]["relative_l2_error"]
            ),
            "mlp_rel_l2_fresh_only": fresh["mlp_inverter"]["relative_l2_error"],
            "mlp_rel_l2_full_bundle": full["mlp_inverter"]["relative_l2_error"],
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
            "recommendation_full_bundle": full["default_on_recommendation"],
            "math_equivalence_note": (
                "fresh_perm_only and fresh_perm_plus_sandwich_plus_pad"
                " both rely on the same per-call fresh N_in / perm / N_out"
                " sampling in run_swiglu_mlp_island; the two bundles"
                " therefore produce identical numerical traces under this"
                " wrapper. The bundle label distinguishes the security"
                " posture (default-on candidate vs. not), not the"
                " numerical visibility surface."
            ),
        }
        if (
            fresh["permutation_recovery"] is not None
            and full["permutation_recovery"] is not None
        ):
            row["permutation_top1_fresh_only"] = fresh["permutation_recovery"][
                "best_top1"
            ]
            row["permutation_top1_full_bundle"] = full["permutation_recovery"][
                "best_top1"
            ]
            row["permutation_top1_delta"] = (
                full["permutation_recovery"]["best_top1"]
                - fresh["permutation_recovery"]["best_top1"]
            )
            row["random_chance_top1"] = fresh["permutation_recovery"][
                "random_chance_top1"
            ]
        if fixed is not None:
            row["linear_rel_l2_fixed_debug"] = fixed["linear_inverter"][
                "relative_l2_error"
            ]
            row["risk_level_fixed_debug"] = fixed["risk_level"]
            if fixed["permutation_recovery"] is not None:
                row["permutation_top1_fixed_debug"] = fixed[
                    "permutation_recovery"
                ]["best_top1"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


_LIMITATIONS = [
    "These are real-activation adaptive proxy attacks, not formal security proofs.",
    "If synthetic fallback is used, results are not real Qwen/TinyLlama activation traces.",
    "Random hidden-state block input is not the same as real token distribution"
    " unless tokenizer/embedding path is added.",
    "No black-box query attack is implemented.",
    "No side-channel attack is implemented.",
    "No real TEE isolation is evaluated.",
    "Dense sandwiching reduces tested recovery but does not imply semantic security.",
    "This stage does not implement full model-level generation or KV cache runtime.",
    "fresh_perm_only and fresh_perm_plus_sandwich_plus_pad share the same per-call"
    " mask sampling under the Stage 6.4b wrapper; the two bundles produce"
    " numerically identical traces. fixed_permutation_debug is a debug baseline"
    " only — never recommended for deployment.",
]


def run_real_activation_attacks(
    config: RealActivationAttackConfig,
) -> dict[str, Any]:
    """Drive trace collection, run all attackers, build the comparison table."""
    bundles = tuple(config.mitigation_bundles)
    per_tensor: dict[str, dict[str, dict[str, Any]]] = {b: {} for b in bundles}
    trace_summaries: dict[str, Any] = {}
    model_loading: dict[str, Any] | None = None
    source: str | None = None
    block_spec: dict[str, Any] | None = None
    for bundle_idx, bundle in enumerate(bundles):
        trace_pkg = _maybe_fixed_seed_traces(config, bundle)
        trace_summaries[bundle] = trace_pkg["trace_summary"]
        if model_loading is None:
            model_loading = trace_pkg["model_loading"]
            source = trace_pkg["source"]
            block_spec = trace_pkg["block_spec"]
        for tensor in config.target_tensors:
            pair = trace_pkg["traces"].get(tensor)
            if pair is None:
                continue
            per_tensor[bundle][tensor] = _per_tensor_attacker_run(
                tensor, pair["plain"], pair["visible"], config,
                seed_offset=37 * (bundle_idx + 1),
            )

    comparison = _compare_bundles(per_tensor, config.target_tensors)
    # High-level summary stats (per bundle).
    attacker_summary = _summarise(per_tensor, config.target_tensors)
    full = attacker_summary.get("fresh_perm_plus_sandwich_plus_pad", {})
    fresh = attacker_summary.get("fresh_perm_only", {})

    if full.get("max_risk_level") in {"low"}:
        recommendation = "acceptable_with_mitigation_under_real_activation_proxy"
    elif full.get("max_risk_level") == "medium":
        recommendation = "needs_more_evaluation_under_real_activation_proxy"
    else:
        recommendation = "unsafe_default_on_under_real_activation_proxy"

    return {
        "config": asdict(config),
        "model_loading": model_loading or {},
        "source": source,
        "block_spec": block_spec or {},
        "trace_summary": trace_summaries,
        "target_tensor_results": per_tensor,
        "bundle_comparison": comparison,
        "attacker_summary": attacker_summary,
        "recommendation": {
            "default_on_recommendation_full_bundle": recommendation,
            "default_on_recommendation_fresh_only": (
                "unsafe_default_on_under_real_activation_proxy"
                if fresh.get("max_risk_level") == "high"
                else "needs_more_evaluation_under_real_activation_proxy"
                if fresh.get("max_risk_level") == "medium"
                else "acceptable_with_mitigation_under_real_activation_proxy"
            ),
            "security_profile_detail_with_real_activation": (
                "real-activation-adaptive-proxy-evaluated, not formal"
            ),
        },
        "limitations": list(_LIMITATIONS),
    }


def _summarise(
    per_tensor: dict[str, dict[str, dict[str, Any]]],
    target_tensors: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Per-bundle headline statistics — used by the markdown emitter."""
    order = {"low": 0, "medium": 1, "high": 2}
    inv_order = {0: "low", 1: "medium", 2: "high"}
    out: dict[str, dict[str, Any]] = {}
    for bundle, by_tensor in per_tensor.items():
        if not by_tensor:
            continue
        risks = [v["risk_level"] for v in by_tensor.values()]
        max_risk = inv_order[max(order[r] for r in risks)]
        out[bundle] = {
            "bundle": bundle,
            "tensors_covered": list(by_tensor.keys()),
            "max_risk_level": max_risk,
            "risk_counts": {
                level: sum(1 for r in risks if r == level)
                for level in ("low", "medium", "high")
            },
            "mean_linear_rel_l2": float(sum(
                v["linear_inverter"]["relative_l2_error"]
                for v in by_tensor.values()
            ) / max(1, len(by_tensor))),
            "mean_mlp_rel_l2": float(sum(
                v["mlp_inverter"]["relative_l2_error"]
                for v in by_tensor.values()
            ) / max(1, len(by_tensor))),
            "mean_linkability_cosine": float(sum(
                v["linkability"]["visible_vs_plain_cosine"]
                for v in by_tensor.values()
            ) / max(1, len(by_tensor))),
        }
    return out


__all__ = [
    "EXTENDED_BUNDLES",
    "PERMUTATION_TARGET_TENSORS",
    "RealActivationAttackConfig",
    "run_real_activation_attacks",
]
