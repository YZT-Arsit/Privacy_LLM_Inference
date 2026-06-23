"""Stage 8.2 -- top-conference evaluation metrics (compact, scalar-only).

Device-independent metric functions for:

* Group C -- output-boundary ablation (plaintext_logits / masked_logits /
  hidden_to_tee) as a correctness + analytic cost model;
* Group D -- leakage/attack accounting:
    - D1 token recovery from (plaintext vs masked) embeddings,
    - D2 masked-logit argmax/top-k alignment (visible vs recovered),
    - D3 hidden-state structural leakage (what orthogonal masks do NOT hide).

All functions return plain Python scalars (no tensors are stored or dumped).
They run identically on CPU (tiny models) and CUDA (real ModelScope
checkpoints). No security beyond explicit leakage accounting is claimed:
orthogonal masks preserve norms / pairwise geometry, and sequence length and
attention scores remain visible.
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.ops.causal_lm_boundaries import (
    VocabLogitMask,
    apply_vocab_logit_mask,
    recover_vocab_logits,
)

__all__ = [
    "attack_masked_logits",
    "attack_token_recovery",
    "hidden_structure_leakage",
    "output_boundary_ablation",
    "run_topconf_evals",
]

_DTYPE_BYTES = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4,
                torch.float64: 8}


def _bytes(dtype: torch.dtype) -> int:
    return _DTYPE_BYTES.get(dtype, 4)


# ---------------------------------------------------------------------------
# Group C: output-boundary ablation (correctness + analytic cost model)
# ---------------------------------------------------------------------------


def output_boundary_ablation(
    *, hidden_size: int, vocab_size: int, batch_size: int, decode_steps: int,
    transfer_dtype: torch.dtype, masked_token_match_rate: float,
    masked_recovered_logits_err: float,
    masked_latency_ms: float | None = None,
    extracted_latency_ms: float | None = None,
) -> list[dict[str, Any]]:
    """Three output-boundary modes scored on the SAME decoder stack.

    Cost model counts the *deployment* logit positions = ``1`` (last prefill) +
    ``decode_steps`` per batch element (a real system computes logits only for
    the positions it samples), not the full prefill width."""
    nb = _bytes(transfer_dtype)
    positions = batch_size * (1 + decode_steps)
    # LM-head GEMM FLOPs for one logit row: 2 * hidden * vocab.
    lm_head_flops = positions * 2 * hidden_size * vocab_size
    recovery_flops = positions * vocab_size * 2          # perm + scale
    sampling_flops = positions * vocab_size              # argmax scan
    logits_bytes = positions * vocab_size * nb
    hidden_bytes = positions * hidden_size * nb

    rows = [
        {
            "boundary_mode": "plaintext_logits",
            "gpu_visible": "plaintext_logits",
            "tee_compute_flops": int(sampling_flops),
            "transfer_bytes": int(logits_bytes),
            "token_match_rate": 1.0,
            "recovered_logits_err": 0.0,
            "latency_ms": extracted_latency_ms,
            "latency_kind": "measured" if extracted_latency_ms is not None
            else "n/a",
            "security_caveat": "UNSAFE baseline: GPU/extractor sees plaintext "
            "logits (direct token-distribution exposure).",
        },
        {
            "boundary_mode": "masked_logits",
            "gpu_visible": "masked_logits (vocab permutation + positive scale)",
            "tee_compute_flops": int(recovery_flops + sampling_flops),
            "transfer_bytes": int(logits_bytes),
            "token_match_rate": round(float(masked_token_match_rate), 6),
            "recovered_logits_err": float(masked_recovered_logits_err),
            "latency_ms": masked_latency_ms,
            "latency_kind": "measured" if masked_latency_ms is not None
            else "n/a",
            "security_caveat": "vocab permutation + positive diagonal scaling "
            "hides index/logit alignment but is WEAKER than dense vocab "
            "masking; no semantic security; sequence length visible.",
        },
        {
            "boundary_mode": "hidden_to_tee",
            "gpu_visible": "masked_hidden_state",
            "tee_compute_flops": int(lm_head_flops + recovery_flops
                                     + sampling_flops),
            "transfer_bytes": int(hidden_bytes),
            "token_match_rate": 1.0,
            "recovered_logits_err": 0.0,
            "latency_ms": None,
            "latency_kind": "analytical_cost_model",
            "security_caveat": "GPU never sees logits; lowest logit leakage but "
            "TEE runs the full LM head (high TEE compute). Cost-model only "
            "(not a runtime measurement).",
        },
    ]
    for r in rows:
        r["transfer_bytes_per_hidden_vs_vocab"] = round(
            hidden_size / max(vocab_size, 1), 6)
    return rows


# ---------------------------------------------------------------------------
# Group D1: token recovery from embeddings
# ---------------------------------------------------------------------------


def _topk_recovery(query: torch.Tensor, table: torch.Tensor,
                   true_ids: torch.Tensor, k: int = 5) -> tuple[float, float]:
    """Cosine nearest-neighbour recovery of ``true_ids`` from ``query`` rows
    against ``table`` rows. Returns ``(top1_rate, topk_rate)``."""
    q = torch.nn.functional.normalize(query.float(), dim=-1)
    t = torch.nn.functional.normalize(table.float(), dim=-1)
    sims = q @ t.t()                                   # [Q, vocab]
    kk = min(k, sims.shape[-1])
    topk = sims.topk(kk, dim=-1).indices              # [Q, k]
    true_ids = true_ids.to(topk.device).view(-1, 1)
    top1 = (topk[:, :1] == true_ids).any(dim=-1).float().mean().item()
    topk_rate = (topk == true_ids).any(dim=-1).float().mean().item()
    return float(top1), float(topk_rate)


def attack_token_recovery(
    embed_table: torch.Tensor, query_token_ids: torch.Tensor,
    n_res: torch.Tensor, wrong_n_res: torch.Tensor | None = None,
    *, max_queries: int = 256, k: int = 5,
) -> dict[str, Any]:
    """D1: how well an attacker recovers tokens from embeddings.

    ``plaintext``: NN of the true embedding to the table (trivially perfect).
    ``masked``: NN of ``x @ n_res`` (what the GPU sees) to the ORIGINAL table.
    ``masked_wrong_mask``: NN after de-masking with a WRONG mask.
    Expected: plaintext ~1.0; masked / wrong-mask near the random baseline."""
    vocab = int(embed_table.shape[0])
    ids = query_token_ids.reshape(-1)
    if ids.numel() > max_queries:
        ids = ids[:max_queries]
    x = embed_table[ids]                               # [Q, H] plaintext
    x_tilde = x @ n_res                                # [Q, H] masked (GPU view)

    p1, p5 = _topk_recovery(x, embed_table, ids, k)
    m1, m5 = _topk_recovery(x_tilde, embed_table, ids, k)
    if wrong_n_res is not None:
        x_wrong = x_tilde @ wrong_n_res.transpose(-2, -1)
        w1, w5 = _topk_recovery(x_wrong, embed_table, ids, k)
    else:
        w1 = w5 = None

    return {
        "vocab_size": vocab,
        "num_queries": int(ids.numel()),
        "random_baseline_top1": round(1.0 / vocab, 8),
        "random_baseline_top5": round(min(k, vocab) / vocab, 8),
        "plaintext_top1_token_recovery": round(p1, 6),
        "plaintext_top5_token_recovery": round(p5, 6),
        "masked_top1_token_recovery": round(m1, 6),
        "masked_top5_token_recovery": round(m5, 6),
        "masked_wrong_mask_top1_token_recovery":
            round(w1, 6) if w1 is not None else None,
        "masked_wrong_mask_top5_token_recovery":
            round(w5, 6) if w5 is not None else None,
    }


# ---------------------------------------------------------------------------
# Group D2: masked-logit leakage / argmax alignment
# ---------------------------------------------------------------------------


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D vectors."""
    ra = a.float().argsort().argsort().float()
    rb = b.float().argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).item()
    if denom == 0:
        return 0.0
    return float((ra @ rb).item() / denom)


def attack_masked_logits(
    plain_logits_last: torch.Tensor, vocab_mask: VocabLogitMask,
    *, k: int = 5, rank_corr: bool = True,
) -> dict[str, Any]:
    """D2: does the GPU-visible masked logit vector leak the true token?

    ``plain_logits_last`` is ``[B, vocab]`` (last-position plain logits). The
    GPU sees ``L_tilde = L P D``; an attacker reading its argmax *as an original
    vocab index* should NOT recover the true token. Correct TEE recovery must."""
    L = plain_logits_last.float()
    L_tilde = apply_vocab_logit_mask(L, vocab_mask).float()   # GPU view
    L_rec = recover_vocab_logits(L_tilde, vocab_mask).float()  # TEE recovery

    plain_arg = L.argmax(dim=-1)
    visible_arg = L_tilde.argmax(dim=-1)        # interpreted as original index
    rec_arg = L_rec.argmax(dim=-1)

    kk = min(k, L.shape[-1])
    plain_top = L.topk(kk, dim=-1).indices
    visible_top = L_tilde.topk(kk, dim=-1).indices
    overlap = [
        len(set(plain_top[i].tolist()) & set(visible_top[i].tolist())) / kk
        for i in range(L.shape[0])
    ]

    rc = None
    if rank_corr:
        rc = sum(_spearman(L[i], L_tilde[i]) for i in range(L.shape[0])) \
            / L.shape[0]

    return {
        "batch": int(L.shape[0]),
        "vocab_size": int(L.shape[-1]),
        "random_baseline_argmax_match": round(1.0 / int(L.shape[-1]), 8),
        "gpu_visible_argmax_matches_plaintext": round(
            (visible_arg == plain_arg).float().mean().item(), 6),
        "top5_overlap_plain_vs_masked_visible": round(
            sum(overlap) / len(overlap), 6),
        "rank_correlation_plain_vs_masked_visible":
            round(rc, 6) if rc is not None else None,
        "recovered_argmax_matches_plaintext": round(
            (rec_arg == plain_arg).float().mean().item(), 6),
    }


# ---------------------------------------------------------------------------
# Group D3: hidden-state structural leakage (orthogonal masks preserve geometry)
# ---------------------------------------------------------------------------


def _pairwise_corr(Xp: torch.Tensor, Xt: torch.Tensor,
                   metric: str) -> float:
    if metric == "dist":
        Dp = torch.cdist(Xp, Xp)
        Dt = torch.cdist(Xt, Xt)
    else:  # cosine
        Pp = torch.nn.functional.normalize(Xp, dim=-1)
        Pt = torch.nn.functional.normalize(Xt, dim=-1)
        Dp = Pp @ Pp.t()
        Dt = Pt @ Pt.t()
    n = Dp.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    a = Dp[iu[0], iu[1]]
    b = Dt[iu[0], iu[1]]
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom else 1.0


def hidden_structure_leakage(
    hidden_plain: torch.Tensor, n_res: torch.Tensor,
    *, max_points: int = 256,
) -> dict[str, Any]:
    """D3: structural leakage of an orthogonal residual mask.

    Reports norm preservation + pairwise distance / cosine correlation +
    nearest-neighbour identity preservation between ``X`` and ``X @ n_res``.
    For an orthogonal mask these are ~1.0 -- an honest statement that masking
    hides coordinates but NOT relative geometry."""
    X = hidden_plain.reshape(-1, hidden_plain.shape[-1]).float()
    if X.shape[0] > max_points:
        X = X[:max_points]
    Xt = X @ n_res.float()

    norm_ratio = (Xt.norm(dim=-1) / X.norm(dim=-1).clamp_min(1e-12))
    dist_corr = _pairwise_corr(X, Xt, "dist")
    cos_corr = _pairwise_corr(X, Xt, "cosine")

    # nearest-neighbour identity preservation (exclude self).
    Dp = torch.cdist(X, X)
    Dt = torch.cdist(Xt, Xt)
    eye = torch.eye(X.shape[0], device=X.device).bool()
    Dp = Dp.masked_fill(eye, float("inf"))
    Dt = Dt.masked_fill(eye, float("inf"))
    nn_preserved = (Dp.argmin(dim=-1) == Dt.argmin(dim=-1)).float().mean().item()

    return {
        "num_points": int(X.shape[0]),
        "norm_preservation_ratio_mean": round(norm_ratio.mean().item(), 6),
        "norm_preservation_ratio_max_abs_dev": round(
            (norm_ratio - 1.0).abs().max().item(), 8),
        "pairwise_distance_correlation": round(dist_corr, 6),
        "pairwise_cosine_correlation": round(cos_corr, 6),
        "nearest_neighbor_identity_preservation": round(nn_preserved, 6),
        "leakage_note": "orthogonal residual masks preserve norms and relative "
        "geometry (distances/cosines/NN); they hide coordinate identities, not "
        "structure. This is leakage accounting, not a security guarantee.",
    }


# ---------------------------------------------------------------------------
# Driver: load a real ModelScope checkpoint ONCE and run every C/D eval.
# ---------------------------------------------------------------------------


def run_topconf_evals(config: Any) -> dict[str, Any]:
    """Run Groups C + D against one checkpoint (shares a single model load).

    ``config`` is a ``ModelScopeRealCheckpointProbeConfig``. Returns a compact
    dict with ``boundary_ablation`` / ``attack_token_recovery`` /
    ``attack_masked_logits`` / ``hidden_structure_leakage`` + metadata. Skips
    cleanly (status != "ok") if the checkpoint/deps are unavailable."""
    import time

    from pllo.experiments.modelscope_real_checkpoint_probe import (
        REQUIRED_STATEMENT, build_probe_inputs, load_modelscope_checkpoint,
        resolve_dtype)
    from pllo.hf_wrappers.hf_causal_lm_skeleton import (
        HFCausalLMSkeletonConfig, extract_hf_causal_lm_skeleton_weights,
        generate_hf_causal_lm_masks, hf_causal_lm_masked_greedy_decode,
        hf_causal_lm_masked_only_decode, hf_causal_lm_plain_prefill,
        make_residual_mask)

    device = config.device
    model_dtype = resolve_dtype(config.dtype, device)
    folding_dtype = resolve_dtype(config.folding_dtype, device)
    runtime_dtype = resolve_dtype(config.folded_weight_runtime_dtype, device)
    recovery_dtype = resolve_dtype(config.recovery_dtype, device)

    base = {
        "stage": "8.2_topconf_evals", "config": _config_dict(config),
        "required_statement": REQUIRED_STATEMENT,
    }
    loaded = load_modelscope_checkpoint(config.model_id, config.cache_dir,
                                        model_dtype, device)
    if loaded["status"] != "ok":
        return {**base, "status": loaded["status"],
                "reason": loaded.get("reason")}

    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    model_config = model.config
    total_layers = int(getattr(model_config, "num_hidden_layers", 0))
    max_layers = (None if str(config.max_layers) == "all"
                  else int(config.max_layers))

    folded_rt = runtime_dtype if runtime_dtype != folding_dtype else None
    skel_cfg = HFCausalLMSkeletonConfig(
        model_family=str(getattr(model_config, "model_type", "qwen2")),
        prefill_seq_len=config.prefill_seq_len, decode_steps=config.decode_steps,
        max_layers=max_layers, dtype=folding_dtype, device=device,
        seed=config.seed, mask_mode=config.mask_mode,
        residual_mask_strategy=config.residual_mask_strategy,
        mask_block_size=config.block_size,
        allow_dense_large_mask=config.allow_dense_large_mask,
        folded_runtime_dtype=folded_rt, recovery_dtype=recovery_dtype)

    weights, layer_configs, extract_meta = \
        extract_hf_causal_lm_skeleton_weights(
            model, model_config, max_layers=max_layers, dtype=folding_dtype,
            device=device)
    masks = generate_hf_causal_lm_masks(weights, layer_configs, skel_cfg)
    vocab = extract_meta["vocab_size"]
    hidden = extract_meta["hidden_size"]
    input_ids, input_meta = build_probe_inputs(tokenizer, vocab, config, device)

    with torch.no_grad():
        # Correctness + plain logits.
        dec = hf_causal_lm_masked_greedy_decode(
            input_ids, weights, layer_configs, masks, skel_cfg)
        pre = dec["prefill_metrics"]
        plain = hf_causal_lm_plain_prefill(input_ids, weights, layer_configs,
                                           masks, skel_cfg)
        # Masked-only latency for the boundary ablation.
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        hf_causal_lm_masked_only_decode(input_ids, weights, layer_configs,
                                        masks, skel_cfg)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        masked_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # Wrong residual mask for the D1 wrong-mask attack.
        g = torch.Generator(device=masks.residual_masks[0].device)
        g.manual_seed(config.seed + 13577)
        wrong_n, _ = make_residual_mask(
            hidden, config.mask_mode, masks.residual_masks[0].dtype,
            masks.residual_masks[0].device, g, block_size=config.block_size,
            allow_dense_large=config.allow_dense_large_mask)

        boundary = output_boundary_ablation(
            hidden_size=hidden, vocab_size=vocab, batch_size=input_ids.shape[0],
            decode_steps=config.decode_steps, transfer_dtype=model_dtype,
            masked_token_match_rate=dec["token_match_rate"],
            masked_recovered_logits_err=pre["recovered_logits_max_abs_error"],
            masked_latency_ms=masked_ms)
        d1 = attack_token_recovery(
            weights.embed_tokens_weight, input_ids, masks.residual_masks[0],
            wrong_n)
        d2 = attack_masked_logits(
            plain["logits_plain"][:, -1, :], masks.vocab_mask)
        d3 = hidden_structure_leakage(
            plain["embeddings_plain"], masks.residual_masks[0])

    del model, weights
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        **base, "status": "ok", "model_id": config.model_id,
        "local_path": loaded["local_path"],
        "model_type": str(getattr(model_config, "model_type", "unknown")),
        "total_layers": total_layers,
        "max_layers_executed": extract_meta["num_layers_extracted"],
        "hidden_size": hidden, "vocab_size": vocab,
        "input": {k: input_meta[k] for k in (
            "input_source", "prompt_count", "input_ids_shape",
            "attention_mask_explicit", "tokenized_length_stats")
            if k in input_meta},
        "mask_mode": masks.metadata["mask_mode"],
        "residual_mask_strategy": masks.metadata["residual_mask_strategy"],
        "token_match_rate_vs_extracted": dec["token_match_rate"],
        "boundary_ablation": boundary,
        "attack_token_recovery": d1,
        "attack_masked_logits": d2,
        "hidden_structure_leakage": d3,
    }


def _config_dict(config: Any) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass
    return asdict(config) if is_dataclass(config) else dict(vars(config))
