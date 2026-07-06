"""NN embedding inversion (P0, structural) — cosine/L2/dot + EDNN differential.

Two modes:
  * standard nearest-neighbour of observed embedding-space vectors vs an
    embedding table (cosine/L2/dot). Recovers tokens by top-k.
  * ``differential`` (EDNN, Lin et al. EMNLP'24): match the within-vector
    element-wise differential ``v − lshift(v)``. This is invariant to
    glide-reflection obfuscation and defeats it; on a general mask it degrades.

Threat-model guard: requires the embedding table. Under
``closed_model_no_weight_access`` (no table, no surrogate) it returns ``blocked``.
"""

from __future__ import annotations

import time

import torch

from .metrics import (mean_token_accuracy, nearest_neighbor_scores,
                      sequence_exact_match, token_recovery_topk)
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics, make_attack_config,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "EDNN (Lin et al. EMNLP'24) / embedding-inversion baseline"


def _differential(x: torch.Tensor) -> torch.Tensor:
    """v - lshift(v): adjacent-coordinate differences with wrap-around (EDNN)."""
    return x - torch.roll(x, shifts=-1, dims=-1)


def run_nn_embedding_inversion(
    inputs: AttackInputs, *, target_method: str,
    threat_model: str = "public_model", metric: str = "cosine",
    use_protected: bool = True, surrogate_table: torch.Tensor | None = None,
    model_family: str = "toy", model_name_or_path: str | None = None,
    save_topk_path=None,
) -> AttackResult:
    impl = "full"
    base = dict(attack_id="nn_embedding_inversion",
                attack_name="NN embedding inversion" + (" (EDNN differential)" if metric == "differential" else ""),
                attack_family="structural", target_method=target_method,
                threat_model=threat_model, paper_source=_PAPER, implementation_level=impl,
                model_family=model_family, model_name_or_path=model_name_or_path)
    table = inputs.embedding_table if inputs.embedding_table is not None else surrogate_table
    if table is None:
        return blocked_result(**base, reason="no embedding table / surrogate available "
                              "under this threat model (closed_model_no_weight_access)")
    if inputs.token_ids is None:
        return blocked_result(**base, reason="no ground-truth token ids to score recovery")

    if use_protected and inputs.protected_embeddings is not None:
        observed = inputs.protected_embeddings
    elif inputs.plaintext_embeddings is not None:
        observed = inputs.plaintext_embeddings
    else:
        observed = inputs.observed_intermediate
    if observed is None or observed.shape[-1] != table.shape[-1]:
        return blocked_result(**base, reason="no embedding-space vectors to invert")
    observed = observed.reshape(-1, table.shape[-1])
    true_ids = inputs.token_ids.reshape(-1)

    t0 = time.perf_counter()
    if metric == "differential":
        scores = -torch.cdist(_differential(observed).to(torch.float32),
                              _differential(table).to(torch.float32))
    else:
        scores = nearest_neighbor_scores(observed, table, metric=metric)
    ms = (time.perf_counter() - t0) * 1000.0
    rec = token_recovery_topk(scores, true_ids)
    pred = scores.argmax(-1)

    if save_topk_path is not None:
        import json
        topk = scores.topk(min(10, table.shape[0]), dim=-1).indices.tolist()
        from pathlib import Path
        Path(save_topk_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_topk_path).write_text(json.dumps({"true": true_ids.tolist(), "topk": topk}))

    metrics = empty_metrics(); metrics.update(rec)
    metrics["attack_success_rate"] = rec["token_recovery_top1"]
    metrics["one_minus_attack_success_rate"] = 1.0 - rec["token_recovery_top1"]
    metrics["mean_token_accuracy"] = mean_token_accuracy(pred, true_ids)
    metrics["sequence_exact_match"] = sequence_exact_match(pred, true_ids)
    N, H = observed.shape
    return AttackResult(
        **base, status="measured",
        attacker_knowledge=make_attacker_knowledge(
            has_embedding_table=inputs.embedding_table is not None,
            has_model_weights=threat_model == "weight_leakage_worst_case"),
        input_info=make_input_info(hidden_size=H, vocab_size=table.shape[0], num_samples=N,
                                   dtype=str(observed.dtype)),
        attack_config=make_attack_config(loss_type=metric),
        metrics=metrics, runtime=make_runtime(wall_time_ms=ms),
        notes=f"metric={metric}, {'protected' if use_protected else 'plaintext'} vectors"
              + ("; EDNN differential invariant (breaks glide-reflection; degrades on general mask)"
                 if metric == "differential" else ""))


__all__ = ["run_nn_embedding_inversion"]
