"""BRE / BiSR attack (Chen et al., CCS'24) — forward best_effort, backward blocked.

BiSR = SIP (learned inverter) -> BRE (forward smashed-data matching + backward
gradient matching). We implement the **forward** component (Eq.7-8): continuous
embedding optimization with a COSINE loss against observed smashed data, then
nearest-token projection. This is a faithful reimplementation of the forward
half only — labelled best_effort, explicitly NOT full BiSR.

The **backward** gradient-matching half (Eq.9) needs the activation gradients
``grad(x̃_trk)`` transmitted during split *fine-tuning*; an inference-only harness
cannot produce them, so ``run_bre_backward_gradient_matching`` returns ``blocked``
with the exact missing requirement.
"""

from __future__ import annotations

import time

import torch

from .metrics import cosine_similarity, mean_token_accuracy, mse, nearest_neighbor_scores, token_recovery_topk
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics, make_attack_config,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "BRE/BiSR (Chen et al., ACM CCS'24, arXiv:2409.00960)"


def run_bre_bisr_attack(
    inputs: AttackInputs, *, target_method: str, threat_model: str = "split_inference",
    num_steps: int = 300, lr: float = 0.05, num_restarts: int = 1, seed: int = 0,
    model_family: str = "toy", model_name_or_path: str | None = None,
) -> AttackResult:
    """BiSR(f): forward smashed-data matching (Eq.7 cosine) + Eq.8 NN projection."""
    base = dict(attack_id="bre_bisr_forward", attack_name="BiSR forward (smashed-data matching)",
                attack_family="optimization", target_method=target_method,
                threat_model=threat_model, paper_source=_PAPER, implementation_level="best_effort",
                model_family=model_family, model_name_or_path=model_name_or_path)
    if inputs.downstream is None or inputs.observed_intermediate is None:
        return blocked_result(**base, reason="BiSR(f) needs the Bottom map (downstream) + smashed data")
    target = inputs.observed_intermediate.detach()
    N = target.shape[0]
    H = (inputs.plaintext_embeddings.shape[1] if inputs.plaintext_embeddings is not None
         else inputs.embedding_table.shape[1])
    down = inputs.downstream

    def cos_loss(pred, tgt):    # Eq.7: cosine (not L2) — BiSR's distinguishing choice
        p = torch.nn.functional.normalize(pred.reshape(N, -1), dim=-1)
        t = torch.nn.functional.normalize(tgt.reshape(N, -1), dim=-1)
        return (1.0 - (p * t).sum(-1)).mean()

    t0 = time.perf_counter()
    best, curve0 = None, None
    for restart in range(num_restarts):
        g = torch.Generator().manual_seed(seed + restart)
        emb = torch.randn(N, H, generator=g, requires_grad=True)
        opt = torch.optim.Adam([emb], lr=lr)
        curve = []
        for _ in range(num_steps):
            opt.zero_grad()
            l = cos_loss(down(emb), target)
            l.backward(); opt.step()
            curve.append(float(l.item()))
        if best is None or curve[-1] < best["final"]:
            best = {"final": curve[-1], "emb": emb.detach()}
            curve0 = curve
    ms = (time.perf_counter() - t0) * 1000.0

    metrics = empty_metrics()
    metrics["mse"] = mse(down(best["emb"]), target)
    metrics["cosine_similarity"] = cosine_similarity(down(best["emb"]), target)
    if inputs.embedding_table is not None and inputs.token_ids is not None:
        scores = nearest_neighbor_scores(best["emb"], inputs.embedding_table, "cosine")  # Eq.8
        rec = token_recovery_topk(scores, inputs.token_ids.reshape(-1))
        metrics.update(rec)
        metrics["mean_token_accuracy"] = mean_token_accuracy(scores.argmax(-1), inputs.token_ids.reshape(-1))
        metrics["attack_success_rate"] = rec["token_recovery_top1"]
    return AttackResult(**base, status="measured",
                        attacker_knowledge=make_attacker_knowledge(
                            has_embedding_table=inputs.embedding_table is not None),
                        input_info=make_input_info(hidden_size=H, num_samples=N, dtype=str(target.dtype)),
                        attack_config=make_attack_config(num_steps=num_steps, lr=lr,
                                                         loss_type="cosine", seed=seed),
                        metrics=metrics,
                        runtime=make_runtime(wall_time_ms=ms, num_steps_run=num_steps),
                        notes=f"BiSR FORWARD half only (Eq.7 cosine + Eq.8 NN projection); NOT full BiSR "
                              f"(backward gradient-matching blocked). initial={curve0[0]:.4g} final={best['final']:.4g}")


def run_bre_backward_gradient_matching(inputs: AttackInputs, *, target_method: str,
                                       model_family: str = "toy") -> AttackResult:
    """BiSR(b): gradient matching (Eq.9) — BLOCKED in an inference-only harness."""
    return blocked_result(
        attack_id="bre_bisr_backward", attack_name="BiSR backward (gradient matching)",
        attack_family="optimization", target_method=target_method,
        threat_model="split_inference", paper_source=_PAPER,
        model_family=model_family,
        reason="BiSR backward gradient matching (Eq.9) requires the activation "
               "gradients grad(x̃_trk) transmitted during split FINE-TUNING, plus a "
               "pre-trained Top mimic. An inference-only harness cannot produce "
               "training gradients. NEEDS: a split-learning fine-tuning loop that "
               "records grad(x̃_trk). See docs/server_run_plan.md.")


__all__ = ["run_bre_bisr_attack", "run_bre_backward_gradient_matching"]
