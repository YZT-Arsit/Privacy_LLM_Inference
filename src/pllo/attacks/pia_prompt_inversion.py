"""PIA — Prompt Inversion Attack (Qu et al., IEEE S&P'25) — best_effort.

Faithful reimplementation of the white-box method's two phases:
  * Phase 1 (Alg.1, Eq.13): constrained optimization of a continuous ``v̂`` —
    ``min ‖F(v̂) − A‖² + λ Σ_i min_{t∈V} ‖v̂_i − E(t)‖²`` with per-dim clip to
    ``[L_j, R_j]`` (embedding-table min/max).
  * Phase 2 (Alg.2, Eq.16): adaptive discretization — greedy autoregressive
    activation calibration over a top-K embedding candidate set ``S_e``.

best_effort: we OMIT the oracle-LLM semantic-speculation candidate set ``S_s``
(needs a separate fluent LM). Labelled accordingly. β=0.1, λ=0.1, N=2000, K=10
defaults follow the paper.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from .metrics import edit_distance, mean_token_accuracy, rouge_l_f1, token_recovery_topk
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics, make_attack_config,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "PIA — Prompt Inversion Attack (Qu et al., arXiv:2503.09022, IEEE S&P'25)"


def run_pia_prompt_inversion(
    inputs: AttackInputs, *, target_method: str, threat_model: str = "split_inference",
    num_steps: int = 300, lr: float = 0.1, lam: float = 0.1, topk: int = 10,
    num_restarts: int = 1, seed: int = 0,
    activation_fn: Callable[[list[int]], torch.Tensor] | None = None,
    model_family: str = "toy", model_name_or_path: str | None = None,
) -> AttackResult:
    base = dict(attack_id="pia_prompt_inversion", attack_name="PIA prompt inversion",
                attack_family="optimization", target_method=target_method,
                threat_model=threat_model, paper_source=_PAPER, implementation_level="best_effort",
                model_family=model_family, model_name_or_path=model_name_or_path)
    if inputs.downstream is None or inputs.observed_intermediate is None or inputs.embedding_table is None:
        return blocked_result(**base, reason="PIA needs white-box F (downstream) + activation A + embedding table E")
    A = inputs.observed_intermediate.detach()
    E = inputs.embedding_table
    N = A.shape[0]
    H = E.shape[1]
    down = inputs.downstream
    L = E.min(0).values            # per-dim min over the vocab
    R = E.max(0).values

    # --- Phase 1: constrained optimization (Eq.13) ---
    t0 = time.perf_counter()
    best, curve0 = None, None
    for restart in range(num_restarts):
        g = torch.Generator().manual_seed(seed + restart)
        v = (torch.rand(N, H, generator=g) * (R - L) + L).clone().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=lr)
        curve = []
        for _ in range(num_steps):
            opt.zero_grad()
            act_loss = ((down(v) - A) ** 2).mean()
            # nearest-token pull: min_t ||v_i - E(t)||^2 (stop-grad on the index)
            with torch.no_grad():
                nn_idx = torch.cdist(v, E).argmin(1)
            prior = ((v - E[nn_idx]) ** 2).sum(-1).mean()
            l = act_loss + lam * prior
            l.backward(); opt.step()
            with torch.no_grad():
                v.clamp_(L, R)
            curve.append(float(l.item()))
        if best is None or curve[-1] < best["final"]:
            best = {"final": curve[-1], "v": v.detach()}
            curve0 = curve
    v_hat = best["v"]

    # --- Phase 2: adaptive discretization (Eq.16) ---
    sims = torch.cdist(v_hat, E)                          # (N, V)
    cand = sims.topk(min(topk, E.shape[0]), largest=False).indices   # S_e per position
    if activation_fn is not None:
        # autoregressive activation calibration: pick candidate minimizing ||F(E(prefix+t))_j - A_j||
        recovered = []
        for j in range(N):
            best_t, best_d = None, float("inf")
            for t in cand[j].tolist():
                act = activation_fn(recovered + [t])       # (>=j+1, H') last row = position j
                d = float(((act[-1] - A[j]) ** 2).mean())
                if d < best_d:
                    best_d, best_t = d, t
            recovered.append(best_t)
        pred_ids = torch.tensor(recovered)
    else:
        pred_ids = cand[:, 0]                              # nearest-embedding fallback
    ms = (time.perf_counter() - t0) * 1000.0

    scores = -sims                                        # higher = closer
    rec = token_recovery_topk(scores, inputs.token_ids.reshape(-1)[:N]) if inputs.token_ids is not None else {}
    metrics = empty_metrics(); metrics.update(rec)
    if inputs.token_ids is not None:
        tru = inputs.token_ids.reshape(-1)[:N]
        acc = mean_token_accuracy(pred_ids, tru)
        metrics["mean_token_accuracy"] = acc
        metrics["attack_success_rate"] = acc
        metrics["token_recovery_top1"] = acc
        metrics["rouge_l_f1"] = rouge_l_f1(pred_ids.tolist(), tru.tolist())
        metrics["edit_distance"] = float(edit_distance(pred_ids.tolist(), tru.tolist()))
        metrics["prompt_recovery_success"] = float(acc > 0.9)
    return AttackResult(**base, status="measured",
                        attacker_knowledge=make_attacker_knowledge(
                            has_embedding_table=True, has_model_weights=True,
                            has_intermediate_activations=True),
                        input_info=make_input_info(seq_len=N, hidden_size=H,
                                                   vocab_size=E.shape[0], num_samples=N),
                        attack_config=make_attack_config(num_steps=num_steps, lr=lr,
                                                         num_restarts=num_restarts, seed=seed),
                        metrics=metrics,
                        runtime=make_runtime(wall_time_ms=ms, num_steps_run=num_steps,
                                             requires_gpu=model_family != "toy"),
                        notes=f"PIA best_effort (Phase1 constrained opt + Phase2 "
                              f"{'autoregressive calibration' if activation_fn else 'NN discretization'}); "
                              f"oracle-LLM S_s OMITTED. initial={curve0[0]:.4g} final={best['final']:.4g}")


__all__ = ["run_pia_prompt_inversion"]
