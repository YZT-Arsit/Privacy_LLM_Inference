"""EIA-style optimization inversion (P0, optimization) — Song&Raghunathan family.

SIMPLIFIED / modern adaptation (best_effort, not a verbatim reproduction of any
single system). Optimizes a relaxed input (soft tokens over the vocab, or
continuous embeddings) so a differentiable victim map ``downstream`` reproduces
an observed representation. Supports MSE / cosine / combined loss, Adam/AdamW,
restarts, temperature, entropy regularization, and hard projection for eval.
"""

from __future__ import annotations

import time

import torch

from .metrics import cosine_similarity, mean_token_accuracy, mse
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics, make_attack_config,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "EIA optimization inversion (Song & Raghunathan CCS'20 family)"


def _loss(pred, target, kind):
    if kind == "mse":
        return ((pred - target) ** 2).mean()
    if kind == "cosine":
        p = torch.nn.functional.normalize(pred.reshape(pred.shape[0], -1), dim=-1)
        t = torch.nn.functional.normalize(target.reshape(target.shape[0], -1), dim=-1)
        return (1.0 - (p * t).sum(-1)).mean()
    if kind == "combined":
        return _loss(pred, target, "mse") + _loss(pred, target, "cosine")
    raise ValueError(kind)


def run_eia_optimization(
    inputs: AttackInputs, *, target_method: str, threat_model: str = "public_model",
    mode: str = "optimize_soft_tokens", loss: str = "mse", num_steps: int = 200,
    lr: float = 0.1, num_restarts: int = 1, temperature: float = 0.1,
    entropy_reg: float = 0.0, optimizer: str = "adam", seed: int = 0,
    model_family: str = "toy", model_name_or_path: str | None = None,
) -> AttackResult:
    base = dict(attack_id="eia_optimization", attack_name="EIA optimization inversion",
                attack_family="optimization", target_method=target_method,
                threat_model=threat_model, paper_source=_PAPER, implementation_level="best_effort",
                model_family=model_family, model_name_or_path=model_name_or_path)
    if inputs.downstream is None or inputs.observed_intermediate is None:
        return blocked_result(**base, reason="EIA needs a differentiable downstream map + observed target")
    if mode == "optimize_soft_tokens" and inputs.embedding_table is None:
        return blocked_result(**base, reason="soft-token mode needs the embedding table")
    target = inputs.observed_intermediate.detach()
    N = target.shape[0]
    H = inputs.embedding_table.shape[1] if inputs.embedding_table is not None else inputs.plaintext_embeddings.shape[1]
    down = inputs.downstream
    Opt = torch.optim.AdamW if optimizer == "adamw" else torch.optim.Adam

    t0 = time.perf_counter()
    best, curve0 = None, None
    for restart in range(num_restarts):
        g = torch.Generator().manual_seed(seed + restart)
        if mode == "optimize_soft_tokens":
            var = torch.randn(N, inputs.embedding_table.shape[0], generator=g, requires_grad=True)
        else:
            var = torch.randn(N, H, generator=g, requires_grad=True)
        opt = Opt([var], lr=lr)
        curve = []
        for _ in range(num_steps):
            opt.zero_grad()
            if mode == "optimize_soft_tokens":
                probs = torch.softmax(var / temperature, dim=-1)
                recon = probs @ inputs.embedding_table
                l = _loss(down(recon), target, loss)
                if entropy_reg > 0:
                    l = l + entropy_reg * (probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
            else:
                recon = var
                l = _loss(down(recon), target, loss)
            l.backward()
            opt.step()
            curve.append(float(l.item()))
        if best is None or curve[-1] < best["final"]:
            with torch.no_grad():
                if mode == "optimize_soft_tokens":
                    probs = torch.softmax(var / temperature, dim=-1)
                    recon = probs @ inputs.embedding_table
                    pred_ids = probs.argmax(-1)
                else:
                    recon = var.detach()
                    pred_ids = None
                    if inputs.embedding_table is not None:
                        sims = torch.nn.functional.normalize(recon, dim=-1) @ \
                            torch.nn.functional.normalize(inputs.embedding_table, dim=-1).T
                        pred_ids = sims.argmax(-1)
            best = {"final": curve[-1], "recon": recon.detach(), "pred_ids": pred_ids}
            curve0 = curve
    ms = (time.perf_counter() - t0) * 1000.0

    metrics = empty_metrics()
    metrics["mse"] = mse(down(best["recon"]), target)
    metrics["cosine_similarity"] = cosine_similarity(down(best["recon"]), target)
    if best["pred_ids"] is not None and inputs.token_ids is not None:
        acc = mean_token_accuracy(best["pred_ids"], inputs.token_ids.reshape(-1))
        metrics["mean_token_accuracy"] = acc
        metrics["token_recovery_top1"] = acc
        metrics["attack_success_rate"] = acc
    return AttackResult(**base, status="measured",
                        attacker_knowledge=make_attacker_knowledge(
                            has_embedding_table=inputs.embedding_table is not None),
                        input_info=make_input_info(hidden_size=H, num_samples=N,
                                                   dtype=str(target.dtype)),
                        attack_config=make_attack_config(num_steps=num_steps, num_restarts=num_restarts,
                                                         lr=lr, loss_type=loss, seed=seed),
                        metrics=metrics,
                        runtime=make_runtime(wall_time_ms=ms, num_steps_run=num_steps,
                                             requires_gpu=model_family != "toy"),
                        notes=f"SIMPLIFIED EIA ({mode}, {loss}); initial_loss={curve0[0]:.4g} "
                              f"final_loss={best['final']:.4g}")


__all__ = ["run_eia_optimization"]
