"""Gram-based signed-permutation recovery (weight-leakage worst case).

Complements ArrowMatch. ArrowMatch matches weight COLUMNS by cosine, which is
defeated by per-column SIGN flips — so it under-recovers a *signed*-permutation
mask (our ``N_res``) even though that mask is not secure. This probe instead
matches the column-**Gram** ``G = Wᵀ W``:

    W_obf[:, k] = W[:, perm[k]] · signs[k]   ⇒   G_obf = Pᵀ G_pub P   (signs²=1)

so ``diag(G_obf)[k] = diag(G_pub)[perm[k]]`` and a nearest-diagonal match
recovers ``perm`` **sign-invariantly** whenever the public diagonals are
distinct. This is exactly the self-Gram recovery our own ``ours`` self-audit used
(``scripts/attacks/residual_mask_gram_attack.py``); it breaks STIP's permutation
AND our signed-perm residual mask, but NOT a dense orthogonal mix (ObfuscaTune),
whose ``G_obf = Rᵀ G_pub R`` is not a permutation of ``G_pub``.

Threat model: weight_leakage_worst_case (has_model_weights=True). NOT the default
closed-model deployment.
"""

from __future__ import annotations

import time

import torch

from .metrics import permutation_recovery_accuracy
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics, make_attack_config,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "Gram self-alignment (ours self-audit; cf. STIP/ArrowMatch weight leakage)"


def recover_perm_from_column_gram(w_obf: torch.Tensor, w_pre: torch.Tensor) -> torch.Tensor:
    """perm_hat[k]: the public column matched to obfuscated column k, via nearest
    Gram-diagonal (== ‖column‖²), which is invariant to per-column sign flips."""
    g_obf = (w_obf.to(torch.float64).T @ w_obf.to(torch.float64)).diagonal()
    g_pre = (w_pre.to(torch.float64).T @ w_pre.to(torch.float64)).diagonal()
    return (g_obf.view(-1, 1) - g_pre.view(1, -1)).abs().argmin(dim=1)


def run_gram_weight_recovery(
    inputs: AttackInputs, *, target_method: str,
    model_family: str = "toy", model_name_or_path: str | None = None,
) -> AttackResult:
    base = dict(attack_id="gram_weight_recovery", attack_name="Gram signed-perm recovery",
                attack_family="alignment", target_method=target_method,
                threat_model="weight_leakage_worst_case", paper_source=_PAPER,
                implementation_level="full", model_family=model_family,
                model_name_or_path=model_name_or_path)
    mw = inputs.model_weights
    if not mw or "public" not in mw or "obfuscated" not in mw:
        return blocked_result(**base, reason="Gram recovery needs model_weights={'public',"
                              "'obfuscated'} (weight-leakage worst case); not the default threat model")
    w_pre, w_obf = mw["public"], mw["obfuscated"]
    if w_pre.dim() != 2 or w_obf.dim() != 2:
        return blocked_result(**base, reason="weights must be 2-D matrices")
    t0 = time.perf_counter()
    perm_hat = recover_perm_from_column_gram(w_obf, w_pre)
    ms = (time.perf_counter() - t0) * 1000.0

    # Primary success = did the recovered match reconstruct the columns? A true
    # (signed-)permutation makes |cos(w_obf_k, w_pre_{perm_hat[k]})| == 1 (sign-
    # invariant); a dense orthogonal mix does NOT, so this stays low for
    # ObfuscaTune even though the diagonal match returns *some* permutation.
    a = torch.nn.functional.normalize(w_obf.to(torch.float32), dim=0)
    b = torch.nn.functional.normalize(w_pre.to(torch.float32), dim=0)
    col_cos = (a * b[:, perm_hat]).sum(dim=0).abs()
    align = float((col_cos > 0.99).float().mean())
    metrics = empty_metrics()
    metrics["alignment_accuracy"] = align
    metrics["attack_success_rate"] = align
    true_perm = mw.get("true_permutation")
    if true_perm is not None:
        metrics["permutation_recovery_accuracy"] = float(
            permutation_recovery_accuracy(perm_hat, torch.as_tensor(true_perm)))
    acc = align
    return AttackResult(**base, status="measured",
                        attacker_knowledge=make_attacker_knowledge(
                            has_model_weights=True, has_embedding_table=True),
                        input_info=make_input_info(hidden_size=w_pre.shape[0],
                                                   num_samples=w_pre.shape[1]),
                        attack_config=make_attack_config(loss_type="gram_diagonal_match"),
                        metrics=metrics, runtime=make_runtime(wall_time_ms=ms),
                        notes=f"column-Gram diagonal match; perm acc={acc:.3f}. Sign-invariant, so it "
                              f"recovers STIP's permutation AND our signed-perm residual mask; a dense "
                              f"orthogonal mix (ObfuscaTune) is NOT a Gram permutation and resists it.")


__all__ = ["run_gram_weight_recovery", "recover_perm_from_column_gram"]
