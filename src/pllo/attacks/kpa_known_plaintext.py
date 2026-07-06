"""KPA known-plaintext attack (P0, cryptanalysis) — linear + permutation.

Linear KPA: from known (X, X*=XR) pairs solve R_hat (lstsq/pinv/ridge), evaluate
held-out deobfuscation error, sample-complexity sweep, token recovery. Fresh
per-sample padding defeats it (intended protection). Permutation KPA: recover a
feature permutation from (X, Xπ) by column matching (Hungarian).
"""

from __future__ import annotations

import time

import torch

from .metrics import (hungarian_match, nearest_neighbor_scores,
                      permutation_recovery_accuracy, relative_l2_error,
                      token_recovery_topk)
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics, make_attack_config,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "Known-plaintext attack on linear/permutation obfuscation (ObfuscaTune/STIP)"


def _solve_R(X, Xstar, ridge=0.0):
    Xd, Yd = X.to(torch.float64), Xstar.to(torch.float64)
    if ridge > 0:
        H = Xd.shape[1]
        A = Xd.T @ Xd + ridge * torch.eye(H, dtype=torch.float64)
        return (torch.linalg.solve(A, Xd.T @ Yd)).to(X.dtype)
    return torch.linalg.lstsq(Xd, Yd).solution.to(X.dtype)


def kpa_linear_single(X, Xstar, *, num_pairs, holdout=16, noise=0.0, ridge=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    perm = torch.randperm(N, generator=g)
    tr, te = perm[:num_pairs], perm[num_pairs:num_pairs + holdout]
    if te.numel() == 0:
        te = perm[:holdout]
    Xstar_tr = Xstar[tr]
    if noise > 0:
        Xstar_tr = Xstar_tr + noise * torch.randn(Xstar_tr.shape, generator=g)
    R_hat = _solve_R(X[tr], Xstar_tr, ridge)
    return {"num_pairs": num_pairs, "R_hat": R_hat,
            "deobfuscation_relative_error": relative_l2_error(X[te] @ R_hat, Xstar[te])}


def run_kpa_known_plaintext(
    inputs: AttackInputs, *, target_method: str, mode: str = "linear",
    num_pairs_sweep=None, noise=0.0, ridge=0.0, seed=0,
    model_family: str = "toy", model_name_or_path: str | None = None,
) -> AttackResult:
    base = dict(attack_id="kpa_known_plaintext", attack_name=f"KPA ({mode})",
                attack_family="cryptanalysis", target_method=target_method,
                threat_model="known_plaintext", paper_source=_PAPER,
                implementation_level="full", model_family=model_family,
                model_name_or_path=model_name_or_path)
    if inputs.known_plaintext_pairs is None:
        return blocked_result(**base, reason="KPA needs known-plaintext pairs (X, X*)")
    X, Xstar = inputs.known_plaintext_pairs
    X = X.reshape(X.shape[0], -1)
    Xstar = Xstar.reshape(Xstar.shape[0], -1)
    fresh = bool(inputs.defense_metadata.get("fresh_pad"))
    H = X.shape[1]
    knowledge = make_attacker_knowledge(has_known_plaintext_pairs=True,
                                        has_embedding_table=inputs.embedding_table is not None)
    t0 = time.perf_counter()
    metrics = empty_metrics()

    if mode == "permutation":
        # recover a feature permutation from (X, Xπ) via column matching
        # cost[i,j] = distance between column i of X and column j of Xstar
        cost = torch.cdist(X.T.to(torch.float32), Xstar.T.to(torch.float32))
        rec = hungarian_match(cost)     # rec[i] = matched col of Xstar for X col i
        # convention-free verification: X col i is reconstructed from Xstar[:, rec[i]]
        recon_acc = 1.0 - relative_l2_error(Xstar[:, rec], X)
        true_perm = inputs.defense_metadata.get("true_permutation")
        exact_acc = None
        if true_perm is not None:
            # Xstar[:,k] = X[:,perm[k]] -> correct rec[i] satisfies perm[rec[i]] = i
            perm = torch.as_tensor(true_perm)
            exact_acc = float((perm[rec] == torch.arange(rec.numel())).float().mean())
        acc = exact_acc if exact_acc is not None else recon_acc
        metrics["permutation_recovery_accuracy"] = float(acc)
        metrics["attack_success_rate"] = float(max(0.0, min(1.0, acc)))
        note = f"permutation KPA via column matching; acc={acc:.3f}"
    else:
        if num_pairs_sweep is None:
            num_pairs_sweep = [max(1, H // 4), H // 2, H, 2 * H, 4 * H]
        num_pairs_sweep = [n for n in num_pairs_sweep if n < X.shape[0]] or \
            [min(H, max(1, X.shape[0] // 2))]
        sweep, best = [], None
        for n in num_pairs_sweep:
            r = kpa_linear_single(X, Xstar, num_pairs=n, noise=noise, ridge=ridge, seed=seed)
            sweep.append({"num_pairs": n, "deobf_err": r["deobfuscation_relative_error"]})
            if best is None or r["deobfuscation_relative_error"] < best["deobfuscation_relative_error"]:
                best = r
        err = best["deobfuscation_relative_error"]
        success = max(0.0, min(1.0, 1.0 - err))
        metrics["kpa_matrix_recovery_error"] = err
        metrics["relative_l2_error"] = err
        metrics["matrix_recovery_error"] = err
        metrics["attack_success_rate"] = success
        metrics["one_minus_attack_success_rate"] = 1.0 - success
        if inputs.embedding_table is not None and inputs.token_ids is not None:
            try:
                R_inv = torch.linalg.pinv(best["R_hat"].to(torch.float64)).to(X.dtype)
                scores = nearest_neighbor_scores(Xstar @ R_inv, inputs.embedding_table, "cosine")
                metrics.update(token_recovery_topk(scores, inputs.token_ids.reshape(-1)))
            except Exception:
                pass
        note = "linear KPA lstsq; sweep=" + ",".join(f"{s['num_pairs']}:{s['deobf_err']:.2e}" for s in sweep)
        if fresh:
            note = "INTENDED PROTECTION: fresh per-sample pad -> no global R; KPA fails. " + note
    ms = (time.perf_counter() - t0) * 1000.0
    return AttackResult(**base, status="measured", attacker_knowledge=knowledge,
                        input_info=make_input_info(hidden_size=H, num_samples=X.shape[0],
                                                   dtype=str(X.dtype)),
                        attack_config=make_attack_config(seed=seed),
                        metrics=metrics, runtime=make_runtime(wall_time_ms=ms), notes=note)


__all__ = ["run_kpa_known_plaintext", "kpa_linear_single"]
