"""ArrowMatch / Game of Arrows weight-alignment attack (P1, alignment).

Faithful local reimplementation of the two-stage attack (Wang et al., USENIX
Sec'25). Per weight matrix, column view w_i.
  * S1 (Eq.1): σ(i) = argmin_j cos_dist(w_obf^i, w_pre^j)  — greedy per-column
    argmin (optionally Hungarian for a bijection). Recovers the permutation.
  * S2: scale ŝ_i = ‖w_pre^{σ(i)}‖ / ‖w_obf^i‖.
Exploits cosine invariance to column permutation + positive scaling; direction-
changing matrix mixing defeats it.

Threat model: weight_leakage_worst_case — attacker has obfuscated + public
weights (has_model_weights=True). Also loads external ArrowMatch results.
"""

from __future__ import annotations

import time

import torch

from .metrics import greedy_argmin_match, hungarian_match, permutation_recovery_accuracy
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics, make_attack_config,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "Game of Arrows / ArrowMatch (Wang et al., USENIX Security'25)"


def _cos_dist_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(n, m) cosine DISTANCE (1 - cos_sim) between columns of a and b.

    a, b are (dim, ncols): each COLUMN is a 'vector' (arrow)."""
    an = torch.nn.functional.normalize(a.to(torch.float32), dim=0)
    bn = torch.nn.functional.normalize(b.to(torch.float32), dim=0)
    return 1.0 - (an.T @ bn)


def arrowmatch_single(w_obf: torch.Tensor, w_pre: torch.Tensor, *, bijection=False):
    """Match obf columns to public columns. Returns (sigma, scales, acc_placeholder)."""
    cost = _cos_dist_matrix(w_obf, w_pre)          # (n_obf, n_pre)
    sigma = hungarian_match(cost) if bijection else greedy_argmin_match(cost)  # per obf col
    # S2 length ratio
    obf_norm = w_obf.to(torch.float32).norm(dim=0)
    pre_norm = w_pre.to(torch.float32).norm(dim=0)
    scales = pre_norm[sigma] / obf_norm.clamp_min(1e-12)
    return sigma, scales


def run_arrowmatch_attack(
    inputs: AttackInputs, *, target_method: str, bijection: bool = True,
    model_family: str = "toy", model_name_or_path: str | None = None,
) -> AttackResult:
    base = dict(attack_id="arrowmatch_weight_alignment", attack_name="ArrowMatch",
                attack_family="alignment", target_method=target_method,
                threat_model="weight_leakage_worst_case", paper_source=_PAPER,
                implementation_level="full", model_family=model_family,
                model_name_or_path=model_name_or_path)
    mw = inputs.model_weights
    if not mw or "public" not in mw or "obfuscated" not in mw:
        return blocked_result(**base, reason="ArrowMatch needs model_weights={'public','obfuscated'} "
                              "(weight-leakage worst case); not the default threat model")
    w_pre, w_obf = mw["public"], mw["obfuscated"]
    # columns = vectors; ensure (dim, ncols)
    if w_pre.dim() != 2:
        return blocked_result(**base, reason="weights must be 2-D matrices")
    t0 = time.perf_counter()
    sigma, scales = arrowmatch_single(w_obf, w_pre, bijection=bijection)
    ms = (time.perf_counter() - t0) * 1000.0

    true_perm = mw.get("true_permutation")
    if true_perm is not None:
        acc = permutation_recovery_accuracy(sigma, torch.as_tensor(true_perm))
    else:
        # verify via reconstruction cosine: how well aligned are matched columns
        cd = _cos_dist_matrix(w_obf, w_pre)
        acc = float((cd.gather(1, sigma.view(-1, 1)).squeeze(1) < 0.05).float().mean())
    metrics = empty_metrics()
    metrics["permutation_recovery_accuracy"] = float(acc)
    metrics["alignment_accuracy"] = float(acc)
    metrics["attack_success_rate"] = float(acc)
    # length similarity Sim = 1 - mean relative length error after scaling
    if true_perm is not None:
        tp = torch.as_tensor(true_perm)
        length_err = float(((scales * w_obf.norm(dim=0) - w_pre.norm(dim=0)[sigma]).abs()
                            / w_pre.norm(dim=0)[sigma].clamp_min(1e-9)).mean())
        metrics["relative_l2_error"] = length_err
    return AttackResult(**base, status="measured",
                        attacker_knowledge=make_attacker_knowledge(
                            has_model_weights=True, has_embedding_table=True),
                        input_info=make_input_info(hidden_size=w_pre.shape[0],
                                                   num_samples=w_pre.shape[1]),
                        attack_config=make_attack_config(loss_type="cosine_distance"),
                        metrics=metrics, runtime=make_runtime(wall_time_ms=ms),
                        notes=f"S1 cosine argmin{'+Hungarian' if bijection else ''} + S2 length ratio; "
                              f"perm acc={acc:.3f}. Cosine is invariant to permutation+positive "
                              f"scaling; direction-changing matrix mixing defeats it.")


def load_external_arrowmatch(path) -> list[AttackResult]:
    """Load external ArrowMatch results (JSON/CSV) into the schema (worst case)."""
    import csv
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    if p.suffix == ".json":
        doc = json.loads(p.read_text())
        rows = doc if isinstance(doc, list) else doc.get("results", [doc])
    elif p.suffix == ".csv":
        with p.open() as f:
            rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        m = empty_metrics()
        for k in ("gram_alignment_success_rate", "alignment_accuracy", "token_recovery_top1"):
            if row.get(k) not in (None, ""):
                dst = "alignment_accuracy" if ("align" in k or "gram" in k) else k
                m[dst] = float(row[k])
        if m["attack_success_rate"] is None and m["alignment_accuracy"] is not None:
            m["attack_success_rate"] = m["alignment_accuracy"]
        out.append(AttackResult(
            attack_id="arrowmatch_weight_alignment", attack_name="ArrowMatch (external)",
            attack_family="alignment", target_method=row.get("method_id", "stip_qwen"),
            threat_model="weight_leakage_worst_case", paper_source=_PAPER,
            implementation_level="full",
            attacker_knowledge=make_attacker_knowledge(has_model_weights=True, has_embedding_table=True),
            metrics=m, status="measured",
            notes="EXTERNAL ArrowMatch result (worst case). " + str(row.get("notes", ""))))
    return out


__all__ = ["run_arrowmatch_attack", "arrowmatch_single", "load_external_arrowmatch"]
