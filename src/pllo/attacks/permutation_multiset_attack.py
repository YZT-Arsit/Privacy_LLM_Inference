"""Permutation / multiset / norm leakage attack (Thomas et al., 2505.18332).

Two components:
  * ``structural_leakage_probe``: measures whether a defense preserves the
    coordinate multiset (sorted rows), per-token norm profile, and pairwise
    distance profile — the invariants STIP/permutation schemes leak. Also
    recovers a feature permutation by sorted-vector matching (Hungarian) against
    a plaintext reference.
  * ``run_autoregressive_decode``: the paper's Algorithm 3 (hidden-dim
    permutation) — greedy autoregressive token recovery matching each candidate's
    layer-l last-row hidden state to the observed (permuted) states by SORTED-L1
    within threshold ε. Requires the model + embedding table (public model). Full
    on toy / tiny Qwen.
"""

from __future__ import annotations

import time

import torch

from .metrics import (hungarian_match, permutation_recovery_accuracy,
                      relative_l2_error, sorted_l1_distance)
from .representations import AttackInputs
from .schema import (AttackResult, blocked_result, empty_metrics, make_attack_config,
                     make_attacker_knowledge, make_input_info, make_runtime)

_PAPER = "Permutation-based inference attack (Thomas et al., arXiv:2505.18332)"


def structural_leakage_probe(
    inputs: AttackInputs, *, target_method: str,
    threat_model: str = "closed_model_no_weight_access",
    model_family: str = "toy", model_name_or_path: str | None = None,
) -> AttackResult:
    base = dict(attack_id="multiset_permutation_leakage",
                attack_name="Structural leakage probe (multiset/norm/permutation)",
                attack_family="structural", target_method=target_method,
                threat_model=threat_model, paper_source=_PAPER, implementation_level="full",
                model_family=model_family, model_name_or_path=model_name_or_path)
    p = inputs.plaintext_embeddings
    q = inputs.protected_embeddings
    if p is None or q is None:
        return blocked_result(**base, reason="needs plaintext + protected states")
    p = p.reshape(-1, p.shape[-1]).to(torch.float32)
    q = q.reshape(-1, q.shape[-1]).to(torch.float32)
    t0 = time.perf_counter()

    # 1. coordinate multiset preserved? sorted rows equal up to row matching
    sd = sorted_l1_distance(p, q)                  # (Np, Nq)
    match = hungarian_match(sd)
    multiset_err = float(sd.gather(1, match.view(-1, 1)).mean() / (p.abs().mean() + 1e-9))
    multiset_leak = max(0.0, 1.0 - multiset_err)

    # 2. per-token norm profile preserved?
    pn = p.norm(dim=-1).sort().values
    qn = q.norm(dim=-1).sort().values
    n = min(pn.numel(), qn.numel())
    norm_match = max(0.0, 1.0 - relative_l2_error(qn[:n], pn[:n]))

    # 3. pairwise distance profile preserved?
    dp = torch.pdist(p).sort().values
    dq = torch.pdist(q).sort().values
    m = min(dp.numel(), dq.numel())
    dist_match = max(0.0, 1.0 - relative_l2_error(dq[:m], dp[:m])) if m > 0 else 0.0

    # 4. permutation recovery: match protected rows to plaintext rows by sorted-L1
    perm_acc = float((match == torch.arange(match.numel())).float().mean()) if p.shape == q.shape else None
    ms = (time.perf_counter() - t0) * 1000.0

    metrics = empty_metrics()
    metrics["multiset_leakage_score"] = multiset_leak
    metrics["norm_profile_match_accuracy"] = norm_match
    metrics["distance_profile_match_accuracy"] = dist_match
    if perm_acc is not None:
        metrics["permutation_recovery_accuracy"] = perm_acc
    metrics["attack_success_rate"] = multiset_leak
    return AttackResult(**base, status="measured",
                        attacker_knowledge=make_attacker_knowledge(has_intermediate_activations=True),
                        input_info=make_input_info(num_samples=p.shape[0], hidden_size=p.shape[1]),
                        metrics=metrics, runtime=make_runtime(wall_time_ms=ms),
                        notes="structural leakage probe: multiset/norm/distance preservation + "
                              "sorted-L1 permutation recovery. High => permutation/orthogonal "
                              "defense leaks structure (STIP-like).")


def run_autoregressive_decode(
    model, observed_states: torch.Tensor, true_ids: torch.Tensor, *,
    target_method: str, layer: int = 0, epsilon: float = 5.0, max_candidates: int = 0,
    permuted: bool = True, model_family: str = "qwen", model_name_or_path: str | None = None,
) -> AttackResult:
    """Algorithm 3: greedy autoregressive sorted-L1 decode (hidden-dim permutation).

    ``observed_states`` (S, H) = the attacker-observed layer-``layer`` states
    (permuted). Recovers tokens by, at each position, forwarding each vocab
    candidate and matching its last-row state to the remaining observed states by
    sorted-L1 (permutation-invariant). ``epsilon`` is the accept threshold.
    """
    base = dict(attack_id="multiset_permutation_leakage",
                attack_name="Autoregressive sorted-L1 decode (Alg.3)",
                attack_family="structural", target_method=target_method,
                threat_model="public_model", paper_source=_PAPER, implementation_level="full",
                model_family=model_family, model_name_or_path=model_name_or_path)
    from pllo.attacks.qwen_hooks import QwenActivationCapture
    V = model.config.vocab_size
    S = observed_states.shape[0]
    cand = list(range(V if max_candidates <= 0 else min(V, max_candidates)))
    dist = sorted_l1_distance if permuted else (lambda a, b: torch.cdist(a, b, p=1))

    def state_of(prefix_ids):
        cap = QwenActivationCapture(model, layers=[layer])
        out = cap.run(torch.tensor(prefix_ids).unsqueeze(0))
        cap.remove()
        h = out.get(f"layer{layer}_input")
        return h[0, -1]              # last row (S index -1), (H,)

    t0 = time.perf_counter()
    remaining = list(range(S))
    recovered = []
    obs = observed_states.to(torch.float32)
    for i in range(S):
        best_v, best_d, best_row = None, float("inf"), None
        prefix = recovered
        # Alg.1 (unpermuted): positionally match obs[i]. Alg.3 (permuted): match the
        # remaining set of observed states (sequence order scrambled).
        targets = [i] if not permuted else list(remaining)
        for v in cand:
            g = state_of(prefix + [v]).to(torch.float32).unsqueeze(0)   # (1, H)
            rows = obs[targets]                                         # (R, H)
            d = dist(g, rows)[0]                                        # (R,)
            j = int(d.argmin())
            if float(d[j]) < best_d:
                best_d, best_v, best_row = float(d[j]), v, targets[j]
            if float(d[j]) < epsilon:
                best_d, best_v, best_row = float(d[j]), v, targets[j]
                break
        recovered.append(best_v)
        if permuted and best_row in remaining:
            remaining.remove(best_row)
    ms = (time.perf_counter() - t0) * 1000.0

    rec_t = torch.tensor(recovered)
    tru = true_ids.reshape(-1)[:S]
    acc = float((rec_t == tru).float().mean())
    exact = float(torch.equal(rec_t, tru))
    metrics = empty_metrics()
    metrics["mean_token_accuracy"] = acc
    metrics["token_recovery_top1"] = acc
    metrics["sequence_exact_match"] = exact
    metrics["attack_success_rate"] = acc
    return AttackResult(**base, status="measured",
                        attacker_knowledge=make_attacker_knowledge(
                            has_embedding_table=True, has_model_weights=True,
                            has_intermediate_activations=True),
                        input_info=make_input_info(seq_len=S, vocab_size=V, num_samples=S),
                        attack_config=make_attack_config(num_steps=S),
                        metrics=metrics, runtime=make_runtime(wall_time_ms=ms),
                        notes=f"Alg.3 sorted-L1 autoregressive decode, eps={epsilon}; token_acc={acc:.3f}")


__all__ = ["structural_leakage_probe", "run_autoregressive_decode"]
