"""Attack-side metrics: token recovery, reconstruction error, alignment, matching."""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Token recovery / reconstruction
# ---------------------------------------------------------------------------
def token_recovery_topk(
    recovered_scores: torch.Tensor,   # (N, vocab)
    true_ids: torch.Tensor,           # (N,)
    ks=(1, 5, 10, 100),
) -> dict[str, float]:
    N, V = recovered_scores.shape
    out = {}
    maxk = min(max(ks), V)
    topk = recovered_scores.topk(maxk, dim=-1).indices
    true = true_ids.view(-1, 1)
    for k in ks:
        kk = min(k, V)
        out[f"token_recovery_top{k}"] = float((topk[:, :kk] == true).any(-1).float().mean().item())
    return out


def nearest_neighbor_scores(observed: torch.Tensor, table: torch.Tensor,
                            metric: str = "cosine") -> torch.Tensor:
    o = observed.to(torch.float32)
    t = table.to(torch.float32)
    if metric == "cosine":
        o = torch.nn.functional.normalize(o, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)
        return o @ t.transpose(0, 1)
    if metric == "l2":
        return -(torch.cdist(o, t) ** 2)
    if metric in ("dot", "dot_product"):
        return o @ t.transpose(0, 1)
    raise ValueError(f"unknown metric {metric!r}")


def sequence_exact_match(pred_ids: torch.Tensor, true_ids: torch.Tensor) -> float:
    return float(torch.equal(pred_ids.reshape(-1), true_ids.reshape(-1)))


def mean_token_accuracy(pred_ids: torch.Tensor, true_ids: torch.Tensor) -> float:
    a = pred_ids.reshape(-1)
    b = true_ids.reshape(-1)
    if a.numel() == 0:
        return 0.0
    return float((a == b).float().mean().item())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.reshape(-1).to(torch.float32)
    bf = b.reshape(-1).to(torch.float32)
    denom = af.norm() * bf.norm()
    return 0.0 if denom == 0 else float((af @ bf / denom).item())


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).to(torch.float32) ** 2).mean().item())


def relative_l2_error(approx: torch.Tensor, ref: torch.Tensor) -> float:
    r = ref.to(torch.float32)
    denom = r.norm().item()
    return 0.0 if denom == 0 else float(((approx.to(torch.float32) - r).norm().item()) / denom)


# ---------------------------------------------------------------------------
# Sequence-string metrics (token-id sequences)
# ---------------------------------------------------------------------------
def edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein distance between two token-id sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def rouge_l_f1(pred: list[int], ref: list[int]) -> float:
    """ROUGE-L F1 over token-id sequences (LCS-based)."""
    n, m = len(pred), len(ref)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if pred[i - 1] == ref[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / n, lcs / m
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Matching / permutation recovery
# ---------------------------------------------------------------------------
def hungarian_match(cost: torch.Tensor) -> torch.Tensor:
    """Return assignment col-index per row minimizing total cost.

    Uses scipy if available, else a greedy fallback (row-wise argmin with
    column masking). Returns a LongTensor of length n_rows.
    """
    c = cost.detach().to(torch.float64).cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment
        _, col = linear_sum_assignment(c)
        return torch.as_tensor(col, dtype=torch.long)
    except Exception:
        n, m = c.shape
        used = set()
        out = []
        order = sorted(range(n), key=lambda i: c[i].min())
        assign = {}
        for i in order:
            js = sorted(range(m), key=lambda j: c[i][j])
            for j in js:
                if j not in used:
                    used.add(j)
                    assign[i] = j
                    break
            else:
                assign[i] = int(c[i].argmin())
        for i in range(n):
            out.append(assign.get(i, int(c[i].argmin())))
        return torch.as_tensor(out, dtype=torch.long)


def greedy_argmin_match(cost: torch.Tensor) -> torch.Tensor:
    """Independent per-row argmin (ArrowMatch Eq.1 style; no bijection)."""
    return cost.argmin(dim=-1)


def permutation_recovery_accuracy(recovered: torch.Tensor, true_perm: torch.Tensor) -> float:
    """Fraction of positions where the recovered index equals the ground truth."""
    r = recovered.reshape(-1)
    t = true_perm.reshape(-1)
    n = min(r.numel(), t.numel())
    if n == 0:
        return 0.0
    return float((r[:n] == t[:n]).float().mean().item())


def sorted_l1_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant sorted-L1 distance between rows (Permutation attack Alg.3).

    a: (Na, d), b: (Nb, d) -> (Na, Nb) matrix of L1 distances between sorted rows.
    """
    sa = a.to(torch.float32).sort(dim=-1).values
    sb = b.to(torch.float32).sort(dim=-1).values
    return torch.cdist(sa, sb, p=1)


__all__ = [
    "token_recovery_topk", "nearest_neighbor_scores", "sequence_exact_match",
    "mean_token_accuracy", "cosine_similarity", "mse", "relative_l2_error",
    "edit_distance", "rouge_l_f1", "hungarian_match", "greedy_argmin_match",
    "permutation_recovery_accuracy", "sorted_l1_distance",
]
