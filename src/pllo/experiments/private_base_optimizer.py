"""O1 -- optimizer equivalence + preconditioner-leakage analysis (private-base).

Three profiles for masked-domain LoRA-SGD under the private-base convention where a
masked leaf is a two-sided transform of the plaintext leaf ``theta_t = P theta Q``
(P from the input/rank masks, Q from the output/rank masks):

* **O1-A** orthogonal feature+rank masks, direct GPU masked SGD (2 trusted inv/step).
  ``P P^T = Q^T Q = I`` so naive masked SGD already equals plaintext SGD and NO Gram
  is exposed. Correctness control -- but orthogonal masks are exactly what the
  self-Gram alignment attack recovers, so this is NOT the strong private-weight
  profile.
* **O1-B** non-orthogonal feature masks, PRECONDITIONED masked SGD (2 trusted
  inv/step). Under a non-orthogonal mask, naive masked SGD is WRONG; the update that
  recovers plaintext SGD is
      theta_t_next = theta_t - lr (P P^T) grad_t (Q^T Q)
  i.e. it needs the mask **Gram** ``P P^T`` / ``Q^T Q``. Whether this is paper-safe
  hinges on whether that Gram becomes GPU-visible (see leakage audit below).
* **O1-C** non-orthogonal feature masks, packed LoRA grads to TDX, trusted optimizer
  (3 trusted inv/step). The preconditioner/optimizer runs inside the boundary, so no
  Gram is exposed to the GPU -- deployable reference at the cost of one extra crossing.

Two structural constraints (verified numerically in :func:`verify_matrix`):
1. **RMSNorm forces an orthogonal residual mask under paper_safe.** ``rmsnorm_core``
   is invariant under ``N`` iff ``N N^T = I``. So the residual/hidden mask (the one
   the norm sees) cannot be non-orthogonal without a trusted reduction. Non-orthogonal
   masking is therefore confined to sub-bases that skip RMSNorm (attention Q/K ``R``,
   V ``S``, LM-head ``D``); it cannot protect the residual-basis weight chain
   ``W_tilde_l = N_l^{-1} W_l N_{l+1}`` under paper_safe.
2. **The O1-B preconditioner IS the mask Gram.** Applying it on the GPU re-exposes
   exactly ``P P^T`` / ``Q^T Q`` -- the Gram the non-orthogonal masking was meant to
   hide -- which is strictly more than the self-Gram attack already extracts.

CPU/fp64 analysis. No security is claimed for any profile here; O1-B is NOT paper-safe
until the leakage question is resolved (custom-backward or trusted-side application).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

DT = torch.float64
_inv = torch.linalg.inv

__all__ = [
    "O1Profile",
    "O1_PROFILES",
    "rand_orthogonal",
    "rand_conditioned",
    "masked_sgd_orthogonal",
    "masked_sgd_preconditioned",
    "trusted_optimizer_step",
    "verify_matrix",
    "preconditioner_visibility",
]


@dataclass(frozen=True)
class O1Profile:
    name: str
    masks: str                       # "orthogonal" | "non_orthogonal"
    update: str                      # "direct_masked_sgd" | "preconditioned" | "trusted"
    trusted_invocations_per_step: int
    gram_visible_to_gpu: bool
    paper_safe_eligible: bool
    role: str


O1_PROFILES: Dict[str, O1Profile] = {
    "O1-A": O1Profile("O1-A", "orthogonal", "direct_masked_sgd", 2,
                      gram_visible_to_gpu=False, paper_safe_eligible=True,
                      role="correctness_control_not_strong_private_weight"),
    "O1-B": O1Profile("O1-B", "non_orthogonal", "preconditioned", 2,
                      gram_visible_to_gpu=True, paper_safe_eligible=False,
                      role="low_interaction_candidate_leaks_gram_unless_custom_backward"),
    "O1-C": O1Profile("O1-C", "non_orthogonal", "trusted", 3,
                      gram_visible_to_gpu=False, paper_safe_eligible=True,
                      role="deployable_reference_extra_trusted_crossing"),
}


def rand_orthogonal(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=DT))
    return q


def rand_conditioned(n: int, cond: float, seed: int) -> torch.Tensor:
    """Random invertible matrix with exact condition number ``cond`` (>=1)."""
    g = torch.Generator().manual_seed(seed)
    u, _, v = torch.linalg.svd(torch.randn(n, n, generator=g, dtype=DT))
    s = torch.linspace(1.0, cond, n, dtype=DT)
    return u @ torch.diag(s) @ v


# -- update rules (masked leaf theta_t = P theta Q) -------------------------


def masked_sgd_orthogonal(theta_t, grad_t, lr):
    """O1-A: valid ONLY when P,Q orthogonal (P P^T = Q^T Q = I)."""
    return theta_t - lr * grad_t


def masked_sgd_preconditioned(theta_t, grad_t, lr, P, Q):
    """O1-B: recovers plaintext SGD under any invertible P,Q. Needs the mask Gram."""
    return theta_t - lr * (P @ P.T) @ grad_t @ (Q.T @ Q)


def trusted_optimizer_step(theta_plain, grad_plain, lr, *, optimizer="sgd",
                           state=None):
    """O1-C: the optimizer runs on the recovered plaintext leaf inside the boundary.
    Returns (theta_next, state). AdamW supported here because it is trusted-side."""
    if optimizer == "sgd":
        return theta_plain - lr * grad_plain, state
    if optimizer == "adamw":
        state = state or {"m": torch.zeros_like(theta_plain),
                          "v": torch.zeros_like(theta_plain), "t": 0}
        b1, b2, eps = 0.9, 0.999, 1e-8
        state["t"] += 1
        state["m"] = b1 * state["m"] + (1 - b1) * grad_plain
        state["v"] = b2 * state["v"] + (1 - b2) * grad_plain.pow(2)
        mh = state["m"] / (1 - b1 ** state["t"])
        vh = state["v"] / (1 - b2 ** state["t"])
        return theta_plain - lr * mh / (vh.sqrt() + eps), state
    raise ValueError(optimizer)


# -- verification matrix ----------------------------------------------------


def _grad_of(theta, x, tgt):
    th = theta.clone().requires_grad_(True)
    ((x @ th - tgt) ** 2).mean().backward()
    return th.grad.detach()


def verify_matrix(conds: Tuple[float, ...] = (1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0),
                  *, m: int = 6, n: int = 8, lr: float = 0.1, seed: int = 0
                  ) -> List[Dict[str, float]]:
    """For each condition number: does the preconditioned update recover plaintext
    SGD, does the naive update fail, and how far is the Gram from I? Also records the
    RMSNorm invariance error at that condition."""
    torch.manual_seed(seed)
    theta = torch.randn(m, n, dtype=DT)
    x = torch.randn(4, m, dtype=DT)
    tgt = torch.randn(4, n, dtype=DT)
    g_plain = _grad_of(theta, x, tgt)
    plain_next = theta - lr * g_plain

    def rms(z, eps=1e-6):
        return z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps)
    h = torch.randn(3, n, dtype=DT)

    rows = []
    for c in conds:
        if c <= 1.0 + 1e-9:
            P, Q = rand_orthogonal(m, seed + 1), rand_orthogonal(n, seed + 2)
            Nfeat = rand_orthogonal(n, seed + 3)
        else:
            P, Q = rand_conditioned(m, c, seed + 1), rand_conditioned(n, c, seed + 2)
            Nfeat = rand_conditioned(n, c, seed + 3)
        theta_t = P @ theta @ Q
        tt = theta_t.clone().requires_grad_(True)
        ((x @ (_inv(P) @ tt @ _inv(Q)) - tgt) ** 2).mean().backward()
        g_t = tt.grad.detach()
        pre_next = masked_sgd_preconditioned(theta_t, g_t, lr, P, Q)
        pre_rec = _inv(P) @ pre_next @ _inv(Q)
        naive_next = _inv(P) @ masked_sgd_orthogonal(theta_t, g_t, lr) @ _inv(Q)
        rows.append({
            "cond": float(c),
            "precond_recovery_err": float((pre_rec - plain_next).abs().max()),
            "naive_recovery_err": float((naive_next - plain_next).abs().max()),
            "gram_dist_from_I": float((P @ P.T - torch.eye(m, dtype=DT)).abs().max()),
            "rmsnorm_invariance_err": float((rms(h @ Nfeat) - rms(h) @ Nfeat).abs().max()),
        })
    return rows


def preconditioner_visibility() -> Dict[str, Dict[str, object]]:
    """Where each profile's preconditioner/Gram lives, and what the GPU learns."""
    return {
        "O1-A": {"preconditioner": "identity (orthogonal masks)",
                 "gram_visible_to_gpu": False,
                 "note": "no Gram exposed; but orthogonal mixer is recoverable by self-Gram alignment"},
        "O1-B": {"preconditioner": "P P^T (left) and Q^T Q (right) = the mask Gram",
                 "gram_visible_to_gpu": True,
                 "note": "applying the update on the GPU exposes the exact mask Gram -- "
                         "strictly more than the self-Gram attack already extracts; "
                         "defeats the non-orthogonal defense unless moved into a custom "
                         "backward that never materialises the Gram on the GPU (open)"},
        "O1-C": {"preconditioner": "applied inside the trusted optimizer (TDX)",
                 "gram_visible_to_gpu": False,
                 "note": "GPU only sends packed masked grads; +1 trusted crossing"},
    }
