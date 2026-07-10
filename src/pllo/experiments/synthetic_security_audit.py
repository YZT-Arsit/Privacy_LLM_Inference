"""CPU-only SYNTHETIC security red-team (Section F).

Every result here is algebraic / synthetic. It is NOT a real-model attack and
does NOT replace attacks on real Qwen/Llama activations (deferred to real-GPU).
Covers: F1 cross-Gram boundary, F2 value-multiset leakage, F3 LoRA subspace
trajectory, F4 compensation second-moment convergence, F5 one-view vs multi-view
(exploratory), F6 packed-buffer metadata leakage.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles

from pllo.experiments import permutation_nonlinear_backward_audit as pnb


def _rng(seed):
    return np.random.default_rng(seed)


def _orth(dim, g):
    q, _ = np.linalg.qr(g.standard_normal((dim, dim)))
    return q


def _gl(dim, g, cond=10.0):
    q1, _ = np.linalg.qr(g.standard_normal((dim, dim)))
    q2, _ = np.linalg.qr(g.standard_normal((dim, dim)))
    s = np.geomspace(1.0, cond, dim)
    return q1 @ np.diag(s) @ q2.T


# ---------------------------------------------------------------------------
# F1 — cross-Gram boundary over many seeds (reuse pnb.cross_gram_boundary)
# ---------------------------------------------------------------------------
def f1_cross_gram(trials=300) -> dict:
    r = pnb.cross_gram_boundary(trials=trials)
    # add rank/spectrum of a representative nonlinear-region Gram
    g = _rng(0)
    Z = g.standard_normal((8, 32)); GZ = g.standard_normal((8, 32))
    gram = Z @ GZ.T
    sv = np.linalg.svd(gram, compute_uv=False)
    return {
        "trials": r["trials"],
        "general_region_mean_rel_error": r["mean_rel_error_general"],
        "general_region_mean_corr": r["mean_corr_general"],
        "general_region_exact": r["general_region_exact"],
        "nonlinear_region_mean_rel_error": r["mean_rel_error_z"],
        "nonlinear_region_mean_corr": r["mean_corr_z"],
        "nonlinear_region_exact": r["nonlinear_region_exact"],
        "nonlinear_gram_rank": int(np.linalg.matrix_rank(gram)),
        "nonlinear_gram_top_singular_values": [round(float(x), 4) for x in sv[:4]],
        "synthetic": True,
    }


# ---------------------------------------------------------------------------
# F2 — value-multiset leakage; fixed-Pi vs fresh-Pi column linkability
# ---------------------------------------------------------------------------
def f2_value_multiset(m=32, d=24, seed=0) -> dict:
    g = _rng(seed)
    Z = g.standard_normal((m, d))
    perm1 = g.permutation(d)
    perm2 = g.permutation(d)
    Z1 = Z[:, perm1]                      # masked view 1 (fixed Pi)
    Z1b = (Z + 1e-9 * g.standard_normal((m, d)))[:, perm1]   # second sample, SAME Pi
    Z2 = Z[:, perm2]                      # fresh Pi

    multiset_err = float(np.abs(np.sort(Z1, axis=1) - np.sort(Z, axis=1)).max())

    # fixed-Pi column linkability: columns of Z1 and Z1b align -> high correlation match
    def link_accuracy(Xa, Xb):
        C = np.abs(np.corrcoef(Xa.T, Xb.T)[:d, d:])
        match = C.argmax(1)
        return float(np.mean(match == np.arange(d)))

    fixed_link = link_accuracy(Z1, Z1b)   # same Pi -> should link ~1.0
    fresh_link = link_accuracy(Z1, Z2)    # different Pi -> direct linking should fail

    # synthetic candidate dictionary: match a masked row to its plaintext row by sorted values
    cand = np.sort(Z, axis=1)
    obs = np.sort(Z1, axis=1)
    dmat = np.linalg.norm(obs[:, None, :] - cand[None, :, :], axis=2)
    match_rows = dmat.argmin(1)
    candidate_match = float(np.mean(match_rows == np.arange(m)))
    return {
        "multiset_exact_equality_error": multiset_err,
        "multiset_invariant": bool(multiset_err < 1e-9),
        "fixed_Pi_column_linkability": round(fixed_link, 3),
        "fresh_Pi_column_linkability": round(fresh_link, 3),
        "candidate_dictionary_match_success": round(candidate_match, 3),
        "synthetic": True,
        "note": "permutation hides column identity but NOT the per-row sorted multiset; "
                "fresh Pi blocks direct column linking, sorted-row multiset stays invariant.",
    }


# ---------------------------------------------------------------------------
# F3 — LoRA subspace trajectory: rank/singular-value/principal-angle invariance
# ---------------------------------------------------------------------------
def f3_subspace(d_in=16, r=4, steps=6, seed=0) -> dict:
    g = _rng(seed)
    # masked gradA columns over steps; masked coordinate = left transform N_in^T
    grads = [g.standard_normal((d_in, r)) for _ in range(steps)]
    M = np.concatenate(grads, axis=1)                 # [d_in, r*steps]
    N = _gl(d_in, g, cond=10.0)                        # general invertible left transform
    Q = _orth(d_in, g)                                 # orthogonal transform

    rank_M = int(np.linalg.matrix_rank(M))
    rank_gl = int(np.linalg.matrix_rank(N.T @ M))      # rank invariant under invertible left mult
    sv = lambda X: np.linalg.svd(X, compute_uv=False)
    sv_change_gl = float(np.linalg.norm(sv(N.T @ M) - sv(M)) / (np.linalg.norm(sv(M)) + 1e-30))
    sv_change_orth = float(np.linalg.norm(sv(Q.T @ M) - sv(M)) / (np.linalg.norm(sv(M)) + 1e-30))

    def top_subspace(X, k=r):
        u, _, _ = np.linalg.svd(X, full_matrices=False)
        return u[:, :k]

    # Angle BETWEEN two subspaces (two halves of the trajectory), and whether the
    # SAME transform applied to both preserves that angle. Orthogonal preserves it;
    # a general GL distorts it.
    U1 = top_subspace(M[:, : r * steps // 2])
    U2 = top_subspace(M[:, r * steps // 2:])
    base_ang = float(np.max(subspace_angles(U1, U2)))
    orth_ang = float(np.max(subspace_angles(_orth_apply(Q, U1), _orth_apply(Q, U2))))
    gl_ang = float(np.max(subspace_angles(_gl_apply(N, U1), _gl_apply(N, U2))))
    return {
        "rank_invariant_under_left_transform": bool(rank_M == rank_gl),
        "rank_M": rank_M, "rank_after_GL": rank_gl,
        "singular_value_change_orthogonal": round(sv_change_orth, 8),
        "singular_value_change_general_GL": round(sv_change_gl, 4),
        "orthogonal_preserves_singular_values": bool(sv_change_orth < 1e-8),
        "gl_changes_singular_values": bool(sv_change_gl > 1e-3),
        "base_subspace_angle_rad": round(base_ang, 6),
        "angle_after_orthogonal_rad": round(orth_ang, 6),
        "angle_after_general_GL_rad": round(gl_ang, 6),
        "orthogonal_preserves_pairwise_angle": bool(abs(orth_ang - base_ang) < 1e-8),
        "gl_changes_pairwise_angle": bool(abs(gl_ang - base_ang) > 1e-3),
        "synthetic": True,
        "allowed_claim": "attacker estimates a masked-coordinate trajectory, not the "
                         "plaintext subspace.",
    }


def _orth_apply(Q, U):
    """Apply orthogonal Q to subspace U and re-orthonormalize the basis."""
    qu, _ = np.linalg.qr(Q.T @ U)
    return qu


def _gl_apply(N, U):
    qu, _ = np.linalg.qr(N.T @ U)
    return qu


# ---------------------------------------------------------------------------
# F4 — compensation second-moment convergence
#   C_k = T_k Delta N_out, T_k ~ N(0, sigma_T^2). E[(1/K) sum C_k^T C_k]
#   -> m sigma_T^2 N_out^T Delta^T Delta N_out
# ---------------------------------------------------------------------------
def f4_compensation(m=8, d=12, d_out=12, rank_delta=6, sigma_T=1.0, seed=0) -> list[dict]:
    g = _rng(seed)
    # low-rank Delta
    Delta = (g.standard_normal((d, rank_delta)) @ g.standard_normal((rank_delta, d_out)))
    rows = []
    for family, Nout in (("orthogonal", _orth(d_out, g)),
                         ("dense_non_orthogonal", _gl(d_out, g, cond=8.0)),
                         ("diagonal", np.diag(np.exp(g.uniform(-1, 1, d_out))))):
        theory = m * sigma_T ** 2 * (Nout.T @ Delta.T @ Delta @ Nout)
        DtD = Delta.T @ Delta
        for K in (50, 200, 1000):
            acc = np.zeros((d_out, d_out))
            for _ in range(K):
                T = g.standard_normal((m, d)) * sigma_T
                C = T @ Delta @ Nout
                acc += C.T @ C
            emp = acc / K
            est_err = float(np.linalg.norm(emp - theory) / (np.linalg.norm(theory) + 1e-30))
            # spectral preservation: eigenvalues of masked Gram vs plaintext Delta^T Delta
            ev_masked = np.sort(np.linalg.eigvalsh(Nout.T @ DtD @ Nout))[::-1]
            ev_plain = np.sort(np.linalg.eigvalsh(DtD))[::-1]
            spec_preserved = float(np.linalg.norm(ev_masked - ev_plain) /
                                   (np.linalg.norm(ev_plain) + 1e-30))
            rows.append({
                "n_out_family": family, "K": K,
                "estimator_rel_error": round(est_err, 4),
                "spectral_preservation_error": round(spec_preserved, 4),
                "spectrum_exactly_preserved": bool(spec_preserved < 1e-8),
                "rank_masked_gram": int(np.linalg.matrix_rank(Nout.T @ DtD @ Nout)),
                "rank_plain_gram": int(np.linalg.matrix_rank(DtD)),
                "synthetic": True,
            })
    return rows


# ---------------------------------------------------------------------------
# F5 — one-view vs multi-view (EXPLORATORY, not a theorem)
# ---------------------------------------------------------------------------
def f5_multiview(d=12, d_out=10, seed=0) -> list[dict]:
    g = _rng(seed)
    W = g.standard_normal((d, d_out)) / d ** 0.5
    rows = []
    for n_views in (1, 2, 4, 8):
        views = []
        for _ in range(n_views):
            N = _gl(d, g, cond=6.0); Nout = _gl(d_out, g, cond=6.0)
            views.append(np.linalg.inv(N) @ W @ Nout)     # masked view N^-1 W Nout
        # attacker: no plaintext calibration -> average of masked views (a naive estimator)
        est = np.mean(views, axis=0)
        cos = float((est.ravel() @ W.ravel()) /
                    (np.linalg.norm(est) * np.linalg.norm(W) + 1e-30))
        recon = float(np.linalg.norm(est - W) / (np.linalg.norm(W) + 1e-30))
        # functional error on random probes
        X = g.standard_normal((16, d))
        func = float(np.linalg.norm(X @ est - X @ W) / (np.linalg.norm(X @ W) + 1e-30))
        rows.append({
            "n_views": n_views, "param_cosine": round(cos, 4),
            "reconstruction_rel_error": round(recon, 4),
            "functional_output_rel_error": round(func, 4),
            "solution_multiplicity": "high (fresh masks per view leave rotational ambiguity)",
            "synthetic": True,
            "disclaimer": "This experiment evaluates a hypothesis; it does not prove that "
                          "one view is always safer than multiple views.",
        })
    return rows


# ---------------------------------------------------------------------------
# F6 — packed-buffer metadata / structural leakage
# ---------------------------------------------------------------------------
def f6_packed_buffer(layers=8, r=4, d=16, steps=6, seed=0) -> dict:
    g = _rng(seed)
    # a packed message reveals: layer count, per-layer shapes/ranks, per-layer grad norms
    # simple step-index classifier from decaying per-layer grad norms
    feats, labels = [], []
    for step in range(steps):
        scale = 1.0 / (1 + step)                      # grads shrink as training progresses
        norms = np.abs(g.standard_normal(layers)) * scale
        feats.append(norms); labels.append(step)
    feats = np.array(feats); labels = np.array(labels)
    # nearest-centroid step classifier (leave-one-out)
    correct = 0
    for i in range(steps):
        d2 = [np.linalg.norm(feats[i] - feats[j]) if j != i else np.inf for j in range(steps)]
        correct += int(labels[int(np.argmin(d2))] == labels[i] or
                       abs(labels[int(np.argmin(d2))] - labels[i]) <= 1)
    return {
        "structural_fields_exposed": ["layer_count", "per_layer_rank", "per_layer_shape",
                                      "target_module_type", "per_layer_grad_norm"],
        "layer_count_visible": layers,
        "step_index_classifier_neighbor_acc": round(correct / steps, 3),
        "synthetic": True,
        "note": "metadata/structural audit only: packing reveals layer count, ranks, module "
                "types and per-layer grad norms; training progress is inferable from grad-norm decay.",
    }


def run_security(seed=0) -> dict:
    return {
        "F1_cross_gram": f1_cross_gram(),
        "F2_value_multiset": f2_value_multiset(seed=seed),
        "F3_subspace_trajectory": f3_subspace(seed=seed),
        "F4_compensation_second_moment": f4_compensation(seed=seed),
        "F5_one_vs_multi_view": f5_multiview(seed=seed),
        "F6_packed_buffer": f6_packed_buffer(seed=seed),
    }
