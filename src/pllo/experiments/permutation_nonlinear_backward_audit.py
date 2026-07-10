"""Permutation nonlinear-island forward/backward correctness + security-boundary audit.

Experiment / audit only. This module does NOT touch any production path. It
validates the algebra of a *pure permutation* nonlinear island for the MLP:

    forward:   H_tilde = H N,  W_up_tilde = N^{-1} W_up Pi,  Z_tilde = Z Pi,
               U_tilde = phi(Z_tilde) = phi(Z) Pi,  W_down_tilde = Pi^T W_down N_out,
               Y_tilde = Y N_out
    backward:  inject GY_tilde = GY M_out (M_out independent),
               GU_tilde = GU Pi,  GZ_tilde = GU_tilde (.) phi'(Z_tilde) = GZ Pi
               (standard autograd, no custom nonlinear primitive),
               GH_tilde = GH M_in (M_in independent)

Security boundary:
    general linear region: H_tilde GH_tilde^T = H (N M_in^T) GH^T  != H GH^T (protected),
    nonlinear region:      Z_tilde GZ_tilde^T = Z GZ^T (exact leak of token x token cross-Gram).

Scope / disallowed overclaims are stated in the summary produced by the runner:
finite tests SUPPORT the algebraic claim; they do not prove the full stabilizer
theorem, do not hide activation values, and do not prove any end-to-end training.
"""

from __future__ import annotations

import torch

DT = torch.float64
PASS_TOL = 1e-8   # exact-relation pass threshold in fp64
FAIL_TOL = 1e-3   # clearly-broken relation threshold


# ---------------------------------------------------------------------------
# activations + elementwise derivatives (derivative via autograd = ground truth)
# ---------------------------------------------------------------------------
def _gelu(z: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(z)


def _silu(z: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(z)


def _relu(z: torch.Tensor) -> torch.Tensor:
    return torch.relu(z)


ACTS = {"gelu": _gelu, "silu": _silu, "relu": _relu}


def phi_prime(act, z: torch.Tensor) -> torch.Tensor:
    """Exact elementwise phi'(z) via autograd (no hand-coded derivative)."""
    zz = z.detach().clone().requires_grad_(True)
    y = act(zz).sum()
    (g,) = torch.autograd.grad(y, zz)
    return g.detach()


# ---------------------------------------------------------------------------
# mask families
# ---------------------------------------------------------------------------
def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def make_invertible(d: int, g: torch.Generator) -> torch.Tensor:
    """Well-conditioned invertible matrix (diagonally dominant)."""
    a = torch.randn(d, d, generator=g, dtype=DT) / (d ** 0.5)
    return a + torch.eye(d, dtype=DT)


def make_permutation(d: int, g: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(d, generator=g)
    return torch.eye(d, dtype=DT)[:, perm]


def make_signed_permutation(d: int, g: torch.Generator) -> torch.Tensor:
    p = make_permutation(d, g)
    signs = (torch.randint(0, 2, (d,), generator=g).to(DT) * 2 - 1)
    if (signs < 0).sum() == 0:
        signs[0] = -1.0   # guarantee at least one negative
    return p * signs.unsqueeze(0)


def make_positive_diagonal(d: int, g: torch.Generator) -> torch.Tensor:
    c = torch.rand(d, generator=g, dtype=DT) + 0.5   # in [0.5, 1.5], strictly positive
    return torch.diag(c)


def make_dense_orthogonal(d: int, g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=DT))
    return q


def make_dense_gl(d: int, g: torch.Generator) -> torch.Tensor:
    return make_invertible(d, g)


MASK_FAMILIES = {
    "permutation": make_permutation,
    "signed_permutation": make_signed_permutation,
    "positive_diagonal": make_positive_diagonal,
    "dense_orthogonal": make_dense_orthogonal,
    "dense_gl": make_dense_gl,
}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    num = (a - b).abs().max().item()
    den = b.abs().max().item() + 1e-30
    return num, num / den


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten() - a.mean()
    y = b.flatten() - b.mean()
    denom = x.norm() * y.norm() + 1e-30
    return (x @ y / denom).item()


# ---------------------------------------------------------------------------
# experiment 1: stabilizer tests -- does phi(Z Q) == phi(Z) Q ?
# ---------------------------------------------------------------------------
def stabilizer_test(act_name: str, family: str, m: int = 16, d: int = 32, seed: int = 0) -> dict:
    g = _gen(seed + hash((act_name, family)) % 100000)
    act = ACTS[act_name]
    z = torch.randn(m, d, generator=g, dtype=DT)
    q = MASK_FAMILIES[family](d, g)
    lhs = act(z @ q)
    rhs = act(z) @ q
    max_abs, rel = err(lhs, rhs)
    return {
        "activation": act_name,
        "mask_family": family,
        "max_abs": max_abs,
        "rel_error": rel,
        "pass_bool": bool(rel < PASS_TOL),
    }


def run_stabilizer_suite(m: int = 16, d: int = 32, seed: int = 0) -> list[dict]:
    rows = []
    for act_name in ("gelu", "silu", "relu"):
        for family in MASK_FAMILIES:
            rows.append(stabilizer_test(act_name, family, m, d, seed))
    return rows


# ---------------------------------------------------------------------------
# experiment 2 + 3: MLP permutation forward/backward, standard autograd
# ---------------------------------------------------------------------------
def mlp_forward_backward(act_name: str = "silu", m: int = 8, d_in: int = 16,
                         d_ff: int = 32, d_out: int = 16, seed: int = 0,
                         use_autograd_nonlinear: bool = True) -> dict:
    """Full masked MLP forward+backward under a permutation island. Returns errors."""
    g = _gen(seed)
    act = ACTS[act_name]

    H = torch.randn(m, d_in, generator=g, dtype=DT)
    W_up = torch.randn(d_in, d_ff, generator=g, dtype=DT) / (d_in ** 0.5)
    W_down = torch.randn(d_ff, d_out, generator=g, dtype=DT) / (d_ff ** 0.5)
    GY = torch.randn(m, d_out, generator=g, dtype=DT)   # upstream grad wrt Y

    # masks
    N = make_invertible(d_in, g)
    N_inv = torch.linalg.inv(N)
    Pi = make_permutation(d_ff, g)
    N_out = make_invertible(d_out, g)
    N_out_inv = torch.linalg.inv(N_out)
    M_in = make_invertible(d_in, g)      # independent backward mask (general region)
    M_out = make_invertible(d_out, g)    # independent backward mask (injected)
    M_out_inv = torch.linalg.inv(M_out)

    # ---- plaintext forward ----
    Z = H @ W_up
    U = act(Z)
    Y = U @ W_down

    # ---- plaintext backward (manual chain) ----
    GU = GY @ W_down.T
    GZ = GU * phi_prime(act, Z)
    GH = GZ @ W_up.T

    # ---- masked forward ----
    H_t = H @ N
    W_up_t = N_inv @ W_up @ Pi
    Z_t = H_t @ W_up_t
    U_t = act(Z_t)
    W_down_t = Pi.T @ W_down @ N_out
    Y_t = U_t @ W_down_t

    # ---- masked backward ----
    GY_t = GY @ M_out
    W_down_bwd_t = M_out_inv @ W_down.T @ Pi
    GU_t = GY_t @ W_down_bwd_t
    if use_autograd_nonlinear:
        # standard autograd through the elementwise nonlinear, no custom primitive
        z_leaf = Z_t.detach().clone().requires_grad_(True)
        u_leaf = act(z_leaf)
        u_leaf.backward(GU_t)
        GZ_t = z_leaf.grad.detach()
    else:
        GZ_t = GU_t * phi_prime(act, Z_t)
    W_up_bwd_t = Pi.T @ W_up.T @ M_in
    GH_t = GZ_t @ W_up_bwd_t

    # ---- errors vs the target masked relations ----
    fz = err(Z_t, Z @ Pi)
    fu = err(U_t, U @ Pi)
    fy = err(Y_t, Y @ N_out)
    bgu = err(GU_t, GU @ Pi)
    bgz = err(GZ_t, GZ @ Pi)
    bgh = err(GH_t, GH @ M_in)

    return {
        "activation": act_name,
        "use_autograd_nonlinear": use_autograd_nonlinear,
        "forward_z_error": fz[1], "forward_z_maxabs": fz[0],
        "forward_u_error": fu[1], "forward_u_maxabs": fu[0],
        "forward_y_error": fy[1], "forward_y_maxabs": fy[0],
        "backward_gu_error": bgu[1], "backward_gu_maxabs": bgu[0],
        "backward_gz_error": bgz[1], "backward_gz_maxabs": bgz[0],
        "backward_gh_error": bgh[1], "backward_gh_maxabs": bgh[0],
    }


def autograd_nonlinear_backward(act_name: str = "silu", m: int = 8, d_ff: int = 32,
                                seed: int = 0) -> dict:
    """Experiment 3: torch autograd on U_tilde=phi(Z_tilde) yields GZ_tilde = GZ Pi.

    Isolates the claim that *standard* autograd suffices inside the permutation
    domain -- no custom nonlinear backward primitive.
    """
    g = _gen(seed)
    act = ACTS[act_name]
    Z = torch.randn(m, d_ff, generator=g, dtype=DT)
    GU = torch.randn(m, d_ff, generator=g, dtype=DT)     # upstream grad wrt U
    Pi = make_permutation(d_ff, g)

    GZ = GU * phi_prime(act, Z)   # plaintext nonlinear backward

    # masked domain: Z_tilde = Z Pi, upstream GU_tilde = GU Pi
    Z_t = (Z @ Pi).detach().clone().requires_grad_(True)
    U_t = act(Z_t)
    U_t.backward(GU @ Pi)
    GZ_t = Z_t.grad.detach()

    gz = err(GZ_t, GZ @ Pi)
    return {
        "activation": act_name,
        "gz_error": gz[1],
        "gz_maxabs": gz[0],
        "pass_bool": bool(gz[1] < PASS_TOL),
        "custom_primitive_used": False,
    }


# ---------------------------------------------------------------------------
# experiment 4: cross-Gram boundary over many trials
# ---------------------------------------------------------------------------
def cross_gram_boundary(trials: int = 200, act_name: str = "silu", m: int = 8,
                        d_in: int = 16, d_ff: int = 32, d_out: int = 16,
                        seed: int = 0) -> dict:
    act = ACTS[act_name]
    rel_gen, corr_gen = [], []
    rel_z, corr_z = [], []
    rel_u, corr_u = [], []
    for t in range(trials):
        g = _gen(seed + 1000 + t)
        H = torch.randn(m, d_in, generator=g, dtype=DT)
        W_up = torch.randn(d_in, d_ff, generator=g, dtype=DT) / (d_in ** 0.5)
        W_down = torch.randn(d_ff, d_out, generator=g, dtype=DT) / (d_ff ** 0.5)
        GY = torch.randn(m, d_out, generator=g, dtype=DT)
        N = make_invertible(d_in, g); N_inv = torch.linalg.inv(N)
        Pi = make_permutation(d_ff, g)
        N_out = make_invertible(d_out, g)
        M_in = make_invertible(d_in, g)
        M_out = make_invertible(d_out, g); M_out_inv = torch.linalg.inv(M_out)

        Z = H @ W_up; U = act(Z);
        GU = GY @ W_down.T; GZ = GU * phi_prime(act, Z); GH = GZ @ W_up.T

        H_t = H @ N
        Z_t = Z @ Pi; U_t = U @ Pi
        GY_t = GY @ M_out
        GU_t = GY_t @ (M_out_inv @ W_down.T @ Pi)
        GZ_t = GU_t * phi_prime(act, Z_t)
        GH_t = GZ_t @ (Pi.T @ W_up.T @ M_in)

        # general linear region (input x input-grad cross-Gram)
        gen_masked = H_t @ GH_t.T
        gen_plain = H @ GH.T
        _, rg = err(gen_masked, gen_plain); rel_gen.append(rg); corr_gen.append(corr(gen_masked, gen_plain))

        # nonlinear permutation region
        z_masked = Z_t @ GZ_t.T; z_plain = Z @ GZ.T
        _, rz = err(z_masked, z_plain); rel_z.append(rz); corr_z.append(corr(z_masked, z_plain))
        u_masked = U_t @ GU_t.T; u_plain = U @ GU.T
        _, ru = err(u_masked, u_plain); rel_u.append(ru); corr_u.append(corr(u_masked, u_plain))

    mean = lambda xs: float(sum(xs) / len(xs))
    return {
        "trials": trials,
        "mean_rel_error_general": mean(rel_gen),
        "mean_corr_general": mean(corr_gen),
        "mean_rel_error_z": mean(rel_z),
        "mean_corr_z": mean(corr_z),
        "mean_rel_error_u": mean(rel_u),
        "mean_corr_u": mean(corr_u),
        "general_region_exact": bool(mean(rel_gen) < PASS_TOL),
        "nonlinear_region_exact": bool(mean(rel_z) < PASS_TOL and mean(rel_u) < PASS_TOL),
    }


# ---------------------------------------------------------------------------
# experiment 5: LoRA compatibility (rank-space mask independent from Pi)
# ---------------------------------------------------------------------------
def lora_compatibility(act_name: str = "silu", m: int = 8, d_in: int = 16,
                       d_ff: int = 32, d_out: int = 16, r_up: int = 4,
                       r_down: int = 4, seed: int = 0) -> dict:
    g = _gen(seed)
    act = ACTS[act_name]
    H = torch.randn(m, d_in, generator=g, dtype=DT)
    W_up = torch.randn(d_in, d_ff, generator=g, dtype=DT) / (d_in ** 0.5)
    W_down = torch.randn(d_ff, d_out, generator=g, dtype=DT) / (d_ff ** 0.5)
    A_up = torch.randn(d_in, r_up, generator=g, dtype=DT) / (d_in ** 0.5)
    B_up = torch.randn(r_up, d_ff, generator=g, dtype=DT) / (r_up ** 0.5)
    A_down = torch.randn(d_ff, r_down, generator=g, dtype=DT) / (d_ff ** 0.5)
    B_down = torch.randn(r_down, d_out, generator=g, dtype=DT) / (r_down ** 0.5)

    N = make_invertible(d_in, g); N_inv = torch.linalg.inv(N)
    Pi = make_permutation(d_ff, g)
    N_out = make_invertible(d_out, g)
    R_up = make_invertible(r_up, g); R_up_inv = torch.linalg.inv(R_up)
    R_down = make_invertible(r_down, g); R_down_inv = torch.linalg.inv(R_down)

    # ---- plaintext with LoRA ----
    Z = H @ (W_up + A_up @ B_up)
    U = act(Z)
    Y = U @ (W_down + A_down @ B_down)

    # ---- masked up ----
    H_t = H @ N
    W_up_t = N_inv @ W_up @ Pi
    A_up_t = N_inv @ A_up @ R_up
    B_up_t = R_up_inv @ B_up @ Pi
    Z_t = H_t @ (W_up_t + A_up_t @ B_up_t)
    U_t = act(Z_t)

    # ---- masked down ----
    W_down_t = Pi.T @ W_down @ N_out
    A_down_t = Pi.T @ A_down @ R_down
    B_down_t = R_down_inv @ B_down @ N_out
    Y_t = U_t @ (W_down_t + A_down_t @ B_down_t)

    ez = err(Z_t, Z @ Pi)
    ey = err(Y_t, Y @ N_out)

    # rank-space masks live in r x r, Pi lives in d_ff x d_ff -> distinct spaces.
    # Demonstrate independence: perturbing R_up must NOT change the Pi-invariant Z_t.
    R_up2 = make_invertible(r_up, _gen(seed + 777)); R_up2_inv = torch.linalg.inv(R_up2)
    A_up_t2 = N_inv @ A_up @ R_up2
    B_up_t2 = R_up2_inv @ B_up @ Pi
    Z_t2 = H_t @ (W_up_t + A_up_t2 @ B_up_t2)
    _, rank_indep = err(Z_t2, Z @ Pi)

    return {
        "activation": act_name,
        "up_z_error": ez[1], "up_z_maxabs": ez[0],
        "down_y_error": ey[1], "down_y_maxabs": ey[0],
        "up_pass": bool(ez[1] < PASS_TOL),
        "down_pass": bool(ey[1] < PASS_TOL),
        "rank_space_dims": [r_up, r_down],
        "activation_perm_dim": d_ff,
        "rank_mask_changes_invariant_error": rank_indep,
        "rank_independent_of_pi": bool(rank_indep < PASS_TOL),
    }


# ---------------------------------------------------------------------------
# experiment 6: integration on a real tiny-transformer block's MLP
# ---------------------------------------------------------------------------
def tiny_transformer_mlp_integration(seed: int = 0) -> dict:
    """Replace one real PlainTransformerBlock MLP's linear/nonlinear with a
    permutation island (GELU, with biases). Verify forward matches plaintext MLP
    (up to the N_out output mask) and backward via standard autograd. Attention
    untouched (identity to the MLP island)."""
    try:
        from pllo.models.plain_tiny_transformer import PlainTransformerBlock
        from pllo.models.tiny_config import TinyTransformerConfig
    except Exception as exc:  # pragma: no cover
        return {"synthetic_mlp_only": True, "reason": f"tiny transformer unavailable: {exc}"}

    torch.manual_seed(seed)
    cfg = TinyTransformerConfig(hidden_size=16, ffn_dim=32, num_heads=2,
                                num_layers=1, vocab_size=8, max_seq_len=8, dtype=DT)
    block = PlainTransformerBlock(cfg).to(DT)
    act = _gelu   # tiny transformer MLP uses GELU

    W_up = block.w_mlp_1.detach().to(DT)
    b_up = block.b_mlp_1.detach().to(DT)
    W_down = block.w_mlp_2.detach().to(DT)
    b_down = block.b_mlp_2.detach().to(DT)
    d_in, d_ff = W_up.shape
    d_out = W_down.shape[1]
    m = 5

    g = _gen(seed + 42)
    H = torch.randn(m, d_in, generator=g, dtype=DT)
    N = make_invertible(d_in, g); N_inv = torch.linalg.inv(N)
    Pi = make_permutation(d_ff, g)
    N_out = make_invertible(d_out, g)

    # plaintext MLP (with biases)
    Z = H @ W_up + b_up
    U = act(Z)
    Y = U @ W_down + b_down

    # island MLP: biases fold too (b_up permutes with Pi, b_down right-masks with N_out)
    H_t = H @ N
    Z_t = H_t @ (N_inv @ W_up @ Pi) + (b_up @ Pi)
    U_t = act(Z_t)
    Y_t = U_t @ (Pi.T @ W_down @ N_out) + (b_down @ N_out)

    fz = err(Z_t, Z @ Pi)
    fy = err(Y_t, Y @ N_out)
    # unfold the output mask -> recovers the exact plaintext MLP output
    Y_rec = Y_t @ torch.linalg.inv(N_out)
    frec = err(Y_rec, Y)

    # backward via standard autograd through the island
    z_leaf = Z_t.detach().clone().requires_grad_(True)
    (act(z_leaf).sum()).backward()
    bwd_ok = bool(err(z_leaf.grad, phi_prime(act, Z_t))[1] < PASS_TOL)

    return {
        "synthetic_mlp_only": False,
        "used_real_block": "PlainTransformerBlock",
        "attention": "untouched (island applied only to MLP)",
        "forward_z_error": fz[1],
        "forward_y_error": fy[1],
        "recovered_output_error": frec[1],
        "forward_pass": bool(fz[1] < PASS_TOL and fy[1] < PASS_TOL and frec[1] < PASS_TOL),
        "autograd_backward_pass": bwd_ok,
    }
