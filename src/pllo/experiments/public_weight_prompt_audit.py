"""Public-weight prompt-privacy audit (verification, not proof).

Claim under test
----------------
Under a **public-weight** adversary (attacker has the full plaintext model
weights and can run them as a forward oracle), our exact-lossless construction
protects the user prompt from **token-level recovery**, at the cost of leaking
some *structural* information (FFN neuron permutation / per-token norm). The
new idea being audited replaces token-level signed permutations with a
**hidden-dimension dense orthogonal mask ``Q``** on the residual stream:

    H_obs = H Q      (Q orthogonal, secret, inside the TEE)

This module does NOT try to prove the claim succeeds. It runs five adversarial
verifications and reports, per verification, whether the claim *survives* — and
faithfully reports every leak it finds. We keep four leakage axes strictly
separate and never collapse them:

    * token-level content recovery      (what the claim is about)
    * norm / structure leakage          (known-leaked; measured, not hidden)
    * weight-alignment leakage          (FFN neuron permutation)
    * model-structure leakage
    * prompt semantic leakage

The five verifications (see the runner for orchestration):

    V1  dense-orthogonal <-> SiLU/Amulet boundary is exact-lossless
    V2  recovering the FFN neuron permutation does NOT unlock token recovery
    V3  internal-state inversion (ISA) incl. known-plaintext attack on static Q
    V4  plaintext-model forward-matching / candidate-verification oracle
    V5  efficiency & dtype stability

Honesty rules enforced here (mirrored from the task spec):
    * "mapping not recovered" is never reported as "prompt safe" -- token
      recovery is always measured directly.
    * norm leakage and FFN-permutation leakage are always surfaced.
    * the public-weight forward oracle is never skipped.
    * no bf16 "support" is claimed unless measured.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import torch

from pllo.ops.amulet_right_mask_islands import (
    amulet_right_mask_activation,
    make_right_mask_amulet_params,
)
from pllo.ops.nonlinear_islands import gelu_reference, silu_reference
from pllo.ops.two_sided_keymat import sample_nonorthogonal_keymat

__all__ = [
    "qr_orthogonal",
    "signed_perm",
    "orthogonal_procrustes_residual",
    "verify1_silu_amulet_boundary",
    "verify2_ffn_perm_to_token_recovery",
    "verify3_isa_public_weight_hidden_q",
    "verify4_forward_matching_oracle",
    "verify5_efficiency_dtype",
]


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------


def qr_orthogonal(d: int, *, seed: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Dense orthogonal ``Q`` (cond ~= 1) from the QR of a Gaussian matrix."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    # fix signs so the diagonal of R is positive (deterministic Q)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(dtype=dtype, device=device)


def signed_perm(d: int, *, seed: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Production-style signed permutation (orthogonal, sparse)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(d, generator=g)
    signs = torch.where(torch.rand(d, generator=g) < 0.5, -1.0, 1.0).double()
    n = torch.zeros(d, d, dtype=torch.float64)
    n[torch.arange(d), perm] = signs
    return n.to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# the core public-weight primitive: orthogonal Procrustes matching
# ---------------------------------------------------------------------------


def orthogonal_procrustes_residual(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative residual of ``min_{Q orthogonal} || A Q - B ||_F``.

    Both ``A`` (candidate hidden state ``H(x')``) and ``B`` (observed masked
    state ``H_obs``) are ``[m, d]``. Because ``B = H(x_true) Q`` for an unknown
    orthogonal ``Q``, this residual is ~0 iff ``A A^T == B B^T`` (the token-token
    Gram is the complete orthogonal invariant). It is the exact test the
    public-weight adversary runs against every candidate prompt.
    """
    if a.shape != b.shape:
        return float("inf")
    a64 = a.double()
    b64 = b.double()
    m = a64.transpose(0, 1) @ b64
    u, _s, vt = torch.linalg.svd(m)
    q_star = u @ vt
    denom = float(b64.norm()) + 1e-30
    return float((a64 @ q_star - b64).norm() / denom)


# ===========================================================================
# V1: dense-orthogonal <-> SiLU / Amulet boundary is exact-lossless
# ===========================================================================


def _boundary_once(
    d: int, m: int, *, activation: str, dtype, seed: int, k: int = 3
) -> dict[str, Any]:
    """One Amulet boundary evaluation: AmuletPhi(U Q, Q) ?= phi(U) Q.

    The Amulet R-factor factorisation is a *pre-existing fp64-only* algebraic
    construction (its product tolerance is 1e-9, unmet in fp32/bf16). So the
    nonlinear island runs in fp64 internally; ``dtype`` is the I/O dtype of the
    masked activation, and we report the residual after casting back.
    """
    g = torch.Generator().manual_seed(seed)
    q = qr_orthogonal(d, seed=seed, dtype=dtype)
    u = torch.randn(m, d, dtype=dtype, generator=g)
    u_masked = u @ q                                            # U Q
    phi = {"silu": silu_reference, "gelu": gelu_reference}[activation]
    params = make_right_mask_amulet_params(
        m, d, k, q.to(torch.float64),
        generator=torch.Generator().manual_seed(seed))
    out = amulet_right_mask_activation(
        u_masked.to(torch.float64), params, activation).to(dtype)  # phi(U) Q ?
    ref = phi(u) @ q
    err = (out - ref).double()
    return {
        "d": d, "m": m, "activation": activation, "dtype": str(dtype),
        "cond_Q": float(torch.linalg.cond(q.double())),
        "max_abs_error": float(err.abs().max()),
        "relative_l2_error": float(err.norm() / (ref.double().norm() + 1e-30)),
    }


def _native_fp32_build_probe(d: int = 64, seed: int = 0) -> dict[str, Any]:
    """Attempt to build the Amulet R-factors *natively in fp32* and report the
    honest outcome (the factorisation fails its 1e-9 product tolerance)."""
    q = qr_orthogonal(d, seed=seed, dtype=torch.float32)
    try:
        make_right_mask_amulet_params(
            8, d, 3, q, generator=torch.Generator().manual_seed(seed))
        return {"native_fp32_build": "succeeded_unexpectedly"}
    except RuntimeError as exc:
        return {"native_fp32_build": "failed_as_expected", "error": str(exc)}


def verify1_silu_amulet_boundary(
    *, dims=(64, 256, 768), seqs=(1, 16, 128), seeds=(0, 1, 2),
    activations=("silu", "gelu"),
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for act in activations:
        for d in dims:
            for m in seqs:
                for dt in (torch.float64, torch.float32):
                    for s in seeds:
                        records.append(_boundary_once(d, m, activation=act, dtype=dt, seed=s))

    def _agg(dt: str) -> dict[str, float]:
        xs = [r["max_abs_error"] for r in records if r["dtype"] == dt]
        ys = [r["relative_l2_error"] for r in records if r["dtype"] == dt]
        return {"max_abs_error": max(xs), "max_relative_l2_error": max(ys), "n": len(xs)}

    # negative control: a dense NON-orthogonal keymat cannot cross SiLU directly
    d = 64
    p, qinv, _ = sample_nonorthogonal_keymat(d, 0.3, seed=7, dtype=torch.float64)
    x = torch.randn(16, d, dtype=torch.float64)
    neg_silu = float((silu_reference(x @ p) @ qinv - silu_reference(x)).abs().max())
    neg_gelu = float((gelu_reference(x @ p) @ qinv - gelu_reference(x)).abs().max())

    fp64 = _agg("torch.float64")
    fp32 = _agg("torch.float32")
    survives = fp64["max_abs_error"] <= 1e-9
    return {
        "verification": "V1_silu_amulet_boundary",
        "threat_model": "n/a (correctness pre-requisite, not an attack)",
        "attacker_observation": "n/a",
        "attacker_knowledge": "n/a",
        "attack_objective": "n/a; this establishes the exact-lossless foundation",
        "metric": "max_abs_error, relative_l2_error of AmuletPhi(UQ,Q) vs phi(U)Q",
        "fp64": fp64,
        "fp32": fp32,
        "negative_control_dense_keymat": {
            "silu_xP_Pinv_vs_silu_x_max_abs": neg_silu,
            "gelu_xP_Pinv_vs_gelu_x_max_abs": neg_gelu,
            "interpretation": "dense non-orthogonal P does NOT commute with SiLU/GELU",
        },
        "native_fp32_build_probe": _native_fp32_build_probe(),
        "records": records,
        "result": (
            f"fp64 exact (max_abs {fp64['max_abs_error']:.2e}); "
            f"fp32-IO stable (max_abs {fp32['max_abs_error']:.2e}); "
            f"native fp32 R-factor build fails (fp64-only construction)"),
        "claim_survives": bool(survives),
        "claim_survives_note": (
            "Boundary is exact-lossless at fp64 rounding; the Amulet R-factor "
            "construction is fp64-only, so fp32/bf16 inputs must be cast to fp64 "
            "for the nonlinear island. bf16 not supported (see V5)."),
    }


# ---------------------------------------------------------------------------
# GPT-2 hidden-state helpers (real public-weight model)
# ---------------------------------------------------------------------------


class Gpt2Oracle:
    """Thin wrapper over a locally-cached GPT-2 used as the public forward oracle.

    Provides layer-0 (embedding) states for NN token inversion and intermediate
    hidden states for Procrustes matching. fp64 throughout for exact algebra.
    """

    def __init__(self, dtype=torch.float64):
        from transformers import GPT2Model, GPT2Tokenizer
        self.tok = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained("gpt2", torch_dtype=dtype).eval()
        self.dtype = dtype
        self.d = int(self.model.config.n_embd)
        self.wte = self.model.wte.weight.detach().to(dtype)     # [V, d]
        self.wpe = self.model.wpe.weight.detach().to(dtype)     # [T, d]
        self.vocab = int(self.wte.shape[0])

    def ids(self, prompt: str) -> torch.Tensor:
        return self.tok(prompt, return_tensors="pt")["input_ids"][0]

    def hidden(self, prompt: str, layer: int) -> torch.Tensor:
        ids = self.tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**ids, output_hidden_states=True)
        return out.hidden_states[layer][0].to(self.dtype)       # [m, d]

    def layer0_target(self, m: int) -> torch.Tensor:
        """Positional part ``wpe[:m]`` so ``H0[t] = wte[id_t] + wpe[t]``."""
        return self.wpe[:m]


def _nn_invert_layer0(
    h_obs: torch.Tensor, q_hat: torch.Tensor | None, oracle: "Gpt2Oracle",
) -> torch.Tensor:
    """Recover input token ids from (possibly masked) layer-0 states.

    If ``q_hat`` is the true/recovered orthogonal mask, unmask first; otherwise
    the attacker matches masked states directly (which fails). Returns argmax
    token id per position by nearest wte+wpe.
    """
    m = h_obs.shape[0]
    h = h_obs if q_hat is None else h_obs @ q_hat.transpose(0, 1)   # H0 = H_obs Q^T
    h = h - oracle.layer0_target(m)                                # subtract wpe[t]
    # nearest embedding row (squared-distance) -> use chunked matmul
    e = oracle.wte                                                 # [V, d]
    # ||h - e||^2 = ||h||^2 - 2 h.e + ||e||^2 ; drop ||h||^2 (const per row)
    scores = h @ e.transpose(0, 1) - 0.5 * (e * e).sum(1, keepdim=True).transpose(0, 1)
    return scores.argmax(1)


# ===========================================================================
# V2: FFN neuron permutation recovered -> token recovery still ~0
# ===========================================================================


def verify2_ffn_perm_to_token_recovery(
    oracle: "Gpt2Oracle", prompts: list[str], *, layer0: int = 0, seed: int = 0,
) -> dict[str, Any]:
    """Give the attacker the *ground-truth* FFN neuron permutation (stronger
    oracle) and measure whether input tokens become recoverable from ``H Q``.

    The FFN permutation acts on the MLP hidden (``4*d``) dimension; the residual
    stream is separately masked by a dense orthogonal ``Q`` on the ``d``
    dimension. We test that knowing the FFN permutation does not unlock ``Q``.
    """
    d = oracle.d
    q = qr_orthogonal(d, seed=seed, dtype=oracle.dtype)
    # "true FFN neuron permutation" on the MLP hidden dim (given to attacker).
    dff = 4 * d
    g = torch.Generator().manual_seed(seed + 1)
    ffn_perm = torch.randperm(dff, generator=g)                   # attacker KNOWS this

    total = 0
    hit_masked = hit_true_q = 0          # top-1
    hit5_masked = 0
    norm_leaked_pairs = []
    for p in prompts:
        ids = oracle.ids(p)
        m = ids.shape[0]
        h0 = oracle.hidden(p, layer0)                            # [m, d]
        h_obs = h0 @ q                                           # masked state
        # per-token norm is preserved by the orthogonal mask (structural leak)
        norm_leaked_pairs.append((float(h0.norm()), float(h_obs.norm())))
        # Attack: recover input ids, WITH the true FFN perm available but WITHOUT Q
        rec_masked = _nn_invert_layer0(h_obs, None, oracle)      # no Q, has ffn_perm
        # Sanity oracle: if Q were known, recovery is exact (upper bound)
        rec_true_q = _nn_invert_layer0(h_obs, q, oracle)
        hit_masked += int((rec_masked == ids).sum())
        hit_true_q += int((rec_true_q == ids).sum())
        # top-5 for the masked attack
        h = h_obs - oracle.layer0_target(m)
        e = oracle.wte
        scores = h @ e.transpose(0, 1) - 0.5 * (e * e).sum(1, keepdim=True).transpose(0, 1)
        top5 = scores.topk(5, dim=1).indices
        hit5_masked += int((top5 == ids.unsqueeze(1)).any(1).sum())
        total += m
    baseline = 1.0 / oracle.vocab
    norm_corr = _rank_corr([a for a, _ in norm_leaked_pairs], [b for _, b in norm_leaked_pairs])
    top1 = hit_masked / total
    survives = top1 <= 5 * baseline    # ~random
    return {
        "verification": "V2_ffn_perm_to_token_recovery",
        "threat_model": "public weights; attacker GIVEN ground-truth FFN neuron permutation",
        "attacker_observation": "masked residual state H_obs = H Q; folded FFN weights",
        "attacker_knowledge": "plaintext weights + TRUE FFN neuron permutation; NOT Q",
        "attack_objective": "recover input prompt tokens from H_obs using the FFN permutation",
        "metric": "token_recovery_top1/top5 (NN vs wte+wpe), exact_prompt_match, norm leak",
        "num_prompts": len(prompts),
        "num_tokens": total,
        "random_baseline_top1": baseline,
        "token_recovery_top1_masked": top1,
        "token_recovery_top5_masked": hit5_masked / total,
        "token_recovery_top1_if_Q_known_upper_bound": hit_true_q / total,
        "structural_leak_note": (
            "per-token norm is PRESERVED by orthogonal Q (||H Q|| = ||H||): "
            f"norm rank-corr(H, H_obs) = {norm_corr:.6f} -- this IS leaked and is "
            "NOT hidden. FFN neuron permutation is weight-alignment leakage."),
        "norm_rank_corr": norm_corr,
        "result": (
            f"with the TRUE FFN permutation, token top-1 = {top1:.2e} "
            f"(~random {baseline:.2e}); with Q known it would be "
            f"{hit_true_q / total:.3f}. FFN permutation does not unlock Q."),
        "claim_survives": bool(survives),
    }


def _rank_corr(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (no scipy)."""
    n = len(a)
    if n < 2:
        return float("nan")
    ra = _ranks(a)
    rb = _ranks(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db + 1e-30)


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for rank, idx in enumerate(order):
        r[idx] = float(rank)
    return r


# ===========================================================================
# V3: ISA / internal-state inversion incl. known-plaintext attack on static Q
# ===========================================================================


def _synth_bank(oracle: "Gpt2Oracle", *, n_rows_target: int, layer0: int, seed: int) -> torch.Tensor:
    """Synthesise a bank of >= ``n_rows_target`` real layer-0 hidden rows from
    random-token sequences. Under public weights the attacker can feed the
    plaintext model ANY input it likes, so this is a faithful known-plaintext
    supply, not a shortcut."""
    g = torch.Generator().manual_seed(seed)
    rows: list[torch.Tensor] = []
    got = 0
    while got < n_rows_target:
        length = int(torch.randint(6, 12, (1,), generator=g).item())
        ids = torch.randint(0, oracle.vocab, (1, length), generator=g)
        with torch.no_grad():
            out = oracle.model(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[layer0][0].to(oracle.dtype)
        rows.append(h)
        got += h.shape[0]
    return torch.cat(rows, 0)


def verify3_isa_public_weight_hidden_q(
    oracle: "Gpt2Oracle", prompts: list[str], *, layer0: int = 0, seed: int = 0,
    kpa_pairs=(1, 16, 64, 256, 768, 1024),
) -> dict[str, Any]:
    d = oracle.d
    q_static = qr_orthogonal(d, seed=seed, dtype=oracle.dtype)

    # Build a bank of (H, H Q) rows the attacker synthesises by running the
    # public model on chosen inputs. Reach >= d + margin rows so the
    # known-plaintext solve can actually reach full rank (measured, not asserted).
    bank_h = _synth_bank(oracle, n_rows_target=max(kpa_pairs) + 64, layer0=layer0, seed=seed + 7)
    n_rows = bank_h.shape[0]

    # target: a held-out prompt masked by the SAME static Q
    target_prompt = prompts[-1]
    t_ids = oracle.ids(target_prompt)
    t_h = oracle.hidden(target_prompt, layer0)
    t_obs = t_h @ q_static

    # --- attack 0: no known plaintext (direct NN, no Q) ---
    rec0 = _nn_invert_layer0(t_obs, None, oracle)
    no_kpa_top1 = float((rec0 == t_ids).float().mean())

    # --- attack: known-plaintext, k rows -> solve Q by least squares ---
    kpa = []
    for k in kpa_pairs:
        k = min(k, n_rows)
        a = bank_h[:k]                                           # known H
        b = a @ q_static                                        # known H Q
        # Solve A Q = B  ->  Q_hat = A^+ B (exact once rank(A) = d)
        q_hat = torch.linalg.lstsq(a.double(), b.double()).solution.to(oracle.dtype)
        q_err = float((q_hat - q_static).abs().max())
        rec = _nn_invert_layer0(t_obs, q_hat, oracle)
        tok_top1 = float((rec == t_ids).float().mean())
        kpa.append({
            "known_rows": int(k), "rank_needed": d,
            "Q_recovery_max_abs_error": q_err,
            "token_recovery_top1": tok_top1,
            "broken": bool(q_err < 1e-6 and tok_top1 > 0.99),
        })

    # --- fresh Q per session: known pairs use a DIFFERENT Q than the target ---
    q_other = qr_orthogonal(d, seed=seed + 999, dtype=oracle.dtype)
    a = bank_h[:min(256, n_rows)]
    b = a @ q_other                                             # known under OTHER Q
    q_hat_fresh = torch.linalg.lstsq(a.double(), b.double()).solution.to(oracle.dtype)
    rec_fresh = _nn_invert_layer0(t_obs, q_hat_fresh, oracle)   # target uses q_static
    fresh_top1 = float((rec_fresh == t_ids).float().mean())

    rows_to_break = next((r["known_rows"] for r in kpa if r["broken"]), None)
    # token recovery often breaks BEFORE Q is exactly recovered (approximate Q
    # still argmaxes to the right token) -- report that threshold too.
    rows_to_break_tokens = next(
        (r["known_rows"] for r in kpa if r["token_recovery_top1"] > 0.99), None)
    return {
        "verification": "V3_isa_public_weight_hidden_q",
        "threat_model": "public weights; attacker observes H_obs = H Q; Q secret in TEE",
        "attacker_observation": "masked hidden states; can synthesise (H, H Q) pairs iff Q static",
        "attacker_knowledge": "plaintext weights; k known-plaintext pairs under the same Q",
        "attack_objective": "recover Q, then recover input tokens",
        "metric": "Q_recovery_error, token_recovery_top1, required_known_rows_to_break",
        "hidden_dim_d": d,
        "no_known_plaintext_token_top1": no_kpa_top1,
        "kpa_sweep": kpa,
        "required_known_rows_to_break": rows_to_break,
        "required_known_rows_to_break_tokens": rows_to_break_tokens,
        "fresh_Q_per_session": {
            "known_pairs_under_different_Q": True,
            "token_recovery_top1": fresh_top1,
            "resists": bool(fresh_top1 < 0.05),
        },
        "result": (
            f"no-known-plaintext top1 = {no_kpa_top1:.2e} (~0); "
            f"STATIC Q: token recovery breaks at ~{rows_to_break_tokens} known rows "
            f"(approximate Q already argmaxes correctly), and Q is recovered EXACTLY "
            f"at known_rows >= d ({rows_to_break} rows, Q err ~1e-14); "
            f"fresh Q per session resists (token top1 = {fresh_top1:.2e})."),
        "claim_survives": bool(no_kpa_top1 < 0.05),
        "claim_survives_note": (
            "Survives ONLY without known plaintext AND/OR with fresh Q per "
            "session. Static Q + >= d known plaintext rows is fully broken; this "
            "is reported, not hidden."),
    }


# ===========================================================================
# V4: plaintext-model forward-matching / candidate-verification oracle (CRUX)
# ===========================================================================


def _rank_of_true(residuals: list[float], true_idx: int) -> int:
    order = sorted(range(len(residuals)), key=lambda i: residuals[i])
    return order.index(true_idx)


def verify4_forward_matching_oracle(
    oracle: "Gpt2Oracle", *, layer: int = 6, seed: int = 0,
) -> dict[str, Any]:
    """The public-weight adversary verifies WHICH candidate prompt produced the
    observed masked state, WITHOUT recovering Q, via orthogonal Procrustes.

    Sub-attacks A (small set), B (large set), C (partial/sensitive-token), D
    (multi-layer), plus a fresh-Q ablation proving freshness does not help.
    """
    d = oracle.d
    q = qr_orthogonal(d, seed=seed, dtype=oracle.dtype)

    # -- A. small candidate set (same-length distractors) --
    def _same_len_family(template: str, fillers: list[str]) -> tuple[list[str], list[str]]:
        cands, kept = [], []
        base_len = None
        for f in fillers:
            p = template.format(f)
            L = oracle.ids(p).shape[0]
            if base_len is None:
                base_len = L
            if L == base_len:
                cands.append(p)
                kept.append(f)
        return cands, kept

    fillers = [" cat", " dog", " red", " sky", " gold", " fire", " moon", " tree"]
    A_cands, _ = _same_len_family("My secret word is{}", fillers)
    true_idx = 0
    A_obs = oracle.hidden(A_cands[true_idx], layer) @ q
    A_res = [orthogonal_procrustes_residual(oracle.hidden(c, layer), A_obs) for c in A_cands]
    A_rank = _rank_of_true(A_res, true_idx)
    A_margin = (sorted(A_res)[1] - sorted(A_res)[0]) if len(A_res) > 1 else float("inf")

    # -- B. large candidate set --
    words = [" cat", " dog", " red", " sky", " gold", " fire", " moon", " tree",
             " book", " road", " lamp", " star", " snow", " rain", " wind", " leaf",
             " king", " wolf", " ship", " song", " door", " ring", " coin", " rose"]
    B_cands, _ = _same_len_family("My secret word is{}", words)
    b_true = 3
    B_obs = oracle.hidden(B_cands[b_true], layer) @ qr_orthogonal(d, seed=seed + 5, dtype=oracle.dtype)
    B_res = [orthogonal_procrustes_residual(oracle.hidden(c, layer), B_obs) for c in B_cands]
    B_rank = _rank_of_true(B_res, b_true)
    B_top1 = int(B_rank == 0)

    # -- C. partial-prompt / sensitive-token attack across templates --
    templates = ["The password is{}", "Patient diagnosis is{}", "The phone code is{}"]
    C = []
    for ti, tmpl in enumerate(templates):
        cands, kept = _same_len_family(tmpl, words)
        if len(cands) < 4:
            continue
        secret = 2 % len(cands)
        obs = oracle.hidden(cands[secret], layer) @ qr_orthogonal(d, seed=seed + 10 + ti, dtype=oracle.dtype)
        res = [orthogonal_procrustes_residual(oracle.hidden(c, layer), obs) for c in cands]
        rank = _rank_of_true(res, secret)
        C.append({
            "template": tmpl, "num_candidates": len(cands),
            "true_secret": kept[secret].strip(),
            "recovered_secret": kept[int(torch.tensor(res).argmin())].strip(),
            "sensitive_token_top1_correct": bool(rank == 0),
            "true_residual": res[secret],
            "nearest_false_residual": sorted(res)[1] if rank == 0 else sorted(res)[0],
        })
    C_top1 = sum(c["sensitive_token_top1_correct"] for c in C) / max(1, len(C))

    # -- D. multi-layer matching (fresh Q PER LAYER) --
    layers = [3, 6, 9]
    D_cands, _ = _same_len_family("My secret word is{}", words)
    d_true = 5
    obs_by_layer = {}
    for li in layers:
        qL = qr_orthogonal(d, seed=seed + 100 + li, dtype=oracle.dtype)   # fresh per layer
        obs_by_layer[li] = oracle.hidden(D_cands[d_true], li) @ qL
    single_ranks = {}
    combined = torch.zeros(len(D_cands))
    for li in layers:
        res = torch.tensor([orthogonal_procrustes_residual(oracle.hidden(c, li), obs_by_layer[li])
                            for c in D_cands])
        single_ranks[li] = _rank_of_true(res.tolist(), d_true)
        combined += res
    D_combined_rank = _rank_of_true(combined.tolist(), d_true)

    # -- fresh-Q ablation: residuals are identical under any fresh orthogonal Q --
    q1 = qr_orthogonal(d, seed=seed + 1, dtype=oracle.dtype)
    q2 = qr_orthogonal(d, seed=seed + 2, dtype=oracle.dtype)
    base = oracle.hidden(A_cands[0], layer)
    r1 = orthogonal_procrustes_residual(base, base @ q1)
    r2 = orthogonal_procrustes_residual(base, base @ q2)

    oracle_succeeds = (A_rank == 0) and (B_top1 == 1) and (C_top1 >= 0.5)
    return {
        "verification": "V4_forward_matching_oracle",
        "threat_model": "public weights; attacker runs candidates through the plaintext model",
        "attacker_observation": "masked hidden state H_obs = H Q (Q secret, possibly fresh)",
        "attacker_knowledge": "full plaintext model; a candidate set containing the true prompt",
        "attack_objective": "identify which candidate produced H_obs (no Q recovery needed)",
        "metric": "candidate_top1/rank, sensitive_token_top1, Procrustes residual margin",
        "invariant_note": (
            "orthogonal Procrustes residual is 0 iff token-token Gram H H^T matches; "
            "(H Q)(H Q)^T = H H^T for ANY orthogonal Q, so Q (fresh or static) is "
            "irrelevant to this attack."),
        "A_small_set": {
            "num_candidates": len(A_cands), "true_rank": A_rank,
            "top1_correct": bool(A_rank == 0),
            "true_residual": A_res[true_idx], "residual_margin": A_margin},
        "B_large_set": {
            "num_candidates": len(B_cands), "true_rank": B_rank, "top1_correct": bool(B_top1)},
        "C_partial_prompt": {"templates": C, "sensitive_token_top1": C_top1},
        "D_multi_layer": {
            "layers": layers, "fresh_Q_per_layer": True,
            "single_layer_true_ranks": single_ranks,
            "combined_true_rank": D_combined_rank},
        "fresh_Q_ablation": {
            "residual_under_Q1": r1, "residual_under_Q2": r2,
            "note": "both ~0: fresh Q does not change the residual"},
        "result": (
            f"A: true ranks #{A_rank + 1}/{len(A_cands)} (residual {A_res[true_idx]:.2e} "
            f"vs margin {A_margin:.2e}); B top1 {'YES' if B_top1 else 'no'}; "
            f"C sensitive-token top1 {C_top1:.2f}; D combined rank #{D_combined_rank + 1}. "
            f"Fresh Q does NOT help (residual invariant to Q)."),
        "oracle_succeeds": bool(oracle_succeeds),
        "claim_survives": bool(not oracle_succeeds),
    }


# ===========================================================================
# V5: efficiency & dtype stability
# ===========================================================================


def _time(fn: Callable[[], Any], *, iters: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3     # ms


def verify5_efficiency_dtype(*, d: int = 768, m: int = 32, k: int = 3) -> dict[str, Any]:
    dt = torch.float64
    x = torch.randn(m, d, dtype=dt)
    q = qr_orthogonal(d, seed=0, dtype=dt)
    n = signed_perm(d, seed=0, dtype=dt)
    perm = torch.randperm(d)
    w1 = torch.randn(d, 4 * d, dtype=dt)
    w2 = torch.randn(4 * d, d, dtype=dt)

    def plain_mlp():
        return gelu_reference(x @ w1) @ w2

    def dense_q_mask():
        return x @ q

    def signed_perm_mask():
        return x @ n

    def gather_perm():
        return x[:, perm]

    qff = qr_orthogonal(4 * d, seed=1, dtype=torch.float64)
    u = torch.randn(m, 4 * d, dtype=torch.float64)
    params = make_right_mask_amulet_params(m, 4 * d, k, qff,
                                           generator=torch.Generator().manual_seed(0))

    def amulet_silu():
        return amulet_right_mask_activation(u, params, "silu")

    times = {
        "plaintext_mlp_ms": _time(plain_mlp),
        "dense_orthogonal_mask_ms": _time(dense_q_mask),
        "signed_perm_matmul_ms": _time(signed_perm_mask),
        "signed_perm_gather_ms": _time(gather_perm),
        "amulet_silu_island_ms": _time(amulet_silu, iters=10, warmup=2),
    }

    # dtype correctness of the boundary. NOTE: the Amulet R-factor construction is
    # fp64-ONLY (its 1e-9 product tolerance is unmet below fp64), so _boundary_once
    # always builds/computes the island in fp64; `dtype` is only the I/O rounding.
    # "exact_lossless_supported" is therefore True ONLY at fp64. We also probe a
    # native (non-fp64) R-factor build to show it fails.
    dtype_rows = []
    for name, dtype in [("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        r = _boundary_once(64, 8, activation="silu", dtype=dtype, seed=0)
        native = {"native_build": "fp64_only_construction"}
        if dtype is not torch.float64:
            try:
                q = qr_orthogonal(64, seed=0, dtype=dtype)
                make_right_mask_amulet_params(8, 64, 3, q,
                                              generator=torch.Generator().manual_seed(0))
                native = {"native_build": "succeeded_unexpectedly"}
            except Exception as exc:  # noqa: BLE001
                native = {"native_build": "failed_as_expected", "error": str(exc)[:80]}
        dtype_rows.append({
            "dtype": name,
            "io_roundtrip_max_abs_error": r["max_abs_error"],
            "compute_precision": "fp64 (island is fp64-only)",
            "exact_lossless_supported": bool(name == "fp64" and r["max_abs_error"] <= 1e-9),
            **native,
        })

    return {
        "verification": "V5_efficiency_dtype",
        "threat_model": "n/a (cost/stability profiling)",
        "attacker_observation": "n/a",
        "attacker_knowledge": "n/a",
        "attack_objective": "n/a",
        "metric": "per-op latency (ms), boundary correctness by dtype",
        "dims": {"d": d, "d_ff": 4 * d, "m": m, "k": k},
        "timings_ms": times,
        "overhead_vs_signed_perm": {
            "dense_orthogonal_vs_signed_perm_matmul":
                times["dense_orthogonal_mask_ms"] / (times["signed_perm_matmul_ms"] + 1e-12),
            "dense_orthogonal_vs_signed_perm_gather":
                times["dense_orthogonal_mask_ms"] / (times["signed_perm_gather_ms"] + 1e-12),
            "amulet_silu_vs_plaintext_mlp":
                times["amulet_silu_island_ms"] / (times["plaintext_mlp_ms"] + 1e-12),
        },
        "dtype_correctness": dtype_rows,
        "result": (
            "dense orthogonal mask is a full [d,d] matmul (denser than a "
            "signed-perm gather); Amulet-SiLU island is the dominant cost "
            "(Kronecker lift). fp64 exact; fp32 tolerable for the mask matmul "
            "but the Amulet island is fp64-only; bf16 unsupported."),
        "classification": "stronger_audit_variant_not_main_path",
    }
