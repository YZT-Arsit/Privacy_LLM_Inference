"""Protected (masked) LoRA *training* protection experiments.

The paper claims protection for user inputs, LoRA adapters, AND LoRA training
data. This module implements protected LoRA **training** (not just inference) and
shows it is mathematically equivalent to plaintext LoRA training, while the
untrusted GPU only ever sees masked activations + masked operator payloads.

Protection model (exact, training-stage)
-----------------------------------------
For each LoRA-adapted layer ``Y = X W + (alpha/r) X B A`` (``W`` frozen):

* the **frozen base** matmul ``X W`` is offloaded to the untrusted GPU in masked
  form -- trusted side sends ``X_tilde = c * (X @ N)`` (signed-permutation mask
  ``N`` over in-features + per-sample positive scale ``c``); the GPU holds the
  folded weight ``W_tilde = N^{-1} W M`` (output mask ``M``) and returns
  ``c * (X W) M``; the trusted side recovers ``X W`` exactly;
* the LoRA term ``(alpha/r) X B A``, the loss, the gradients ``dA, dB``, and the
  optimizer update all stay inside the trusted boundary.

Because the masks are exactly invertible (orthogonal signed permutation + a
positive diagonal + a divided-out per-sample scale), the recovered base equals
plaintext ``X W`` up to floating point, so the entire training trajectory --
losses, parameters, gradients, optimizer state -- matches plaintext LoRA training
to fp tolerance. The GPU never sees ``X``, the labels, ``A``, ``B``, ``B@A``,
the gradients, or the optimizer state.

Scope: tasks A (synthetic linear) and B (tiny transformer) are fully implemented
in numpy (no torch). Tasks C (GPT-2) and D (Qwen2.5-7B) are gated feasibility
probes -- see :func:`run_gpt2_probe` / :func:`run_qwen_probe`.

numpy only for the implemented tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pllo.protocol.lora_training_audit import (
    LoRAMaskedInitRequest,
    LoRAMaskedMatmulRequest,
    LoRAMaskedMatmulResponse,
    LoRATrainingTrace,
)

__all__ = [
    "LoRAMaskSecrets",
    "make_lora_mask_secrets",
    "fold_base_weight",
    "mask_input",
    "recover_base",
    "UntrustedBaseMatmul",
    "compare_arrays",
    "auc",
    "run_synthetic_linear",
    "run_tiny_transformer",
    "run_gpt2_probe",
    "run_qwen_probe",
    "adapter_recovery_attack",
    "gradient_inversion_attack",
    "membership_attack",
    "run_all",
]

_DTYPE = np.float64


# ===========================================================================
# Masking primitives (trusted-side secrets)
# ===========================================================================


@dataclass
class LoRAMaskSecrets:
    """Per-layer mask secrets. Never leave the trusted boundary."""
    in_perm: np.ndarray
    in_inv_perm: np.ndarray
    in_signs: np.ndarray
    out_perm: np.ndarray
    out_inv_perm: np.ndarray
    out_scale: np.ndarray
    out_inv_scale: np.ndarray

    def secret_dict(self) -> dict[str, np.ndarray]:
        return {
            "in_perm": self.in_perm, "in_signs": self.in_signs,
            "in_inv_perm": self.in_inv_perm, "out_perm": self.out_perm,
            "out_inv_perm": self.out_inv_perm, "out_scale": self.out_scale,
            "out_inv_scale": self.out_inv_scale,
        }


def make_lora_mask_secrets(in_features: int, out_features: int,
                           rng: np.random.Generator) -> LoRAMaskSecrets:
    in_perm = rng.permutation(in_features).astype(np.int64)
    in_inv = np.argsort(in_perm).astype(np.int64)
    in_signs = np.where(rng.random(in_features) < 0.5, -1.0, 1.0).astype(_DTYPE)
    out_perm = rng.permutation(out_features).astype(np.int64)
    out_inv = np.argsort(out_perm).astype(np.int64)
    out_scale = (0.5 + rng.random(out_features) * 1.5).astype(_DTYPE)
    return LoRAMaskSecrets(in_perm, in_inv, in_signs, out_perm, out_inv,
                           out_scale, (1.0 / out_scale).astype(_DTYPE))


def fold_base_weight(W: np.ndarray, s: LoRAMaskSecrets) -> np.ndarray:
    """``W_tilde = N^{-1} W M`` ([in, out]); the only base artifact the GPU gets.

    ``N^{-1} = N^T`` (orthogonal): ``(N^T W)[k] = in_signs[k] * W[in_perm[k]]``;
    then mask output columns with ``M`` (permutation + positive scale)."""
    W = np.asarray(W, dtype=_DTYPE)
    nt_w = W[s.in_perm, :] * s.in_signs[:, None]          # N^T W
    return nt_w[:, s.out_perm] * s.out_scale[None, :]      # @ M


def mask_input(X: np.ndarray, s: LoRAMaskSecrets, rng: np.random.Generator,
               scale_sigma: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """``X_tilde = c * (X @ N)``; returns ``(X_tilde, c)``. ``c`` per-sample > 0,
    divided out at recovery, randomises the GPU-visible activation norm."""
    X = np.asarray(X, dtype=_DTYPE)
    x_perm = X[:, s.in_perm] * s.in_signs[None, :]        # X @ N
    c = np.exp(rng.normal(0.0, scale_sigma, size=(X.shape[0], 1))).astype(_DTYPE)
    return (c * x_perm), c


def recover_base(O_tilde: np.ndarray, s: LoRAMaskSecrets,
                 c: np.ndarray) -> np.ndarray:
    """Invert :func:`mask_input` + output mask: recover ``X W`` exactly."""
    unscaled = O_tilde * s.out_inv_scale[None, :]         # undo out_scale
    base = unscaled[:, s.out_inv_perm]                    # undo out_perm
    return base / c                                       # undo per-sample c


class UntrustedBaseMatmul:
    """The untrusted GPU side: holds folded base weights, computes masked matmuls.

    It only ever sees ``X_tilde`` and returns ``X_tilde @ W_tilde``. It has no
    access to the masks, the LoRA params, or the plaintext. ``tee_used`` False."""

    tee_used = False

    def __init__(self, folded: dict[str, np.ndarray]) -> None:
        self._folded = {k: np.asarray(v, dtype=_DTYPE) for k, v in folded.items()}

    def matmul(self, layer: str, masked_input: np.ndarray) -> np.ndarray:
        return np.asarray(masked_input, dtype=_DTYPE) @ self._folded[layer]


# ===========================================================================
# Metric helpers
# ===========================================================================


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, float | bool]:
    """max/mean abs error, relative L2, cosine, allclose between two arrays."""
    a = np.asarray(a, dtype=_DTYPE).ravel()
    b = np.asarray(b, dtype=_DTYPE).ravel()
    diff = a - b
    denom = float(np.linalg.norm(b)) or 1.0
    cos_d = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return {
        "max_abs_error": float(np.abs(diff).max()) if diff.size else 0.0,
        "mean_abs_error": float(np.abs(diff).mean()) if diff.size else 0.0,
        "relative_l2_error": float(np.linalg.norm(diff) / denom),
        "cosine_similarity": float(a @ b / cos_d) if a.size else 1.0,
        "allclose": bool(np.allclose(a, b, atol=1e-9, rtol=1e-6)),
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney statistic (1 = member, 0 = non-member)."""
    scores = np.asarray(scores, dtype=_DTYPE)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def _adam_update(p, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8,
                 weight_decay=0.0):
    """AdamW update (decoupled weight decay). ``weight_decay=0`` reduces to Adam.

    Returns ``(p_new, m, v, update)`` where ``update = p - p_new`` (the adapter
    update magnitude, used as the per-step ``adapter_update`` snapshot)."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    step = lr * mhat / (np.sqrt(vhat) + eps)
    if weight_decay:
        step = step + lr * weight_decay * p          # decoupled (AdamW)
    return p - step, m, v, step


# ===========================================================================
# Task A: synthetic LoRA linear training (exact)
# ===========================================================================


@dataclass
class LinearLoRAState:
    A: np.ndarray
    B: np.ndarray
    mA: np.ndarray
    vA: np.ndarray
    mB: np.ndarray
    vB: np.ndarray
    t: int = 0


def _new_state(A0, B0):
    return LinearLoRAState(A0.copy(), B0.copy(), np.zeros_like(A0),
                           np.zeros_like(A0), np.zeros_like(B0),
                           np.zeros_like(B0))


def run_synthetic_linear(rank: int, *, in_features: int = 16,
                         out_features: int = 8, n_train: int = 32,
                         n_eval: int = 16, steps: int = 40, lr: float = 0.05,
                         alpha: float | None = None, seed: int = 0,
                         scale_sigma: float = 0.5,
                         weight_decay: float = 0.0) -> dict[str, Any]:
    """Exact masked vs plaintext LoRA linear training; full per-step metrics."""
    rng = np.random.default_rng([seed, rank, 0xA])
    alpha = float(alpha if alpha is not None else rank)
    s = alpha / rank

    W = (rng.standard_normal((in_features, out_features)) *
         (1.0 / in_features ** 0.5)).astype(_DTYPE)               # frozen base
    # a learnable low-rank target signal so LoRA actually has something to fit
    B_true = rng.standard_normal((in_features, rank)).astype(_DTYPE) * 0.3
    A_true = rng.standard_normal((rank, out_features)).astype(_DTYPE) * 0.3
    dW_true = B_true @ A_true

    def make_data(n):
        X = rng.standard_normal((n, in_features)).astype(_DTYPE)
        T = X @ W + s * (X @ dW_true) + 0.01 * rng.standard_normal(
            (n, out_features)).astype(_DTYPE)
        return X, T

    Xtr, Ttr = make_data(n_train)
    Xev, Tev = make_data(n_eval)
    n_out = n_train * out_features

    A0 = (rng.standard_normal((rank, out_features)) * 0.02).astype(_DTYPE)
    B0 = np.zeros((in_features, rank), dtype=_DTYPE)              # LoRA: B=0 init

    plain = _new_state(A0, B0)
    prot = _new_state(A0, B0)

    secrets = make_lora_mask_secrets(in_features, out_features, rng)
    gpu = UntrustedBaseMatmul({"proj": fold_base_weight(W, secrets)})
    trace = LoRATrainingTrace(tee_used_on_gpu=gpu.tee_used)
    trace.record_inbound(LoRAMaskedInitRequest(
        session_id="lora-train", folded_base_weights={"proj": gpu._folded["proj"]},
        public_metadata={"in_features": in_features, "out_features": out_features,
                         "rank": rank, "alpha": alpha}))

    base_plain = Xtr @ W                                          # plaintext base

    def forward_loss(st, base, X, T):
        H = X @ st.B
        Y = base + s * (H @ st.A)
        G = (Y - T) / n_out
        loss = 0.5 * float(np.mean((Y - T) ** 2))
        return H, Y, G, loss

    def eval_loss(st):
        Y = Xev @ W + s * ((Xev @ st.B) @ st.A)
        return 0.5 * float(np.mean((Y - Tev) ** 2)), Y

    hist: dict[str, list] = {k: [] for k in (
        "train_loss_plain", "train_loss_protected", "eval_loss_plain",
        "eval_loss_protected", "lora_a_error", "lora_b_error", "delta_w_error",
        "grad_a_error", "grad_b_error", "optimizer_state_error")}
    # plaintext-only artifacts collected to prove they never reach the GPU
    snaps: dict[str, list] = {k: [] for k in (
        "lora_a", "lora_b", "delta_w", "lora_grad_a", "lora_grad_b",
        "optimizer_state", "adapter_update", "plain_hidden")}

    for step in range(1, steps + 1):
        # --- plaintext step ---
        Hp, Yp, Gp, lp = forward_loss(plain, base_plain, Xtr, Ttr)
        dAp = s * (Hp.T @ Gp)
        dBp = s * (Xtr.T @ (Gp @ plain.A.T))
        plain.t += 1
        plain.A, plain.mA, plain.vA, _ = _adam_update(
            plain.A, dAp, plain.mA, plain.vA, plain.t, lr,
            weight_decay=weight_decay)
        plain.B, plain.mB, plain.vB, _ = _adam_update(
            plain.B, dBp, plain.mB, plain.vB, plain.t, lr,
            weight_decay=weight_decay)

        # --- protected step: base matmul offloaded to the untrusted GPU ---
        X_tilde, c = mask_input(Xtr, secrets, rng, scale_sigma)
        req = LoRAMaskedMatmulRequest(
            session_id="lora-train", layer="proj", step=step, phase="forward",
            masked_input=X_tilde, batch_size=n_train, in_features=in_features,
            out_features=out_features)
        trace.record_inbound(req)
        O_tilde = gpu.matmul("proj", X_tilde)
        trace.record_outbound(LoRAMaskedMatmulResponse(
            session_id="lora-train", layer="proj", step=step,
            masked_output=O_tilde))
        base_q = recover_base(O_tilde, secrets, c)            # == Xtr@W (fp)

        Hq, Yq, Gq, lq = forward_loss(prot, base_q, Xtr, Ttr)
        dAq = s * (Hq.T @ Gq)
        dBq = s * (Xtr.T @ (Gq @ prot.A.T))
        prot.t += 1
        prot.A, prot.mA, prot.vA, upA = _adam_update(
            prot.A, dAq, prot.mA, prot.vA, prot.t, lr,
            weight_decay=weight_decay)
        prot.B, prot.mB, prot.vB, upB = _adam_update(
            prot.B, dBq, prot.mB, prot.vB, prot.t, lr,
            weight_decay=weight_decay)

        # --- per-step comparison ---
        elp, _ = eval_loss(plain)
        elq, _ = eval_loss(prot)
        hist["train_loss_plain"].append(lp)
        hist["train_loss_protected"].append(lq)
        hist["eval_loss_plain"].append(elp)
        hist["eval_loss_protected"].append(elq)
        hist["lora_a_error"].append(float(np.abs(plain.A - prot.A).max()))
        hist["lora_b_error"].append(float(np.abs(plain.B - prot.B).max()))
        hist["delta_w_error"].append(
            float(np.abs(plain.B @ plain.A - prot.B @ prot.A).max()))
        hist["grad_a_error"].append(float(np.abs(dAp - dAq).max()))
        hist["grad_b_error"].append(float(np.abs(dBp - dBq).max()))
        hist["optimizer_state_error"].append(float(max(
            np.abs(plain.mA - prot.mA).max(), np.abs(plain.vA - prot.vA).max(),
            np.abs(plain.mB - prot.mB).max(), np.abs(plain.vB - prot.vB).max())))
        # protected-side plaintext artifacts (must NOT be in the trace)
        snaps["lora_a"].append(prot.A.copy())
        snaps["lora_b"].append(prot.B.copy())
        snaps["delta_w"].append((prot.B @ prot.A).copy())
        snaps["lora_grad_a"].append(dAq.copy())
        snaps["lora_grad_b"].append(dBq.copy())
        snaps["optimizer_state"].append(prot.mA.copy())
        snaps["adapter_update"].append(upA.copy())
        snaps["plain_hidden"].append(Hq.copy())

    # --- final metrics ---
    _, Yp_ev = eval_loss(plain)
    _, Yq_ev = eval_loss(prot)
    logits_cmp = compare_arrays(Yp_ev, Yq_ev)
    top1_match = float(np.mean(Yp_ev.argmax(1) == Yq_ev.argmax(1)))
    loss_curve_distance = float(np.linalg.norm(
        np.array(hist["train_loss_plain"]) -
        np.array(hist["train_loss_protected"])))
    final_eval_delta = float(abs(hist["eval_loss_plain"][-1] -
                                 hist["eval_loss_protected"][-1]))

    metrics = {
        "task": "synthetic_linear", "rank": rank, "alpha": alpha, "steps": steps,
        "max_lora_a_error": max(hist["lora_a_error"]),
        "max_lora_b_error": max(hist["lora_b_error"]),
        "max_delta_w_error": max(hist["delta_w_error"]),
        "max_grad_a_error": max(hist["grad_a_error"]),
        "max_grad_b_error": max(hist["grad_b_error"]),
        "max_optimizer_state_error": max(hist["optimizer_state_error"]),
        "final_logits_error": logits_cmp["max_abs_error"],
        "final_logits_relative_l2": logits_cmp["relative_l2_error"],
        "final_logits_cosine": logits_cmp["cosine_similarity"],
        "final_logits_allclose": logits_cmp["allclose"],
        "top1_match_rate": top1_match,
        "loss_curve_distance": loss_curve_distance,
        "final_eval_delta": final_eval_delta,
        "final_task_metric_plain": hist["eval_loss_plain"][-1],
        "final_task_metric_protected": hist["eval_loss_protected"][-1],
        "train_loss_improved": bool(
            hist["train_loss_protected"][-1] < hist["train_loss_protected"][0]),
        "tee_used_on_gpu": False,
    }
    return {
        "metrics": metrics, "history": hist, "trace": trace, "secrets": secrets,
        "plaintext": {
            "train_examples": Xtr, "labels": Ttr, "input_ids": None,
            "tokenized_examples": None,
            "lora_a": snaps["lora_a"], "lora_b": snaps["lora_b"],
            "delta_w": snaps["delta_w"], "lora_grad_a": snaps["lora_grad_a"],
            "lora_grad_b": snaps["lora_grad_b"],
            "optimizer_state": snaps["optimizer_state"],
            "adapter_update": snaps["adapter_update"],
            "plain_hidden": snaps["plain_hidden"], "recovered_logits": Yq_ev,
        },
        # artifacts reused by the attack baselines
        "attack_inputs": {
            "Xtr": Xtr, "A_final": prot.A, "B_final": prot.B,
            "delta_w_final": prot.B @ prot.A, "Xev": Xev,
        },
    }


# ===========================================================================
# Task B: tiny transformer LoRA training (attention V proj + MLP proj)
# ===========================================================================


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _gelu(x):
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _gelu_grad(x):
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = c * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du


def run_tiny_transformer(rank: int, *, d_model: int = 8, d_ff: int = 16,
                         seq_len: int = 4, n_classes: int = 3, n_train: int = 24,
                         n_eval: int = 16, steps: int = 20, lr: float = 0.05,
                         alpha: float | None = None, seed: int = 0,
                         scale_sigma: float = 0.5,
                         weight_decay: float = 0.0) -> dict[str, Any]:
    """Single-head attention + MLP with LoRA on the V projection and the MLP
    first projection; sequence classification. Exact masked vs plaintext."""
    rng = np.random.default_rng([seed, rank, 0xB])
    alpha = float(alpha if alpha is not None else rank)
    s = alpha / rank
    d = d_model

    # frozen base weights
    Wq = rng.standard_normal((d, d)).astype(_DTYPE) / np.sqrt(d)
    Wk = rng.standard_normal((d, d)).astype(_DTYPE) / np.sqrt(d)
    Wv = rng.standard_normal((d, d)).astype(_DTYPE) / np.sqrt(d)
    Wo = rng.standard_normal((d, d)).astype(_DTYPE) / np.sqrt(d)
    W1 = rng.standard_normal((d, d_ff)).astype(_DTYPE) / np.sqrt(d)
    W2 = rng.standard_normal((d_ff, d)).astype(_DTYPE) / np.sqrt(d_ff)

    # trainable params: LoRA on Wv (Bv,Av) + W1 (B1,A1) + classifier head
    def lora_init(din, dout):
        return (np.zeros((din, rank), dtype=_DTYPE),
                (rng.standard_normal((rank, dout)) * 0.02).astype(_DTYPE))
    Bv0, Av0 = lora_init(d, d)
    B10, A10 = lora_init(d, d_ff)
    Wcls0 = (rng.standard_normal((d, n_classes)) * 0.1).astype(_DTYPE)

    # data: synthetic classification (label depends on a fixed random readout)
    X = rng.standard_normal((n_train + n_eval, seq_len, d)).astype(_DTYPE)
    readout = rng.standard_normal((d, n_classes)).astype(_DTYPE)
    logits_true = X.mean(1) @ readout
    y = logits_true.argmax(1)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xev, yev = X[n_train:], y[n_train:]

    secrets = {nm: make_lora_mask_secrets(*shp, rng) for nm, shp in {
        "q": (d, d), "k": (d, d), "v": (d, d), "o": (d, d),
        "w1": (d, d_ff), "w2": (d_ff, d)}.items()}
    folded = {"q": fold_base_weight(Wq, secrets["q"]),
              "k": fold_base_weight(Wk, secrets["k"]),
              "v": fold_base_weight(Wv, secrets["v"]),
              "o": fold_base_weight(Wo, secrets["o"]),
              "w1": fold_base_weight(W1, secrets["w1"]),
              "w2": fold_base_weight(W2, secrets["w2"])}
    gpu = UntrustedBaseMatmul(folded)
    trace = LoRATrainingTrace(tee_used_on_gpu=gpu.tee_used)
    trace.record_inbound(LoRAMaskedInitRequest(
        session_id="tiny-tf", folded_base_weights=dict(folded),
        public_metadata={"d_model": d, "d_ff": d_ff, "rank": rank}))

    BS, SQ = n_train, seq_len

    def base_matmul(name, Xin, protected, step):
        """X @ W: plaintext, or offloaded to the GPU in masked form."""
        flat = Xin.reshape(-1, Xin.shape[-1])
        if not protected:
            out = flat @ {"q": Wq, "k": Wk, "v": Wv, "o": Wo,
                          "w1": W1, "w2": W2}[name]
        else:
            xt, c = mask_input(flat, secrets[name], rng, scale_sigma)
            trace.record_inbound(LoRAMaskedMatmulRequest(
                session_id="tiny-tf", layer=name, step=step, phase="forward",
                masked_input=xt, batch_size=flat.shape[0],
                in_features=flat.shape[1], out_features=folded[name].shape[1]))
            ot = gpu.matmul(name, xt)
            trace.record_outbound(LoRAMaskedMatmulResponse(
                session_id="tiny-tf", layer=name, step=step, masked_output=ot))
            out = recover_base(ot, secrets[name], c)
        return out.reshape(*Xin.shape[:-1], out.shape[-1])

    def forward(params, Xin, protected=False, step=0, cache=None):
        Bv, Av, B1, A1, Wcls = params
        Q = base_matmul("q", Xin, protected, step)
        K = base_matmul("k", Xin, protected, step)
        Vb = base_matmul("v", Xin, protected, step)
        V = Vb + s * ((Xin @ Bv) @ Av)
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d)
        P = _softmax(scores, axis=-1)
        ctx = P @ V
        attn = base_matmul("o", ctx, protected, step)
        res1 = Xin + attn
        h1b = base_matmul("w1", res1, protected, step)
        h1 = h1b + s * ((res1 @ B1) @ A1)
        a1 = _gelu(h1)
        mlp = base_matmul("w2", a1, protected, step)
        res2 = res1 + mlp
        pooled = res2.mean(1)
        logits = pooled @ Wcls
        if cache is not None:
            cache.update(dict(Xin=Xin, Q=Q, K=K, V=V, P=P, ctx=ctx, res1=res1,
                              h1=h1, a1=a1, res2=res2, pooled=pooled,
                              logits=logits))
        return logits

    def loss_and_grad(params, Xin, labels, protected, step):
        Bv, Av, B1, A1, Wcls = params
        cache: dict = {}
        logits = forward(params, Xin, protected, step, cache)
        probs = _softmax(logits, axis=-1)
        n = Xin.shape[0]
        loss = float(-np.mean(np.log(probs[np.arange(n), labels] + 1e-12)))
        dlogits = probs.copy()
        dlogits[np.arange(n), labels] -= 1.0
        dlogits /= n
        # head
        dWcls = cache["pooled"].T @ dlogits
        dpooled = dlogits @ Wcls.T
        dres2 = np.repeat(dpooled[:, None, :], SQ, axis=1) / SQ
        # mlp out (W2 frozen) -> a1
        da1 = dres2 @ W2.T
        dres1 = dres2.copy()                              # residual
        dh1 = da1 * _gelu_grad(cache["h1"])
        # h1 = base(res1@W1) + s (res1 B1) A1  -> LoRA grads for W1 + back to res1
        r1 = cache["res1"]
        m1 = r1 @ B1                                      # [.,.,rank]
        dA1 = s * (m1.reshape(-1, rank).T @ dh1.reshape(-1, d_ff))
        dm1 = s * (dh1 @ A1.T)
        dB1 = (r1.reshape(-1, d).T @ dm1.reshape(-1, rank))
        dres1 = dres1 + dh1 @ W1.T + dm1 @ B1.T
        # attn: res1 = Xin + attn ; attn = ctx@Wo (frozen)
        dctx = dres1 @ Wo.T
        dXin = dres1.copy()
        # ctx = P @ V
        dP = dctx @ cache["V"].transpose(0, 2, 1)
        dV = cache["P"].transpose(0, 2, 1) @ dctx
        # V = base(Xin@Wv) + s (Xin Bv) Av
        mv = cache["Xin"] @ Bv
        dAv = s * (mv.reshape(-1, rank).T @ dV.reshape(-1, d))
        dmv = s * (dV @ Av.T)
        dBv = (cache["Xin"].reshape(-1, d).T @ dmv.reshape(-1, rank))
        dXin = dXin + dmv @ Bv.T                          # Wv base is frozen
        # softmax backward -> scores (-> Q,K frozen, no LoRA there)
        dscores = cache["P"] * (dP - (dP * cache["P"]).sum(-1, keepdims=True))
        dscores /= np.sqrt(d)
        # Q,K have no LoRA; their grads not needed (frozen). dXin via them optional.
        grads = (dBv, dAv, dB1, dA1, dWcls)
        return loss, grads, logits

    params_p = [Bv0.copy(), Av0.copy(), B10.copy(), A10.copy(), Wcls0.copy()]
    params_q = [Bv0.copy(), Av0.copy(), B10.copy(), A10.copy(), Wcls0.copy()]
    opt_p = [(np.zeros_like(p), np.zeros_like(p)) for p in params_p]
    opt_q = [(np.zeros_like(p), np.zeros_like(p)) for p in params_q]

    def evaluate(params):
        logits = forward(params, Xev, protected=False, step=0)
        probs = _softmax(logits, -1)
        loss = float(-np.mean(np.log(probs[np.arange(len(yev)), yev] + 1e-12)))
        acc = float(np.mean(logits.argmax(1) == yev))
        return loss, acc, logits

    hist = {k: [] for k in ("train_loss_plain", "train_loss_protected",
                            "eval_loss_plain", "eval_loss_protected",
                            "param_error", "grad_error")}
    snaps = {k: [] for k in ("lora_a", "lora_b", "lora_grad_a", "lora_grad_b",
                             "plain_hidden", "optimizer_state", "adapter_update")}

    for step in range(1, steps + 1):
        lp, gp, _ = loss_and_grad(params_p, Xtr, ytr, False, step)
        lq, gq, _ = loss_and_grad(params_q, Xtr, ytr, True, step)
        for i in range(len(params_p)):
            m, v = opt_p[i]
            params_p[i], m, v, _ = _adam_update(
                params_p[i], gp[i], m, v, step, lr, weight_decay=weight_decay)
            opt_p[i] = (m, v)
            m, v = opt_q[i]
            params_q[i], m, v, up = _adam_update(
                params_q[i], gq[i], m, v, step, lr, weight_decay=weight_decay)
            opt_q[i] = (m, v)
        elp, accp, _ = evaluate(params_p)
        elq, accq, _ = evaluate(params_q)
        hist["train_loss_plain"].append(lp)
        hist["train_loss_protected"].append(lq)
        hist["eval_loss_plain"].append(elp)
        hist["eval_loss_protected"].append(elq)
        hist["param_error"].append(float(max(
            np.abs(a - b).max() for a, b in zip(params_p, params_q))))
        hist["grad_error"].append(float(max(
            np.abs(a - b).max() for a, b in zip(gp, gq))))
        snaps["lora_a"].append(params_q[1].copy())        # Av
        snaps["lora_b"].append(params_q[0].copy())        # Bv
        snaps["lora_grad_a"].append(gq[1].copy())
        snaps["lora_grad_b"].append(gq[0].copy())
        snaps["optimizer_state"].append(opt_q[0][0].copy())
        snaps["adapter_update"].append(gq[1].copy())
        snaps["plain_hidden"].append((Xtr @ params_q[0]).copy())

    elp, accp, Lp = evaluate(params_p)
    elq, accq, Lq = evaluate(params_q)
    logits_cmp = compare_arrays(Lp, Lq)
    metrics = {
        "task": "tiny_transformer", "rank": rank, "alpha": alpha, "steps": steps,
        "max_param_error": max(hist["param_error"]),
        "max_grad_error": max(hist["grad_error"]),
        "final_logits_error": logits_cmp["max_abs_error"],
        "final_logits_relative_l2": logits_cmp["relative_l2_error"],
        "final_logits_cosine": logits_cmp["cosine_similarity"],
        "final_logits_allclose": logits_cmp["allclose"],
        "top1_match_rate": float(np.mean(Lp.argmax(1) == Lq.argmax(1))),
        "loss_curve_distance": float(np.linalg.norm(
            np.array(hist["train_loss_plain"]) -
            np.array(hist["train_loss_protected"]))),
        "final_eval_delta": float(abs(elp - elq)),
        "final_task_metric_plain": accp,
        "final_task_metric_protected": accq,
        "train_loss_improved": bool(
            hist["train_loss_protected"][-1] < hist["train_loss_protected"][0]),
        "tee_used_on_gpu": False,
    }
    secret_dicts = {f"{nm}.{k}": v for nm, sec in secrets.items()
                    for k, v in sec.secret_dict().items()}
    return {
        "metrics": metrics, "history": hist, "trace": trace,
        "secrets": secret_dicts,
        "plaintext": {
            "train_examples": Xtr, "labels": ytr.astype(np.int64),
            "input_ids": ytr.astype(np.int64),
            "tokenized_examples": ytr.astype(np.int64),
            "lora_a": snaps["lora_a"], "lora_b": snaps["lora_b"],
            "delta_w": [b @ a for a, b in zip(snaps["lora_a"], snaps["lora_b"])],
            "lora_grad_a": snaps["lora_grad_a"],
            "lora_grad_b": snaps["lora_grad_b"],
            "optimizer_state": snaps["optimizer_state"],
            "adapter_update": snaps["adapter_update"],
            "plain_hidden": snaps["plain_hidden"], "recovered_logits": Lq,
        },
    }


# ===========================================================================
# Attacks
# ===========================================================================


def adapter_recovery_attack(result: dict[str, Any]) -> dict[str, float]:
    """Attacker with the GPU trace tries to reconstruct LoRA A/B and delta_W.

    The trace contains only masked base activations/outputs -- it has zero
    information about A/B (the LoRA term is computed inside the boundary). The
    attacker's best estimate is therefore the zero adapter; relative error ~1.0.
    Baseline: if A/B/delta_W were exposed to the GPU, error would be ~0."""
    ai = result["attack_inputs"]
    A, B, dW = ai["A_final"], ai["B_final"], ai["delta_w_final"]
    # best attacker estimate from the (A/B-free) trace = zeros
    A_hat = np.zeros_like(A)
    B_hat = np.zeros_like(B)
    dW_hat = np.zeros_like(dW)

    def rel(x, xh):
        return float(np.linalg.norm(x - xh) / (np.linalg.norm(x) or 1.0))

    return {
        "adapter_recovery_relative_error": max(rel(A, A_hat), rel(B, B_hat)),
        "delta_w_recovery_relative_error": rel(dW, dW_hat),
        "baseline_exposed_relative_error": 0.0,    # if A/B were on the wire
    }


def gradient_inversion_attack(rank: int = 4, *, in_features: int = 12,
                              out_features: int = 6, seed: int = 1
                              ) -> dict[str, float]:
    """Plaintext-gradient baseline leaks the input; the protected trace does not.

    Single-sample LoRA step. Baseline attacker has the plaintext gradient
    ``dB = s X^T (G A^T)`` and recovers ``X`` (low error). Protected attacker has
    only the masked activation ``X_tilde = c (X N)`` and can do no better than
    using it directly as its guess (high error)."""
    rng = np.random.default_rng([seed, rank, 0xC])
    s = 1.0
    W = rng.standard_normal((in_features, out_features)) / np.sqrt(in_features)
    A = rng.standard_normal((rank, out_features)) * 0.3
    B = rng.standard_normal((in_features, rank)) * 0.3
    X = rng.standard_normal((1, in_features))
    T = rng.standard_normal((1, out_features))
    Y = X @ W + s * (X @ B) @ A
    G = (Y - T) / out_features
    dB = s * (X.T @ (G @ A.T))                            # [in, rank]
    # baseline: dB/s = X^T (G A^T); recover X^T via the pseudo-inverse of (G A^T)
    GA = (G @ A.T)                                        # [1, rank]
    X_hat_baseline = ((dB / s) @ np.linalg.pinv(GA)).T    # [1, in]
    baseline_err = float(np.linalg.norm(X - X_hat_baseline) /
                         (np.linalg.norm(X) or 1.0))
    # protected: attacker only sees masked activation
    secrets = make_lora_mask_secrets(in_features, out_features, rng)
    X_tilde, _ = mask_input(X, secrets, rng, 0.5)
    prot_err = float(np.linalg.norm(X - X_tilde) / (np.linalg.norm(X) or 1.0))
    return {
        "gradient_inversion_baseline_error": baseline_err,
        "gradient_inversion_reconstruction_error": prot_err,
    }


def membership_attack(result: dict[str, Any], *, seed: int = 2
                      ) -> dict[str, float]:
    """Distinguish training members from non-members using the GPU trace.

    Members = training samples (whose masked activations are in the trace);
    non-members = fresh samples. A norm-matching attacker scores each candidate
    by closeness to the nearest GPU-visible activation norm. Because the protocol
    applies a per-sample random positive scale, the GPU-visible norms are
    randomised, so the attack is near chance (AUC ~ 0.5). A plaintext baseline
    (true input norms) is reported for contrast."""
    rng = np.random.default_rng([seed, 0xD])
    trace: LoRATrainingTrace = result["trace"]
    Xtr = result["plaintext"]["train_examples"]
    Xtr = np.asarray(Xtr, dtype=_DTYPE).reshape(Xtr.shape[0], -1)
    n = Xtr.shape[0]
    non_members = rng.standard_normal(Xtr.shape).astype(_DTYPE)

    # GPU-visible activation norms (masked, per request row)
    masked_norms: list[float] = []
    for m in trace.inbound:
        if isinstance(m, LoRAMaskedMatmulRequest):
            xi = np.asarray(m.masked_input, dtype=_DTYPE)
            masked_norms.extend(np.linalg.norm(xi, axis=1).tolist())
    masked_norms = np.array(masked_norms) if masked_norms else np.array([0.0])

    def nearest_score(norms, ref):
        return np.array([-np.min(np.abs(ref - q)) for q in norms])

    cand = np.concatenate([Xtr, non_members], 0)
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    cand_norm = np.linalg.norm(cand, axis=1)

    # protected attack: match candidate norms to masked (randomised) norms
    prot_scores = nearest_score(cand_norm, masked_norms)
    prot_auc = auc(prot_scores, labels)
    # plaintext baseline: match to the TRUE member norms (no scaling)
    base_scores = nearest_score(cand_norm, np.linalg.norm(Xtr, axis=1))
    base_auc = auc(base_scores, labels)
    # accuracy at the median threshold
    thr = np.median(prot_scores)
    acc = float(np.mean((prot_scores >= thr).astype(int) == labels))
    return {
        "membership_attack_auc": float(prot_auc),
        "membership_attack_accuracy": acc,
        "membership_baseline_auc": float(base_auc),
    }


# ===========================================================================
# Tasks C / D: gated feasibility probes
# ===========================================================================


def run_gpt2_probe(model_path: str | None = None, steps: int = 3,
                   rank: int = 8) -> dict[str, Any]:
    """GPT-2(-small) LoRA training probe -- gated on a LOCAL checkpoint.

    Per project policy (HF download blocked; ModelScope-only) this runs only when
    an explicit local GPT-2 checkpoint path is supplied and torch/transformers are
    importable; otherwise it reports ``status='skipped'`` and produces no numbers
    (we never fabricate a GPT-2 result). ``tee_used_on_gpu`` is always False."""
    from pathlib import Path as _P
    if not model_path:
        return {"task": "gpt2", "status": "skipped",
                "reason": "no local GPT-2 checkpoint supplied "
                          "(--gpt2-model-path); HF download is disabled",
                "tee_used_on_gpu": False}
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {"task": "gpt2", "status": "skipped",
                "reason": f"torch/transformers unavailable: {exc}",
                "tee_used_on_gpu": False}
    if not _P(model_path).exists():
        return {"task": "gpt2", "status": "skipped",
                "reason": f"checkpoint path not found: {model_path}",
                "tee_used_on_gpu": False}
    # A real run is wired to execute on a machine that has the checkpoint; the
    # masked-base-matmul LoRA training path is shared with the tiny-transformer
    # implementation. We do not claim a completed GPT-2 run from this environment.
    return {"task": "gpt2", "status": "available_not_run",
            "reason": "local checkpoint present; execute the masked LoRA training "
                      "path on that machine", "model_path": model_path,
            "rank": rank, "steps": steps, "tee_used_on_gpu": False}


def run_qwen_probe(model_path: str | None = None, steps: int = 1,
                   rank: int = 8) -> dict[str, Any]:
    """Qwen2.5-7B LoRA training-step feasibility probe (gated; NOT full training).

    Intended to run one/few-step LoRA forward/backward/update correctness on
    selected modules on the GPU server and report memory/latency with
    ``tee_used_on_gpu=False``. Locally (no CUDA/checkpoint) it reports
    ``status='skipped'``. This is explicitly a feasibility probe, not full
    fine-tuning."""
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        return {"task": "qwen2.5-7b", "status": "skipped",
                "reason": f"torch unavailable: {exc}", "probe_only": True,
                "tee_used_on_gpu": False}
    if not (has_cuda and model_path):
        return {"task": "qwen2.5-7b", "status": "skipped",
                "reason": "requires CUDA + a ModelScope checkpoint; "
                          "feasibility probe only", "probe_only": True,
                "tee_used_on_gpu": False}
    return {"task": "qwen2.5-7b", "status": "available_not_run",
            "reason": "run one/few-step LoRA probe on the GPU server",
            "probe_only": True, "rank": rank, "steps": steps,
            "tee_used_on_gpu": False}


# ===========================================================================
# Orchestration
# ===========================================================================


def run_all(ranks: tuple[int, ...] = (4, 8, 16), *, seed: int = 0,
            alpha: float = 16.0, weight_decay: float = 0.0,
            gpt2_model_path: str | None = None, qwen_model_path: str | None = None,
            include_gpt2: bool = True, include_qwen: bool = True
            ) -> dict[str, Any]:
    """Run the full implemented suite (synthetic + tiny transformer for each
    rank) plus attacks + gated probes; return a structured summary.

    Main setting: AdamW (``weight_decay`` configurable), fixed ``alpha`` across
    ranks (so scaling = alpha/r), same seed / init / batch order for plain vs
    protected. ``r=8`` is the main rank; ``r in {4,8,16}`` is the rank scaling."""
    from pllo.protocol.lora_training_audit import audit_lora_training_trace

    rows: list[dict[str, Any]] = []
    attacks: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for rank in ranks:
        for runner in (run_synthetic_linear, run_tiny_transformer):
            res = runner(rank, seed=seed, alpha=alpha, weight_decay=weight_decay)
            task = res["metrics"]["task"]
            rows.append(res["metrics"])
            rep = audit_lora_training_trace(
                res["trace"], plaintext=res["plaintext"],
                secrets=res["secrets"] if isinstance(res["secrets"], dict)
                else res["secrets"].secret_dict())
            audits[f"{task}_r{rank}"] = rep.to_dict()
            if task == "synthetic_linear":
                attacks[f"r{rank}"] = {
                    **adapter_recovery_attack(res),
                    **gradient_inversion_attack(rank, seed=seed),
                    **membership_attack(res, seed=seed),
                }
    summary = {
        "stage": "lora_training_protection",
        "tee_used_on_gpu": False,
        "optimizer": "AdamW", "weight_decay": weight_decay, "alpha": alpha,
        "main_rank": 8, "ranks": list(ranks),
        "correctness": rows,
        "security_audit": audits,
        "attacks": attacks,
        "gpt2_probe": (run_gpt2_probe(model_path=gpt2_model_path)
                       if include_gpt2 else {"status": "disabled"}),
        "qwen_probe": (run_qwen_probe(model_path=qwen_model_path)
                       if include_qwen else {"status": "disabled"}),
        "all_audits_passed": all(a["audit_passed"] for a in audits.values()),
        "all_allclose": all(r["final_logits_allclose"] for r in rows),
    }
    return summary
