"""E7: minimal private LoRA update prototype (trusted boundary owns everything
sensitive; the untrusted GPU does only the masked frozen-base matmul).

This is a DEFENSIBLE MINIMAL prototype, not production fine-tuning. On a tiny
synthetic linear task (configurable "target modules" = independent linears) we
update a private LoRA adapter while:

* the training inputs ``X``, labels ``Y*``, raw LoRA ``A``/``B``, gradients
  ``dA``/``dB``, optimizer state, the loss, and the mask secrets all stay on the
  trusted boundary;
* the GPU receives ONLY the masked frozen-base matmul: ``X_tilde = X @ N_in`` ->
  ``(X @ W) @ N_out`` (the folded base ``W_tilde = N_in^{-1} W N_out`` is public);
* the LoRA branch ``scaling * (X @ A) @ B``, the loss, and the gradient/optimizer
  step are computed trusted-side.

The recorded GPU trace is audited with the existing protocol audit
(:func:`pllo.protocol.lora_training_audit.audit_lora_training_trace`) so the
security claims are checked against the EXACT messages that crossed.

numpy + standard library only (no torch / CUDA / checkpoint).
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
    audit_lora_training_trace,
)

__all__ = [
    "DEFAULT_TARGET_MODULES", "ModuleTask", "make_synthetic_tasks",
    "signed_perm", "run_private_lora_training", "LIMITATIONS",
]

DEFAULT_TARGET_MODULES = ("q_proj",)

LIMITATIONS = [
    "Minimal prototype on a tiny synthetic linear task (configurable independent "
    "'target modules'), NOT full Qwen LoRA fine-tuning.",
    "Only the frozen-base matmul is offloaded (masked) to the untrusted GPU; the "
    "LoRA branch, loss, gradients, and optimizer step run trusted-side. A fully "
    "GPU-offloaded masked backward is future work.",
    "Plain SGD, batch size 1, 1-5 steps; no scheduler / weight decay / dropout.",
    "Synthetic regression target (low-rank ground truth) demonstrates the adapter "
    "update affects the output; it is not a language-modeling loss.",
    "No formal cryptographic security is claimed for the masking (signed "
    "permutation + scaling); it prevents the GPU from reading raw A/B / data / "
    "labels / gradients, which is what this prototype audits.",
]


def signed_perm(dim: int, rng: np.random.Generator, scale_low=0.5,
                scale_high=2.0):
    """A signed-permutation-with-scaling mask ``[dim,dim]`` + its inverse.
    Orthogonal-up-to-scaling; used as the frozen-base input/output mask."""
    perm = rng.permutation(dim)
    signs = rng.choice(np.array([-1.0, 1.0]), size=dim)
    scale = rng.uniform(scale_low, scale_high, size=dim)
    m = np.zeros((dim, dim), dtype=np.float64)
    m[perm, np.arange(dim)] = signs * scale
    m_inv = np.linalg.inv(m)
    return m, m_inv


@dataclass
class ModuleTask:
    """One independent synthetic linear 'module': frozen base ``W`` (public),
    a low-rank ground-truth delta, and the private LoRA factors ``A``/``B``."""
    name: str
    W: np.ndarray                 # [in, out] public frozen base
    X: np.ndarray                 # [B, in] private training inputs
    Y: np.ndarray                 # [B, out] private targets (labels)
    A: np.ndarray                 # [in, r] private LoRA
    B: np.ndarray                 # [r, out] private LoRA
    n_in: np.ndarray = field(default=None)       # input mask (secret)
    n_in_inv: np.ndarray = field(default=None)
    n_out: np.ndarray = field(default=None)      # output mask (secret)
    n_out_inv: np.ndarray = field(default=None)
    folded_base: np.ndarray = field(default=None)  # public: N_in^{-1} W N_out


def make_synthetic_tasks(target_modules, *, in_dim=16, out_dim=16, rank=4,
                         batch=1, seed=0):
    """Build one synthetic linear task per target module (deterministic)."""
    rng = np.random.default_rng(seed)
    tasks = {}
    for mi, name in enumerate(target_modules):
        r = np.random.default_rng(seed + 101 * (mi + 1))
        W = r.standard_normal((in_dim, out_dim)) / np.sqrt(in_dim)
        # low-rank ground-truth delta the adapter should learn to approximate
        ga = r.standard_normal((in_dim, rank)) / np.sqrt(in_dim)
        gb = r.standard_normal((rank, out_dim)) / np.sqrt(rank)
        delta_true = ga @ gb
        X = r.standard_normal((batch, in_dim))
        Y = X @ (W + delta_true)
        A = r.standard_normal((in_dim, rank)) / np.sqrt(in_dim)  # standard init
        B = np.zeros((rank, out_dim))                     # B=0 -> dW starts at 0
        n_in, n_in_inv = signed_perm(in_dim, rng)
        n_out, n_out_inv = signed_perm(out_dim, rng)
        folded_base = n_in_inv @ W @ n_out                # public folded base
        tasks[name] = ModuleTask(
            name=name, W=W, X=X, Y=Y, A=A, B=B, n_in=n_in, n_in_inv=n_in_inv,
            n_out=n_out, n_out_inv=n_out_inv, folded_base=folded_base)
    return tasks


def _gpu_masked_base_forward(trace: LoRATrainingTrace, task: ModuleTask,
                             step: int, phase: str) -> np.ndarray:
    """Offload the frozen-base matmul to the (untrusted) GPU in masked form and
    return the recovered plaintext base output (trusted-side). Records the exact
    GPU-channel messages so the audit can verify nothing sensitive crossed."""
    x_tilde = task.X @ task.n_in                          # masked activation
    req = LoRAMaskedMatmulRequest(
        session_id="lora-train", layer=task.name, step=step, phase=phase,
        masked_input=x_tilde, batch_size=int(task.X.shape[0]),
        in_features=int(task.X.shape[1]), out_features=int(task.W.shape[1]))
    trace.record_inbound(req)
    masked_output = x_tilde @ task.folded_base            # GPU compute (masked)
    resp = LoRAMaskedMatmulResponse(
        session_id="lora-train", layer=task.name, step=step,
        masked_output=masked_output)
    trace.record_outbound(resp)
    return masked_output @ task.n_out_inv                 # trusted recovery


def run_private_lora_training(target_modules=DEFAULT_TARGET_MODULES, *,
                              rank=4, alpha=8.0, steps=3, lr=0.05, in_dim=16,
                              out_dim=16, batch=4, seed=0,
                              max_grad_norm=1.0) -> dict:
    """Run the minimal private LoRA update + audit; return the E7 report dict."""
    target_modules = list(target_modules)
    scaling = float(alpha) / float(rank)
    tasks = make_synthetic_tasks(target_modules, in_dim=in_dim, out_dim=out_dim,
                                 rank=rank, batch=batch, seed=seed)
    trace = LoRATrainingTrace(tee_used_on_gpu=False)

    # init: hand the GPU only the PUBLIC folded base weights + public metadata
    trace.record_inbound(LoRAMaskedInitRequest(
        session_id="lora-train",
        folded_base_weights={t.name: t.folded_base for t in tasks.values()},
        public_metadata={"rank": rank, "alpha": alpha,
                         "target_modules": target_modules, "steps": steps}))

    dW_before = {name: (t.A @ t.B).copy() for name, t in tasks.items()}
    grad_snaps: list[np.ndarray] = []

    def total_loss() -> float:
        tot = 0.0
        for t in tasks.values():
            base = _gpu_masked_base_forward(trace, t, -1, "eval")
            y = base + scaling * (t.X @ t.A) @ t.B
            diff = y - t.Y
            tot += float(np.mean(diff * diff))
        return tot / len(tasks)

    loss_before = total_loss()
    for step in range(steps):
        for t in tasks.values():
            base = _gpu_masked_base_forward(trace, t, step, "forward")  # masked
            xa = t.X @ t.A                                  # trusted LoRA branch
            y = base + scaling * xa @ t.B
            n = float(t.X.shape[0])
            dy = (2.0 / n) * (y - t.Y)                      # trusted loss grad
            dB = scaling * (xa.T @ dy)                      # trusted grad
            dA = scaling * (t.X.T @ (dy @ t.B.T))
            # trusted gradient clipping (keeps the bilinear A,B step stable)
            gnorm = float(np.sqrt(np.sum(dA * dA) + np.sum(dB * dB)))
            if max_grad_norm and gnorm > max_grad_norm:
                s = max_grad_norm / (gnorm + 1e-12)
                dA, dB = dA * s, dB * s
            grad_snaps.extend([dA.copy(), dB.copy()])
            t.A = t.A - lr * dA                             # trusted SGD step
            t.B = t.B - lr * dB
    loss_after = total_loss()
    adapter_delta_norm = float(np.sqrt(sum(
        float(np.sum((t.A @ t.B - dW_before[name]) ** 2))
        for name, t in tasks.items())))

    # audit the EXACT recorded GPU trace against the trusted-side artifacts
    plaintext = {
        "train_examples": [t.X for t in tasks.values()],
        "labels": [t.Y for t in tasks.values()],
        "lora_a": [t.A for t in tasks.values()],
        "lora_b": [t.B for t in tasks.values()],
        "lora_grad_a": grad_snaps, "lora_grad_b": grad_snaps,
        "delta_w": [t.A @ t.B for t in tasks.values()],
    }
    secrets = {}
    for t in tasks.values():
        secrets["%s_n_in" % t.name] = t.n_in
        secrets["%s_n_out" % t.name] = t.n_out
        secrets["%s_n_in_inv" % t.name] = t.n_in_inv
        secrets["%s_n_out_inv" % t.name] = t.n_out_inv
    audit = audit_lora_training_trace(trace, plaintext, secrets,
                                      raise_on_fail=False)

    raw_lora_vis = bool(audit.gpu_visible_lora_a or audit.gpu_visible_lora_b
                        or audit.gpu_visible_delta_w)
    plaintext_fields = []
    if audit.gpu_visible_train_examples:
        plaintext_fields.append("train_examples")
    if audit.gpu_visible_labels:
        plaintext_fields.append("labels")
    if audit.gpu_visible_recovered_logits:
        plaintext_fields.append("recovered_logits")
    if audit.gpu_visible_plain_hidden:
        plaintext_fields.append("plain_hidden")

    return {
        "stage": "private_lora_training_probe",
        "task": "tiny_synthetic_linear",
        "training_steps": int(steps), "rank": int(rank), "alpha": float(alpha),
        "scaling": scaling, "target_modules": target_modules,
        "lr": lr, "in_dim": in_dim, "out_dim": out_dim, "batch_size": batch,
        "loss_before": loss_before, "loss_after": loss_after,
        "loss_decreased": bool(loss_after < loss_before),
        "adapter_delta_norm": adapter_delta_norm,
        "raw_lora_visible_to_gpu": raw_lora_vis,
        "optimizer_state_visible_to_gpu": bool(
            audit.gpu_visible_optimizer_state),
        "training_data_visible_to_gpu": bool(audit.gpu_visible_train_examples),
        "labels_visible_to_gpu": bool(audit.gpu_visible_labels),
        "gradients_visible_to_gpu": bool(audit.gpu_visible_lora_grad_a
                                         or audit.gpu_visible_lora_grad_b),
        "worker_has_mask_secrets": bool(audit.leaked_secret_fields),
        "tee_used_on_gpu": bool(audit.tee_used_on_gpu),
        "gpu_visible_plaintext_fields": plaintext_fields,
        "leaked_secret_fields": audit.leaked_secret_fields,
        "forbidden_field_names": audit.forbidden_field_names,
        "audit_passed": bool(audit.audit_passed),
        "gpu_calls": trace.gpu_calls, "gpu_bytes": trace.gpu_bytes,
        "limitations": LIMITATIONS,
    }
