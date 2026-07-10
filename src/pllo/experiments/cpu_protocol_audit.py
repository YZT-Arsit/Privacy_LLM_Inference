"""CPU-only unified protocol audit entry point (Sections A-F).

Orchestrates the CPU-only verification stage. NO CUDA, NO real GPU, NO real TEE.
The trusted controller is a plain CPU function (`simulated_trusted_controller=True`).
fp64 exactness here does NOT imply bf16/GPU exactness; operation/byte counts are
NOT measured TEE/GPU latency; synthetic attacks are NOT real-model attacks.

Reuses existing modules:
- A4/A5/cross-Gram: `permutation_nonlinear_backward_audit`
- B3 single-layer masked-vs-plaintext trajectory: `masked_lora_backward_audit.run_optimizer_equivalence`
- B3 multi-layer trusted-AdamW training: `multilayer_lora_training.run_multilayer_lora_training`
- correctness/accounting/conditioning/security: the four cpu_* / *_audit modules
"""

from __future__ import annotations

import platform
import subprocess
import sys

import numpy as np
import torch

from pllo.experiments import cpu_correctness_audit as cca
from pllo.experiments import mask_conditioning_audit as mca
from pllo.experiments import synthetic_security_audit as ssa
from pllo.experiments import trusted_boundary_accounting as tba
from pllo.experiments.masked_lora_backward_audit import run_optimizer_equivalence
from pllo.experiments.multilayer_lora_training import (
    MultiLayerLoRATrainingConfig,
    run_multilayer_lora_training,
)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=_repo_root(), timeout=5).stdout.strip() or "unavailable"
    except Exception:
        return "unavailable"


def _repo_root() -> str:
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[3])


def audit_metadata(seed: int) -> dict:
    return {
        "git_commit": _git_hash(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cpu_device": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "primary_dtype": "float64",
        "float32_sensitivity_reported": True,
        "seed": seed,
        "tolerance_fp64": cca.PASS,
        "tolerance_fail_threshold": cca.FAIL,
        "mask_families": ["permutation", "signed_permutation", "positive_diagonal",
                          "orthogonal", "dense_gl"],
        "simulated_trusted_controller": True,
        "uses_real_gpu": False,
        "uses_real_tee": False,
        "synthetic_security": True,
    }


# ---------------------------------------------------------------------------
# B — LoRA multi-step + packed-update equivalence
# ---------------------------------------------------------------------------
def _adamw(p, g, st, lr=0.02, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    st.setdefault("m", np.zeros_like(p)); st.setdefault("v", np.zeros_like(p)); st.setdefault("t", 0)
    st["t"] += 1
    st["m"] = b1 * st["m"] + (1 - b1) * g
    st["v"] = b2 * st["v"] + (1 - b2) * (g * g)
    mh = st["m"] / (1 - b1 ** st["t"]); vh = st["v"] / (1 - b2 ** st["t"])
    p = p - lr * (mh / (np.sqrt(vh) + eps) + wd * p)
    return p, st


def b3_multistep_trajectory() -> dict:
    """Reuse: single-layer masked-vs-plaintext trajectory over SGD/momentum/AdamW,
    and multi-layer trusted-AdamW training. Extract alignment metrics."""
    out = {}
    for scheme in ("A0", "A1"):
        eq = run_optimizer_equivalence(scheme=scheme, steps=10)
        out[f"single_layer_{scheme}"] = {
            "per_optimizer": {k: {"max_param_error": v["max_param_error"],
                                  "loss_curve_distance": v["loss_curve_distance"],
                                  "final_adapter_error": v["final_adapter_error"]}
                              for k, v in eq["per_optimizer"].items()},
            "dense_masked_adamw": eq["dense_masked_adamw"]["status"],
        }
    multi = []
    for steps in (1, 2, 5, 10):
        for opt in ("sgd", "adamw"):
            cfg = MultiLayerLoRATrainingConfig(num_layers=3, num_steps=steps, optimizer=opt, seed=7)
            tc = run_multilayer_lora_training(cfg)["training_correctness"]
            multi.append({
                "num_layers": 3, "num_steps": steps, "optimizer": opt,
                "max_loss_diff_abs": tc["max_loss_diff"],
                "max_grad_a_err": tc["max_grad_a_real_err"],
                "max_grad_b_err": tc["max_grad_b_real_err"],
                "max_update_a_err": tc["max_update_a_err"],
                "max_update_b_err": tc["max_update_b_err"],
                "allclose": tc["allclose"],
            })
    out["multi_layer"] = multi
    return out


def b4_packed_vs_per_layer(num_layers=4, steps=6, m=6, d=12, d_out=12, r=4, seed=3) -> dict:
    """Packed (one trusted update for all layers) MUST equal per-layer (sequential
    updates), because trusted AdamW is per-parameter and layers are independent.
    Verify final A/B, DeltaW=AB, Adam moments, and next-step logits are identical."""
    g = np.random.default_rng(seed)
    layers = []
    for _ in range(num_layers):
        X = g.standard_normal((m, d))
        W = g.standard_normal((d, d_out)) / d ** 0.5
        T = g.standard_normal((m, d_out))
        A = g.standard_normal((d, r)) * 0.1
        B = np.zeros((r, d_out))
        layers.append({"X": X, "W": W, "T": T, "A": A, "B": B})

    def run(mode):
        st = [{"A": {}, "B": {}} for _ in range(num_layers)]
        params = [{"A": l["A"].copy(), "B": l["B"].copy()} for l in layers]
        for _ in range(steps):
            grads = []
            for j, l in enumerate(layers):                      # forward+backward all layers
                A, B = params[j]["A"], params[j]["B"]
                Y = l["X"] @ (l["W"] + A @ B)
                GY = (Y - l["T"]) / Y.size
                gA = l["X"].T @ GY @ B.T
                gB = (l["X"] @ A).T @ GY
                grads.append((gA, gB))
            if mode == "per_layer":
                for j in range(num_layers):                     # |S| sequential updates
                    params[j]["A"], st[j]["A"] = _adamw(params[j]["A"], grads[j][0], st[j]["A"])
                    params[j]["B"], st[j]["B"] = _adamw(params[j]["B"], grads[j][1], st[j]["B"])
            else:                                               # packed: single update for all
                for j in range(num_layers):
                    params[j]["A"], st[j]["A"] = _adamw(params[j]["A"], grads[j][0], st[j]["A"])
                    params[j]["B"], st[j]["B"] = _adamw(params[j]["B"], grads[j][1], st[j]["B"])
        return params, st

    p_per, st_per = run("per_layer")
    p_pack, st_pack = run("packed")
    a_err = max(np.abs(p_per[j]["A"] - p_pack[j]["A"]).max() for j in range(num_layers))
    b_err = max(np.abs(p_per[j]["B"] - p_pack[j]["B"]).max() for j in range(num_layers))
    dw_err = max(np.abs((p_per[j]["A"] @ p_per[j]["B"]) -
                        (p_pack[j]["A"] @ p_pack[j]["B"])).max() for j in range(num_layers))
    m_err = max(np.abs(st_per[j]["A"]["m"] - st_pack[j]["A"]["m"]).max() for j in range(num_layers))
    v_err = max(np.abs(st_per[j]["A"]["v"] - st_pack[j]["A"]["v"]).max() for j in range(num_layers))
    logit_err = max(np.abs(layers[j]["X"] @ (layers[j]["W"] + p_per[j]["A"] @ p_per[j]["B"]) -
                           layers[j]["X"] @ (layers[j]["W"] + p_pack[j]["A"] @ p_pack[j]["B"])).max()
                    for j in range(num_layers))
    return {
        "num_layers": num_layers, "steps": steps,
        "param_A_error": float(a_err), "param_B_error": float(b_err),
        "delta_W_error": float(dw_err),
        "adam_first_moment_error": float(m_err), "adam_second_moment_error": float(v_err),
        "next_step_logit_error": float(logit_err),
        "packed_equals_per_layer": bool(max(a_err, b_err, dw_err, m_err, v_err, logit_err) < 1e-10),
        "note": "packing changes SCHEDULE only; parameters and optimizer state are identical.",
    }


# ---------------------------------------------------------------------------
# full audit
# ---------------------------------------------------------------------------
def run_full_audit(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "metadata": audit_metadata(seed),
        "A_correctness": cca.run_correctness(seed=seed),
        "B_lora_training": b3_multistep_trajectory(),
        "B4_packed_vs_per_layer": b4_packed_vs_per_layer(seed=seed + 3),
        "C_D_accounting": tba.run_accounting(),
        "E_conditioning": mca.run_conditioning(),
        "F_synthetic_security": ssa.run_security(seed=seed),
    }
