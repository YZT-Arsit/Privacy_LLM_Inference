"""Simulated ObfuscaTune modules with TEE / outside-TEE phase accounting.

These modules do NOT provide a real TEE. They split each computation into a
``tee`` phase (embedding obfuscation/de-obfuscation, LayerNorm, softmax,
activation) and an ``outside`` phase (the big obfuscated matmuls), and report
simulator counters: matmul counts per phase, bytes crossing the boundary, and
the numerical error versus the plaintext reference. Use them to study
correctness and the boundary-crossing cost structure of the paper's scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .config import ObfuscaTuneConfig
from .linear_obfuscation import (
    math_weight,
    module_bias,
    obfuscate_weight_left,
    obfuscate_weight_output_right,
)
from .random_matrices import (
    condition_number,
    gaussian_invertible_matrix,
    matrix_with_condition_number,
    orthogonal_matrix,
)


def _elem_bytes(dtype: torch.dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def select_matrix(
    dim: int, config: ObfuscaTuneConfig, seed: int
) -> tuple[torch.Tensor, torch.Tensor, float, str]:
    """Return ``(R, R_inv, cond_value, inverse_method)`` per the config policy."""
    dtype = config.torch_dtype()
    dev = config.device
    if config.random_matrix_type == "random":
        r, r_inv = gaussian_invertible_matrix(dim, seed, dtype, dev, config.gaussian_jitter)
        return r, r_inv, condition_number(r), "torch.linalg.inv"
    if config.random_matrix_type == "cond" and config.condition_number > 1.0:
        r, r_inv = matrix_with_condition_number(dim, config.condition_number, seed, dtype, dev)
        return r, r_inv, condition_number(r), "svd_factors"
    r, r_inv = orthogonal_matrix(dim, seed, dtype, dev)
    return r, r_inv, 1.0, "transpose"


@dataclass
class PhaseMetrics:
    outside_matmul_count: int = 0
    tee_matmul_count: int = 0
    tee_transfer_bytes: int = 0
    boundary_calls: int = 0
    condition_numbers: list[float] = field(default_factory=list)
    max_abs_err: float = 0.0

    def merge(self, other: "PhaseMetrics") -> None:
        self.outside_matmul_count += other.outside_matmul_count
        self.tee_matmul_count += other.tee_matmul_count
        self.tee_transfer_bytes += other.tee_transfer_bytes
        self.boundary_calls += other.boundary_calls
        self.condition_numbers += other.condition_numbers
        self.max_abs_err = max(self.max_abs_err, other.max_abs_err)

    def to_dict(self) -> dict[str, Any]:
        cns = self.condition_numbers
        return {
            "outside_matmul_count": self.outside_matmul_count,
            "tee_matmul_count": self.tee_matmul_count,
            "tee_transfer_bytes": self.tee_transfer_bytes,
            "boundary_calls": self.boundary_calls,
            "condition_number_mean": (sum(cns) / len(cns)) if cns else 1.0,
            "condition_number_max": max(cns) if cns else 1.0,
            "max_abs_err": self.max_abs_err,
        }


class ObfuscaTuneLinearSimulator:
    """A single obfuscated linear (``torch.nn.Linear`` or GPT-2 ``Conv1D``).

    ``direction='input'`` uses input obfuscation (X* = X@Ra, W* = Ra^{-1}@W) and
    returns the TRUE ``Y = X@W`` on the untrusted side. ``direction='output'``
    uses output obfuscation (W* = W@Rb) and de-obfuscates ``Y = Y*@Rb^{-1}``
    inside the TEE.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        config: ObfuscaTuneConfig | None = None,
        *,
        direction: str = "input",
        seed: int | None = None,
    ) -> None:
        self.config = config or ObfuscaTuneConfig()
        self.direction = direction
        self.seed = self.config.seed if seed is None else seed
        dtype = self.config.torch_dtype()
        self.w = math_weight(module).detach().to(dtype)   # (d_in, d_out)
        b = module_bias(module)
        self.bias = None if b is None else b.detach().to(dtype)
        self.d_in, self.d_out = self.w.shape

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PhaseMetrics]:
        x = x.to(self.w.dtype)
        m = PhaseMetrics()
        ebytes = _elem_bytes(self.w.dtype)
        if self.direction == "input":
            r, r_inv, cond, _ = select_matrix(self.d_in, self.config, self.seed)
            x_star = x @ r                       # TEE: obfuscate input
            w_star = obfuscate_weight_left(self.w, r_inv)  # (precomputed offline)
            y = x_star @ w_star                  # OUTSIDE: untrusted matmul
            m.tee_matmul_count += 1
            m.outside_matmul_count += 1
            m.boundary_calls += 1
            m.tee_transfer_bytes += x_star.numel() * ebytes + y.numel() * ebytes
        elif self.direction == "output":
            r, r_inv, cond, _ = select_matrix(self.d_out, self.config, self.seed)
            w_star = obfuscate_weight_output_right(self.w, r)
            y_star = x @ w_star                  # OUTSIDE: untrusted matmul
            y = y_star @ r_inv                   # TEE: de-obfuscate
            m.tee_matmul_count += 1
            m.outside_matmul_count += 1
            m.boundary_calls += 1
            m.tee_transfer_bytes += x.numel() * ebytes + y_star.numel() * ebytes
        else:
            raise ValueError(f"direction must be 'input' or 'output', got {self.direction!r}")
        if self.bias is not None:
            y = y + self.bias                    # bias added in TEE after de-obf
        m.condition_numbers.append(cond)
        y_plain = x @ self.w + (self.bias if self.bias is not None else 0.0)
        m.max_abs_err = float((y - y_plain).abs().max().item()) if y.numel() else 0.0
        return y, m


def _layernorm(x: torch.Tensor, weight: torch.Tensor | None,
               bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


class ObfuscaTuneAttentionProjectionSimulator:
    """Q/K/V input projections + output projection under ObfuscaTune.

    Q/K/V are produced on the untrusted side as the TRUE plaintext values (the
    input mask cancels). The softmax attention is a non-linearity and runs in
    the TEE on those plaintext Q/K/V; the output projection is output-obfuscated
    and de-obfuscated back in the TEE.
    """

    def __init__(self, config: ObfuscaTuneConfig | None = None,
                 *, n_heads: int = 4, seed: int | None = None) -> None:
        self.config = config or ObfuscaTuneConfig()
        self.n_heads = n_heads
        self.seed = self.config.seed if seed is None else seed

    def forward(
        self,
        h: torch.Tensor,
        wq: torch.nn.Module,
        wk: torch.nn.Module,
        wv: torch.nn.Module,
        wo: torch.nn.Module,
    ) -> tuple[torch.Tensor, dict[str, Any], PhaseMetrics]:
        cfg = self.config
        metrics = PhaseMetrics()
        q_sim = ObfuscaTuneLinearSimulator(wq, cfg, direction="input", seed=self.seed)
        k_sim = ObfuscaTuneLinearSimulator(wk, cfg, direction="input", seed=self.seed + 1)
        v_sim = ObfuscaTuneLinearSimulator(wv, cfg, direction="input", seed=self.seed + 2)
        q, mq = q_sim.forward(h)
        k, mk = k_sim.forward(h)
        v, mv = v_sim.forward(h)
        for mm in (mq, mk, mv):
            metrics.merge(mm)
        S, d = q.shape[-2], q.shape[-1]
        hd = d // self.n_heads
        qh = q.view(S, self.n_heads, hd).transpose(0, 1)
        kh = k.view(S, self.n_heads, hd).transpose(0, 1)
        vh = v.view(S, self.n_heads, hd).transpose(0, 1)
        att = torch.softmax(qh @ kh.transpose(-1, -2) / (hd ** 0.5), dim=-1)  # TEE
        metrics.boundary_calls += 1  # softmax non-linearity forces a TEE crossing
        ctx = (att @ vh).transpose(0, 1).reshape(S, d)
        o_sim = ObfuscaTuneLinearSimulator(wo, cfg, direction="output", seed=self.seed + 3)
        out, mo = o_sim.forward(ctx)
        metrics.merge(mo)
        audit = {"exposed_plaintext_tensors": ["Q_plaintext", "K_plaintext", "V_plaintext"]}
        return out, audit, metrics


class ObfuscaTuneMLPSimulator:
    """MLP block: first linear (input-obfuscated) -> GELU in TEE -> second linear.

    The GELU is applied inside the TEE on the de-obfuscated intermediate, which
    is exposed in plaintext to the untrusted side (the first linear's mask
    cancels, exactly like Q/K/V).
    """

    def __init__(self, config: ObfuscaTuneConfig | None = None,
                 *, seed: int | None = None, activation: str = "gelu") -> None:
        self.config = config or ObfuscaTuneConfig()
        self.seed = self.config.seed if seed is None else seed
        self.activation = activation

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return torch.nn.functional.gelu(x)
        if self.activation == "relu":
            return torch.relu(x)
        raise ValueError(f"unsupported activation {self.activation!r}")

    def forward(
        self, h: torch.Tensor, fc: torch.nn.Module, proj: torch.nn.Module
    ) -> tuple[torch.Tensor, dict[str, Any], PhaseMetrics]:
        cfg = self.config
        metrics = PhaseMetrics()
        fc_sim = ObfuscaTuneLinearSimulator(fc, cfg, direction="input", seed=self.seed)
        inter, m1 = fc_sim.forward(h)
        metrics.merge(m1)
        inter = self._act(inter)              # TEE: activation on plaintext hidden
        metrics.boundary_calls += 1
        proj_sim = ObfuscaTuneLinearSimulator(proj, cfg, direction="output", seed=self.seed + 1)
        out, m2 = proj_sim.forward(inter)
        metrics.merge(m2)
        audit = {"exposed_plaintext_tensors": ["mlp_intermediate_plaintext"]}
        return out, audit, metrics


__all__ = [
    "PhaseMetrics",
    "select_matrix",
    "ObfuscaTuneLinearSimulator",
    "ObfuscaTuneAttentionProjectionSimulator",
    "ObfuscaTuneMLPSimulator",
    "_layernorm",
]
