"""Real-path nonlinear dispatch for the folded Qwen worker (design A vs B).

This is the wiring that makes the selectable nonlinear *design* a genuinely
EXECUTED path in the real folded-package pipeline (no longer a metadata tag):

* ``current``          -> the trusted-boundary inline nonlinearity (SiLU/SwiGLU
  evaluated in the boundary, softmax/RMSNorm with the trusted reduction). This is
  numerically identical to the historical folded path (``silu_reference`` /
  ``rmsnorm_core`` / ``torch.softmax``) -- existing ``current`` artifacts are
  unaffected -- and it records *trusted* op counters.
* ``trusted_shortcut`` -> the Amulet-style migrated backend
  (:mod:`pllo.nonlinear.amulet_backend`): the MLP activation (SiLU/SwiGLU, or
  GELU) is *lifted* onto the untrusted accelerator via a selector lift (exact
  after the folded squeeze), and softmax / RMSNorm are migrated onto the
  accelerator keeping only a small trusted reduction shortcut. It records *lift*
  counters (``amulet_lift_executed`` / ``lifted_nonlinear_ops_count`` /
  ``lift_k`` / ``lifted_gpu_bytes``) plus the migrated reduction stats.

Correctness first: the selector lift gathers the valid column (scale 1), so the
activation output is bit-identical to the direct activation; RMSNorm is the same
``x * rsqrt(mean(x^2)+eps)`` formula; softmax is the standard stable softmax.
Security is NOT formally claimed for ``trusted_shortcut`` (selector-leak caveat,
under discussion) -- this module measures correctness + the trust/accelerator
cost split only.

torch is used (this is worker-side compute, never a measured-boundary stdlib
module). The design->op-backend mapping comes from the torch-free
:mod:`pllo.experiments.nonlinear_designs` registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from pllo.experiments.nonlinear_designs import (
    normalize_nonlinear_backend,
    op_backend_for_design,
)
from pllo.nonlinear.backends import tensor_bytes
from pllo.nonlinear.registry import make_nonlinear_backend
from pllo.ops.nonlinear_islands import rmsnorm_core as _rmsnorm_core_fn
from pllo.ops.nonlinear_islands import silu_reference as _silu_fn

__all__ = [
    "NonlinearExecAccumulator",
    "FoldedNonlinearRunner",
    "make_folded_nonlinear_runner",
    "UnsupportedNonlinearOp",
]

# Activations migrated via a selector *lift* onto the accelerator (design B).
_LIFT_ACTIVATIONS = ("gelu", "silu", "swiglu")
# Reductions migrated onto the accelerator keeping a trusted reduction shortcut.
_MIGRATED_REDUCTIONS = ("softmax", "rmsnorm", "layernorm")


class UnsupportedNonlinearOp(RuntimeError):
    """A nonlinear op was requested that the selected design cannot migrate.

    Raised loudly so a ``trusted_shortcut`` run never silently falls back to the
    ``current`` trusted path for an op it does not actually migrate."""


@dataclass
class NonlinearExecAccumulator:
    """Per-session counters of where the nonlinear work actually ran.

    These are *measured* from real :class:`~pllo.nonlinear.backends.NonlinearOpResult`
    counters during execution -- never fabricated. The execution-evidence subset
    (``amulet_lift_executed`` / ``lifted_nonlinear_ops_count`` / ``lift_k`` /
    ``lifted_gpu_bytes``) is what
    :func:`pllo.experiments.nonlinear_designs.report_has_amulet_execution` checks.
    """

    nonlinear_backend: str
    nonlinear_op_backend: str
    amulet_lift_executed: bool = False
    amulet_backend_used: bool = False
    lifted_nonlinear_ops_count: int = 0
    lift_k: int = 0
    lifted_gpu_bytes: int = 0
    # A_rightmul (compatible right-multiply) accounting
    right_multiply_executed: bool = False
    right_multiply_ops_count: int = 0
    right_multiply_gpu_bytes: int = 0
    # amulet_secure_R accounting
    secure_right_multiply_executed: bool = False
    secure_right_multiply_ops_count: int = 0
    secure_right_multiply_gpu_bytes: int = 0
    secure_R_enabled: bool = False
    secure_zero_decoys: bool = True            # default True == NOT secure (no run)
    secure_selector_visible: bool = True       # default True == NOT secure (no run)
    trusted_nonlinear_ops_count: int = 0
    trusted_calls: int = 0
    trusted_bytes: int = 0
    gpu_bytes: int = 0
    migrated_ops_by_type: Dict[str, int] = field(default_factory=dict)
    unsupported_ops: List[str] = field(default_factory=list)

    def _bump(self, op_type: str) -> None:
        self.migrated_ops_by_type[op_type] = \
            self.migrated_ops_by_type.get(op_type, 0) + 1

    def record_trusted(self, op_type: str, x: torch.Tensor,
                       out: torch.Tensor) -> None:
        """A trusted-boundary nonlinear island (design A): one boundary crossing,
        no accelerator payload."""
        self._bump(op_type)
        self.trusted_nonlinear_ops_count += 1
        self.trusted_calls += 1
        self.trusted_bytes += tensor_bytes(x) + tensor_bytes(out)

    def record_lift(self, op_type: str, result: Any) -> None:
        """A selector-*lift* activation migrated onto the accelerator (design B)."""
        self._bump(op_type)
        self.amulet_lift_executed = True
        self.amulet_backend_used = True
        self.lifted_nonlinear_ops_count += 1
        gb = int(result.gpu_bytes)
        self.lifted_gpu_bytes += gb
        self.gpu_bytes += gb
        self.trusted_calls += int(result.trusted_calls)
        self.trusted_bytes += int(result.trusted_bytes)
        k = int(result.extra.get("lift_k", 0) or 0)
        if k > self.lift_k:
            self.lift_k = k

    def record_migrated(self, op_type: str, result: Any) -> None:
        """A reduction (softmax/RMSNorm) migrated onto the accelerator keeping a
        small trusted reduction shortcut (design B)."""
        self._bump(op_type)
        self.amulet_backend_used = True
        self.gpu_bytes += int(result.gpu_bytes)
        self.trusted_calls += int(result.trusted_calls)
        self.trusted_bytes += int(result.trusted_bytes)

    def record_right_multiply(self, op_type: str, result: Any) -> None:
        """An A_rightmul compatible right-multiply island run in place on the
        accelerator (design A_rightmul): ZERO trusted crossings."""
        self._bump(op_type)
        self.right_multiply_executed = True
        self.right_multiply_ops_count += 1
        gb = int(result.gpu_bytes)
        self.right_multiply_gpu_bytes += gb
        self.gpu_bytes += gb
        # right-multiply islands never cross the trusted boundary
        self.trusted_calls += int(result.trusted_calls)   # == 0 by construction
        self.trusted_bytes += int(result.trusted_bytes)   # == 0 by construction

    def record_secure_right_multiply(self, op_type: str, result: Any) -> None:
        """An amulet_secure_R island run on the accelerator (dense single-one R
        for GELU/SiLU; direct masked-state reduction for softmax/RMSNorm). ZERO
        trusted crossings; captures the secure-R condition flags."""
        self._bump(op_type)
        self.secure_right_multiply_executed = True
        self.secure_right_multiply_ops_count += 1
        gb = int(result.gpu_bytes)
        self.secure_right_multiply_gpu_bytes += gb
        self.gpu_bytes += gb
        self.trusted_calls += int(result.trusted_calls)   # == 0 by construction
        self.trusted_bytes += int(result.trusted_bytes)   # == 0 by construction
        ex = getattr(result, "extra", {}) or {}
        if ex.get("secure_R_enabled"):
            self.secure_R_enabled = True
        # any op must report no zero decoys / selector hidden to STAY secure
        if ex.get("zero_decoys") is False:
            self.secure_zero_decoys = False
        if ex.get("selector_visible_to_gpu") is False:
            self.secure_selector_visible = False

    def to_report_fields(self) -> Dict[str, Any]:
        """The runtime execution-evidence fields a paper-facing report stamps
        AFTER ``nonlinear_design_report_fields`` (override the default tag-only
        stamp with measured counters)."""
        if self.secure_right_multiply_executed:
            status = "secure_right_multiply_on_accelerator"
        elif self.right_multiply_executed:
            status = "right_multiply_on_accelerator"
        elif self.amulet_lift_executed:
            status = "lifted_on_accelerator"
        elif self.nonlinear_op_backend == "amulet_migrated":
            status = "migrated_with_trusted_shortcut"
        else:
            status = "executed_trusted_boundary_inline"
        out = {
            "nonlinear_backend": self.nonlinear_backend,
            "nonlinear_op_backend": self.nonlinear_op_backend,
            "nonlinear_real_path_executed": True,
            "amulet_lift_executed": bool(self.amulet_lift_executed),
            "amulet_backend_used": bool(self.amulet_backend_used),
            "lifted_nonlinear_ops_count": int(self.lifted_nonlinear_ops_count),
            "lift_k": int(self.lift_k),
            "lifted_gpu_bytes": int(self.lifted_gpu_bytes),
            "trusted_nonlinear_ops_count": int(self.trusted_nonlinear_ops_count),
            "nonlinear_trusted_calls": int(self.trusted_calls),
            "nonlinear_trusted_bytes": int(self.trusted_bytes),
            "nonlinear_accelerator_bytes": int(self.gpu_bytes),
            "migrated_ops_by_type": dict(self.migrated_ops_by_type),
            "unsupported_nonlinear_ops": list(self.unsupported_ops),
            "nonlinear_execution_status": status,
        }
        if self.nonlinear_op_backend == "compatible_right_multiply":
            # A_rightmul measured execution evidence (single TEE entry/exit;
            # every nonlinear island ran on the accelerator, zero trusted calls).
            out.update({
                "right_multiply_nonlinear_executed":
                    bool(self.right_multiply_executed),
                "right_multiply_nonlinear_ops_count":
                    int(self.right_multiply_ops_count),
                "right_multiply_gpu_bytes": int(self.right_multiply_gpu_bytes),
                "nonlinear_masking_mode":
                    "compatible_right_multiply_or_permutation",
                "nonlinear_single_tee_entry_exit": True,
                "linear_boundary_pad": True,
            })
        if self.nonlinear_op_backend == "amulet_secure_R":
            # amulet_secure_R measured execution evidence (single TEE entry/exit;
            # zero trusted nonlinear crossings; secure-R conditions captured).
            out.update({
                "secure_right_multiply_executed":
                    bool(self.secure_right_multiply_executed),
                "secure_right_multiply_ops_count":
                    int(self.secure_right_multiply_ops_count),
                "secure_right_multiply_gpu_bytes":
                    int(self.secure_right_multiply_gpu_bytes),
                "secure_R_enabled": bool(self.secure_R_enabled),
                "zero_decoys": bool(self.secure_zero_decoys),
                "selector_visible_to_gpu": bool(self.secure_selector_visible),
                "valid_channel_observable": bool(self.secure_selector_visible),
                "nonlinear_masking_mode": "amulet_secure_right_multiply",
                "nonlinear_single_tee_entry_exit": True,
                "linear_boundary_pad": True,
            })
        return out


class FoldedNonlinearRunner:
    """Dispatch the folded worker's nonlinear ops through the selected design.

    For ``current`` the ops run exactly as before (``silu_reference`` /
    ``rmsnorm_core`` / ``torch.softmax``) and are counted as trusted islands. For
    ``trusted_shortcut`` they dispatch through the Amulet-migrated backend and the
    measured lift/migration counters are accumulated in :attr:`acc`."""

    def __init__(self, nonlinear_backend: str = "current", *,
                 lift_k: int = 2, seed: int = 2035) -> None:
        self.nonlinear_backend = normalize_nonlinear_backend(nonlinear_backend)
        self.op_backend = op_backend_for_design(self.nonlinear_backend)
        self.lift_k = int(lift_k)
        self._amulet = None
        self._rightmul = None
        self._secure = None
        if self.op_backend == "amulet_migrated":
            self._amulet = make_nonlinear_backend(
                "amulet_migrated", lift_k=self.lift_k, seed=int(seed))
        elif self.op_backend == "compatible_right_multiply":
            # A_rightmul: every nonlinear island runs on the accelerator over the
            # masked state; no trusted crossing, no fallback to the 'current'
            # trusted-island path.
            self._rightmul = make_nonlinear_backend("compatible_right_multiply")
        elif self.op_backend == "amulet_secure_R":
            # amulet_secure_R: GELU/SiLU via dense single-one secure-R lift,
            # softmax/RMSNorm directly on the masked state; zero trusted calls,
            # no per-op reduction shortcut, no fallback to the trusted path.
            self._secure = make_nonlinear_backend(
                "amulet_secure_R", lift_k=max(2, self.lift_k), seed=int(seed))
        self.acc = NonlinearExecAccumulator(
            nonlinear_backend=self.nonlinear_backend,
            nonlinear_op_backend=self.op_backend)

    # -- MLP activation (lifted for design B; right-multiply for A_rightmul) ---
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        if self._secure is not None:
            r = self._secure.silu(x)
            self.acc.record_secure_right_multiply("silu", r)
            return r.output
        if self._rightmul is not None:
            r = self._rightmul.silu(x)
            self.acc.record_right_multiply("silu", r)
            return r.output
        if self._amulet is None:
            out = _silu_fn(x)
            self.acc.record_trusted("silu", x, out)
            return out
        r = self._amulet.silu(x)
        self.acc.record_lift("silu", r)
        return r.output

    def gelu(self, x: torch.Tensor) -> torch.Tensor:
        if self._secure is not None:
            r = self._secure.gelu(x)
            self.acc.record_secure_right_multiply("gelu", r)
            return r.output
        if self._rightmul is not None:
            r = self._rightmul.gelu(x)
            self.acc.record_right_multiply("gelu", r)
            return r.output
        if self._amulet is None:
            out = torch.nn.functional.gelu(x)
            self.acc.record_trusted("gelu", x, out)
            return out
        r = self._amulet.gelu(x)
        self.acc.record_lift("gelu", r)
        return r.output

    # -- RMSNorm core (weight is folded into the linear; weight-free here) -----
    def rmsnorm_core(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        if self._secure is not None:
            r = self._secure.rmsnorm(x, weight=None, eps=eps)
            self.acc.record_secure_right_multiply("rmsnorm", r)
            return r.output
        if self._rightmul is not None:
            r = self._rightmul.rmsnorm(x, weight=None, eps=eps)
            self.acc.record_right_multiply("rmsnorm", r)
            return r.output
        if self._amulet is None:
            out = _rmsnorm_core_fn(x, eps)
            self.acc.record_trusted("rmsnorm", x, out)
            return out
        r = self._amulet.rmsnorm(x, weight=None, eps=eps)
        self.acc.record_migrated("rmsnorm", r)
        return r.output

    # -- attention softmax ----------------------------------------------------
    def softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self._secure is not None:
            r = self._secure.softmax(x, dim=dim)
            self.acc.record_secure_right_multiply("softmax", r)
            return r.output
        if self._rightmul is not None:
            r = self._rightmul.softmax(x, dim=dim)
            self.acc.record_right_multiply("softmax", r)
            return r.output
        if self._amulet is None:
            out = torch.softmax(x, dim=dim)
            self.acc.record_trusted("softmax", x, out)
            return out
        r = self._amulet.softmax(x, dim=dim)
        self.acc.record_migrated("softmax", r)
        return r.output

    def fail_unsupported(self, op_type: str) -> None:
        """Record + raise loudly for an op the selected design cannot migrate
        (no silent fallback to the ``current`` path under a trusted_shortcut tag)."""
        if op_type not in self.acc.unsupported_ops:
            self.acc.unsupported_ops.append(op_type)
        raise UnsupportedNonlinearOp(
            "nonlinear design %r (op_backend=%s) cannot migrate op %r; refusing "
            "to silently run the 'current' trusted path under a %r tag"
            % (self.nonlinear_backend, self.op_backend, op_type,
               self.nonlinear_backend))

    def execution_evidence(self) -> Dict[str, Any]:
        return self.acc.to_report_fields()


def make_folded_nonlinear_runner(nonlinear_backend: str | None = None, *,
                                 lift_k: int = 2, seed: int = 2035
                                 ) -> FoldedNonlinearRunner:
    """Construct a :class:`FoldedNonlinearRunner` (defaulting to ``current``)."""
    return FoldedNonlinearRunner(nonlinear_backend or "current",
                                 lift_k=lift_k, seed=seed)
