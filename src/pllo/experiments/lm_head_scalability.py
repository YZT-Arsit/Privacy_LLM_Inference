"""Stage 7.7a -- Scalable LM-head masking.

The dense ``N_vocab in R^{V x V}`` orthogonal mask used in the tiny
baseline does not scale to real LLM vocab sizes (32k / 50k / 100k):
``O(V^2)`` storage and ``O(V^2)`` recovery dominate any plausible
boundary cost.

This module compares three exact LM-head masking strategies plus one
exploratory partial-recovery strategy under CPU local emulation:

* ``dense_vocab_mask_baseline`` -- the tiny baseline. Orthogonal
  ``[V, V]`` mask; recovery is one ``[B, S, V] @ [V, V]`` matmul. Only
  feasible for small ``V`` -- larger ``V`` is estimated symbolically.

* ``vocab_permutation_mask`` -- ``P_vocab`` is a permutation (a
  symmetric group element). ``z_tilde[..., i] = z[..., perm[i]]``;
  trusted recovery is an inverse-permutation index_select. Storage
  ``O(V)``, recovery ``O(V)``. Hides token-index alignment but
  preserves the *multiset* of logits.

* ``block_diagonal_vocab_mask`` -- partition vocab into chunks of
  size ``b`` and apply an independent orthogonal mask per chunk.
  Storage ``O(V * b)``, recovery ``O(V * b)``. Hides intra-block
  ordering / values; cross-block ordering remains via block id.

* ``topk_trusted_recovery_mode`` -- the trusted side recovers the
  full logits and then truncates to top-k. For greedy decoding (top-1)
  this is exact; if downstream sampling depends on the full
  distribution the recovery must include all logits.

The report explicitly states the dense baseline is NOT scalable to
real LLM vocab sizes, and that none of these provide formal
cryptographic security.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LMHeadScalabilityConfig:
    seed: int = 2026
    hidden_size: int = 64
    batch_size: int = 2
    seq_len: int = 4
    block_size: int = 64
    topk: int = 8
    # Vocab sizes to actually run on CPU. Larger sizes are reported
    # symbolically using the same formulas.
    real_vocab_sizes: Tuple[int, ...] = (97, 1024, 4096)
    # Vocab sizes we estimate (do not allocate dense V x V).
    estimated_vocab_sizes: Tuple[int, ...] = (16_384, 50_000)
    # Dense baseline is run only for these (small) sizes to avoid
    # allocating gigabytes for V=50_000.
    dense_max_real_v: int = 4_096


# ---------------------------------------------------------------------------
# Mask sampling helpers
# ---------------------------------------------------------------------------


def _sample_orthogonal(
    dim: int, *, dtype: torch.dtype, device: str, generator: torch.Generator
) -> torch.Tensor:
    raw = torch.randn(dim, dim, dtype=dtype, device=device, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


# ---------------------------------------------------------------------------
# Mode runners (return masked + recovered logits + metrics dict)
# ---------------------------------------------------------------------------


def _run_dense(
    z: torch.Tensor, *, generator: torch.Generator,
) -> Dict[str, Any]:
    """Exact, O(V^2) memory + recovery."""
    V = z.shape[-1]
    n = _sample_orthogonal(V, dtype=z.dtype, device=str(z.device), generator=generator)
    n_inv = n.transpose(-2, -1)
    z_tilde = z @ n
    z_rec = z_tilde @ n_inv
    err = float((z_rec - z).abs().max().item())
    plain_argmax = z.argmax(dim=-1)
    rec_argmax = z_rec.argmax(dim=-1)
    greedy = float((plain_argmax == rec_argmax).float().mean().item())
    return {
        "mode": "dense_vocab_mask_baseline",
        "vocab_size": V,
        "exactness": "exact",
        "max_abs_error": err,
        "greedy_token_match_rate": greedy,
        "memory_bytes_mask": int(V * V * z.element_size()),
        "preprocessing_ops_estimate": V ** 3,        # QR cost ~ O(V^3)
        "online_recovery_ops_estimate": z.shape[0] * z.shape[1] * V * V,
        "trusted_compute_ops_estimate": z.shape[0] * z.shape[1] * V * V,
        "accelerator_compute_ops_estimate": z.shape[0] * z.shape[1] * V * V,
        "leakage_notes": (
            "Dense orthogonal mask exact; storage and recovery O(V^2); "
            "not feasible for real LLM vocab sizes."
        ),
        "feasibility": "feasible_only_for_small_V",
    }


def _run_permutation(
    z: torch.Tensor, *, generator: torch.Generator,
) -> Dict[str, Any]:
    V = z.shape[-1]
    perm = torch.randperm(V, generator=generator)
    inv_perm = torch.argsort(perm)
    z_tilde = z.index_select(dim=-1, index=perm)
    z_rec = z_tilde.index_select(dim=-1, index=inv_perm)
    err = float((z_rec - z).abs().max().item())
    plain_argmax = z.argmax(dim=-1)
    rec_argmax = z_rec.argmax(dim=-1)
    greedy = float((plain_argmax == rec_argmax).float().mean().item())
    # Multiset invariance: sorted values of z_tilde and z match.
    sorted_plain, _ = z.sort(dim=-1)
    sorted_tilde, _ = z_tilde.sort(dim=-1)
    multiset_err = float((sorted_plain - sorted_tilde).abs().max().item())
    return {
        "mode": "vocab_permutation_mask",
        "vocab_size": V,
        "exactness": "exact",
        "max_abs_error": err,
        "greedy_token_match_rate": greedy,
        "memory_bytes_mask": int(V * 8),  # int64 indices
        "preprocessing_ops_estimate": V,
        "online_recovery_ops_estimate": z.shape[0] * z.shape[1] * V,
        "trusted_compute_ops_estimate": z.shape[0] * z.shape[1] * V,
        "accelerator_compute_ops_estimate": z.shape[0] * z.shape[1] * V,
        "logit_multiset_preserved_max_abs_error": multiset_err,
        "leakage_notes": (
            "Permutation mask preserves the multiset of logits exactly. "
            "Token-index alignment is hidden, but the sorted logit "
            "vector is observable. Not formal cryptographic security."
        ),
        "feasibility": "scalable",
    }


def _run_block_diagonal(
    z: torch.Tensor,
    *,
    block_size: int,
    generator: torch.Generator,
) -> Dict[str, Any]:
    V = z.shape[-1]
    if V % block_size != 0:
        # Pad to a multiple; downstream we crop back. For simplicity
        # we require block_size | V at this stage's tiny configs.
        pad = block_size - (V % block_size)
        z_pad = torch.nn.functional.pad(z, (0, pad))
    else:
        pad = 0
        z_pad = z
    V_pad = z_pad.shape[-1]
    num_blocks = V_pad // block_size
    blocks: List[torch.Tensor] = []
    block_invs: List[torch.Tensor] = []
    for _ in range(num_blocks):
        n_b = _sample_orthogonal(
            block_size, dtype=z.dtype, device=str(z.device), generator=generator
        )
        blocks.append(n_b)
        block_invs.append(n_b.transpose(-2, -1))
    z_blocks = z_pad.view(*z_pad.shape[:-1], num_blocks, block_size)
    z_tilde = torch.stack(
        [z_blocks[..., i, :] @ blocks[i] for i in range(num_blocks)],
        dim=-2,
    ).reshape(*z_pad.shape)
    z_rec = torch.stack(
        [z_tilde.view(*z_pad.shape[:-1], num_blocks, block_size)[..., i, :]
         @ block_invs[i] for i in range(num_blocks)],
        dim=-2,
    ).reshape(*z_pad.shape)
    if pad:
        z_rec = z_rec[..., :V]
    err = float((z_rec - z).abs().max().item())
    plain_argmax = z.argmax(dim=-1)
    rec_argmax = z_rec.argmax(dim=-1)
    greedy = float((plain_argmax == rec_argmax).float().mean().item())
    return {
        "mode": "block_diagonal_vocab_mask",
        "vocab_size": V,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "exactness": "exact",
        "max_abs_error": err,
        "greedy_token_match_rate": greedy,
        "memory_bytes_mask": int(num_blocks * block_size * block_size * z.element_size()),
        "preprocessing_ops_estimate": num_blocks * block_size ** 3,
        "online_recovery_ops_estimate":
            z.shape[0] * z.shape[1] * num_blocks * block_size * block_size,
        "trusted_compute_ops_estimate":
            z.shape[0] * z.shape[1] * num_blocks * block_size * block_size,
        "accelerator_compute_ops_estimate":
            z.shape[0] * z.shape[1] * num_blocks * block_size * block_size,
        "leakage_notes": (
            "Block-diagonal orthogonal mask. Within-block values are "
            "hidden via orthogonal rotation; block membership of each "
            "vocab index is still observable unless the block "
            "partition is itself permuted. Not formal cryptographic "
            "security."
        ),
        "feasibility": "scalable_with_block_size_tunable",
    }


def _run_topk(
    z: torch.Tensor, *, topk: int, generator: torch.Generator,
) -> Dict[str, Any]:
    """Trusted-side recovers full logits via permutation, then keeps
    top-k indices/values. Greedy (top-1) is exact; full distribution
    is exact only if all logits are kept.
    """
    V = z.shape[-1]
    perm = torch.randperm(V, generator=generator)
    inv_perm = torch.argsort(perm)
    z_tilde = z.index_select(dim=-1, index=perm)
    z_rec_full = z_tilde.index_select(dim=-1, index=inv_perm)
    full_err = float((z_rec_full - z).abs().max().item())
    topk_vals, topk_idx = z_rec_full.topk(topk, dim=-1)
    plain_argmax = z.argmax(dim=-1)
    rec_argmax = topk_idx[..., 0]
    greedy = float((plain_argmax == rec_argmax).float().mean().item())
    return {
        "mode": "topk_trusted_recovery_mode",
        "vocab_size": V,
        "topk": topk,
        "exactness": "exact_for_greedy_top1__not_full_softmax_unless_full_recovery",
        "max_abs_error_full_recovery": full_err,
        "greedy_token_match_rate": greedy,
        "memory_bytes_mask": int(V * 8),
        "preprocessing_ops_estimate": V,
        "online_recovery_ops_estimate": z.shape[0] * z.shape[1] * V,
        "trusted_compute_ops_estimate": z.shape[0] * z.shape[1] * V,
        "accelerator_compute_ops_estimate": z.shape[0] * z.shape[1] * V,
        "leakage_notes": (
            "TEE recovers full logits via inverse permutation, then "
            "returns only the top-k values + indices. Greedy decoding "
            "is exact; if the downstream sampler depends on the full "
            "distribution, full recovery must be performed before "
            "truncation. Not formal cryptographic security."
        ),
        "feasibility": "scalable_top1_only",
    }


def _symbolic_estimate(
    V: int, *, mode: str, block_size: int, hidden: int, batch: int, seq: int,
    dtype_bytes: int = 8,
) -> Dict[str, Any]:
    """Symbolic O(.) estimate for vocab sizes we do not actually run."""
    BS = batch * seq
    if mode == "dense_vocab_mask_baseline":
        return {
            "mode": mode,
            "vocab_size": V,
            "symbolic_estimate_only": True,
            "memory_bytes_mask": V * V * dtype_bytes,
            "online_recovery_ops_estimate": BS * V * V,
            "feasibility": (
                "infeasible_for_real_llm_vocab" if V >= 16_000 else "tight"
            ),
            "leakage_notes": "Dense N_vocab not allocated; estimated.",
        }
    if mode == "vocab_permutation_mask":
        return {
            "mode": mode,
            "vocab_size": V,
            "symbolic_estimate_only": True,
            "memory_bytes_mask": V * 8,
            "online_recovery_ops_estimate": BS * V,
            "feasibility": "scalable",
            "leakage_notes": "Sorted logit vector preserved.",
        }
    if mode == "block_diagonal_vocab_mask":
        num_blocks = (V + block_size - 1) // block_size
        return {
            "mode": mode,
            "vocab_size": V,
            "block_size": block_size,
            "symbolic_estimate_only": True,
            "memory_bytes_mask": num_blocks * block_size * block_size * dtype_bytes,
            "online_recovery_ops_estimate": BS * num_blocks * block_size * block_size,
            "feasibility": "scalable_with_block_size_tunable",
            "leakage_notes": "Block-membership of vocab index observable.",
        }
    raise ValueError(f"unknown mode {mode!r}")


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------


def run_lm_head_scalability(
    *, cfg: Optional[LMHeadScalabilityConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = LMHeadScalabilityConfig()
    torch.manual_seed(cfg.seed)
    dtype = torch.float64

    real_results: Dict[str, List[Dict[str, Any]]] = {
        "dense_vocab_mask_baseline": [],
        "vocab_permutation_mask": [],
        "block_diagonal_vocab_mask": [],
        "topk_trusted_recovery_mode": [],
    }
    for V in cfg.real_vocab_sizes:
        g_z = torch.Generator(device="cpu").manual_seed(cfg.seed + V)
        z = torch.randn(
            cfg.batch_size, cfg.seq_len, V, dtype=dtype, generator=g_z
        )
        g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed + V + 1)
        if V <= cfg.dense_max_real_v:
            real_results["dense_vocab_mask_baseline"].append(_run_dense(z, generator=g_mask))
        else:
            real_results["dense_vocab_mask_baseline"].append(
                _symbolic_estimate(
                    V, mode="dense_vocab_mask_baseline",
                    block_size=cfg.block_size,
                    hidden=cfg.hidden_size,
                    batch=cfg.batch_size, seq=cfg.seq_len,
                )
            )
        g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed + V + 2)
        real_results["vocab_permutation_mask"].append(_run_permutation(z, generator=g_mask))
        if V % cfg.block_size == 0 or V < cfg.block_size:
            g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed + V + 3)
            real_results["block_diagonal_vocab_mask"].append(
                _run_block_diagonal(z, block_size=min(cfg.block_size, V), generator=g_mask)
            )
        else:
            g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed + V + 3)
            real_results["block_diagonal_vocab_mask"].append(
                _run_block_diagonal(z, block_size=cfg.block_size, generator=g_mask)
            )
        g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed + V + 4)
        real_results["topk_trusted_recovery_mode"].append(
            _run_topk(z, topk=min(cfg.topk, V), generator=g_mask)
        )

    estimated_results: Dict[str, List[Dict[str, Any]]] = {
        "dense_vocab_mask_baseline": [],
        "vocab_permutation_mask": [],
        "block_diagonal_vocab_mask": [],
    }
    for V in cfg.estimated_vocab_sizes:
        for mode in estimated_results:
            estimated_results[mode].append(
                _symbolic_estimate(
                    V, mode=mode, block_size=cfg.block_size,
                    hidden=cfg.hidden_size,
                    batch=cfg.batch_size, seq=cfg.seq_len,
                )
            )

    report = {
        "status": "ok",
        "stage": "7.7a",
        "main_mode": "lm_head_scalability",
        "device": "cpu",
        "dtype": str(dtype),
        "config": {
            "real_vocab_sizes": list(cfg.real_vocab_sizes),
            "estimated_vocab_sizes": list(cfg.estimated_vocab_sizes),
            "hidden_size": cfg.hidden_size,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "block_size": cfg.block_size,
            "topk": cfg.topk,
            "dense_max_real_v": cfg.dense_max_real_v,
        },
        "real_runs": real_results,
        "symbolic_estimates": estimated_results,
        "modes_evaluated": [
            "dense_vocab_mask_baseline",
            "vocab_permutation_mask",
            "block_diagonal_vocab_mask",
            "topk_trusted_recovery_mode",
        ],
        "limitations": [
            "CPU local emulation only; no real TEE / GPU.",
            "Dense N_vocab not feasible for real LLM vocab sizes "
            "(V >= 16k); estimated symbolically only.",
            "Permutation mask preserves the multiset of logits; this "
            "is observable side information, not formal cryptographic "
            "security.",
            "Block-diagonal mask reveals block membership of each vocab "
            "index unless the block partition is itself permuted.",
            "topk_trusted_recovery_mode is exact for top-1 greedy "
            "decoding; for sampling that depends on the full "
            "distribution, full recovery must be performed before "
            "truncation.",
            "This is NOT formal cryptographic / semantic / "
            "differential-privacy security.",
        ],
        "paper_safe_wording": (
            "Dense orthogonal N_vocab is not scalable to real LLM "
            "vocab sizes. Permutation and block-diagonal masks scale "
            "but disclose either the sorted logit multiset or the "
            "block partition; we present them as scalable algebraic "
            "alternatives with explicit leakage notes, not formal "
            "cryptographic security."
        ),
        "unsafe_wording_to_avoid": [
            "Dense vocab mask is scalable.",
            "Permutation mask cryptographically hides logits.",
            "topk_trusted_recovery_mode preserves full softmax.",
            "This is formal cryptographic security.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if x == 0.0:
            return "0.0"
        if abs(x) >= 1e-3:
            return f"{x:.6g}"
        return f"{x:.3e}"
    return str(x)


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Scalable LM-Head Masking")
    w()
    w(
        "_Stage 7.7a: compare scalable LM-head masking strategies under "
        "CPU local emulation; dense ``N_vocab`` is not feasible for "
        "real LLM vocab sizes._"
    )
    w()
    w("## Configuration")
    w()
    cfg = report["config"]
    w("| Field | Value |")
    w("|---|---|")
    for k in ("real_vocab_sizes", "estimated_vocab_sizes", "hidden_size",
              "batch_size", "seq_len", "block_size", "topk",
              "dense_max_real_v"):
        w(f"| {k} | {cfg[k]} |")
    w()

    w("## Real CPU Runs")
    w()
    w(
        "| Mode | V | exactness | max_abs_error | greedy_match | "
        "memory_bytes_mask | online_recovery_ops_estimate | feasibility |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for mode in report["modes_evaluated"]:
        for r in report["real_runs"][mode]:
            err = r.get("max_abs_error", r.get("max_abs_error_full_recovery"))
            err_s = "n/a" if err is None else _fmt(err)
            w(
                f"| `{mode}` | {r['vocab_size']} | {r['exactness']} | "
                f"{err_s} | {r.get('greedy_token_match_rate', 'n/a')} | "
                f"{r['memory_bytes_mask']} | "
                f"{r['online_recovery_ops_estimate']} | "
                f"{r['feasibility']} |"
            )
    w()

    w("## Symbolic Estimates (No Dense Allocation)")
    w()
    w("| Mode | V | memory_bytes_mask | online_recovery_ops_estimate | feasibility |")
    w("|---|---|---|---|---|")
    for mode, rows in report["symbolic_estimates"].items():
        for r in rows:
            w(
                f"| `{mode}` | {r['vocab_size']} | "
                f"{r['memory_bytes_mask']} | "
                f"{r['online_recovery_ops_estimate']} | "
                f"{r['feasibility']} |"
            )
    w()

    w("## Limitations")
    w()
    for item in report["limitations"]:
        w(f"- {item}")
    w()
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w("## Unsafe Wording to Avoid")
    w()
    for item in report["unsafe_wording_to_avoid"]:
        w(f"- {item}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any], *, outputs_dir: Path,
    json_filename: str = "lm_head_scalability.json",
    md_filename: str = "lm_head_scalability.md",
) -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = outputs_dir / json_filename
    md_path = outputs_dir / md_filename
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "LMHeadScalabilityConfig",
    "render_markdown",
    "run_lm_head_scalability",
    "write_reports",
]
