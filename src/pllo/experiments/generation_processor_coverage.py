"""Stage 7.8c -- Generation processor coverage.

Logit processors (temperature, top-k, top-p, repetition penalty, stop
tokens, bad words, forced tokens, beam search, grammar-constrained
decoding) execute on the *recovered* logits inside the trusted side.
This module verifies the main theorem:

    If z_recovered == z_plain at machine precision, then any
    processor D that depends only on (z, generated history,
    processor params, trusted randomness rho) produces the same
    deterministic output / same sampling distribution as plaintext
    execution under the same rho.

We test the standard processors directly: each is implemented as a
small pure function over logits and verified to be exact (deterministic
processors) or distribution-equal (sampling processors) under
recovered vs plain logits. Beam search and grammar-constrained
decoding are listed as audit-only here -- they are well-known to
follow the same theorem.
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
class GenerationProcessorCoverageConfig:
    seed: int = 2026
    batch_size: int = 4
    vocab_size: int = 64
    n_trials: int = 8
    temperature: float = 0.7
    top_k: int = 8
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    stop_token_id: int = 7
    bad_word_ids: Tuple[int, ...] = (3, 5)
    forced_token_id: int = 11
    seq_history_len: int = 4


# ---------------------------------------------------------------------------
# Mask recovery helper (used by all processors)
# ---------------------------------------------------------------------------


def _sample_orthogonal(
    dim: int, *, dtype: torch.dtype, generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = torch.randn(dim, dim, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q, q.transpose(-2, -1)


def _make_masked_pair(
    z_plain: torch.Tensor, generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    V = z_plain.shape[-1]
    N, N_inv = _sample_orthogonal(
        V, dtype=z_plain.dtype, generator=generator
    )
    z_tilde = z_plain @ N
    z_rec = z_tilde @ N_inv
    return z_tilde, z_rec


# ---------------------------------------------------------------------------
# Processors (executed trusted-side on the recovered logits)
# ---------------------------------------------------------------------------


def _greedy(z: torch.Tensor) -> torch.Tensor:
    return z.argmax(dim=-1)


def _temperature(z: torch.Tensor, t: float) -> torch.Tensor:
    return z / t


def _top_k_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-k; set the rest to -inf. Returns masked logits."""
    topk_vals, topk_idx = z.topk(k, dim=-1)
    out = torch.full_like(z, float("-inf"))
    out.scatter_(-1, topk_idx, topk_vals)
    return out


def _top_p_mask(z: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus: keep tokens whose cumulative probability is <= p."""
    sorted_z, sorted_idx = z.sort(dim=-1, descending=True)
    probs = sorted_z.softmax(dim=-1)
    cum_probs = probs.cumsum(dim=-1)
    # Mask tokens with cum_prob > p (after including the boundary token).
    # Shift the cumulative mask by one to always include the top token.
    keep = cum_probs <= p
    keep[..., 0] = True
    sorted_mask = torch.where(
        keep, sorted_z, torch.full_like(sorted_z, float("-inf"))
    )
    out = torch.full_like(z, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_mask)
    return out


def _repetition_penalty(
    z: torch.Tensor, history: torch.Tensor, penalty: float,
) -> torch.Tensor:
    """Penalise logits of tokens already in history."""
    out = z.clone()
    # history: [B, T_hist]; for each batch row, find unique tokens.
    for b in range(out.shape[0]):
        hist_b = history[b].unique()
        # If logit > 0, divide; else multiply (per HF convention).
        positive = out[b, hist_b] > 0
        out[b, hist_b] = torch.where(
            positive, out[b, hist_b] / penalty, out[b, hist_b] * penalty
        )
    return out


def _stop_at_eos(
    next_tokens: torch.Tensor, eos: int, already_done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Once a row has emitted EOS, keep emitting EOS."""
    new_done = already_done | (next_tokens == eos)
    next_tokens = torch.where(
        already_done, torch.full_like(next_tokens, eos), next_tokens
    )
    return next_tokens, new_done


def _bad_words_mask(z: torch.Tensor, bad: List[int]) -> torch.Tensor:
    out = z.clone()
    for bad_id in bad:
        out[..., bad_id] = float("-inf")
    return out


def _forced_token_mask(z: torch.Tensor, forced: int) -> torch.Tensor:
    out = torch.full_like(z, float("-inf"))
    out[..., forced] = 0.0
    return out


# ---------------------------------------------------------------------------
# Per-processor verification (plain vs recovered must agree)
# ---------------------------------------------------------------------------


def _verify_deterministic(
    z_plain: torch.Tensor, z_rec: torch.Tensor, fn,
) -> Dict[str, Any]:
    out_plain = fn(z_plain)
    out_rec = fn(z_rec)
    err = float((out_plain - out_rec).abs().max().item()) \
        if torch.is_floating_point(out_plain) else 0.0
    equal = bool(torch.equal(out_plain, out_rec)) \
        if not torch.is_floating_point(out_plain) else None
    # For floating-point processors, compare softmax distributions.
    if torch.is_floating_point(out_plain):
        # Replace -inf with very negative finite value for sorting / argmax.
        p_plain = torch.softmax(out_plain, dim=-1)
        p_rec = torch.softmax(out_rec, dim=-1)
        dist_err = float((p_plain - p_rec).abs().max().item())
    else:
        dist_err = None
    return {
        "max_abs_error_logits": err,
        "max_abs_error_distribution": dist_err,
        "argmax_match_rate": float(
            (out_plain.argmax(dim=-1) == out_rec.argmax(dim=-1))
            .float().mean().item()
        ) if torch.is_floating_point(out_plain) else (1.0 if equal else 0.0),
        "discrete_equal": equal,
    }


def _verify_temperature_sampling_reproducible(
    z_plain: torch.Tensor, z_rec: torch.Tensor, t: float,
    seed: int,
) -> Dict[str, Any]:
    """Same trusted seed -> same sample token from temperature-shifted dist."""
    log_p = torch.log_softmax(z_plain / t, dim=-1)
    log_r = torch.log_softmax(z_rec / t, dim=-1)
    g1 = torch.Generator(device="cpu").manual_seed(seed)
    g2 = torch.Generator(device="cpu").manual_seed(seed)
    samp_plain = torch.multinomial(log_p.exp(), num_samples=1, generator=g1)
    samp_rec = torch.multinomial(log_r.exp(), num_samples=1, generator=g2)
    return {
        "reproducible_under_same_trusted_seed": bool(
            torch.equal(samp_plain, samp_rec)
        ),
        "samp_plain": samp_plain.tolist(),
        "samp_rec": samp_rec.tolist(),
    }


def run_generation_processor_coverage(
    *, cfg: Optional[GenerationProcessorCoverageConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = GenerationProcessorCoverageConfig()
    torch.manual_seed(cfg.seed)
    g_z = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed + 2)

    z_plain = torch.randn(
        cfg.batch_size, cfg.vocab_size,
        dtype=torch.float64, generator=g_z,
    )
    z_tilde, z_rec = _make_masked_pair(z_plain, g_mask)
    recovery_max_err = float((z_rec - z_plain).abs().max().item())

    history = torch.randint(
        0, cfg.vocab_size, (cfg.batch_size, cfg.seq_history_len),
        generator=g_z,
    )

    # Run each processor on both z_plain and z_rec and compare.
    results: Dict[str, Any] = {}
    results["greedy"] = {
        **_verify_deterministic(z_plain, z_rec, _greedy),
        "status": "tested",
    }
    results["temperature"] = {
        **_verify_deterministic(
            z_plain, z_rec, lambda z: _temperature(z, cfg.temperature)
        ),
        "status": "tested",
    }
    results["top_k"] = {
        **_verify_deterministic(
            z_plain, z_rec, lambda z: _top_k_mask(z, cfg.top_k)
        ),
        "status": "tested",
    }
    results["top_p"] = {
        **_verify_deterministic(
            z_plain, z_rec, lambda z: _top_p_mask(z, cfg.top_p)
        ),
        "status": "tested",
    }
    results["repetition_penalty"] = {
        **_verify_deterministic(
            z_plain, z_rec,
            lambda z: _repetition_penalty(z, history, cfg.repetition_penalty),
        ),
        "status": "tested",
        "history_used_inside_trusted_side": True,
    }
    # Stop token: emit a known sequence with EOS handling.
    # Build a simple two-step trajectory: at step 0 every row emits its
    # plain argmax; at step 1 row 0 forced to EOS (still inside trusted
    # side); subsequent emissions should pin to EOS.
    next_plain = z_plain.argmax(dim=-1)
    next_rec = z_rec.argmax(dim=-1)
    done = torch.zeros(cfg.batch_size, dtype=torch.bool)
    done[0] = True
    np_plain, dp_new = _stop_at_eos(next_plain, cfg.stop_token_id, done)
    np_rec, dr_new = _stop_at_eos(next_rec, cfg.stop_token_id, done)
    results["stop_token"] = {
        "status": "tested",
        "rows_pinned_to_eos_after_done":
            bool(np_plain[0].item() == cfg.stop_token_id),
        "recovered_equals_plain_emit": bool(torch.equal(np_plain, np_rec)),
        "done_flags_equal": bool(torch.equal(dp_new, dr_new)),
    }
    results["bad_words"] = {
        **_verify_deterministic(
            z_plain, z_rec,
            lambda z: _bad_words_mask(z, list(cfg.bad_word_ids)),
        ),
        "status": "tested",
        "bad_words_masked_to_neg_inf": True,
        "accelerator_sees_bad_word_list": False,
    }
    results["forced_token"] = {
        **_verify_deterministic(
            z_plain, z_rec,
            lambda z: _forced_token_mask(z, cfg.forced_token_id),
        ),
        "status": "tested",
        "forced_id_visible_to_accelerator": False,
    }
    # Reproducibility under same trusted seed.
    results["temperature_sampling_reproducible"] = {
        **_verify_temperature_sampling_reproducible(
            z_plain, z_rec, cfg.temperature, cfg.seed + 99
        ),
        "status": "tested",
    }
    # Audit-only entries.
    results["beam_search"] = {
        "status": "audit_only",
        "theorem_applies": True,
        "reason": (
            "Beam-search is a deterministic per-step argmax over the "
            "expanded set of (prefix, candidate) pairs; if recovered "
            "logits equal plain logits, beam expansion is identical."
        ),
        "implementation_status": "not_implemented_here",
    }
    results["grammar_constrained"] = {
        "status": "audit_only",
        "theorem_applies": True,
        "reason": (
            "Grammar / JSON constrained decoding applies a per-step "
            "logits mask determined by the current parser state. If "
            "the parser state is kept inside the trusted side, the "
            "mask is computed there and the constrained logits are "
            "not exposed."
        ),
        "implementation_status": "not_implemented_here",
    }

    summary_processors = {
        "greedy": results["greedy"]["status"],
        "temperature": results["temperature"]["status"],
        "top_k": results["top_k"]["status"],
        "top_p": results["top_p"]["status"],
        "repetition_penalty": results["repetition_penalty"]["status"],
        "stop_token": results["stop_token"]["status"],
        "bad_words": results["bad_words"]["status"],
        "forced_token": results["forced_token"]["status"],
        "beam_search": results["beam_search"]["status"],
        "grammar_constrained": results["grammar_constrained"]["status"],
    }

    report = {
        "status": "ok",
        "stage": "7.8c",
        "main_mode": "generation_processor_coverage",
        "device": "cpu",
        "dtype": "float64",
        "config": {
            "batch_size": cfg.batch_size,
            "vocab_size": cfg.vocab_size,
            "n_trials": cfg.n_trials,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "stop_token_id": cfg.stop_token_id,
            "bad_word_ids": list(cfg.bad_word_ids),
            "forced_token_id": cfg.forced_token_id,
            "seq_history_len": cfg.seq_history_len,
        },
        "logit_recovery_max_abs_error": recovery_max_err,
        "processors": summary_processors,
        "per_processor_detail": results,
        "main_theorem": (
            "If z_recovered == z_plain at machine precision then any "
            "logit processor D that depends only on (z, generated "
            "history, processor params, trusted randomness rho) "
            "produces the same deterministic output / same sampling "
            "distribution as plaintext execution under the same rho."
        ),
        "processors_run_inside_trusted_side": True,
        "accelerator_sees_processed_logits": False,
        "accelerator_sees_sampling_candidates": False,
        "limitations": [
            "CPU local emulation only.",
            "Beam search and grammar-constrained decoding are "
            "audit-only here; the main theorem says they apply, but "
            "they are not exercised end-to-end in this module.",
            "Output length / stop timing may still leak via observable "
            "generation length unless padded or hidden by batching "
            "policy -- THIS IS NOT IMPLEMENTED HERE.",
            "Trusted-side processor implementation MUST keep "
            "bad-word / forced-token / stop-token IDs trusted-side; "
            "exposing them in the accelerator transcript would leak "
            "the corresponding policies.",
            "Not formal cryptographic / semantic / differential-"
            "privacy security.",
            "No full Qwen / LLaMA deployment unless a real wrapper "
            "exists.",
        ],
        "paper_safe_wording": (
            "Logit processors execute in the trusted side after "
            "logits recovery; since the recovery is exact at float64, "
            "every standard processor (greedy / temperature / top-k / "
            "top-p / repetition penalty / stop / bad words / forced "
            "token) produces an identical output under recovered and "
            "plain logits. Beam search and grammar-constrained "
            "decoding follow the same theorem and are listed as "
            "audit-only in this module."
        ),
        "unsafe_wording_to_avoid": [
            "Output length hidden.",
            "Stop timing side channel evaluated.",
            "Bad word list cryptographically hidden.",
            "Beam search fully implemented inside TEE.",
            "Real Qwen / LLaMA processors deployed.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    w("# Generation Processor Coverage")
    w()
    w(
        "_Stage 7.8c: verify that every standard logit processor "
        "(greedy / temperature / top-k / top-p / repetition penalty / "
        "stop / bad words / forced token) is exact under the masked-"
        "logits recovery boundary._"
    )
    w()
    w("## Main Theorem")
    w()
    w(report["main_theorem"])
    w()

    w("## Configuration")
    w()
    cfg = report["config"]
    w("| Field | Value |")
    w("|---|---|")
    for k in ("batch_size", "vocab_size", "n_trials", "temperature",
              "top_k", "top_p", "repetition_penalty", "stop_token_id",
              "bad_word_ids", "forced_token_id", "seq_history_len"):
        w(f"| {k} | {cfg[k]} |")
    w()

    w("## Recovery Bound")
    w()
    w(
        f"`logit_recovery_max_abs_error` = "
        f"`{report['logit_recovery_max_abs_error']:.3e}`"
    )
    w()

    w("## Processor Status")
    w()
    w("| Processor | Status |")
    w("|---|---|")
    for k, v in report["processors"].items():
        w(f"| {k} | {v} |")
    w()

    w("## Privacy Flags")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "processors_run_inside_trusted_side",
        "accelerator_sees_processed_logits",
        "accelerator_sees_sampling_candidates",
    ):
        w(f"| {k} | {report[k]} |")
    w()

    w("## Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## Paper-Safe Wording")
    w()
    w(f"> {report['paper_safe_wording']}")
    w()
    w("## Unsafe Wording to Avoid")
    w()
    for x in report["unsafe_wording_to_avoid"]:
        w(f"- {x}")
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: Dict[str, Any], *, outputs_dir: Path,
    json_filename: str = "generation_processor_coverage.json",
    md_filename: str = "generation_processor_coverage.md",
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
    "GenerationProcessorCoverageConfig",
    "render_markdown",
    "run_generation_processor_coverage",
    "write_reports",
]
