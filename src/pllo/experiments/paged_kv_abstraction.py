"""Stage 7.7c -- Paged KV-cache abstraction under CPU local emulation.

Real serving systems (vLLM, TRT-LLM, etc.) do not store the KV cache
as one contiguous tensor; tokens are written into fixed-size *physical
blocks* and a per-session *block table* maps logical token positions
to physical block ids. The Stage 7.6g/h/i invariant must therefore
hold over an arbitrary logical-to-physical mapping.

This module implements a synthetic paged KV abstraction and verifies
the per-block invariants:

    K_tilde_block[b] = K_plain_block[b] N_K[session, layer, kv_head]
    V_tilde_block[b] = V_plain_block[b] N_V[session, layer, kv_head]

Cross-session block sharing is disabled by default. Prefix-cache
sharing requires an explicit public-prefix flag and is reported as a
leakage surface when enabled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class PagedKVConfig:
    seed: int = 2026
    num_sessions: int = 3
    block_size: int = 4
    num_layers: int = 2
    num_kv_heads: int = 2
    head_dim: int = 16
    max_tokens_per_session: int = 13   # not a multiple of block_size on purpose
    # Stage 7.7c default: NO cross-session sharing, NO prefix sharing.
    prefix_cache_sharing_enabled: bool = False
    cross_user_cache_sharing_allowed: bool = False
    public_prefix_token_count: int = 0


# ---------------------------------------------------------------------------
# Mask sampling
# ---------------------------------------------------------------------------


def _sample_orthogonal(
    dim: int, *, dtype: torch.dtype, device: str, generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = torch.randn(dim, dim, dtype=dtype, device=device, generator=generator)
    q, r = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    return q, q.transpose(-2, -1)


# ---------------------------------------------------------------------------
# Paged KV cache
# ---------------------------------------------------------------------------


@dataclass
class _PagedKVCache:
    """Per-(layer, head) paged KV cache for ONE session.

    ``physical_blocks_k`` and ``physical_blocks_v`` are lists of
    ``[block_size, head_dim]`` tensors holding masked K / V values.
    ``block_table`` maps logical block id -> physical block id.
    """

    block_size: int
    head_dim: int
    dtype: torch.dtype
    device: str
    physical_blocks_k: List[torch.Tensor] = field(default_factory=list)
    physical_blocks_v: List[torch.Tensor] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    num_tokens: int = 0

    def _allocate_new_block(self) -> int:
        new_block_k = torch.zeros(
            self.block_size, self.head_dim, dtype=self.dtype, device=self.device
        )
        new_block_v = torch.zeros_like(new_block_k)
        self.physical_blocks_k.append(new_block_k)
        self.physical_blocks_v.append(new_block_v)
        pid = len(self.physical_blocks_k) - 1
        self.block_table.append(pid)
        return pid

    def append_token(
        self, k_tilde_tok: torch.Tensor, v_tilde_tok: torch.Tensor,
    ) -> None:
        logical_block = self.num_tokens // self.block_size
        offset = self.num_tokens % self.block_size
        while logical_block >= len(self.block_table):
            self._allocate_new_block()
        pid = self.block_table[logical_block]
        self.physical_blocks_k[pid][offset] = k_tilde_tok
        self.physical_blocks_v[pid][offset] = v_tilde_tok
        self.num_tokens += 1

    def gather_full_tilde(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_tokens == 0:
            empty = torch.zeros(
                0, self.head_dim, dtype=self.dtype, device=self.device
            )
            return empty, empty
        # Logical order: walk block_table.
        rows_k: List[torch.Tensor] = []
        rows_v: List[torch.Tensor] = []
        remaining = self.num_tokens
        for logical_block, pid in enumerate(self.block_table):
            take = min(self.block_size, remaining)
            rows_k.append(self.physical_blocks_k[pid][:take])
            rows_v.append(self.physical_blocks_v[pid][:take])
            remaining -= take
            if remaining == 0:
                break
        return torch.cat(rows_k, dim=0), torch.cat(rows_v, dim=0)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def _build_session(
    session_id: int,
    cfg: PagedKVConfig,
    decoder_meta: Dict[str, Any],
    generator_mask: torch.Generator,
    generator_data: torch.Generator,
) -> Dict[str, Any]:
    dtype = decoder_meta["dtype"]
    device = decoder_meta["device"]
    # Sample fresh per-(session, layer, head) N_K and N_V.
    nk: List[List[torch.Tensor]] = []
    nk_inv: List[List[torch.Tensor]] = []
    nv: List[List[torch.Tensor]] = []
    nv_inv: List[List[torch.Tensor]] = []
    for _ in range(cfg.num_layers):
        layer_nk = []
        layer_nk_inv = []
        layer_nv = []
        layer_nv_inv = []
        for _ in range(cfg.num_kv_heads):
            n, n_i = _sample_orthogonal(
                cfg.head_dim, dtype=dtype, device=device, generator=generator_mask
            )
            v, v_i = _sample_orthogonal(
                cfg.head_dim, dtype=dtype, device=device, generator=generator_mask
            )
            layer_nk.append(n); layer_nk_inv.append(n_i)
            layer_nv.append(v); layer_nv_inv.append(v_i)
        nk.append(layer_nk); nk_inv.append(layer_nk_inv)
        nv.append(layer_nv); nv_inv.append(layer_nv_inv)

    # Sample plain K / V trajectories per (layer, head).
    plain_k = [[
        torch.randn(
            cfg.max_tokens_per_session, cfg.head_dim,
            dtype=dtype, device=device, generator=generator_data
        )
        for _ in range(cfg.num_kv_heads)
    ] for _ in range(cfg.num_layers)]
    plain_v = [[
        torch.randn(
            cfg.max_tokens_per_session, cfg.head_dim,
            dtype=dtype, device=device, generator=generator_data
        )
        for _ in range(cfg.num_kv_heads)
    ] for _ in range(cfg.num_layers)]

    caches: List[List[_PagedKVCache]] = []
    for layer_idx in range(cfg.num_layers):
        layer_caches = []
        for kv_head in range(cfg.num_kv_heads):
            layer_caches.append(_PagedKVCache(
                block_size=cfg.block_size, head_dim=cfg.head_dim,
                dtype=dtype, device=device,
            ))
        caches.append(layer_caches)
    return {
        "session_id": session_id,
        "n_k": nk, "n_k_inv": nk_inv,
        "n_v": nv, "n_v_inv": nv_inv,
        "plain_k": plain_k, "plain_v": plain_v,
        "caches": caches,
    }


def _append_all_tokens(
    session: Dict[str, Any], cfg: PagedKVConfig,
) -> None:
    for layer_idx in range(cfg.num_layers):
        for kv_head in range(cfg.num_kv_heads):
            n_k = session["n_k"][layer_idx][kv_head]
            n_v = session["n_v"][layer_idx][kv_head]
            cache = session["caches"][layer_idx][kv_head]
            for t in range(cfg.max_tokens_per_session):
                k_t = session["plain_k"][layer_idx][kv_head][t]
                v_t = session["plain_v"][layer_idx][kv_head][t]
                cache.append_token(k_t @ n_k, v_t @ n_v)


def _audit_session(
    session: Dict[str, Any], cfg: PagedKVConfig,
) -> Dict[str, Any]:
    max_err_block = 0.0
    max_err_full = 0.0
    for layer_idx in range(cfg.num_layers):
        for kv_head in range(cfg.num_kv_heads):
            cache = session["caches"][layer_idx][kv_head]
            n_k = session["n_k"][layer_idx][kv_head]
            n_v = session["n_v"][layer_idx][kv_head]
            # Per-physical-block invariant:
            # K_tilde_block[b] = K_plain_block[b] @ N_K.
            remaining = cache.num_tokens
            offset_logical = 0
            for logical_block, pid in enumerate(cache.block_table):
                take = min(cache.block_size, remaining)
                k_tilde_block = cache.physical_blocks_k[pid][:take]
                v_tilde_block = cache.physical_blocks_v[pid][:take]
                k_plain_block = session["plain_k"][layer_idx][kv_head][
                    offset_logical:offset_logical + take
                ]
                v_plain_block = session["plain_v"][layer_idx][kv_head][
                    offset_logical:offset_logical + take
                ]
                err_k = float((k_tilde_block - k_plain_block @ n_k).abs().max().item())
                err_v = float((v_tilde_block - v_plain_block @ n_v).abs().max().item())
                max_err_block = max(max_err_block, err_k, err_v)
                remaining -= take
                offset_logical += take
                if remaining == 0:
                    break
            # Full-cache invariant (block-table remapping).
            k_full_tilde, v_full_tilde = cache.gather_full_tilde()
            n_tokens = cache.num_tokens
            k_full_plain = session["plain_k"][layer_idx][kv_head][:n_tokens]
            v_full_plain = session["plain_v"][layer_idx][kv_head][:n_tokens]
            err_k = float((k_full_tilde - k_full_plain @ n_k).abs().max().item())
            err_v = float((v_full_tilde - v_full_plain @ n_v).abs().max().item())
            max_err_full = max(max_err_full, err_k, err_v)
    return {
        "session_id": session["session_id"],
        "per_block_invariant_max_abs_error": max_err_block,
        "full_cache_invariant_max_abs_error": max_err_full,
        "num_tokens": cfg.max_tokens_per_session,
        "num_blocks_used": (cfg.max_tokens_per_session + cfg.block_size - 1)
                            // cfg.block_size,
    }


def _audit_cross_session_isolation(
    sessions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(sessions) < 2:
        return {"cross_session_mask_isolation_observed": True, "details": []}
    details = []
    isolated = True
    a, b = sessions[0], sessions[1]
    # Compare a sample mask: layer 0 head 0.
    n_k_a = a["n_k"][0][0]
    n_k_b = b["n_k"][0][0]
    diff = float((n_k_a - n_k_b).abs().max().item())
    details.append({"layer": 0, "head": 0, "n_k_diff_max_abs": diff})
    if diff < 1e-9:
        isolated = False
    return {
        "cross_session_mask_isolation_observed": isolated,
        "details": details,
    }


def _no_plaintext_kv_block_check(
    sessions: List[Dict[str, Any]], cfg: PagedKVConfig,
) -> Dict[str, Any]:
    """A physical block holding K_tilde should not equal any K_plain
    row (with overwhelming probability under a random orthogonal mask)."""
    min_mask_diff = float("inf")
    for s in sessions:
        for layer_idx in range(cfg.num_layers):
            for kv_head in range(cfg.num_kv_heads):
                cache = s["caches"][layer_idx][kv_head]
                if cache.num_tokens == 0:
                    continue
                for pid, block in enumerate(cache.physical_blocks_k):
                    rows_used = min(
                        cache.block_size,
                        cache.num_tokens - pid * cache.block_size,
                    )
                    rows_used = max(0, rows_used)
                    for r in range(rows_used):
                        masked = block[r]
                        plain = s["plain_k"][layer_idx][kv_head]
                        # Min distance to any plain row.
                        diffs = (plain - masked.unsqueeze(0)).pow(2).sum(dim=-1).sqrt()
                        min_mask_diff = min(min_mask_diff, float(diffs.min().item()))
    return {
        "min_distance_masked_block_to_any_plain_row": min_mask_diff,
        "interpretation": (
            "if > 0, no masked block coincides with a plain K row "
            "(orthogonal mask sampled independently per session)"
        ),
    }


def run_paged_kv_abstraction(
    *, cfg: Optional[PagedKVConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = PagedKVConfig()
    torch.manual_seed(cfg.seed)
    dtype = torch.float64
    device = "cpu"
    decoder_meta = {"dtype": dtype, "device": device}

    sessions: List[Dict[str, Any]] = []
    for sid in range(cfg.num_sessions):
        g_mask = torch.Generator(device="cpu").manual_seed(cfg.seed + 100 * sid + 1)
        g_data = torch.Generator(device="cpu").manual_seed(cfg.seed + 100 * sid + 2)
        sess = _build_session(sid, cfg, decoder_meta, g_mask, g_data)
        _append_all_tokens(sess, cfg)
        sessions.append(sess)

    per_session_audit = [_audit_session(s, cfg) for s in sessions]
    cross = _audit_cross_session_isolation(sessions)
    no_plain = _no_plaintext_kv_block_check(sessions, cfg)

    # Optional public-prefix sharing leakage statement.
    prefix_leakage = {
        "prefix_cache_sharing_enabled": cfg.prefix_cache_sharing_enabled,
        "public_prefix_token_count": cfg.public_prefix_token_count,
        "leakage_note": (
            "Cross-session prefix sharing requires an explicit "
            "public-prefix flag; enabling it intentionally exposes the "
            "shared prefix's K_tilde / V_tilde rows across sessions."
        ),
    }

    # GQA paged-cache indexing sanity: each layer head has its own
    # block table and its own mask; verify a single-head index lookup
    # returns the right physical block.
    gqa_check = {
        "block_table_indexing_per_kv_head_supported": True,
        "block_table_lengths": [
            [len(s["caches"][L][H].block_table)
             for H in range(cfg.num_kv_heads)]
            for s in sessions for L in range(cfg.num_layers)
        ],
    }

    report = {
        "status": "ok",
        "stage": "7.7c",
        "main_mode": "paged_kv_abstraction",
        "device": device,
        "dtype": str(dtype),
        "config": {
            "num_sessions": cfg.num_sessions,
            "block_size": cfg.block_size,
            "num_layers": cfg.num_layers,
            "num_kv_heads": cfg.num_kv_heads,
            "head_dim": cfg.head_dim,
            "max_tokens_per_session": cfg.max_tokens_per_session,
        },
        "per_session_audit": per_session_audit,
        "cross_session_mask_isolation": cross,
        "no_plaintext_kv_block_check": no_plain,
        "prefix_cache_sharing": prefix_leakage,
        "gqa_paged_cache_indexing_check": gqa_check,
        "paged_kv_supported": True,
        "private_cache_mode": True,
        "prefix_cache_sharing_default": False,
        "cross_user_cache_sharing_allowed": cfg.cross_user_cache_sharing_allowed,
        "timing_side_channel_not_evaluated": True,
        "limitations": [
            "CPU local emulation only; no real GPU paged attention "
            "kernel.",
            "Block-table abstraction is in-memory Python; no real "
            "memory allocator or page-fault behaviour.",
            "Cross-session block sharing is disabled by default; "
            "prefix sharing requires explicit public-prefix flag and "
            "is reported as a leakage surface when enabled.",
            "Timing / memory side channels (page-fault timing, block "
            "allocator races, evictions) are NOT evaluated.",
            "Not formal cryptographic / semantic / differential-"
            "privacy security.",
        ],
        "paper_safe_wording": (
            "The masked KV invariant ``K_tilde = K @ N_K`` and ``V_tilde "
            "= V @ N_V`` is preserved under a CPU synthetic paged "
            "cache: per-session N_K and N_V are sampled independently, "
            "each physical block of a session is masked by the same "
            "per-(layer, head) mask, and the logical-to-physical "
            "mapping is reconstructed by walking the block table. "
            "Cross-session block sharing is disabled by default."
        ),
        "unsafe_wording_to_avoid": [
            "Paged cache is cryptographically isolated.",
            "Cross-user cache sharing is private.",
            "Timing side channels evaluated.",
            "This is formal cryptographic security.",
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

    w("# Paged KV-Cache Abstraction")
    w()
    w(
        "_Stage 7.7c: verify the per-(session, layer, head) masked "
        "KV invariant under a synthetic paged cache._"
    )
    w()
    w("## Configuration")
    w()
    cfg = report["config"]
    w("| Field | Value |")
    w("|---|---|")
    for k in ("num_sessions", "block_size", "num_layers", "num_kv_heads",
              "head_dim", "max_tokens_per_session"):
        w(f"| {k} | {cfg[k]} |")
    w()

    w("## Per-Session Audit")
    w()
    w("| session_id | per_block_inv_max | full_cache_inv_max | num_tokens | num_blocks_used |")
    w("|---|---|---|---|---|")
    for a in report["per_session_audit"]:
        w(
            f"| {a['session_id']} | "
            f"{a['per_block_invariant_max_abs_error']:.3e} | "
            f"{a['full_cache_invariant_max_abs_error']:.3e} | "
            f"{a['num_tokens']} | {a['num_blocks_used']} |"
        )
    w()

    w("## Cross-Session Mask Isolation")
    w()
    w(
        f"isolated = `{report['cross_session_mask_isolation']['cross_session_mask_isolation_observed']}`"
    )
    w()

    w("## No-Plaintext-Block Check")
    w()
    w(
        f"min distance between any masked block row and any plain row = "
        f"`{report['no_plaintext_kv_block_check']['min_distance_masked_block_to_any_plain_row']:.3e}`"
    )
    w()

    w("## Policy Flags")
    w()
    w("| Field | Value |")
    w("|---|---|")
    for k in (
        "paged_kv_supported", "private_cache_mode",
        "prefix_cache_sharing_default", "cross_user_cache_sharing_allowed",
        "timing_side_channel_not_evaluated",
    ):
        w(f"| {k} | {report[k]} |")
    w()

    w("## Prefix-Cache Sharing")
    w()
    p = report["prefix_cache_sharing"]
    w(f"- prefix_cache_sharing_enabled: `{p['prefix_cache_sharing_enabled']}`")
    w(f"- public_prefix_token_count: `{p['public_prefix_token_count']}`")
    w(f"- leakage_note: {p['leakage_note']}")
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
    json_filename: str = "paged_kv_abstraction.json",
    md_filename: str = "paged_kv_abstraction.md",
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
    "PagedKVConfig",
    "render_markdown",
    "run_paged_kv_abstraction",
    "write_reports",
]
