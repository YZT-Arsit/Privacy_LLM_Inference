"""Permutation utilities for STIP-on-Qwen (feature-dimension permutations).

Convention: a permutation is an index vector ``p`` (LongTensor). Applying it to
the last dim is a gather ``x[..., p]`` — this is the right-multiplication ``x π``
where ``π`` is the permutation matrix with ``(xπ)[...,j] = x[...,p[j]]``.

Correctness identity used everywhere (verified in tests): for a linear
``Y = X @ W`` (W in math layout ``(d_in, d_out)``),
    ``X[..., p_in] @ W[p_in][:, p_out] == (X @ W)[..., p_out]``.

Three permutation *kinds* are needed on Qwen2:
  * ``pi`` — GLOBAL residual-stream permutation (ANY permutation of hidden_size;
    it is always undone by ``πᵀ`` before Q/K/V projection, so it never meets RoPE).
  * ``pi_attn`` — per-layer Q/K/V-internal permutation. Must be a **whole-head**
    (GQA-group) permutation so it commutes with RoPE and the head reshape, and it
    is shared by Q,K (cancels in QKᵀ) and V (cancels via o_proj). Two index
    vectors: one over ``n_q*head_dim`` (q/o) and one over ``n_kv*head_dim`` (k/v).
  * ``pi_mlp`` — per-layer intermediate permutation (ANY permutation of
    intermediate_size), shared by gate_proj and up_proj (SwiGLU fix).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def random_permutation(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(n, generator=g)


def inverse_perm(p: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), dtype=p.dtype)
    return inv


def apply_perm_last(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Right-multiply by a permutation matrix: (x π)[..., j] = x[..., p[j]]."""
    return x.index_select(-1, p.to(x.device))


def whole_head_permutation(n_q_heads: int, n_kv_heads: int, head_dim: int,
                           seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (p_q, p_kv) index vectors for a GQA-group whole-head permutation.

    A permutation ``ρ`` of the ``n_kv`` groups is drawn; each group owns
    ``n_rep = n_q // n_kv`` query heads and 1 kv head. Query heads are reordered
    by ρ at the group level (order within a group preserved); kv heads by ρ. The
    within-head ``head_dim`` order is IDENTITY (so RoPE, applied identically per
    head, commutes). Returns index vectors over ``n_q*head_dim`` and
    ``n_kv*head_dim``.
    """
    assert n_q_heads % n_kv_heads == 0, "n_q must be a multiple of n_kv (GQA)"
    n_rep = n_q_heads // n_kv_heads
    g = torch.Generator().manual_seed(int(seed))
    rho = torch.randperm(n_kv_heads, generator=g)          # group permutation

    # kv head order: group g goes to position where rho maps it
    kv_head_order = rho.tolist()
    # q head order: expand each group into its n_rep heads (order preserved)
    q_head_order = []
    for g_new in kv_head_order:
        base = g_new * n_rep
        q_head_order.extend(range(base, base + n_rep))

    def _expand(head_order, head_dim):
        idx = []
        for h in head_order:
            idx.extend(range(h * head_dim, h * head_dim + head_dim))
        return torch.tensor(idx, dtype=torch.long)

    p_q = _expand(q_head_order, head_dim)
    p_kv = _expand(kv_head_order, head_dim)
    return p_q, p_kv


@dataclass
class StipPermutations:
    """The full STIP key set for a Qwen2 model (SECRET — never serialised raw)."""

    pi: torch.Tensor                       # global residual perm (hidden_size)
    pi_c: torch.Tensor                     # classifier/lm_head perm (vocab_size)
    pi_attn_q: list[torch.Tensor] = field(default_factory=list)   # per-layer (n_q*hd)
    pi_attn_kv: list[torch.Tensor] = field(default_factory=list)  # per-layer (n_kv*hd)
    pi_mlp: list[torch.Tensor] = field(default_factory=list)      # per-layer (intermediate)

    def num_layers(self) -> int:
        return len(self.pi_mlp)


def make_stip_permutations(arch, seed: int = 0) -> StipPermutations:
    """Sample a full STIP key set for a QwenArch (see obfuscatune.qwen_config)."""
    pi = random_permutation(arch.hidden_size, seed)
    pi_c = random_permutation(arch.vocab_size, seed + 1)
    q_list, kv_list, mlp_list = [], [], []
    for li in range(arch.num_layers):
        pq, pkv = whole_head_permutation(
            arch.num_attention_heads, arch.num_key_value_heads, arch.head_dim,
            seed=1000 + li)
        q_list.append(pq)
        kv_list.append(pkv)
        mlp_list.append(random_permutation(arch.intermediate_size, 2000 + li))
    return StipPermutations(pi=pi, pi_c=pi_c, pi_attn_q=q_list,
                            pi_attn_kv=kv_list, pi_mlp=mlp_list)


__all__ = [
    "random_permutation", "inverse_perm", "apply_perm_last",
    "whole_head_permutation", "StipPermutations", "make_stip_permutations",
]
