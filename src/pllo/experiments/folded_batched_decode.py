"""Batched greedy decode over the folded_remote path (T2 latency optimisation).

Amortises the per-token TDX<->H800 round trip + logits payload + GPU forward across
B independent prompt streams: one batched prefill and one batched decode call per
step drive the whole batch, instead of one round trip per token per prompt.

Correctness: batching is EXACT (bit-identical to per-stream sequential decode) when
every stream in a batch is at the same position, because the folded worker forward
is a plain tensor op over the batch dim and RoPE/causal masking key off the single
shared ``position``. So we bucket prompts by IDENTICAL tokenised length and batch
within a bucket -- no padding, no per-stream position handling, no result change.
(Variable-length batching via left-pad + per-stream positions is a larger, separate
extension and is intentionally NOT done here.)

Security is unchanged: the boundary still masks each token's embedding and recovers
each stream's logits trusted-side; only masked embeddings [B,1,H] and masked logits
[B,V] cross to the GPU -- exactly the single-stream payloads, stacked. Single TEE
entry/exit per token still holds (one mask + one recover per step, now for B streams).

This module drives an already-built ``LiteBoundary`` + a ``RemoteGpuWorker``-style
client; it does not touch the single-stream predictor path.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import torch

from pllo.protocol.tee_gpu_messages import (
    BoundaryInitRequest, MaskedDecodeRequest, MaskedPrefillRequest)


def _to_np(t: torch.Tensor):
    return np.asarray(t.detach().to("cpu").float().numpy())


def _recover_argmax(boundary, masked_logits) -> list[int]:
    """Trusted-side recover [G,V] -> per-stream greedy token ids."""
    x = (masked_logits if isinstance(masked_logits, torch.Tensor)
         else torch.as_tensor(np.asarray(masked_logits)))
    rec = boundary.recover(x.to(boundary.compute_device, boundary.fdtype))
    return rec.argmax(-1).reshape(-1).tolist()


def _bucket_by_length(id_lists: Sequence[Sequence[int]]) -> list[list[int]]:
    """Indices grouped by identical length (exact-batch => bit-identical)."""
    buckets: dict[int, list[int]] = {}
    for i, ids in enumerate(id_lists):
        buckets.setdefault(len(ids), []).append(i)
    return [buckets[L] for L in sorted(buckets)]


def decode_group(
    boundary: Any, worker: Any, ids: torch.Tensor, *, max_new_tokens: int,
    eos_ids: Iterable[int] = (), meta: dict | None = None, num_layers: int,
    session_id: str = "batch",
) -> list[list[int]]:
    """Batched greedy decode of one equal-length group ``ids`` [G, L].

    Returns G token-id lists (each truncated at its first EOS, like single-stream
    greedy with stop_on_eos). One prefill + (max_new_tokens-1) decode round trips
    for the WHOLE group.
    """
    g, seq_len = int(ids.shape[0]), int(ids.shape[1])
    eos = set(int(e) for e in eos_ids)
    md = meta or boundary.exec_metadata(seq_len=seq_len,
                                        max_new_tokens=max_new_tokens)
    worker.init(BoundaryInitRequest(
        session_id=session_id, hidden_size=int(md["hidden_size"]),
        vocab_size=int(md["vocab_size"]), num_layers=int(num_layers),
        dtype=str(md.get("dtype", "float32")),
        gpu_backend="qwen7b_folded_package", folded_lm_head=None,
        public_metadata=md))

    h = boundary.mask_embeddings(ids)                              # [G, L, H]
    pre = worker.prefill(MaskedPrefillRequest(
        session_id=session_id, masked_embeddings=_to_np(h),
        positions=list(range(seq_len)), batch_size=g, seq_len=seq_len))
    cur = _recover_argmax(boundary, pre.masked_logits)            # [G]
    done = [t in eos for t in cur]
    out: list[list[int]] = [([t] if not d else [t]) for t, d in zip(cur, done)]
    # (the first token is recorded even if it is EOS, matching single-stream greedy
    # which appends the sampled token then stops)

    pos = seq_len
    for step in range(max_new_tokens - 1):
        if all(done):
            break
        x = boundary.mask_token_embedding(torch.tensor(cur))     # [G, 1, H]
        dec = worker.decode(MaskedDecodeRequest(
            session_id=session_id, masked_embedding=_to_np(x), position=pos,
            step=step + 1))
        cur = _recover_argmax(boundary, dec.masked_logits)       # [G]
        for i, t in enumerate(cur):
            if not done[i]:
                out[i].append(t)
                if t in eos:
                    done[i] = True
        pos += 1
    return out


def batched_greedy_decode(
    boundary: Any, worker: Any, id_lists: Sequence[Sequence[int]], *,
    max_new_tokens: int, eos_ids: Iterable[int] = (), num_layers: int,
    meta: dict | None = None,
) -> list[list[int]]:
    """Greedy-decode many prompts, batching equal-length ones. Results are returned
    in the ORIGINAL order of ``id_lists`` and are bit-identical to decoding each
    prompt alone. ``worker`` is reused across buckets (its session is re-init per
    bucket)."""
    results: list[list[int] | None] = [None] * len(id_lists)
    for b, group in enumerate(_bucket_by_length(id_lists)):
        ids = torch.tensor([list(id_lists[i]) for i in group])
        toks = decode_group(
            boundary, worker, ids, max_new_tokens=max_new_tokens,
            eos_ids=eos_ids, meta=meta, num_layers=num_layers,
            session_id="batch-%d" % b)
        for k, i in enumerate(group):
            results[i] = toks[k]
    return [r or [] for r in results]
