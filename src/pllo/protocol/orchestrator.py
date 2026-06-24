"""Trusted-boundary orchestration for the TEE <-> GPU protocol.

Wires the trusted boundary runtime (Stage 8.3 ``pllo.tee``) to an untrusted GPU
worker (:mod:`pllo.protocol.gpu_worker`) over the message protocol and records
everything that crosses to the untrusted side into a
:class:`~pllo.protocol.tee_gpu_messages.ProtocolTrace`.

Trusted domain (this orchestrator + the boundary runtime process):
  raw prompt, tokenization, ``input_ids``, mask handles, the offline weight
  fold, embedding+masking, logit recovery, greedy selection, the generated
  token ids, and remasking each new token for the next step.

Untrusted domain (the GPU worker process):
  receives only masked embeddings + public metadata + the folded head; returns
  only masked logits + a public KV length.

The "decoder" exercised here is the mock identity backend (numpy, no torch), so
the whole round trip runs on CPU and the recovered greedy tokens match the
trusted plaintext reference exactly. The qwen7b backend plugs into the same
protocol on the GPU server. numpy only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pllo.protocol.gpu_worker import LocalGpuWorker, make_gpu_backend
from pllo.protocol.tee_gpu_messages import (
    BoundaryInitRequest,
    MaskedDecodeRequest,
    MaskedPrefillRequest,
    ProtocolTrace,
    RecoveredTokenResponse,
)
from pllo.tee.runtime_api import MaskedLogitsPacket, TEEConfig, make_runtime
from pllo.tee.simulated_runtime import (
    SimulatedTrustedRuntime,
    build_embedding_table,
)

__all__ = ["trusted_tokenize", "fold_lm_head", "run_protocol"]


def trusted_tokenize(prompt: str, vocab_size: int, max_len: int) -> np.ndarray:
    """Deterministic trusted tokenizer stub for the mock model.

    Real tokenization happens in the trusted domain; for the mock model (which
    has no semantic vocab) we map prompt bytes into the vocab range. The result
    is ``input_ids`` -- trusted, and never sent to the GPU worker."""
    ids = [((b * 131) + 7) % vocab_size for b in prompt.encode("utf-8")]
    if not ids:
        ids = [1]
    ids = ids[:max_len]
    return np.asarray(ids, dtype=np.int64)[None, :]         # [1, T]


def fold_lm_head(head_w: np.ndarray, handles, dtype: np.dtype) -> np.ndarray:
    """Offline trusted fold: ``W_tilde = N^{-1} @ W @ M_vocab`` ([H, V]).

    Bakes the residual + vocab masks into the LM head so that
    ``(X @ N) @ W_tilde = X @ W @ M_vocab``. Produces the single untrusted
    artifact the GPU worker is given; the raw masks never leave the trusted
    domain."""
    h = head_w.shape[0]
    n_mat = np.zeros((h, h), dtype=dtype)
    n_mat[handles.residual_perm, np.arange(h)] = \
        handles.residual_signs.astype(dtype)               # N (X @ N == perm)
    w_unmask = n_mat.T @ head_w                             # N^{-1} @ W
    return (w_unmask[:, handles.vocab_perm]
            * handles.vocab_scale.astype(dtype))           # @ M_vocab


def _plaintext_reference(embed: np.ndarray, head_w: np.ndarray,
                         input_ids: np.ndarray, max_new_tokens: int):
    """Trusted plaintext greedy decode with the identity decoder (mock model).

    Next-token logits depend only on the current token's embedding, so we can
    roll the reference sequence forward trusted-side to validate the protocol."""
    x = embed[input_ids]                                   # [B, T, H]
    prompt_logits = x[:, -1, :] @ head_w                   # [B, V]
    tok = prompt_logits.argmax(axis=-1)                    # [B]
    tokens = [tok.copy()]
    step_logits = [prompt_logits]
    for _ in range(max_new_tokens - 1):
        h = embed[tok]                                     # [B, H]
        logits = h @ head_w                                # [B, V]
        tok = logits.argmax(axis=-1)
        tokens.append(tok.copy())
        step_logits.append(logits)
    return np.stack(tokens, axis=1), step_logits           # [B, N], list[B,V]


def run_protocol(
    prompt: str,
    *,
    boundary_backend: str = "process",
    gpu_backend: str = "mock",
    max_new_tokens: int = 8,
    hidden_size: int = 128,
    vocab_size: int = 2000,
    seed: int = 4242,
    seq_len: int = 12,
    dtype: str = "float32",
    gpu_kwargs: dict[str, Any] | None = None,
    gpu_worker_url: str | None = None,
) -> dict[str, Any]:
    """Run the trusted/untrusted decode protocol and return trace + correctness.

    Returns a dict with the :class:`ProtocolTrace`, the trusted plaintext
    reference (for the audit's value checks), the recovered tokens, mask handles
    (trusted-only, for the secret-leak audit), and a wrong-mask control sample.
    Only ``gpu_backend == 'mock'`` runs end-to-end here; ``qwen7b`` requires the
    GPU server.

    If ``gpu_worker_url`` is given, the GPU worker is a **remote** HTTP server
    (cross-machine); otherwise a local spawn-process worker is used. Either way
    the boundary sends only masked tensors + public metadata."""
    np_dtype = np.dtype(dtype)
    cfg = TEEConfig(hidden_size=hidden_size, vocab_size=vocab_size, seed=seed,
                    backend=boundary_backend, dtype=dtype)

    # --- offline trusted preparation (handles + folded head) ---------------
    prep = SimulatedTrustedRuntime(cfg)                    # trusted, in-process
    handles = prep.handles
    embed = build_embedding_table(vocab_size, hidden_size, seed, np_dtype)
    rng = np.random.default_rng([seed, 0x10AD])
    head_w = (rng.standard_normal((hidden_size, vocab_size))
              * (1.0 / hidden_size ** 0.5)).astype(np_dtype)   # trusted W_lm
    folded_head = fold_lm_head(head_w, handles, np_dtype)      # public artifact

    # --- trusted tokenization + plaintext reference ------------------------
    input_ids = trusted_tokenize(prompt, vocab_size, seq_len)
    batch = int(input_ids.shape[0])
    ref_tokens, ref_step_logits = _plaintext_reference(
        embed, head_w, input_ids, max_new_tokens)

    trace = ProtocolTrace(boundary_backend=boundary_backend,
                          gpu_backend=gpu_backend,
                          max_new_tokens=max_new_tokens, tee_used_on_gpu=False)
    trace.trusted_bytes += int(input_ids.nbytes)

    def _record(direction: str, method: str, msg: Any) -> None:
        if direction == "inbound":
            trace.record_gpu_inbound(msg)
        else:
            trace.record_gpu_outbound(msg)

    boundary = make_runtime(cfg)                           # process/simulated
    if gpu_worker_url:
        from pllo.protocol.remote import RemoteGpuWorker    # stdlib client only
        worker = RemoteGpuWorker(gpu_worker_url, gpu_backend, recorder=_record)
    else:
        worker = LocalGpuWorker(gpu_backend, dict(gpu_kwargs or {}),
                                recorder=_record)
    recovered_responses: list[RecoveredTokenResponse] = []
    recovered_tokens: list[list[int]] = []
    masked_logits_first = None
    try:
        # --- init: hand the GPU only public metadata + the folded head -----
        init_req = BoundaryInitRequest(
            session_id="sess-0", hidden_size=hidden_size, vocab_size=vocab_size,
            num_layers=1, dtype=dtype, gpu_backend=gpu_backend,
            folded_lm_head=folded_head,
            public_metadata={"model": "mock-identity", "seq_len": seq_len})
        init_resp = worker.init(init_req)
        trace.tee_used_on_gpu = bool(init_resp.tee_used_on_gpu)

        # --- prefill: trusted embed+mask -> GPU -> masked logits -----------
        emb_pkt = boundary.embed_and_mask(input_ids)       # trusted boundary
        trace.bump_boundary("embed_and_mask")
        prefill_req = MaskedPrefillRequest(
            session_id="sess-0", masked_embeddings=emb_pkt.masked_embeddings,
            positions=list(range(emb_pkt.seq_len)), batch_size=emb_pkt.batch_size,
            seq_len=emb_pkt.seq_len)
        pre_resp = worker.prefill(prefill_req)
        masked_logits_first = np.asarray(pre_resp.masked_logits)

        # --- decode loop ----------------------------------------------------
        position = emb_pkt.seq_len
        masked_logits = masked_logits_first
        for step in range(max_new_tokens):
            logits_pkt = MaskedLogitsPacket(
                masked_logits=masked_logits, batch_size=batch,
                vocab_size=vocab_size, dtype=str(masked_logits.dtype),
                nbytes=int(masked_logits.nbytes))
            recovered = boundary.recover_logits(logits_pkt)    # trusted
            trace.bump_boundary("recover_logits")
            result = boundary.sample(recovered)                # trusted greedy
            trace.bump_boundary("sample")
            tok = result.next_token_ids                        # [B]
            trace.trusted_bytes += int(np.asarray(recovered).nbytes
                                       + tok.nbytes)
            recovered_responses.append(RecoveredTokenResponse(
                step=step, next_token_ids=tok.tolist(),
                recovered_logits_nbytes=int(np.asarray(recovered).nbytes)))
            recovered_tokens.append(tok.tolist())
            trace.recovered_tokens.extend(tok.tolist())
            if step == max_new_tokens - 1:
                break
            # trusted: embed + remask the new token for the next decode step
            next_emb = boundary.embed_and_mask(tok[:, None])   # [B,1,H] masked
            trace.bump_boundary("embed_and_mask")
            dec_req = MaskedDecodeRequest(
                session_id="sess-0",
                masked_embedding=next_emb.masked_embeddings,
                position=position, step=step + 1)
            dec_resp = worker.decode(dec_req)
            masked_logits = np.asarray(dec_resp.masked_logits)
            position += 1
    finally:
        worker.close()
        boundary.close()

    # --- wrong-mask control (trusted-side) ---------------------------------
    wrong_cfg = TEEConfig(hidden_size=hidden_size, vocab_size=vocab_size,
                          seed=seed + 13579, backend="simulated", dtype=dtype)
    wrong_handles = SimulatedTrustedRuntime(wrong_cfg).handles

    gen = np.asarray(recovered_tokens, dtype=np.int64).T if recovered_tokens \
        else np.zeros((batch, 0), dtype=np.int64)          # [B, N]
    return {
        "trace": trace,
        "handles": handles,
        "wrong_handles": wrong_handles,
        "input_ids": input_ids,
        "recovered_tokens": recovered_tokens,
        "reference_tokens": ref_tokens,
        "generated_token_ids": gen,
        "recovered_responses": recovered_responses,
        "masked_logits_first": masked_logits_first,
        "plaintext_logits_first": ref_step_logits[0],
        "tokens_match_reference": bool(
            np.array_equal(gen, ref_tokens)) if recovered_tokens else False,
        "init_response": init_resp,
        "gpu_worker_remote": bool(gpu_worker_url),
        "gpu_worker_url": gpu_worker_url,
    }
