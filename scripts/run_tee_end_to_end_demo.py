"""TEE boundary end-to-end demo (Stage 8.3).

Shows the trusted/untrusted split on a *stub* model (no transformer): trusted
embedding+masking -> untrusted folded LM head (the only big GEMM) -> trusted
recovery+sampling. Demonstrates that the untrusted side never sees plaintext
embeddings or plaintext logits, yet the recovered greedy tokens match the
plaintext reference; and that a wrong mask breaks recovery. numpy only.

The "decoder" here is the identity (hidden = embeddings) -- this demo validates
the BOUNDARY, not a real model. The real masked decoder lives in the untrusted
ML stack (Stage 8.2); it is intentionally out of the TEE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.tee.runtime_api import (  # noqa: E402
    MaskedLogitsPacket,
    TEEConfig,
    make_runtime,
)
from pllo.tee.simulated_runtime import build_embedding_table  # noqa: E402


def _residual_matrix(perm: np.ndarray, signs: np.ndarray,
                     dtype: np.dtype) -> np.ndarray:
    """Dense N with ``N[perm[k], k] = signs[k]`` (so ``X @ N`` == signed perm)."""
    h = perm.shape[0]
    n = np.zeros((h, h), dtype=dtype)
    n[perm, np.arange(h)] = signs.astype(dtype)
    return n


def main() -> int:
    hidden, vocab, batch, seq = 128, 2000, 3, 6
    cfg = TEEConfig(hidden_size=hidden, vocab_size=vocab, seed=4242,
                    backend="simulated")
    rt = make_runtime(cfg)
    h = rt.handles
    dt = np.dtype(cfg.dtype)

    rng = np.random.default_rng(7)
    input_ids = rng.integers(0, vocab, size=(batch, seq), dtype=np.int64)

    # --- trusted-only model weights (stub) -----------------------------
    embed = build_embedding_table(vocab, hidden, cfg.seed, dt)   # trusted
    head_w = (rng.standard_normal((hidden, vocab)) * (1.0 / hidden ** 0.5)
              ).astype(dt)                                       # trusted W_lm

    # plaintext reference (stays inside the trusted side): hidden = embeddings
    x_plain = embed[input_ids]                                   # [B,T,H]
    logits_plain = x_plain @ head_w                              # [B,T,V]
    plain_tokens = logits_plain[:, -1, :].argmax(axis=-1)

    # --- offline fold: untrusted gets ONLY the folded head -------------
    # W_tilde = N^{-1} @ W @ M_vocab ; with X_tilde = X @ N we get
    # X_tilde @ W_tilde = X @ W @ M_vocab = L @ M_vocab (masked logits).
    n_mat = _residual_matrix(h.residual_perm, h.residual_signs, dt)
    w_unmask = n_mat.T @ head_w                                  # N^{-1} @ W
    head_w_tilde = (w_unmask[:, h.vocab_perm]
                    * h.vocab_scale.astype(dt))                  # @ M_vocab

    # --- boundary flow --------------------------------------------------
    # 1) trusted: embed + mask (release masked embeddings only)
    emb_pkt = rt.embed_and_mask(input_ids)
    x_tilde = emb_pkt.masked_embeddings                          # untrusted view
    # 2) untrusted: the only big GEMM -- masked hidden @ folded head
    masked_logits = (x_tilde @ head_w_tilde)[:, -1, :]          # [B,V]
    logits_pkt = MaskedLogitsPacket(masked_logits, batch, vocab,
                                    str(masked_logits.dtype),
                                    int(masked_logits.nbytes))
    # 3) trusted: recover + greedy sample
    recovered = rt.recover_logits(logits_pkt)
    result = rt.sample(recovered)

    recover_err = float(np.abs(recovered - logits_plain[:, -1, :]).max())
    match = bool((result.next_token_ids == plain_tokens).all())

    # --- wrong-mask control --------------------------------------------
    wrong = make_runtime(TEEConfig(hidden_size=hidden, vocab_size=vocab,
                                   seed=999999, backend="simulated"))
    wrong_rec = wrong.recover_logits(logits_pkt)
    wrong_tokens = wrong.sample(wrong_rec).next_token_ids
    wrong_match = bool((wrong_tokens == plain_tokens).all())
    wrong_err = float(np.abs(wrong_rec - logits_plain[:, -1, :]).max())

    print("=== TEE boundary end-to-end demo (stub model) ===")
    print(f"shapes: input_ids={input_ids.shape} masked_emb={x_tilde.shape} "
          f"masked_logits={masked_logits.shape}")
    print(f"untrusted sees: masked embeddings + folded head + masked logits")
    print(f"untrusted NEVER sees: input_ids, plaintext embeddings, "
          f"plaintext logits")
    print(f"recovered-vs-plain max abs err : {recover_err:.3e}")
    print(f"recovered greedy tokens        : {result.next_token_ids.tolist()}")
    print(f"plaintext greedy tokens        : {plain_tokens.tolist()}")
    print(f"correct-mask token match       : {match}")
    print(f"wrong-mask token match         : {wrong_match} "
          f"(max abs err {wrong_err:.3e})")
    print(f"attestation                    : {rt.attest().tee_type} / "
          f"quote_status={rt.attest().quote_status}")

    ok = match and (not wrong_match) and recover_err < 1e-3
    print(f"\nDEMO {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
