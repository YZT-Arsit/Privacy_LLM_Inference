"""Representation-capture interface for attacks.

An ``AttackInputs`` bundle carries everything an attack might consume:
token ids, plaintext/protected embeddings, an observed intermediate, the
embedding table, known-plaintext pairs and defense metadata. Three sources:

  * :func:`toy_representations` -- synthetic linear/embedding model (CPU).
  * :func:`load_representations` -- from a saved ``.pt`` / ``.npz`` / ``.json``.
  * :func:`tiny_qwen_representations` -- a forward-hook capture on a *tiny random*
    Qwen2 (no download). The same hook interface extends to Qwen-7B on a server.

The toy generators also expose a differentiable ``downstream`` callable so the
optimization attacks can backprop through a known "victim" mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class AttackInputs:
    token_ids: torch.Tensor | None = None                 # (N,) or (B, S)
    plaintext_embeddings: torch.Tensor | None = None      # (N, H)
    protected_embeddings: torch.Tensor | None = None      # (N, H)
    observed_intermediate: torch.Tensor | None = None     # (N, H') attacker-visible
    qkv: dict[str, torch.Tensor] | None = None            # {"q":..,"k":..,"v":..}
    attention_scores: torch.Tensor | None = None          # (heads, S, S)
    kv_cache: Any | None = None
    logits: torch.Tensor | None = None                    # (N, V)
    embedding_table: torch.Tensor | None = None           # (V, H)
    model_weights: dict[str, torch.Tensor] | None = None  # {"public": .., "obfuscated": ..}
    known_plaintext_pairs: tuple[torch.Tensor, torch.Tensor] | None = None  # (X, X*)
    downstream: Callable[[torch.Tensor], torch.Tensor] | None = None        # victim map
    defense_metadata: dict[str, Any] = field(default_factory=dict)
    secret_metadata_present_but_not_revealed: bool = True

    def dims(self) -> dict[str, Any]:
        H = self.embedding_table.shape[1] if self.embedding_table is not None else None
        V = self.embedding_table.shape[0] if self.embedding_table is not None else None
        N = None
        for t in (self.token_ids, self.plaintext_embeddings, self.observed_intermediate):
            if t is not None:
                N = t.reshape(-1, t.shape[-1]).shape[0] if t.dim() > 1 else t.numel()
                break
        return {"hidden_size": H, "vocab_size": V, "num_samples": N}


# ---------------------------------------------------------------------------
# Toy synthetic representations
# ---------------------------------------------------------------------------
def toy_representations(
    *,
    vocab_size: int = 128,
    hidden_size: int = 32,
    num_samples: int = 16,
    seed: int = 0,
    obfuscation: str = "none",         # "none" | "orthogonal" | "random" | "fresh_pad"
    dtype: torch.dtype = torch.float32,
    downstream_out: int | None = None,
) -> AttackInputs:
    """Build a synthetic (token -> embedding -> obfuscated) toy dataset.

    ``downstream`` maps an embedding through a fixed random linear "victim" layer
    (the map the optimization attacks try to invert). ``obfuscation`` controls
    ``protected_embeddings`` and, for ``fresh_pad``, uses a *fresh* mask per
    sample (which defeats a single global known-plaintext solve).
    """
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(vocab_size, hidden_size, generator=g, dtype=dtype)
    ids = torch.randint(0, vocab_size, (num_samples,), generator=g)
    plain = table[ids]                                        # (N, H)

    def _orth(d):
        a = torch.randn(d, d, generator=g, dtype=torch.float64)
        q, r = torch.linalg.qr(a)
        q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
        return q.to(dtype)

    meta: dict[str, Any] = {"obfuscation": obfuscation}
    model_weights = None
    if obfuscation == "none":
        protected = plain.clone()
    elif obfuscation == "orthogonal":
        R = _orth(hidden_size)
        protected = plain @ R
        meta["global_matrix"] = True
    elif obfuscation == "random":
        R = torch.randn(hidden_size, hidden_size, generator=g, dtype=dtype)
        protected = plain @ R
        meta["global_matrix"] = True
    elif obfuscation == "permutation":
        # feature permutation (STIP-style): protected = plain π; multiset preserved
        perm = torch.randperm(hidden_size, generator=g)
        protected = plain[:, perm]
        meta["global_matrix"] = True
        meta["true_permutation"] = perm.tolist()
        # obfuscated weights = public weights permuted + positive scaled (ArrowMatch target)
        pub = torch.randn(hidden_size, hidden_size, generator=g, dtype=dtype)
        scales = torch.rand(hidden_size, generator=g, dtype=dtype) + 0.5
        obf = (pub * scales.unsqueeze(0))[:, perm]           # columns permuted + scaled
        model_weights = {"public": pub, "obfuscated": obf, "true_permutation": perm.tolist()}
    elif obfuscation == "fresh_pad":
        # a fresh orthogonal mask per sample -> no single global R exists
        protected = torch.stack([plain[i] @ _orth(hidden_size) for i in range(num_samples)])
        meta["global_matrix"] = False
        meta["fresh_pad"] = True
    else:
        raise ValueError(f"unknown obfuscation {obfuscation!r}")

    dout = downstream_out or hidden_size
    W = torch.randn(hidden_size, dout, generator=g, dtype=dtype)

    def downstream(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x @ W)                              # nonlinear victim map

    observed = downstream(plain).detach()
    return AttackInputs(
        token_ids=ids, plaintext_embeddings=plain, protected_embeddings=protected,
        observed_intermediate=observed, embedding_table=table, model_weights=model_weights,
        known_plaintext_pairs=(plain, protected), downstream=downstream,
        defense_metadata=meta)


# ---------------------------------------------------------------------------
# Saved-file loader
# ---------------------------------------------------------------------------
def load_representations(path: str | Path) -> AttackInputs:
    """Load an AttackInputs bundle from a ``.pt`` (torch.save of a dict)."""
    path = Path(path)
    if path.suffix == ".pt":
        d = torch.load(path, map_location="cpu")
    elif path.suffix == ".npz":
        import numpy as np
        raw = np.load(path)
        d = {k: torch.as_tensor(raw[k]) for k in raw.files}
    else:
        raise ValueError(f"unsupported representation file {path.suffix!r} (use .pt/.npz)")
    kp = None
    if "kp_X" in d and "kp_Xstar" in d:
        kp = (d["kp_X"], d["kp_Xstar"])
    return AttackInputs(
        token_ids=d.get("token_ids"),
        plaintext_embeddings=d.get("plaintext_embeddings"),
        protected_embeddings=d.get("protected_embeddings"),
        observed_intermediate=d.get("observed_intermediate"),
        embedding_table=d.get("embedding_table"),
        known_plaintext_pairs=kp,
        defense_metadata=d.get("defense_metadata", {}) if isinstance(d.get("defense_metadata"), dict) else {},
    )


# ---------------------------------------------------------------------------
# Tiny-Qwen forward-hook capture (no download; same interface for Qwen-7B)
# ---------------------------------------------------------------------------
def tiny_qwen_representations(
    *, num_samples: int = 8, seed: int = 0, capture: str = "layer0_input",
    dtype: torch.dtype = torch.float32,
) -> AttackInputs:
    """Capture an intermediate from a tiny random Qwen2 via a forward hook.

    ``capture``: currently ``layer0_input`` (the hidden states entering decoder
    layer 0). Returns token ids, the embedding table (wte), the plaintext
    embeddings and the captured intermediate. No obfuscation is applied here;
    defenses wrap this separately.
    """
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config

    cfg = make_tiny_qwen_config()
    model = load_qwen(tiny_random_qwen=cfg, seed=seed, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (1, num_samples), generator=g)

    captured = {}

    def hook(_module, inputs, _output):
        captured["h"] = inputs[0].detach()

    handle = model.model.layers[0].register_forward_hook(hook, with_kwargs=False)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()

    table = model.model.embed_tokens.weight.detach()
    plain = model.model.embed_tokens(ids).detach()[0]         # (S, H)
    observed = captured["h"][0]                               # (S, H)
    return AttackInputs(
        token_ids=ids[0], plaintext_embeddings=plain, observed_intermediate=observed,
        embedding_table=table, defense_metadata={"capture": capture, "model": "tiny_random_qwen"})


__all__ = [
    "AttackInputs", "toy_representations", "load_representations",
    "tiny_qwen_representations",
]
