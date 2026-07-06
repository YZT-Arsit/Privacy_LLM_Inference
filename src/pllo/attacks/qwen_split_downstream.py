"""Differentiable split-inference downstream for EIA / BRE / PIA on real Qwen.

Split at decoder layer ``k``: the device runs ``embed -> layers[:k]`` in the TEE
and sends the (obfuscated) smashed state to the untrusted GPU. This module builds
a DIFFERENTIABLE map ``F(soft_embeddings) = layers[:k](embeds)`` so the
optimization attacks can backprop a dummy input to match the observed smashed
state — the white-box split_inference threat model.

``F`` treats the N rows as ONE sequence (reshape to (1, S, H)) so causal
attention/RoPE across positions is preserved. The observed target is the
*protected* smashed state ``mask(h_k)``; ``F`` reproduces the *plaintext* ``h_k``.
So for plaintext the attack matches and recovers the input; for a masked method
the attacker has no mask and cannot match — measured resistance, not blocked.

Efficiency: the partial forward temporarily truncates ``model.model.layers`` to
the first ``k`` (same code path as the model, version-robust; only k layers run).
Both the observed capture and ``F`` use the identical truncated path, so the
final RMSNorm cancels out of the comparison.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .representations import AttackInputs


@contextmanager
def _truncated_layers(model, k: int):
    m = model.model
    saved = m.layers
    try:
        m.layers = saved[:k]
        yield
    finally:
        m.layers = saved


class SplitDownstream:
    def __init__(self, model, k: int, *, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.model = model
        self.k = k
        self.model_device = next(model.parameters()).device
        self.dtype = dtype

    def run_k(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """(1, S, H) inputs_embeds -> (1, S, H) smashed state after k layers."""
        with _truncated_layers(self.model, self.k):
            out = self.model.model(inputs_embeds=inputs_embeds)
        return out.last_hidden_state

    def downstream(self, soft_embeddings: torch.Tensor) -> torch.Tensor:
        """(S, H) -> (S, H): the victim map the optimization attacks invert.

        The optimization attacks run on CPU tensors; the heavy partial forward
        runs on the model's device. ``.to()`` keeps the map differentiable across
        the device boundary, so the CPU optimizer backprops through the GPU model.
        """
        e = soft_embeddings.to(device=self.model_device, dtype=self.dtype).unsqueeze(0)
        out = self.run_k(e)[0]
        return out.to(device="cpu", dtype=torch.float32)


def _apply_method(h: torch.Tensor, method: str, *, seed: int, cond: float) -> torch.Tensor:
    """Apply a defense's cloud-visible transform to the smashed state h (S,H)."""
    from pllo.baselines.obfuscatune.random_matrices import (matrix_with_condition_number,
                                                            orthogonal_matrix)
    s, d = h.shape
    dev = str(h.device)
    g = torch.Generator().manual_seed(seed)
    if method == "plaintext_gpu":
        return h.clone()
    if method == "stip_qwen":
        return h[:, torch.randperm(d, generator=g).to(h.device)]
    if method == "obfuscatune_qwen_orthogonal":
        R, _ = orthogonal_matrix(d, seed=seed, dtype=h.dtype, device=dev)
        return h @ R
    if method == "ours_amulet_style_signed_perm":
        perm = torch.randperm(d, generator=g).to(h.device)
        signs = torch.where(torch.rand(d, generator=g) < 0.5,
                            torch.tensor(-1.0), torch.tensor(1.0)).to(h.device)
        return h[:, perm] * signs.to(h.dtype)
    if method == "ours_amulet_style_fresh_pad":
        return torch.stack([h[i] @ orthogonal_matrix(d, seed=seed + 1 + i, dtype=h.dtype, device=dev)[0]
                            for i in range(s)])
    if method == "ours_non_isometric_variant":
        return torch.stack([h[i] @ matrix_with_condition_number(d, cond=cond, seed=seed + 7 + i,
                                                                dtype=h.dtype, device=dev)[0]
                            for i in range(s)])
    raise ValueError(method)


def build_split_attack_inputs(model, ids: torch.Tensor, *, k: int, method: str,
                              seed: int = 0, cond: float = 5.0, device: str = "cpu",
                              dtype: torch.dtype = torch.float32) -> tuple[AttackInputs, SplitDownstream]:
    """Build AttackInputs whose downstream is the real split map and whose
    observed target is the method's obfuscated smashed state at layer k."""
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    sd = SplitDownstream(model, k, device=device, dtype=dtype)
    with torch.no_grad():
        table = model.model.embed_tokens.weight.detach()
        embeds = model.model.embed_tokens(ids)
        h_k = sd.run_k(embeds)[0].detach()                      # (S, H) plaintext smashed
    protected = _apply_method(h_k, method, seed=seed, cond=cond).detach()
    # the optimization attacks run on CPU (fp32); downstream bridges to the model
    # device internally, so keep the static tensors on CPU here.
    inp = AttackInputs(
        token_ids=ids.reshape(-1).cpu(),
        plaintext_embeddings=model.model.embed_tokens(ids).detach()[0].to("cpu", torch.float32),
        protected_embeddings=protected.to("cpu", torch.float32),
        observed_intermediate=protected.to("cpu", torch.float32),
        embedding_table=table.detach().to("cpu", torch.float32),
        downstream=sd.downstream,
        defense_metadata={"method": method, "split_layer_k": k,
                          "observed_tensor_name": f"masked_smashed_state_at_layer_{k}"},
        secret_metadata_present_but_not_revealed=True)
    return inp, sd


__all__ = ["SplitDownstream", "build_split_attack_inputs"]
