"""Trusted/untrusted split of the validated masked Qwen execution.

Wraps the Stage 8.4 memory-optimized full-layer masked path
(:mod:`pllo.hf_wrappers.qwen_memory_optimized`) into a boundary-split session so
it can be driven over the TEE<->GPU protocol:

* **trusted methods** (run inside the boundary): ``mask_embeddings`` /
  ``mask_token_embedding`` (embedding lookup + residual mask), ``recover``
  (vocab-mask inverse on the masked logits), ``sample`` (greedy / temperature).
* **untrusted worker methods** (run on the GPU): ``worker_prefill`` /
  ``worker_decode`` -- given the masked hidden ``h_tilde`` they stream the masked
  decoder layers (folded weights, chunked down-proj, masked KV cache) and return
  **masked logits** ``logits_tilde``. They never touch the plaintext hidden, the
  input ids, the recovered logits, or the sampled tokens.

Nothing here enters a TEE: the decoder blocks / attention / MLP / KV cache / LM
head run on the untrusted side (``tee_used_on_gpu = False``). The residual /
vocab masks are the boundary's; the worker consumes only folded artifacts +
public rope/config. All numerics reuse the already-validated kernels, so the
recovered logits equal the plaintext reference to fp tolerance (the same property
``masked_prefill_full_logits`` checks).
"""

from __future__ import annotations

from typing import Any

import torch

from pllo.hf_wrappers.hf_causal_lm_skeleton import (
    HFCausalLMSkeletonConfig,
    generate_hf_causal_lm_masks,
)
from pllo.hf_wrappers.llama_qwen_single_block import (
    extract_hf_single_block_weights,
    infer_config_from_hf_layer,
)
from pllo.hf_wrappers.qwen_memory_optimized import (
    MemoryOptimizedConfig,
    align_qk_weights_to_hf_rope,
    _base_model,
    _cfg_to,
    _extract_boundary,
    _masked_block_decode_chunked,
    _masked_block_prefill_chunked,
    _move_folded,
    _resolve_dtype,
    _vocab_to,
    fold_layer_attention_and_up,
)
from pllo.hf_wrappers.qwen_memory_optimized import _boundary_weights  # noqa: E402
from pllo.ops.causal_lm_boundaries import (
    fold_final_norm_lm_head_with_vocab_mask,
    greedy_sample,
    recover_vocab_logits,
    trusted_embedding_lookup,
)
from pllo.ops.nonlinear_islands import rmsnorm_core
from pllo.ops.rope import build_rope_cache

__all__ = ["MaskedQwenSession"]


class MaskedQwenSession:
    """Boundary-split masked Qwen generation over the validated Stage 8.4 path."""

    def __init__(self, model: Any, model_config: Any,
                 config: MemoryOptimizedConfig) -> None:
        self.config = config
        self.compute_device = torch.device(config.device)
        self.fold_device = torch.device(config.folded_weight_device)
        self.fdtype = _resolve_dtype(config.folding_dtype)
        self.chunk = config.mlp_down_chunk_size
        self.model = model
        self.model_config = model_config
        self.base = _base_model(model)
        total = len(self.base.layers)
        self.n = (total if config.num_layers is None
                  else min(int(config.num_layers), total))
        self.eps = float(getattr(model_config, "rms_norm_eps", 1e-5))

        self.boundary = _extract_boundary(model, model_config, self.fdtype,
                                          self.compute_device)
        self.bw = _boundary_weights(self.boundary)
        self.layer_configs = [
            infer_config_from_hf_layer(self.base.layers[i], model_config,
                                       self.fdtype, str(self.fold_device))
            for i in range(self.n)]
        skel = HFCausalLMSkeletonConfig(
            model_family=str(getattr(model_config, "model_type", "qwen2")),
            prefill_seq_len=config.seq_len,
            decode_steps=max(0, config.max_new_tokens - 1), max_layers=self.n,
            dtype=self.fdtype, device=str(self.fold_device), seed=config.seed,
            mask_mode=config.mask_mode,
            residual_mask_strategy=config.residual_mask_strategy,
            mask_block_size=config.mask_block_size)
        self.masks = generate_hf_causal_lm_masks(
            self.boundary, self.layer_configs, skel)

        # trusted-only mask state
        self._n0 = self.masks.residual_masks[0].to(self.compute_device)
        self._vocab_mask = _vocab_to(self.masks.vocab_mask, self.compute_device)
        n_res_inv_last = self.masks.residual_mask_inverses[-1].to(
            self.compute_device)
        # folded final-norm+LM-head (public artifact the worker uses)
        self._w_lm_tilde = fold_final_norm_lm_head_with_vocab_mask(
            self.bw.final_norm_weight, self.bw.lm_head_weight, n_res_inv_last,
            self._vocab_mask)

        head_dim = self.layer_configs[0].head_dim
        rope_theta = self.layer_configs[0].rope_theta
        max_pos = config.seq_len + max(0, config.max_new_tokens - 1) + 1
        self._cos, self._sin = build_rope_cache(
            max_pos, head_dim, rope_theta, self.fdtype, self.compute_device)
        self.tee_used_on_gpu = False

    # -- trusted boundary methods --------------------------------------------
    def embed_plain(self, input_ids: torch.Tensor) -> torch.Tensor:
        return trusted_embedding_lookup(
            input_ids.to(self.compute_device), self.boundary.embed_tokens_weight)

    def mask_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``h_tilde = embed[input_ids] @ N`` -- the only embedding the GPU sees."""
        return self.embed_plain(input_ids) @ self._n0

    def mask_token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.reshape(-1, 1).to(self.compute_device)
        return self.embed_plain(ids) @ self._n0          # [B, 1, H]

    def recover(self, logits_tilde: torch.Tensor) -> torch.Tensor:
        return recover_vocab_logits(logits_tilde, self._vocab_mask)

    @staticmethod
    def sample_greedy(recovered_logits_last: torch.Tensor) -> torch.Tensor:
        return greedy_sample(recovered_logits_last)

    # -- untrusted worker methods (masked logits only) ------------------------
    def _empty_cache(self) -> None:
        if (self.config.empty_cache_between_layers
                and self.compute_device.type == "cuda"):
            torch.cuda.empty_cache()

    def _folded_layer(self, ell: int):
        """Stream-fold one layer's masked weights (untrusted compute)."""
        w = extract_hf_single_block_weights(self.base.layers[ell], self.fdtype,
                                            str(self.fold_device))
        if self.config.align_rope_to_hf:
            c = self.layer_configs[ell]
            w = align_qk_weights_to_hf_rope(w, c.num_heads,
                                            c.num_key_value_heads, c.head_dim)
        bm = self.masks.layer_block_masks[ell]
        folded = fold_layer_attention_and_up(w, bm)
        if self.fold_device != self.compute_device:
            folded = _move_folded(folded, self.compute_device)
        down_info = (w.down_proj_weight, bm["perm"], bm["n_res"],
                     None if w.down_proj_bias is None
                     else (w.down_proj_bias @ bm["n_res"]))
        cfg_c = _cfg_to(self.layer_configs[ell], self.compute_device)
        return folded, down_info, cfg_c

    def _final_head(self, h_tilde: torch.Tensor) -> torch.Tensor:
        return rmsnorm_core(h_tilde, self.eps) @ self._w_lm_tilde

    # -- folded-package export (trusted setup) -------------------------------
    def export_folded_layer_tensors(self, ell: int) -> dict[str, torch.Tensor]:
        """Folded operators for ONE layer as a name->tensor dict, for the folded
        weight package. Contains ONLY folded operators (``*_tilde``): attention
        q/k/v/o, MLP gate/up, and the fully-folded down projection
        ``wdown_tilde = down[perm] @ n_res``. The masks (``perm``/``n_res``) are
        used to compute these but are NEVER part of the output.

        When ``config.use_linear_boundary_pad`` is set, also emits the per-module
        Linear-boundary additive pad ``<w>_xpad_tilde`` (= ``T N_in``) and
        compensation ``<w>_cpad_tilde`` (= ``T W N_out``) for q/k/v/o/gate/up/down
        -- composed offsets only (raw pads/masks never leave the trusted setup)."""
        folded, down_info, _ = self._folded_layer(ell)
        down_w, perm, n_res, bdown_tilde = down_info
        out = {k: v for k, v in folded.items() if isinstance(v, torch.Tensor)}
        out["wdown_tilde"] = down_w.index_select(0, perm) @ n_res
        if bdown_tilde is not None:
            out["bdown_tilde"] = bdown_tilde
        if getattr(self.config, "use_linear_boundary_pad", False):
            from pllo.deployment.linear_boundary_pad import (
                add_input_pads_to_folded_layer)
            # independent pad per (session seed, layer); reproducible builds.
            add_input_pads_to_folded_layer(
                out, seed=int(self.config.seed) + 31 * (ell + 1),
                scale=float(getattr(self.config, "linear_pad_scale", 0.1)))
        return out

    def export_folded_head_tensors(self) -> dict[str, torch.Tensor]:
        """Folded final-norm + LM-head operator (with the vocab mask baked in)
        as ``{"w_lm_tilde": ...}`` -- the only head artifact the worker needs.
        With ``config.use_linear_boundary_pad`` also emits the head's input pad +
        compensation (``w_lm_xpad_tilde`` / ``w_lm_cpad_tilde``)."""
        out = {"w_lm_tilde": self._w_lm_tilde}
        if getattr(self.config, "use_linear_boundary_pad", False):
            from pllo.deployment.linear_boundary_pad import (
                add_input_pad_to_folded_head)
            add_input_pad_to_folded_head(
                out, seed=int(self.config.seed),
                scale=float(getattr(self.config, "linear_pad_scale", 0.1)))
        return out

    def worker_prefill(self, h_tilde: torch.Tensor) -> dict[str, Any]:
        """Masked prefill: ``h_tilde`` -> masked logits ``[B, T, V]`` + masked KV."""
        h = h_tilde.to(self.compute_device)
        kv: list[dict[str, Any]] = []
        for ell in range(self.n):
            folded, down_info, cfg_c = self._folded_layer(ell)
            out = _masked_block_prefill_chunked(
                h, folded, down_info, cfg_c, self._cos, self._sin, self.chunk)
            h = out["y_tilde"]
            kv.append(out["cache"])
            del folded, down_info
            self._empty_cache()
        return {"logits_tilde": self._final_head(h), "kv": kv}

    def worker_decode(self, x_next_tilde: torch.Tensor, kv: list[dict[str, Any]],
                      position: int) -> dict[str, Any]:
        """Masked one-token decode: returns masked logits ``[B, 1, V]`` + KV."""
        h = x_next_tilde.to(self.compute_device)
        new_kv: list[dict[str, Any]] = []
        for ell in range(self.n):
            folded, down_info, cfg_c = self._folded_layer(ell)
            out = _masked_block_decode_chunked(
                h, kv[ell], folded, down_info, cfg_c, self._cos, self._sin,
                position, self.chunk)
            h = out["y_tilde"]
            new_kv.append(out["cache"])
            del folded, down_info
            self._empty_cache()
        return {"logits_tilde": self._final_head(h), "kv": new_kv}
