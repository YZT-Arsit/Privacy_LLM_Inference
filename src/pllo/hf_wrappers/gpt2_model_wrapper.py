"""Obfuscated model-level wrapper for HuggingFace GPT-2 (Stages 4.7 / 4.8)."""

from __future__ import annotations

import torch

from pllo.hf_wrappers.gpt2_block_wrapper import ObfuscatedGPT2BlockWrapper
from pllo.hf_wrappers.gpt2_cache import ObfuscatedGPT2KVCache
from pllo.hf_wrappers.nonlinear_modes import (
    DEFAULT_NONLINEAR_MODE,
    normalize_nonlinear_mode,
)
from pllo.ops.mitigation_bundles import (
    DEFAULT_MITIGATION_BUNDLE,
    normalize_mitigation_bundle,
)


class ObfuscatedGPT2ModelWrapper:
    """Stage 4.7 multi-block obfuscated correctness wrapper for GPT2LMHeadModel.

    Orchestrates per-block obfuscated execution (via ObfuscatedGPT2BlockWrapper)
    and applies a diagonal vocab output mask to the LM head logits.

    The HuggingFace model is never modified. Tied embedding is preserved.

    Engineering simplifications (same as Stage 4.6):
    - Trusted LayerNorm (ln_1, ln_2, ln_f are called on plaintext)
    - Trusted activation (GELU is evaluated in plaintext domain)
    - Vocab output mask is a diagonal matrix (scaling vector) for memory
      efficiency: N_vocab = diag(scale), scale in R^{vocab_size}. For
      sshleifer/tiny-gpt2 (vocab_size=50257) a full [V,V] orthogonal matrix
      would require ~10 GB; the diagonal form is algebraically equivalent for
      correctness purposes.
    - No pad on LM head (lm_head_pad: false). Reason: vocab dimension is
      large; Stage 4.7 uses vocab output mask only.
    - KV cache and generation not implemented in this stage.
    """

    def __init__(
        self,
        model,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        use_pad: bool = True,
        pad_scale: float = 1.0,
        nonlinear_mode: str = DEFAULT_NONLINEAR_MODE,
        mitigation_bundle: str = DEFAULT_MITIGATION_BUNDLE,
    ) -> None:
        self.model = model
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_pad = use_pad
        self.pad_scale = pad_scale
        self.nonlinear_mode = normalize_nonlinear_mode(nonlinear_mode)
        self.mitigation_bundle = normalize_mitigation_bundle(mitigation_bundle)

        model.eval()

        # Per-block obfuscated wrappers (one per transformer block). Every
        # block receives the same ``nonlinear_mode`` and
        # ``mitigation_bundle``; Stage 5.3b / 5.3e do not support per-block
        # mode / bundle mixing.
        self.block_wrappers = [
            ObfuscatedGPT2BlockWrapper(
                block=block,
                config=model.config,
                dtype=dtype,
                device=self.device,
                use_pad=use_pad,
                pad_scale=pad_scale,
                nonlinear_mode=self.nonlinear_mode,
                mitigation_bundle=self.mitigation_bundle,
            )
            for block in model.transformer.h
        ]

        # Diagonal vocab output mask: N_vocab = diag(scale), scale > 0
        vocab_size = model.config.vocab_size
        self._n_vocab_scale, self._n_vocab_scale_inv = self._make_vocab_mask(vocab_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_vocab_mask(self, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagonal vocab output mask (log-normal scale, always positive)."""
        scale = torch.randn(vocab_size, dtype=self.dtype, device=self.device).exp()
        return scale, 1.0 / scale

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def pad_reports(self) -> list[dict]:
        """Per-block pad audit reports (populated after each forward call)."""
        return [w.pad_report for w in self.block_wrappers]

    @property
    def island_reports(self) -> list[dict]:
        """Per-block compatible-island audit reports."""
        return [w.island_report for w in self.block_wrappers]

    @property
    def island_summary(self) -> dict[str, object]:
        """Stage 5.3b — aggregate island audit across all blocks.

        Populated from each block's ``island_report``. When
        ``nonlinear_mode == "compatible_islands"`` every entry in
        ``blocks_with_compatible_islands`` flips to ``True`` after the first
        ``forward()`` / ``prefill()`` / ``decode_step()`` call. The summary
        is recomputed on every read, so callers do not need to invalidate
        it.
        """
        reports = self.island_reports
        num_blocks = len(reports)
        blocks_active = sum(1 for r in reports if r.get("mlp_gelu_island_active"))
        permutation_draws = sum(
            int(r.get("mlp_island_permutation_draws", 0)) for r in reports
        )
        extra_matmul = sum(
            int(r.get("online_extra_matmul_count", 0)) for r in reports
        )
        layernorm_trusted = all(
            bool(r.get("layernorm_remains_trusted", True)) for r in reports
        )
        lm_head_untouched = all(
            bool(r.get("lm_head_not_modified", True)) for r in reports
        )
        generation_untouched = all(
            bool(r.get("generation_path_not_modified", True)) for r in reports
        )
        pad_placements = sorted({
            str(r.get("mlp_island_pad_placement", "n/a")) for r in reports
        })
        pad_placement = (
            pad_placements[0] if len(pad_placements) == 1 else "/".join(pad_placements)
        )
        if self.nonlinear_mode == "compatible_islands":
            security_profile = "proxy-evaluated, not formal"
            security_caveats = [
                "Compatible mask families are weaker than unrestricted dense masks.",
                "Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.",
                "Only GPT-2 model-level wrapper is integrated; BERT/T5 not integrated.",
                "This is not a real TEE measurement.",
            ]
        else:
            security_profile = "n/a"
            security_caveats = []
        # Bundle is the same across all blocks (model-level enforced).
        bundles = sorted({str(r.get("mitigation_bundle")) for r in reports if r.get("mitigation_bundle")})
        bundle_summary = bundles[0] if len(bundles) == 1 else "/".join(bundles)
        dense_sandwich_enabled = all(
            bool(r.get("dense_sandwich_enabled", False)) for r in reports
        ) if reports else False
        fresh_permutation_enabled = all(
            bool(r.get("fresh_permutation_enabled", True)) for r in reports
        ) if reports else True
        boundary_pad_enabled = all(
            bool(r.get("boundary_pad_enabled", False)) for r in reports
        ) if reports else False
        default_on_candidate = (
            self.nonlinear_mode == "compatible_islands"
            and all(
                bool(r.get("default_on_candidate_under_stage_5_4", False))
                for r in reports
            )
            if reports
            else False
        )
        return {
            "nonlinear_mode": self.nonlinear_mode,
            "mitigation_bundle": bundle_summary or self.mitigation_bundle,
            "num_blocks": num_blocks,
            "blocks_with_compatible_islands": blocks_active,
            "total_mlp_island_permutation_draws": permutation_draws,
            "online_extra_matmul_count": extra_matmul,
            "layernorm_remains_trusted": layernorm_trusted,
            "lm_head_not_modified": lm_head_untouched,
            "generation_path_not_modified": generation_untouched,
            "pad_placement": pad_placement,
            "dense_sandwich_enabled": dense_sandwich_enabled,
            "fresh_permutation_enabled": fresh_permutation_enabled,
            "boundary_pad_enabled": boundary_pad_enabled,
            "default_on_candidate_under_stage_5_4": default_on_candidate,
            "security_profile": security_profile,
            "security_caveats": security_caveats,
            "wrapper_integration_scope": "gpt2_model_level",
        }

    # ------------------------------------------------------------------
    # Internal: LM head with vocab output mask (shared by forward / prefill / decode)
    # ------------------------------------------------------------------

    def _apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply final LayerNorm and LM head with diagonal vocab output mask."""
        model = self.model
        hidden_states = model.transformer.ln_f(hidden_states)
        batch, seq, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_size).to(
            dtype=self.dtype, device=self.device
        )
        W_vocab = model.lm_head.weight.detach().T.to(dtype=self.dtype, device=self.device)
        bias = None
        if model.lm_head.bias is not None:
            bias = model.lm_head.bias.detach().to(dtype=self.dtype, device=self.device)
        n_vocab = self._n_vocab_scale.to(dtype=self.dtype, device=self.device)
        n_vocab_inv = self._n_vocab_scale_inv.to(dtype=self.dtype, device=self.device)
        W_vocab_tilde = W_vocab * n_vocab
        logits_tilde = hidden_flat @ W_vocab_tilde
        if bias is not None:
            logits_tilde = logits_tilde + bias * n_vocab
        logits = logits_tilde * n_vocab_inv
        return logits.reshape(batch, seq, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run full GPT-2 forward and return recovered logits [batch, seq, vocab].

        attention_mask is accepted for API compatibility; only None is
        validated in Stage 4.7.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}"
            )
        model = self.model
        seq_len = input_ids.shape[1]

        token_embeds = model.transformer.wte(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = (token_embeds + position_embeds).to(
            dtype=self.dtype, device=self.device
        )

        for wrapper in self.block_wrappers:
            hidden_states = wrapper.forward(hidden_states, attention_mask=None)

        return self._apply_lm_head(hidden_states)

    # ------------------------------------------------------------------
    # Stage 4.8: prefill / decode with internal obfuscated KV cache
    # ------------------------------------------------------------------

    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ObfuscatedGPT2KVCache]:
        """Run prefill on a prompt and return ``(logits, cache)``.

        ``logits`` has shape ``[batch, prompt_len, vocab_size]``; ``cache``
        carries per-layer obfuscated K/V plus TEE-managed masks.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}"
            )
        model = self.model
        seq_len = input_ids.shape[1]

        token_embeds = model.transformer.wte(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = (token_embeds + position_embeds).to(
            dtype=self.dtype, device=self.device
        )

        layer_caches = []
        for wrapper in self.block_wrappers:
            hidden_states, layer_cache = wrapper.prefill(hidden_states, attention_mask=None)
            layer_caches.append(layer_cache)

        logits = self._apply_lm_head(hidden_states)
        cache = ObfuscatedGPT2KVCache(layers=layer_caches, seq_len=seq_len)
        return logits, cache

    def decode_step(
        self,
        input_ids: torch.Tensor,
        cache: ObfuscatedGPT2KVCache,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ObfuscatedGPT2KVCache]:
        """Run one decode step and return ``(logits, updated_cache)``.

        ``input_ids`` must have shape ``[batch, 1]``. Position id for the new
        token is ``cache.seq_len``; cache masks are reused (not re-sampled).
        """
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError(
                f"decode_step expects input_ids with shape [batch, 1], got "
                f"{tuple(input_ids.shape)}"
            )
        if len(cache.layers) != len(self.block_wrappers):
            raise ValueError(
                f"cache has {len(cache.layers)} layers but model has "
                f"{len(self.block_wrappers)} blocks"
            )
        model = self.model

        token_embeds = model.transformer.wte(input_ids)
        position_ids = torch.full(
            (input_ids.shape[0], input_ids.shape[1]),
            cache.seq_len,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_embeds = model.transformer.wpe(position_ids)
        hidden_states = (token_embeds + position_embeds).to(
            dtype=self.dtype, device=self.device
        )

        new_layer_caches = []
        for wrapper, layer_cache in zip(self.block_wrappers, cache.layers):
            hidden_states, new_layer_cache = wrapper.decode_step(hidden_states, layer_cache)
            new_layer_caches.append(new_layer_cache)

        logits = self._apply_lm_head(hidden_states)
        new_cache = ObfuscatedGPT2KVCache(
            layers=new_layer_caches, seq_len=cache.seq_len + 1
        )
        return logits, new_cache

    # ------------------------------------------------------------------
    # Stage 4.9: greedy generation built on prefill + decode_step
    # ------------------------------------------------------------------

    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, dict]:
        """Greedy-decode ``max_new_tokens`` continuations of ``input_ids``.

        Returns ``(generated_ids, trace)`` where ``generated_ids`` has shape
        ``[batch, prompt_len + max_new_tokens]`` and ``trace`` carries the
        per-step recovered logits used for the new tokens, the final cache,
        and the new-token tensor only.

        No sampling, temperature, beam search, or EOS handling. The first new
        token is selected from ``prefill_logits[:, -1, :]``; subsequent new
        tokens are selected from each ``decode_step`` output.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq], got {tuple(input_ids.shape)}"
            )
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        prefill_logits, cache = self.prefill(input_ids)
        step_logits: list[torch.Tensor] = [prefill_logits[:, -1:, :]]
        next_token = prefill_logits[:, -1, :].argmax(dim=-1)
        new_tokens: list[torch.Tensor] = [next_token]

        for _ in range(max_new_tokens - 1):
            decode_input = next_token.unsqueeze(-1)
            decode_logits, cache = self.decode_step(decode_input, cache)
            step_logits.append(decode_logits)
            next_token = decode_logits[:, -1, :].argmax(dim=-1)
            new_tokens.append(next_token)

        new_token_tensor = torch.stack(new_tokens, dim=1)
        generated_ids = torch.cat([input_ids, new_token_tensor], dim=1)

        trace = {
            "step_logits": step_logits,
            "new_tokens": new_token_tensor,
            "final_cache": cache,
            "cache_seq_len": cache.seq_len,
            "prompt_len": int(input_ids.shape[1]),
            "max_new_tokens": int(max_new_tokens),
        }
        return generated_ids, trace
