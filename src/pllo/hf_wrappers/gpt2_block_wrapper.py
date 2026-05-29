"""Obfuscated wrapper for a single HuggingFace GPT-2 block."""

from __future__ import annotations

import torch

from pllo.backends.simulated_tee import SimulatedTEE
from pllo.backends.untrusted_gpu import UntrustedGPUExecutor
from pllo.masks.mask_state import MaskState
from pllo.model_zoo.gpt2_conv1d_adapter import extract_conv1d_as_linear
from pllo.ops.attention import generate_head_masks, split_heads
from pllo.hf_wrappers.gpt2_attention_wrapper import (
    obfuscated_gpt2_attention,
    obfuscated_gpt2_attention_decode,
    obfuscated_gpt2_attention_prefill,
)
from pllo.hf_wrappers.gpt2_cache import ObfuscatedGPT2LayerCache


class ObfuscatedGPT2BlockWrapper:
    """Stage 4.6 obfuscated correctness wrapper for one GPT-2 block.

    The wrapper reads weights from a HuggingFace block but does not modify it.
    LayerNorm and MLP activation remain trusted-side shortcuts in this stage.
    """

    def __init__(
        self,
        block,
        config,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        use_pad: bool = False,
        pad_scale: float = 1.0,
    ) -> None:
        self.block = block
        self.config = config
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_pad = use_pad
        self.pad_scale = pad_scale
        self.tee = SimulatedTEE(dtype=dtype, device=self.device)
        self.executor = UntrustedGPUExecutor()
        self.pad_report: dict[str, object] = self._new_pad_report()

    def _new_pad_report(self) -> dict[str, object]:
        return {
            "use_pad": self.use_pad,
            "attn_c_attn_pad": False,
            "attn_c_proj_pad": False,
            "mlp_c_fc_pad": False,
            "mlp_c_proj_pad": False,
            "compensation_formula": "C_T = T W N_out",
            "input_form": "X_tilde = (X - T) N_in",
            "pad_tensor_ids": [],
            "fresh_pad_count": 0,
            "fresh_pad_unique": True,
            "untrusted_receives_plain_pad": False,
        }

    def _finalize_pad_report(self) -> None:
        pad_ids = self.pad_report.get("pad_tensor_ids", [])
        self.pad_report["fresh_pad_count"] = len(pad_ids)
        self.pad_report["fresh_pad_unique"] = len(set(pad_ids)) == len(pad_ids)
        if not self.use_pad:
            self.pad_report["reason"] = "use_pad is false; all GPT-2 Conv1D paths run mask-only"

    def _initial_hidden_state(self, hidden_states: torch.Tensor) -> tuple[MaskState, torch.Tensor]:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        state = self.tee.create_linear_mask_state(flat, hidden_states.shape[-1], use_pad=False)
        state.n_out = state.n_in
        state.n_out_inv = state.n_in_inv
        return state, (flat @ state.n_out).reshape_as(hidden_states)

    def _recover(self, hidden_tilde: torch.Tensor, state: MaskState) -> torch.Tensor:
        hidden_size = hidden_tilde.shape[-1]
        return self.tee.recover_output(hidden_tilde.reshape(-1, hidden_size), state).reshape_as(hidden_tilde)

    def _obfuscated_mlp(self, x_plain: torch.Tensor, residual_state: MaskState) -> torch.Tensor:
        hidden_size = x_plain.shape[-1]
        flat = x_plain.reshape(-1, hidden_size)
        w_fc, b_fc = extract_conv1d_as_linear(self.block.mlp.c_fc)
        w_proj, b_proj = extract_conv1d_as_linear(self.block.mlp.c_proj)
        w_fc = w_fc.to(dtype=x_plain.dtype, device=x_plain.device)
        b_fc = None if b_fc is None else b_fc.to(dtype=x_plain.dtype, device=x_plain.device)
        w_proj = w_proj.to(dtype=x_plain.dtype, device=x_plain.device)
        b_proj = None if b_proj is None else b_proj.to(dtype=x_plain.dtype, device=x_plain.device)

        fc_state = self.tee.create_linear_mask_state(
            flat,
            w_fc.shape[1],
            use_pad=self.use_pad,
            pad_scale=self.pad_scale,
        )
        fc_compensation = self.tee.make_linear_pad_compensation(w_fc, fc_state)
        self.pad_report["mlp_c_fc_pad"] = fc_state.pad is not None
        if fc_state.pad is not None:
            self.pad_report.setdefault("pad_tensor_ids", []).append(id(fc_state.pad))
        fc_tilde = self.executor.linear_forward(
            self.tee.obfuscate_input(flat, fc_state),
            *self.tee.transform_linear_weight(w_fc, b_fc, fc_state),
            fc_compensation,
        )
        fc_plain = self.tee.recover_output(fc_tilde, fc_state).reshape(*x_plain.shape[:-1], w_fc.shape[1])
        activated = self.block.mlp.act(fc_plain)
        activated_flat = activated.reshape(-1, activated.shape[-1])

        proj_state = self.tee.create_linear_mask_state(
            activated_flat,
            w_proj.shape[1],
            use_pad=self.use_pad,
            pad_scale=self.pad_scale,
        )
        proj_state.n_out = residual_state.n_out
        proj_state.n_out_inv = residual_state.n_out_inv
        proj_compensation = self.tee.make_linear_pad_compensation(w_proj, proj_state)
        self.pad_report["mlp_c_proj_pad"] = proj_state.pad is not None
        if proj_state.pad is not None:
            self.pad_report.setdefault("pad_tensor_ids", []).append(id(proj_state.pad))
        proj_tilde = self.executor.linear_forward(
            self.tee.obfuscate_input(activated_flat, proj_state),
            *self.tee.transform_linear_weight(w_proj, b_proj, proj_state),
            proj_compensation,
        )
        return proj_tilde.reshape(*x_plain.shape[:-1], w_proj.shape[1])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run one GPT-2 block and return recovered hidden states."""
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape [batch, seq, hidden], got {tuple(hidden_states.shape)}")
        self.pad_report = self._new_pad_report()
        hidden_states = hidden_states.to(dtype=self.dtype, device=self.device)
        residual_state, hidden_tilde = self._initial_hidden_state(hidden_states)

        hidden_plain = self._recover(hidden_tilde, residual_state)
        ln1_plain = self.block.ln_1(hidden_plain)
        attn_tilde = obfuscated_gpt2_attention(
            ln1_plain,
            self.block.attn,
            residual_state,
            self.tee,
            self.executor,
            attention_mask=attention_mask,
            use_pad=self.use_pad,
            pad_scale=self.pad_scale,
            pad_audit=self.pad_report,
        )
        hidden_tilde = hidden_tilde + attn_tilde

        hidden_plain = self._recover(hidden_tilde, residual_state)
        ln2_plain = self.block.ln_2(hidden_plain)
        mlp_tilde = self._obfuscated_mlp(ln2_plain, residual_state)
        hidden_tilde = hidden_tilde + mlp_tilde
        self._finalize_pad_report()
        return self._recover(hidden_tilde, residual_state)

    # ------------------------------------------------------------------
    # Stage 4.8 prefill/decode with KV cache
    # ------------------------------------------------------------------

    def _plain_kv_for_test(self, ln1_plain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute plain K/V tensors (test-only) from LayerNorm-ed input."""
        w_qkv, b_qkv = extract_conv1d_as_linear(self.block.attn.c_attn)
        w_qkv = w_qkv.to(dtype=ln1_plain.dtype, device=ln1_plain.device)
        b_qkv = None if b_qkv is None else b_qkv.to(dtype=ln1_plain.dtype, device=ln1_plain.device)
        qkv = ln1_plain @ w_qkv
        if b_qkv is not None:
            qkv = qkv + b_qkv
        hidden_size = ln1_plain.shape[-1]
        _, k_plain, v_plain = torch.split(qkv, hidden_size, dim=-1)
        num_heads = int(self.block.attn.num_heads)
        return split_heads(k_plain, num_heads), split_heads(v_plain, num_heads)

    def prefill(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ObfuscatedGPT2LayerCache]:
        """Run prefill and return ``(recovered_hidden_states, layer_cache)``.

        The layer cache stores per-head K/V masks (TEE-managed) and the
        obfuscated K_tilde / V_tilde tensors (GPU-visible). The plain K/V are
        also kept for cache invariant checks but are not exposed to the
        untrusted executor.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape [batch, seq, hidden], got {tuple(hidden_states.shape)}"
            )
        self.pad_report = self._new_pad_report()
        hidden_states = hidden_states.to(dtype=self.dtype, device=self.device)
        residual_state, hidden_tilde = self._initial_hidden_state(hidden_states)

        hidden_plain = self._recover(hidden_tilde, residual_state)
        ln1_plain = self.block.ln_1(hidden_plain)

        num_heads = int(self.block.attn.num_heads)
        head_dim = int(self.block.attn.head_dim)
        key_masks, key_mask_inverses = generate_head_masks(
            num_heads, head_dim, self.dtype, self.device
        )
        value_masks, value_mask_inverses = generate_head_masks(
            num_heads, head_dim, self.dtype, self.device
        )

        attn_tilde, k_heads_tilde, v_heads_tilde = obfuscated_gpt2_attention_prefill(
            ln1_plain,
            self.block.attn,
            residual_state,
            self.tee,
            self.executor,
            key_masks=key_masks,
            key_mask_inverses=key_mask_inverses,
            value_masks=value_masks,
            value_mask_inverses=value_mask_inverses,
            use_pad=self.use_pad,
            pad_scale=self.pad_scale,
            pad_audit=self.pad_report,
        )
        hidden_tilde = hidden_tilde + attn_tilde

        hidden_plain = self._recover(hidden_tilde, residual_state)
        ln2_plain = self.block.ln_2(hidden_plain)
        mlp_tilde = self._obfuscated_mlp(ln2_plain, residual_state)
        hidden_tilde = hidden_tilde + mlp_tilde
        self._finalize_pad_report()

        k_plain, v_plain = self._plain_kv_for_test(ln1_plain)
        layer_cache = ObfuscatedGPT2LayerCache(
            key_tilde=k_heads_tilde,
            value_tilde=v_heads_tilde,
            key_masks=key_masks,
            key_mask_inverses=key_mask_inverses,
            value_masks=value_masks,
            value_mask_inverses=value_mask_inverses,
            key_plain_for_test=k_plain,
            value_plain_for_test=v_plain,
        )
        return self._recover(hidden_tilde, residual_state), layer_cache

    def decode_step(
        self,
        hidden_states: torch.Tensor,
        layer_cache: ObfuscatedGPT2LayerCache,
    ) -> tuple[torch.Tensor, ObfuscatedGPT2LayerCache]:
        """Run one decode step and return ``(recovered_hidden_states, updated_cache)``."""
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(
                f"decode_step expects hidden_states with shape [batch, 1, hidden], got "
                f"{tuple(hidden_states.shape)}"
            )
        self.pad_report = self._new_pad_report()
        hidden_states = hidden_states.to(dtype=self.dtype, device=self.device)
        residual_state, hidden_tilde = self._initial_hidden_state(hidden_states)

        hidden_plain = self._recover(hidden_tilde, residual_state)
        ln1_plain = self.block.ln_1(hidden_plain)

        attn_tilde, k_new_heads_tilde, v_new_heads_tilde = obfuscated_gpt2_attention_decode(
            ln1_plain,
            self.block.attn,
            residual_state,
            self.tee,
            self.executor,
            past_key_tilde=layer_cache.key_tilde,
            past_value_tilde=layer_cache.value_tilde,
            key_masks=layer_cache.key_masks,
            key_mask_inverses=layer_cache.key_mask_inverses,
            value_masks=layer_cache.value_masks,
            value_mask_inverses=layer_cache.value_mask_inverses,
            use_pad=self.use_pad,
            pad_scale=self.pad_scale,
            pad_audit=self.pad_report,
        )
        hidden_tilde = hidden_tilde + attn_tilde

        hidden_plain = self._recover(hidden_tilde, residual_state)
        ln2_plain = self.block.ln_2(hidden_plain)
        mlp_tilde = self._obfuscated_mlp(ln2_plain, residual_state)
        hidden_tilde = hidden_tilde + mlp_tilde
        self._finalize_pad_report()

        key_tilde_all = torch.cat([layer_cache.key_tilde, k_new_heads_tilde], dim=2)
        value_tilde_all = torch.cat([layer_cache.value_tilde, v_new_heads_tilde], dim=2)

        if (
            layer_cache.key_plain_for_test is not None
            and layer_cache.value_plain_for_test is not None
        ):
            k_new_plain, v_new_plain = self._plain_kv_for_test(ln1_plain)
            key_plain_all = torch.cat([layer_cache.key_plain_for_test, k_new_plain], dim=2)
            value_plain_all = torch.cat(
                [layer_cache.value_plain_for_test, v_new_plain], dim=2
            )
        else:
            key_plain_all = None
            value_plain_all = None

        new_layer_cache = ObfuscatedGPT2LayerCache(
            key_tilde=key_tilde_all,
            value_tilde=value_tilde_all,
            key_masks=layer_cache.key_masks,
            key_mask_inverses=layer_cache.key_mask_inverses,
            value_masks=layer_cache.value_masks,
            value_mask_inverses=layer_cache.value_mask_inverses,
            key_plain_for_test=key_plain_all,
            value_plain_for_test=value_plain_all,
        )
        return self._recover(hidden_tilde, residual_state), new_layer_cache
