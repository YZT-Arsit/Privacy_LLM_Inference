"""Stage 6.4b — modern decoder block inspector tests.

Tests use lightweight ``torch.nn`` mock blocks for the no-transformers
case; the HF-dependent test is gated on ``transformers`` being importable
and skips if the tiny LLaMA checkpoint cannot be reached.
"""

from __future__ import annotations

import pytest
import torch

from pllo.model_zoo.modern_decoder_spec import (
    ModernDecoderBlockSpec,
    extract_linear_row_weights,
    extract_rmsnorm_params,
    inspect_modern_decoder_block,
    spec_to_dict,
)


# ---------------------------------------------------------------------------
# Mock LLaMA-style modules
# ---------------------------------------------------------------------------


class _MockConfig:
    def __init__(
        self,
        *,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        model_type="llama",
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.model_type = model_type


class _MockRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(0.9 + torch.rand(dim) * 0.2)
        self.variance_epsilon = eps


class _MockAttention(torch.nn.Module):
    def __init__(self, hidden, q_heads, kv_heads, head_dim, bias=False):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden, q_heads * head_dim, bias=bias)
        self.k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=bias)
        self.v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=bias)
        self.o_proj = torch.nn.Linear(q_heads * head_dim, hidden, bias=bias)


class _MockMLP(torch.nn.Module):
    def __init__(self, hidden, intermediate, bias=False):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=bias)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=bias)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=bias)


class _MockLlamaBlock(torch.nn.Module):
    def __init__(self, cfg: _MockConfig, bias=False):
        super().__init__()
        self.input_layernorm = _MockRMSNorm(cfg.hidden_size)
        self.self_attn = _MockAttention(
            cfg.hidden_size, cfg.num_attention_heads,
            cfg.num_key_value_heads, cfg.head_dim, bias=bias,
        )
        self.post_attention_layernorm = _MockRMSNorm(cfg.hidden_size)
        self.mlp = _MockMLP(cfg.hidden_size, cfg.intermediate_size, bias=bias)


class _MockLlamaInner(torch.nn.Module):
    def __init__(self, cfg: _MockConfig, num_layers=2, bias=False):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_MockLlamaBlock(cfg, bias=bias) for _ in range(num_layers)]
        )


class _MockLlamaForCausalLM(torch.nn.Module):
    def __init__(self, cfg: _MockConfig, num_layers=2, bias=False):
        super().__init__()
        self.config = cfg
        self.model = _MockLlamaInner(cfg, num_layers=num_layers, bias=bias)


# ---------------------------------------------------------------------------
# Inspector path resolution
# ---------------------------------------------------------------------------


def test_inspect_mock_llama_block_basic_fields() -> None:
    cfg = _MockConfig(hidden_size=32, intermediate_size=64,
                      num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, model_type="llama")
    model = _MockLlamaForCausalLM(cfg)
    spec = inspect_modern_decoder_block(model)
    assert spec.model_family == "llama_like"
    assert spec.hidden_size == 32
    assert spec.intermediate_size == 64
    assert spec.num_attention_heads == 4
    assert spec.num_key_value_heads == 2
    assert spec.head_dim == 8
    assert spec.norm_type == "rmsnorm"
    assert spec.activation_type == "swiglu"
    assert spec.position_encoding_type == "rotary"
    assert spec.attention_variant == "gqa"
    assert spec.rope_base == 10000.0


def test_inspect_resolves_qkv_and_gate_up_down_paths() -> None:
    model = _MockLlamaForCausalLM(_MockConfig())
    spec = inspect_modern_decoder_block(model)
    assert spec.block_path == "model.layers.0"
    assert spec.q_proj_path == "model.layers.0.self_attn.q_proj"
    assert spec.k_proj_path == "model.layers.0.self_attn.k_proj"
    assert spec.v_proj_path == "model.layers.0.self_attn.v_proj"
    assert spec.o_proj_path == "model.layers.0.self_attn.o_proj"
    assert spec.gate_proj_path == "model.layers.0.mlp.gate_proj"
    assert spec.up_proj_path == "model.layers.0.mlp.up_proj"
    assert spec.down_proj_path == "model.layers.0.mlp.down_proj"
    assert spec.input_norm_path == "model.layers.0.input_layernorm"
    assert spec.post_attention_norm_path == "model.layers.0.post_attention_layernorm"


def test_inspect_qwen_family() -> None:
    cfg = _MockConfig(model_type="qwen2", num_attention_heads=4,
                      num_key_value_heads=4, head_dim=8)
    model = _MockLlamaForCausalLM(cfg)
    spec = inspect_modern_decoder_block(model)
    assert spec.model_family == "qwen_like"
    assert spec.attention_variant == "mha"


def test_inspect_mqa_variant() -> None:
    cfg = _MockConfig(num_attention_heads=4, num_key_value_heads=1, head_dim=8)
    model = _MockLlamaForCausalLM(cfg)
    spec = inspect_modern_decoder_block(model)
    assert spec.attention_variant == "mqa"


def test_inspect_unknown_family_raises() -> None:
    cfg = _MockConfig(model_type="gpt_neo")

    class _UnknownModel(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.model = _MockLlamaInner(cfg)

    with pytest.raises(ValueError, match="modern decoder family"):
        inspect_modern_decoder_block(_UnknownModel(cfg))


# ---------------------------------------------------------------------------
# Weight extraction (row-vector convention; bias=None tolerated)
# ---------------------------------------------------------------------------


def test_extract_linear_row_weights_shape_bias_false() -> None:
    layer = torch.nn.Linear(7, 11, bias=False)
    W_row, b = extract_linear_row_weights(layer)
    assert W_row.shape == (7, 11)
    assert b is None


def test_extract_linear_row_weights_shape_bias_true() -> None:
    layer = torch.nn.Linear(7, 11, bias=True)
    W_row, b = extract_linear_row_weights(layer)
    assert W_row.shape == (7, 11)
    assert b is not None
    assert b.shape == (11,)


def test_extract_linear_row_weights_matches_xWplusb_row_convention() -> None:
    """Y = X @ W_row + b must agree with linear(X)."""
    torch.manual_seed(0)
    layer = torch.nn.Linear(5, 3, bias=True)
    x = torch.randn(2, 4, 5)
    W_row, b = extract_linear_row_weights(layer)
    y_row = x @ W_row + b
    y_torch = layer(x)
    assert torch.allclose(y_row, y_torch, atol=1e-6)


def test_extract_rmsnorm_params_variance_epsilon_path() -> None:
    norm = _MockRMSNorm(8, eps=1e-5)
    w, eps = extract_rmsnorm_params(norm)
    assert w.shape == (8,)
    assert eps == 1e-5


def test_extract_rmsnorm_params_eps_fallback_path() -> None:
    class _AltRMSNorm(torch.nn.Module):
        def __init__(self, dim, eps):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

    norm = _AltRMSNorm(8, 2e-6)
    _, eps = extract_rmsnorm_params(norm)
    assert eps == 2e-6


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_spec_to_dict_is_json_safe() -> None:
    spec = inspect_modern_decoder_block(_MockLlamaForCausalLM(_MockConfig()))
    d = spec_to_dict(spec)
    import json

    json.dumps(d)   # must not raise


# ---------------------------------------------------------------------------
# Real HF tiny-random LLaMA — skip when transformers / hub unreachable
# ---------------------------------------------------------------------------


def test_inspect_real_tiny_llama_when_available() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tiny-random LLaMA unavailable: {exc}")
    spec = inspect_modern_decoder_block(
        model, model_id="hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    assert spec.model_family in {"llama_like", "tinyllama"}
    assert spec.hidden_size > 0
    assert spec.intermediate_size > 0
    assert spec.num_attention_heads > 0
    assert spec.head_dim > 0
    assert spec.norm_type == "rmsnorm"
    assert spec.activation_type == "swiglu"
    assert spec.position_encoding_type == "rotary"
