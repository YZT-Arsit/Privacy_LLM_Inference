from __future__ import annotations

import json

import pytest
import torch

from pllo.baselines.obfuscatune_lora_v2.artifacts import (
    apply_adapter,
    export_adapter,
    load_adapter,
    load_training_checkpoint,
    save_training_checkpoint,
)
from pllo.baselines.obfuscatune_lora_v2.generation import greedy_generate
from pllo.baselines.obfuscatune_lora_v2.optimizer import AdamWHyperparameters, TransformedAdamW
from pllo.baselines.obfuscatune_lora_v2.qwen_block import AdaptedQwenBlock
from pllo.baselines.obfuscatune_lora_v2.qwen_ops import KVCache, causal_mask, repeat_kv
from pllo.baselines.obfuscatune_lora_v2.qwen_model import AdaptedQwenCausalLM
from pllo.baselines.obfuscatune_lora_v2.runtime import ExternalLoRALinear, RuntimeCounters, TrustedRuntime
from pllo.baselines.obfuscatune_lora_v2.transforms import orthogonal_matrix, restore_factors

DTYPE = torch.float64


def _linear(direction="input", *, din=6, dout=5, rank=2, seed=4):
    torch.manual_seed(seed)
    w = torch.randn(din, dout, dtype=DTYPE)
    dim = din if direction == "input" else dout
    r = orthogonal_matrix(dim, seed + 10, dtype=DTYPE)
    return ExternalLoRALinear(w, r, rank=rank, alpha=4, direction=direction), w


def _block(seed=2):
    torch.manual_seed(seed)
    h, inter = 8, 12
    shapes = {
        "q_proj": (h, h), "k_proj": (h, 4), "v_proj": (h, 4), "o_proj": (h, h),
        "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h),
    }
    directions = {"o_proj": "output", "down_proj": "output"}
    weights = {name: torch.randn(*shape, dtype=DTYPE) * 0.05 for name, shape in shapes.items()}
    rotations = {}
    for i, (name, shape) in enumerate(shapes.items()):
        dim = shape[1] if directions.get(name) == "output" else shape[0]
        rotations[name] = orthogonal_matrix(dim, 1000 + i, dtype=DTYPE)
    counters = RuntimeCounters()
    trusted = TrustedRuntime(counters)
    block = AdaptedQwenBlock(hidden_size=h, intermediate_size=inter, num_heads=4, num_kv_heads=2,
                             rotations=rotations, weights=weights, trusted=trusted, rank=2, alpha=4)
    return block, trusted


def _rope(seq, dim):
    pos = torch.arange(seq, dtype=DTYPE)[:, None]
    half = dim // 2
    freq = torch.arange(half, dtype=DTYPE)[None, :] / 13
    angles = pos * freq
    return torch.cat((torch.cos(angles), torch.cos(angles)), -1), torch.cat((torch.sin(angles), torch.sin(angles)), -1)


def _optimizer(block):
    return TransformedAdamW(block.transformed_parameters(), AdamWHyperparameters(lr=1e-3, weight_decay=0.01))


def _train_step(block, optimizer, x):
    optimizer.parameters = block.transformed_parameters()
    out, _, _ = block(x, *_rope(x.shape[1], 2))
    loss = out.square().mean()
    loss.backward()
    grads = {name: value.grad.detach().clone() for name, value in block.transformed_parameters().items()}
    optimizer.step(grads)
    optimizer.parameters = block.transformed_parameters()
    for value in block.transformed_parameters().values():
        value.grad = None
    return float(loss.detach())


def test_01_transformed_dense_forward():
    layer, w = _linear("input")
    x = torch.randn(3, 6, dtype=DTYPE)
    assert torch.allclose(layer(x), x @ w, atol=1e-12)


def test_02_transformed_dense_backward():
    layer, w = _linear("output")
    x = torch.randn(3, 6, dtype=DTYPE, requires_grad=True)
    y = layer(x).sum(); y.backward()
    assert torch.allclose(x.grad, torch.ones(3, 5, dtype=DTYPE) @ w.T, atol=1e-12)


def test_03_lora_forward_in_canonical_coordinates():
    layer, w = _linear("input")
    with torch.no_grad(): layer.b_star.normal_()
    a, b = restore_factors(layer.a_star, layer.b_star, layer.rotation, "input")
    x = torch.randn(4, 6, dtype=DTYPE)
    assert torch.allclose(layer(x), x @ (w + layer.scale * a @ b), atol=1e-12)


def test_04_da_db_match_autograd_reference():
    layer, _ = _linear("output")
    with torch.no_grad(): layer.b_star.normal_()
    x = torch.randn(4, 6, dtype=DTYPE)
    layer(x).square().sum().backward()
    a = layer.a_star.detach().clone().requires_grad_(); b = layer.b_star.detach().clone().requires_grad_()
    merged = layer.weight_star + layer.scale * a @ b
    ((x @ merged) @ layer.rotation.T).square().sum().backward()
    assert torch.allclose(layer.a_star.grad, a.grad) and torch.allclose(layer.b_star.grad, b.grad)


def test_05_input_gradient():
    layer, _ = _linear("input")
    x = torch.randn(2, 6, dtype=DTYPE, requires_grad=True)
    torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-5)


def test_06_one_step_optimizer():
    block, _ = _block(); opt = _optimizer(block)
    assert torch.isfinite(torch.tensor(_train_step(block, opt, torch.randn(2, 3, 8, dtype=DTYPE))))
    assert opt.step_index == 1


def test_07_ten_step_optimizer():
    block, _ = _block(); opt = _optimizer(block)
    for _ in range(10): _train_step(block, opt, torch.randn(2, 3, 8, dtype=DTYPE))
    assert opt.step_index == 10 and all(torch.isfinite(x).all() for x in opt.parameters.values())


def test_08_zero_gradient_weight_decay():
    layer, _ = _linear(); opt = TransformedAdamW({"x.a_star": layer.a_star}, AdamWHyperparameters(lr=.1, weight_decay=.2))
    before = layer.a_star.detach().clone(); opt.step({"x.a_star": torch.zeros_like(layer.a_star)})
    assert torch.allclose(layer.a_star, before * .98)


def test_09_checkpoint_roundtrip(tmp_path):
    block, trusted = _block(); opt = _optimizer(block); _train_step(block, opt, torch.randn(1, 2, 8, dtype=DTYPE))
    binding = {"base_hash": "base", "config_hash": "cfg", "rank": 2, "alpha": 4}
    path = tmp_path / "state.pt"; digest = save_training_checkpoint(path, parameters=block.transformed_parameters(), optimizer=opt,
                                                                    binding=binding, step=1, counters=trusted.counters.to_dict())
    payload = load_training_checkpoint(path, parameters=block.transformed_parameters(), optimizer=opt, expected_binding=binding)
    assert len(digest) == 64 and payload["step"] == 1


def test_10_restart_continuation(tmp_path):
    x = torch.randn(1, 2, 8, dtype=DTYPE); b1, t1 = _block(); o1 = _optimizer(b1); _train_step(b1, o1, x)
    binding = {"run": "restart"}; p = tmp_path / "c.pt"
    save_training_checkpoint(p, parameters=b1.transformed_parameters(), optimizer=o1, binding=binding, step=1, counters=t1.counters.to_dict())
    b2, _ = _block(); o2 = _optimizer(b2); load_training_checkpoint(p, parameters=b2.transformed_parameters(), optimizer=o2, expected_binding=binding)
    _train_step(b1, o1, x); _train_step(b2, o2, x)
    assert all(torch.equal(o1.parameters[n], o2.parameters[n]) for n in o1.parameters)


def test_11_rmsnorm():
    trusted = TrustedRuntime(); x = torch.randn(2, 3, 8, dtype=DTYPE); w = torch.randn(8, dtype=DTYPE)
    ref = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(trusted.rmsnorm(x, w, 1e-6), ref, atol=1e-7)


def test_12_rope_preserves_norm():
    trusted = TrustedRuntime(); q = torch.randn(1, 4, 3, 2, dtype=DTYPE); k = torch.randn(1, 2, 3, 2, dtype=DTYPE)
    q2, k2 = trusted.rope(q, k, *_rope(3, 2))
    assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-12)
    assert torch.allclose(k.norm(dim=-1), k2.norm(dim=-1), atol=1e-12)


def test_13_gqa_repeat():
    x = torch.randn(1, 2, 3, 4); y = repeat_kv(x, 2)
    assert y.shape == (1, 4, 3, 4) and torch.equal(y[:, 0], y[:, 1])


def test_14_attention_prefill_decode():
    block, _ = _block(); x = torch.randn(1, 3, 8, dtype=DTYPE)
    _, cache, p = block(x, *_rope(3, 2)); assert cache.length == 3 and p.shape[-2:] == (3, 3)
    _, cache, p2 = block(torch.randn(1, 1, 8, dtype=DTYPE), *_rope(1, 2), cache)
    assert cache.length == 4 and p2.shape[-2:] == (1, 4)


def test_15_kv_cache_and_causal_mask():
    cache = KVCache(); k = torch.randn(1, 2, 3, 2); cache.append(k, k); cache.append(k[:, :, :1], k[:, :, :1])
    assert cache.length == 4
    assert torch.equal(causal_mask(2, 4, offset=2, device="cpu"), torch.tensor([[1,1,1,0],[1,1,1,1]], dtype=torch.bool))


def test_16_swiglu():
    trusted = TrustedRuntime(); g = torch.randn(2, 3); u = torch.randn(2, 3)
    assert torch.equal(trusted.swiglu(g, u), torch.nn.functional.silu(g) * u)


def test_17_complete_qwen_block_backward():
    block, trusted = _block(); x = torch.randn(2, 3, 8, dtype=DTYPE, requires_grad=True)
    out, _, _ = block(x, *_rope(3, 2)); out.sum().backward()
    assert out.shape == x.shape and trusted.counters.transformed_linear_calls == 7
    assert trusted.counters.backward_boundary_calls == 7


def test_18_tiny_decoder_generation():
    vocab = 7
    def step(ids, cache):
        offset = 0 if cache is None else cache
        logits = torch.zeros(ids.shape[0], ids.shape[1], vocab)
        logits[..., (offset + 1) % vocab] = 1
        return logits, offset + ids.shape[1]
    assert torch.equal(greedy_generate(step, torch.tensor([[1, 2]]), max_new_tokens=3), torch.tensor([[1, 3, 4]]))


def test_19_adapter_export_reload(tmp_path):
    block, _ = _block(); binding = {"base": "hash", "targets": list(block.TARGETS)}
    export_adapter(tmp_path / "adapter", parameters=block.transformed_parameters(), binding=binding, final_step=750)
    tensors = load_adapter(tmp_path / "adapter", expected_binding=binding)
    clone, _ = _block(); apply_adapter(clone.transformed_parameters(), tensors)
    assert all(torch.equal(block.transformed_parameters()[n], clone.transformed_parameters()[n]) for n in tensors)


def test_20_negative_manifest_and_shape_checks(tmp_path):
    block, _ = _block(); binding = {"base": "ok"}; directory = tmp_path / "adapter"
    export_adapter(directory, parameters=block.transformed_parameters(), binding=binding, final_step=1)
    manifest = json.loads((directory / "adapter_manifest.json").read_text()); manifest["binding"] = {"base": "wrong"}
    (directory / "adapter_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binding"):
        load_adapter(directory, expected_binding=binding)


def test_21_real_hf_qwen_block_and_logits_oracle():
    from transformers import Qwen2ForCausalLM
    from pllo.baselines.obfuscatune.qwen_config import make_tiny_qwen_config
    torch.manual_seed(11)
    model = Qwen2ForCausalLM(make_tiny_qwen_config(num_hidden_layers=1)).to(DTYPE).eval()
    adapted = AdaptedQwenCausalLM(model, trusted=TrustedRuntime(), rank=2, alpha=4)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        reference = model(ids).logits
        actual, _, _ = adapted(ids)
    assert (reference - actual).abs().max().item() < 5e-7
    assert torch.equal(reference.argmax(-1), actual.argmax(-1))


def test_22_real_hf_qwen_cached_generation_token_agreement():
    from transformers import Qwen2ForCausalLM
    from pllo.baselines.obfuscatune.qwen_config import make_tiny_qwen_config
    torch.manual_seed(12)
    model = Qwen2ForCausalLM(make_tiny_qwen_config(num_hidden_layers=1)).to(DTYPE).eval()
    adapted = AdaptedQwenCausalLM(model, trusted=TrustedRuntime(), rank=2, alpha=4)
    ids = torch.tensor([[1, 2, 3, 4]])
    ours = adapted.generate_greedy(ids, max_new_tokens=5)
    reference = model.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=0)[:, ids.shape[1]:]
    assert torch.equal(ours, reference)
