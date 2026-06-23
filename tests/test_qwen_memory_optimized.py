"""Stage 8.4 tests -- memory-optimized full-layer masked execution.

Covers chunked down-projection equivalence, layerwise non-retention of folded
weights, and a tiny 28-layer dry run (no Qwen checkpoint). CPU only; the masked
path is the untrusted GPU pipeline (torch), NOT the TEE.
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import pytest
import torch

from pllo.hf_wrappers import qwen_memory_optimized as QMO
from pllo.hf_wrappers.qwen_memory_optimized import (
    MemoryOptimizedConfig,
    chunked_folded_down_projection,
    run_memory_optimized_masked,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_model(num_layers: int, vocab: int = 256):
    pytest.importorskip("transformers")
    from pllo.hf_wrappers.hf_causal_lm_skeleton import (
        HFCausalLMSkeletonConfig, make_random_tiny_hf_causal_lm)
    skel = HFCausalLMSkeletonConfig(
        model_family="qwen2", max_layers=num_layers, max_vocab_size=vocab,
        dtype=torch.float32, device="cpu", seed=7)
    return make_random_tiny_hf_causal_lm(skel)


# ---------------------------------------------------------------------------
# Chunked down-projection equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 16, 64, 1000])
def test_chunked_down_projection_equivalence(chunk) -> None:
    torch.manual_seed(0)
    inter, hidden, B, T = 64, 24, 2, 5
    hidden_act = torch.randn(B, T, inter, dtype=torch.float64)
    down = torch.randn(inter, hidden, dtype=torch.float64)
    perm = torch.randperm(inter)
    n_res = torch.linalg.qr(torch.randn(hidden, hidden, dtype=torch.float64))[0]
    bdown = torch.randn(hidden, dtype=torch.float64)

    full = hidden_act @ (down.index_select(0, perm) @ n_res) + bdown
    got = chunked_folded_down_projection(hidden_act, down, perm, n_res, bdown,
                                         chunk_size=chunk)
    assert torch.allclose(got, full, atol=1e-10), \
        f"chunk={chunk} max err {(got - full).abs().max().item():.2e}"


def test_chunked_down_projection_no_bias_and_default_chunk() -> None:
    torch.manual_seed(1)
    inter, hidden = 50, 16
    hidden_act = torch.randn(1, 3, inter, dtype=torch.float64)
    down = torch.randn(inter, hidden, dtype=torch.float64)
    perm = torch.randperm(inter)
    n_res = torch.linalg.qr(torch.randn(hidden, hidden, dtype=torch.float64))[0]
    full = hidden_act @ (down.index_select(0, perm) @ n_res)
    got = chunked_folded_down_projection(hidden_act, down, perm, n_res, None,
                                         chunk_size=0)  # 0 -> single chunk
    assert torch.allclose(got, full, atol=1e-10)


# ---------------------------------------------------------------------------
# 28-layer dry-run control flow + correctness vs plain reference
# ---------------------------------------------------------------------------


def test_dry_run_28_layers_control_flow_and_correctness() -> None:
    model, mc = _tiny_model(28)
    ids = torch.randint(0, mc.vocab_size, (1, 16))
    cfg = MemoryOptimizedConfig(
        num_layers=28, batch_size=1, seq_len=16, max_new_tokens=4, device="cpu",
        dtype="float32", folding_dtype="float32", folded_weight_device="cpu",
        mlp_down_chunk_size=16, seed=7)
    r = run_memory_optimized_masked(model, mc, ids, cfg)
    assert r["status"] == "ok"
    # explicit, not silently reduced
    assert r["executed_layers"] == 28
    assert r["total_layers"] == 28
    assert r["requested_layers"] == 28
    assert r["tee_used"] is False
    # correctness vs the extracted-weight plaintext reference
    assert r["top1_match_rate"] == 1.0
    assert r["greedy_token_match"] == 1.0
    assert r["max_abs_error"] < 1e-4
    assert len(r["per_layer_memory"]) == 28


def test_chunk_size_does_not_change_result() -> None:
    model, mc = _tiny_model(6)
    ids = torch.randint(0, mc.vocab_size, (1, 8))
    base = dict(num_layers=6, batch_size=1, seq_len=8, max_new_tokens=2,
                device="cpu", dtype="float32", folding_dtype="float32",
                folded_weight_device="cpu", seed=7)
    r_small = run_memory_optimized_masked(
        model, mc, ids, MemoryOptimizedConfig(**base, mlp_down_chunk_size=4))
    r_big = run_memory_optimized_masked(
        model, mc, ids, MemoryOptimizedConfig(**base, mlp_down_chunk_size=10_000))
    assert r_small["generated_masked_tokens"] == r_big["generated_masked_tokens"]
    assert r_small["top1_match_rate"] == r_big["top1_match_rate"] == 1.0


# ---------------------------------------------------------------------------
# Layerwise mode must NOT keep folded weights for previous layers
# ---------------------------------------------------------------------------


def test_layerwise_does_not_retain_previous_folded(monkeypatch) -> None:
    model, mc = _tiny_model(8)
    ids = torch.randint(0, mc.vocab_size, (1, 8))

    markers: list = []
    real = QMO.fold_layer_attention_and_up

    def spy(weights, bm):
        # before producing the next layer's folded weights, every previous
        # folded set must already be collectable (i.e. not retained).
        gc.collect()
        alive = [m for m in markers if m() is not None]
        assert not alive, f"{len(alive)} previous folded weight set(s) retained"
        d = real(weights, bm)

        class _Marker:
            pass
        mk = _Marker()
        d["__marker__"] = mk
        markers.append(weakref.ref(mk))
        return d

    monkeypatch.setattr(QMO, "fold_layer_attention_and_up", spy)
    cfg = MemoryOptimizedConfig(
        num_layers=8, batch_size=1, seq_len=8, max_new_tokens=2, device="cpu",
        dtype="float32", folding_dtype="float32", folded_weight_device="cpu",
        mlp_down_chunk_size=8, seed=7)
    r = run_memory_optimized_masked(model, mc, ids, cfg)
    assert r["status"] == "ok" and r["executed_layers"] == 8
    # folds happened (>= one per layer for prefill, plus per decode step)
    assert len(markers) >= 8


def test_oom_records_layer_index(monkeypatch) -> None:
    """A simulated OOM mid-stream is recorded (status + oom_layer_index +
    executed_layers), not silently swallowed."""
    model, mc = _tiny_model(8)
    ids = torch.randint(0, mc.vocab_size, (1, 8))
    real = QMO.fold_layer_attention_and_up
    state = {"calls": 0}

    def boom(weights, bm):
        if state["calls"] == 3:
            raise RuntimeError("CUDA out of memory (simulated)")
        state["calls"] += 1
        return real(weights, bm)

    monkeypatch.setattr(QMO, "fold_layer_attention_and_up", boom)
    cfg = MemoryOptimizedConfig(
        num_layers=8, batch_size=1, seq_len=8, max_new_tokens=1, device="cpu",
        dtype="float32", folding_dtype="float32", folded_weight_device="cpu",
        seed=7)
    r = run_memory_optimized_masked(model, mc, ids, cfg)
    assert r["status"] == "stopped_oom"
    assert r["oom_layer_index"] == 3
    assert r["executed_layers"] == 3
    assert r["total_layers"] == 8
    assert "out of memory" in r["reason"].lower()


# ---------------------------------------------------------------------------
# No TEE coupling in the GPU pipeline
# ---------------------------------------------------------------------------


def test_gpu_pipeline_does_not_import_tee() -> None:
    import re
    src = (REPO_ROOT / "src" / "pllo" / "hf_wrappers"
           / "qwen_memory_optimized.py").read_text(encoding="utf-8")
    script = (REPO_ROOT / "scripts"
              / "run_qwen7b_full_layer_masked_memory_optimized.py").read_text(
                  encoding="utf-8")
    forbidden = re.compile(r"^\s*(?:import|from)\s+pllo\.tee\b", re.MULTILINE)
    assert not forbidden.search(src), "module imports pllo.tee"
    assert not forbidden.search(script), "script imports pllo.tee"
    # neither references a TEE runtime symbol at all
    assert "pllo.tee" not in src and "pllo.tee" not in script


# ---------------------------------------------------------------------------
# HF-aligned prompt building, decode-start, and generation comparison (script)
# ---------------------------------------------------------------------------


import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_qwen7b_full_layer_masked_memory_optimized as SCRIPT  # noqa: E402


class _MockTok:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        return "CHAT:" + messages[0]["content"]

    def __call__(self, text):
        return {"input_ids": [(ord(c) % 200) + 1 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "txt"


def test_short_prompt_keeps_real_length_no_pad() -> None:
    tok = _MockTok()
    chat = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                   tokenize=False, add_generation_prompt=True)
    expected = tok(chat)["input_ids"]
    ids, mask, meta = SCRIPT.build_chat_inputs(
        tok, ["hi"], max_prompt_len=128, use_chat_template=True, device="cpu")
    assert list(ids.shape) == [1, len(expected)]
    # the input_ids used are exactly the chat-template tokenization (so the HF
    # baseline and the masked path consume identical input_ids)
    assert ids[0].tolist() == expected
    assert len(expected) < 128                      # short prompt
    assert meta["requested_seq_len"] == 128
    assert meta["effective_prompt_len"] == len(expected)
    assert meta["real_prompt_len"] == len(expected)
    assert meta["truncated"] is False
    assert int(mask.sum().item()) == len(expected)  # no padding


def test_decode_start_index_equals_real_prompt_len_not_seq_len() -> None:
    tok = _MockTok()
    _ids, _mask, meta = SCRIPT.build_chat_inputs(
        tok, ["hello there"], max_prompt_len=128, use_chat_template=True,
        device="cpu")
    assert meta["decode_start_index"] == meta["effective_prompt_len"]
    assert meta["decode_start_index"] != meta["requested_seq_len"]


def test_long_prompt_truncates_deterministically() -> None:
    tok = _MockTok()
    chat = tok.apply_chat_template([{"role": "user", "content": "hello world"}],
                                   tokenize=False, add_generation_prompt=True)
    full = tok(chat)["input_ids"]
    assert len(full) > 4
    ids, mask, meta = SCRIPT.build_chat_inputs(
        tok, ["hello world"], max_prompt_len=4, use_chat_template=True,
        device="cpu")
    assert list(ids.shape) == [1, 4]
    assert ids[0].tolist() == full[:4]              # deterministic prefix
    assert meta["truncated"] is True
    assert meta["effective_prompt_len"] == 4
    assert meta["decode_start_index"] == 4
    assert meta["real_prompt_len"] == len(full)


def test_compare_generations_status_ok() -> None:
    cmp = SCRIPT.compare_generations([1, 2, 3], [1, 2, 3], [1, 2, 3])
    assert cmp["status"] == "ok"
    assert cmp["sequence_exact_match_hf_masked"] is True
    assert cmp["hf_vs_masked_token_match_rate"] == 1.0


def test_compare_generations_hf_mismatch() -> None:
    # plain and masked agree, but neither matches official HF
    cmp = SCRIPT.compare_generations([9, 9, 9], [1, 2, 3], [1, 2, 3])
    assert cmp["status"] == "hf_mismatch"
    assert cmp["sequence_exact_match_plain_masked"] is True
    assert cmp["sequence_exact_match_hf_masked"] is False
    assert cmp["plain_vs_masked_token_match_rate"] == 1.0


def test_compare_generations_internal_mismatch() -> None:
    cmp = SCRIPT.compare_generations([1, 2], [1, 2], [3, 4])
    assert cmp["status"] == "internal_mismatch"
    assert cmp["sequence_exact_match_plain_masked"] is False


def test_script_does_not_import_tee() -> None:
    import re
    script_src = (REPO_ROOT / "scripts"
                  / "run_qwen7b_full_layer_masked_memory_optimized.py").read_text(
                      encoding="utf-8")
    diag_src = (REPO_ROOT / "scripts"
                / "diagnose_qwen_hf_parity.py").read_text(encoding="utf-8")
    pat = re.compile(r"^\s*(?:import|from)\s+pllo\.tee\b", re.MULTILINE)
    assert not pat.search(script_src) and "pllo.tee" not in script_src
    assert not pat.search(diag_src) and "pllo.tee" not in diag_src
    assert "tee_used" in script_src and "tee_used" in diag_src


# ---------------------------------------------------------------------------
# HF RoPE parity fixes: rope_theta reading + half-split alignment
# ---------------------------------------------------------------------------


def test_rope_theta_read_from_nested_config() -> None:
    """transformers>=5 nests rope_theta in rope_parameters; infer must read the
    real value (1e6 for Qwen2.5), not silently default to 10000."""
    pytest.importorskip("transformers")
    from transformers import Qwen2Config
    from pllo.hf_wrappers.llama_qwen_single_block import _read_rope_theta
    mc = Qwen2Config(hidden_size=256, num_attention_heads=2,
                     num_key_value_heads=1, rope_theta=1000000.0)
    assert _read_rope_theta(mc) == 1000000.0


def test_hf_rope_alignment_matches_halfsplit_scores() -> None:
    """adjacent-pair RoPE on the permuted q/k must reproduce HF half-split RoPE
    attention scores exactly (the q.k dot is convention-invariant)."""
    from pllo.hf_wrappers.qwen_memory_optimized import hf_rope_interleave_index
    from pllo.ops.rope import apply_rope, build_rope_cache
    torch.manual_seed(0)
    T, hd, nh = 7, 8, 1
    half = hd // 2
    q = torch.randn(1, nh, T, hd, dtype=torch.float64)
    k = torch.randn(1, nh, T, hd, dtype=torch.float64)

    # HF half-split RoPE (cos = cat(freqs, freqs))
    inv = 10000.0 ** (-(torch.arange(0, half, dtype=torch.float64) * 2) / hd)
    t = torch.arange(T, dtype=torch.float64)
    emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
    cos_hs, sin_hs = emb.cos(), emb.sin()

    def halfsplit(x):
        rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos_hs + rot * sin_hs

    hs_scores = halfsplit(q) @ halfsplit(k).transpose(-2, -1)

    # adjacent-pair RoPE on permuted q,k
    p = hf_rope_interleave_index(nh, hd)        # length nh*hd == hd here
    cos_adj, sin_adj = build_rope_cache(T, hd, 10000.0, torch.float64, "cpu")
    qa = apply_rope(q.index_select(-1, p), cos_adj, sin_adj)
    ka = apply_rope(k.index_select(-1, p), cos_adj, sin_adj)
    adj_scores = qa @ ka.transpose(-2, -1)

    assert torch.allclose(hs_scores, adj_scores, atol=1e-10)


def test_alignment_and_theta_give_hf_logit_parity() -> None:
    """With theta fix + RoPE alignment, the internal reference's next-token
    logits match HF, and alignment is strictly better than without it."""
    pytest.importorskip("transformers")
    import diagnose_qwen_hf_parity as DIAG
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    mc = Qwen2Config(vocab_size=512, hidden_size=256, intermediate_size=512,
                     num_hidden_layers=3, num_attention_heads=2,
                     num_key_value_heads=1, max_position_embeddings=256,
                     rms_norm_eps=1e-6, rope_theta=1000000.0,
                     tie_word_embeddings=False)
    model = Qwen2ForCausalLM(mc).eval()
    ids = torch.randint(0, 512, (1, 48))
    with torch.no_grad():
        hf = model(input_ids=ids, use_cache=False, return_dict=True)
    hf_logits = hf.logits[:, -1, :]
    hf_top1 = int(hf_logits.argmax(-1).item())

    def rell2(a, b):
        return float(((a - b).float().pow(2).sum().sqrt()
                      / b.float().pow(2).sum().sqrt()).item())

    ref_a = DIAG.internal_reference(model, mc, ids, torch.float32, "cpu",
                                    align=True)
    ref_n = DIAG.internal_reference(model, mc, ids, torch.float32, "cpu",
                                    align=False)
    assert ref_a["rope_theta"] == 1000000.0              # theta fix applied
    la = ref_a["logits"][:, -1, :]
    ln = ref_n["logits"][:, -1, :]
    assert int(la.argmax(-1).item()) == hf_top1          # aligned top1 == HF
    assert rell2(la, hf_logits) < 1e-3                    # aligned near-exact
    assert rell2(la, hf_logits) < rell2(ln, hf_logits)   # alignment is better
