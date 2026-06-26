"""Tests for the TRUSTED-SIDE generation-config processor (repetition_penalty).

The folded remote path decodes by raw argmax over recovered logits; the plaintext
baseline (``model.generate``) applies Qwen's ``repetition_penalty``. This adds the
SAME penalty trusted-side -- after logits recovery, before argmax -- so nothing new
crosses to the GPU. These tests confirm:

* ``_apply_generation_processors`` matches HF ``RepetitionPenaltyLogitsProcessor``;
* default OFF is a bit-identical no-op (raw greedy unchanged);
* turning it on actually changes the trusted-side token choice;
* it runs entirely trusted-side -- the GPU requests carry NO token history /
  recovered logits / forbidden fields, audited on the exact recorded traffic.

No real model / GPU / server: a fake boundary + fake worker drive the real
``_decode_loop``. Run: python -m pytest tests/test_trusted_generation_processor.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.real_predictors import _RemoteMaskedPredictor  # noqa: E402
from pllo.protocol.remote import forbidden_fields_in_payload  # noqa: E402
from pllo.protocol.tee_gpu_messages import ProtocolTrace  # noqa: E402
from pllo.protocol.wire import encode_message  # noqa: E402


def _bare(align, penalty):
    p = object.__new__(_RemoteMaskedPredictor)
    p._align_gen = align
    p._rep_penalty = penalty
    p._gen_processor_applied = False
    return p


# ---- HF parity -------------------------------------------------------------

def test_matches_hf_repetition_penalty() -> None:
    from transformers import RepetitionPenaltyLogitsProcessor
    torch.manual_seed(0)
    rec = torch.randn(1, 50)
    seen = [3, 7, 7, 40, 1, 3]
    got = _bare(True, 1.3)._apply_generation_processors(rec.clone(), seen)
    hf = RepetitionPenaltyLogitsProcessor(1.3)(
        torch.tensor([sorted(set(seen))]), rec.clone())
    assert torch.allclose(got, hf, atol=1e-6)


def test_sign_rule_matches_spec() -> None:
    # logit<0 -> *penalty (more negative); logit>0 -> /penalty (smaller)
    rec = torch.tensor([[2.0, -2.0, 5.0]])
    out = _bare(True, 2.0)._apply_generation_processors(rec.clone(), [0, 1])
    assert abs(float(out[0, 0]) - 1.0) < 1e-6        # 2.0 / 2
    assert abs(float(out[0, 1]) - (-4.0)) < 1e-6     # -2.0 * 2
    assert abs(float(out[0, 2]) - 5.0) < 1e-6        # unseen -> unchanged


def test_only_seen_tokens_affected() -> None:
    rec = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    out = _bare(True, 2.0)._apply_generation_processors(rec.clone(), [1])
    assert float(out[0, 1]) == 0.5
    assert [float(out[0, i]) for i in (0, 2, 3)] == [1.0, 1.0, 1.0]


def test_disabled_is_identity_no_mutation() -> None:
    rec = torch.randn(1, 20)
    p = _bare(False, 1.3)
    out = p._apply_generation_processors(rec.clone(), [1, 2, 3])
    assert torch.equal(out, rec)
    assert p._gen_processor_applied is False


def test_penalty_one_is_identity() -> None:
    rec = torch.randn(1, 20)
    p = _bare(True, 1.0)
    out = p._apply_generation_processors(rec.clone(), [1, 2, 3])
    assert torch.equal(out, rec)
    assert p._gen_processor_applied is False


def test_applied_flag_set_when_firing() -> None:
    p = _bare(True, 1.3)
    p._apply_generation_processors(torch.randn(1, 10), [2, 4])
    assert p._gen_processor_applied is True


def test_out_of_range_seen_ids_ignored() -> None:
    rec = torch.tensor([[1.0, 1.0, 1.0]])
    out = _bare(True, 2.0)._apply_generation_processors(rec.clone(), [1, 99])
    assert float(out[0, 1]) == 0.5                   # 99 silently dropped


# ---- end-to-end trusted-side decode (fake boundary + fake worker) ----------

class _FakeBoundary:
    compute_device = torch.device("cpu")
    fdtype = torch.float32

    def __init__(self, logits):
        self._logits = logits

    def mask_embeddings(self, ids):
        return torch.zeros(1, int(ids.shape[1]), 4)

    def mask_token_embedding(self, t):
        return torch.zeros(1, 1, 4)

    def recover(self, x):                 # identity: worker already returns logits
        return torch.as_tensor(np.asarray(self._logits)).float()


class _Resp:
    def __init__(self, logits):
        self.masked_logits = np.asarray(logits, dtype="float32")
        self.worker_timing = None


class _FakeWorker:
    def __init__(self, logits):
        self._logits = logits
        self.sent = []

    def prefill(self, req):
        self.sent.append(req)
        return _Resp(self._logits)

    def decode(self, req):
        self.sent.append(req)
        return _Resp(self._logits)


def _run_loop(align, penalty, *, prompt_ids, logits, max_new_tokens=4):
    p = object.__new__(_RemoteMaskedPredictor)
    p.max_new_tokens = max_new_tokens
    p._align_gen = align
    p._rep_penalty = penalty
    p._gen_processor_applied = False
    p._boundary = _FakeBoundary(logits)
    p._worker = _FakeWorker(logits)
    p._trace = ProtocolTrace(boundary_backend="process",
                             gpu_backend="qwen7b_folded_package",
                             max_new_tokens=max_new_tokens, tee_used_on_gpu=False)
    p._profiler = None
    p._schedule = None
    gen, _ = p._decode_loop(torch.tensor([prompt_ids], dtype=torch.long))
    return p, gen


def test_default_off_is_raw_greedy_unchanged() -> None:
    # constant recovered logits -> raw greedy picks index 3 every step
    logits = [[0.0, 0.0, 0.0, 10.0, 9.5, 0.0]]
    _, gen_off = _run_loop(False, 1.3, prompt_ids=[3], logits=logits)
    assert gen_off == [3, 3, 3, 3]


def test_enabled_changes_token_choice_trusted_side() -> None:
    logits = [[0.0, 0.0, 0.0, 10.0, 9.5, 0.0]]
    p_off, gen_off = _run_loop(False, 2.0, prompt_ids=[3], logits=logits)
    p_on, gen_on = _run_loop(True, 2.0, prompt_ids=[3], logits=logits)
    assert gen_off == [3, 3, 3, 3]
    assert gen_on != gen_off                      # penalty steered away from 3
    assert gen_on[0] == 4                          # 10/2=5 < 9.5 -> index 4 wins
    assert p_on._gen_processor_applied is True
    assert p_off._gen_processor_applied is False


def test_no_forbidden_fields_or_token_history_in_gpu_requests() -> None:
    logits = [[0.0, 0.0, 0.0, 10.0, 9.5, 0.0]]
    prompt = [3, 4, 3]
    p, gen = _run_loop(True, 2.0, prompt_ids=prompt, logits=logits)
    assert p._worker.sent                          # requests were actually sent
    history = set(prompt) | set(gen)
    for req in p._worker.sent:
        payload = encode_message(req)
        # no forbidden GPU-boundary field crosses
        assert forbidden_fields_in_payload(payload) == []
        # no token-history / generated ids / prompt key anywhere in the request
        keys = _walk_keys(payload)
        assert "input_ids" not in keys and "generated_tokens" not in keys
        assert "prompt" not in keys and "recovered_logits" not in keys
        # the masked request must not carry plaintext token ids as a list payload
        assert "token_ids" not in keys


def _walk_keys(obj):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(k)
            out |= _walk_keys(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _walk_keys(v)
    return out
