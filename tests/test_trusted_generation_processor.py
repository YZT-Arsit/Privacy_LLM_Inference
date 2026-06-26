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

    def mask_embeddings(self, ids):
        return torch.zeros(1, int(ids.shape[1]), 4)

    def mask_token_embedding(self, t):
        return torch.zeros(1, 1, 4)

    def recover(self, x):                 # identity: worker already returns logits
        return torch.as_tensor(np.asarray(x)).float()


class _Resp:
    def __init__(self, logits):
        self.masked_logits = np.asarray(logits, dtype="float32")
        self.worker_timing = None


class _FakeWorker:
    """Returns row i for the i-th call (last row repeats), so a test can make the
    argmax become EOS at a chosen step."""

    def __init__(self, rows):
        self._rows = rows
        self._i = -1
        self.sent = []

    def _next(self, req):
        self.sent.append(req)
        self._i += 1
        return _Resp(self._rows[min(self._i, len(self._rows) - 1)])

    prefill = _next
    decode = _next


def _new_predictor(*, max_new_tokens, align=False, penalty=1.0, stop_on_eos=False,
                   eos_ids=None, pad_token_id=None, length_hide=False,
                   dummy_token_id=0):
    p = object.__new__(_RemoteMaskedPredictor)
    p.max_new_tokens = max_new_tokens
    p._align_gen = align
    p._rep_penalty = penalty
    p._gen_processor_applied = False
    p._stop_on_eos = stop_on_eos
    p._eos_ids = set(eos_ids or ())
    p._pad_token_id = pad_token_id
    p._examples_finish = []
    p._length_hide = length_hide
    p._dummy_token_id = dummy_token_id
    p._lh_true_latency_s = 0.0
    p._lh_dummy_latency_s = 0.0
    p._lh_gpu_rounds = 0
    p._lh_returned_tokens = 0
    p._trace = ProtocolTrace(boundary_backend="process",
                             gpu_backend="qwen7b_folded_package",
                             max_new_tokens=max_new_tokens, tee_used_on_gpu=False)
    p._profiler = None
    p._schedule = None
    p._boundary = _FakeBoundary()
    return p


def _run_loop(align, penalty, *, prompt_ids, logits=None, logits_seq=None,
              max_new_tokens=4, eos_ids=None, stop_on_eos=False,
              pad_token_id=None, length_hide=False, dummy_token_id=0):
    rows = logits_seq if logits_seq is not None else [logits]
    p = _new_predictor(max_new_tokens=max_new_tokens, align=align, penalty=penalty,
                       stop_on_eos=stop_on_eos, eos_ids=eos_ids,
                       pad_token_id=pad_token_id, length_hide=length_hide,
                       dummy_token_id=dummy_token_id)
    p._worker = _FakeWorker(rows)
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


# ---- EOS-id normalisation (int / list / None) -----------------------------

def test_normalize_eos_ids_int_list_none() -> None:
    from pllo.benchmarks.real_predictors import _normalize_eos_ids
    assert _normalize_eos_ids(151645) == {151645}
    assert _normalize_eos_ids([151645, 151643]) == {151645, 151643}
    assert _normalize_eos_ids((1, 2, 2)) == {1, 2}
    assert _normalize_eos_ids(None) == set()


# ---- trusted-side EOS stopping --------------------------------------------

_ARGMAX3 = [[0.0, 0.0, 0.0, 10.0, 9.5, 0.0]]      # argmax -> token 3
_ARGMAX5 = [[0.0, 0.0, 0.0, 0.0, 0.0, 10.0]]      # argmax -> token 5


def test_predicted_eos_stops_trusted_side() -> None:
    p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits=_ARGMAX3,
                       max_new_tokens=8, eos_ids=[3], stop_on_eos=True)
    assert gen == [3]                               # stopped right at prefill EOS
    fin = p._examples_finish[0]
    assert fin["finish_reason"] == "eos"
    assert fin["stopped_by_eos"] is True
    assert fin["generated_tokens"] == 1


def test_no_eos_continues_until_max_new_tokens() -> None:
    p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits=_ARGMAX3,
                       max_new_tokens=5, eos_ids=[99], stop_on_eos=True)
    assert gen == [3, 3, 3, 3, 3]                   # never hits EOS -> full length
    fin = p._examples_finish[0]
    assert fin["finish_reason"] == "length"
    assert fin["stopped_by_eos"] is False
    assert fin["generated_tokens"] == 5


def test_eos_mid_sequence_stops_and_includes_eos() -> None:
    seq = [_ARGMAX3, _ARGMAX3, _ARGMAX5]           # token 5 (=eos) at step 2
    p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits_seq=seq,
                       max_new_tokens=8, eos_ids=[5], stop_on_eos=True)
    assert gen == [3, 3, 5]                         # eos included then stop
    fin = p._examples_finish[0]
    assert fin["finish_reason"] == "eos" and fin["generated_tokens"] == 3


def test_multi_id_eos_set_stops() -> None:
    p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits=_ARGMAX3,
                       max_new_tokens=8, eos_ids=[3, 5], stop_on_eos=True)
    assert gen == [3]                               # 3 is in the eos set


def test_disable_eos_stop_keeps_fixed_length() -> None:
    p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits=_ARGMAX3,
                       max_new_tokens=4, eos_ids=[3], stop_on_eos=False)
    assert gen == [3, 3, 3, 3]                      # old fixed-length behaviour
    assert p._examples_finish[0]["finish_reason"] == "length"


def test_eos_and_repetition_penalty_together() -> None:
    # rep-penalty steers prefill to token 4 (10/2 < 9.5); then token 4 is EOS
    p, gen = _run_loop(True, 2.0, prompt_ids=[3], logits=_ARGMAX3,
                       max_new_tokens=8, eos_ids=[4], stop_on_eos=True)
    assert gen == [4]
    assert p._gen_processor_applied is True
    assert p._examples_finish[0]["finish_reason"] == "eos"


def test_per_example_finish_records_accumulate() -> None:
    # one predictor, two examples: first stops on EOS, second runs full length
    p = _new_predictor(max_new_tokens=4, stop_on_eos=True, eos_ids={3})
    p._worker = _FakeWorker([_ARGMAX3])
    p._decode_loop(torch.tensor([[1]], dtype=torch.long))   # -> eos at prefill
    p._eos_ids = {99}                                       # second: never eos
    p._worker = _FakeWorker([_ARGMAX3])
    p._decode_loop(torch.tensor([[1]], dtype=torch.long))   # -> full length
    reasons = [e["finish_reason"] for e in p._examples_finish]
    counts = [e["generated_tokens"] for e in p._examples_finish]
    assert reasons == ["eos", "length"]
    assert counts == [1, 4]


def test_eos_decision_not_in_gpu_requests() -> None:
    seq = [_ARGMAX3, _ARGMAX3, _ARGMAX5]
    p, gen = _run_loop(True, 1.05, prompt_ids=[3, 4, 5], logits_seq=seq,
                       max_new_tokens=8, eos_ids=[5], stop_on_eos=True)
    for req in p._worker.sent:
        keys = _walk_keys(encode_message(req))
        assert forbidden_fields_in_payload(encode_message(req)) == []
        for forbidden in ("input_ids", "generated_tokens", "token_ids",
                          "eos_token_id", "recovered_logits", "prompt"):
            assert forbidden not in keys


# ---- strict length-hiding mode --------------------------------------------

def test_default_mode_gpu_rounds_below_budget_on_eos() -> None:
    # default mode (no length hiding): EOS at prefill -> 1 GPU round (prefill only)
    p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits=_ARGMAX3,
                       max_new_tokens=8, eos_ids=[3], stop_on_eos=True,
                       length_hide=False)
    assert gen == [3]
    fin = p._examples_finish[0]
    assert fin["gpu_decode_rounds"] == 1            # < max_new_tokens=8
    assert fin["dummy_decode_rounds"] == 0
    assert fin["true_generated_tokens"] == 1


def test_length_hiding_fills_gpu_rounds_to_budget_on_eos() -> None:
    # strict mode: EOS at prefill, but GPU still sees max_new_tokens rounds
    p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits=_ARGMAX3,
                       max_new_tokens=8, eos_ids=[3], stop_on_eos=True,
                       length_hide=True, dummy_token_id=0)
    assert gen == [3]                               # returned output UNCHANGED
    fin = p._examples_finish[0]
    assert fin["gpu_decode_rounds"] == 8            # == max_new_tokens (prefill+7)
    assert fin["dummy_decode_rounds"] == 7
    assert fin["true_generated_tokens"] == 1        # real tokens unchanged
    assert fin["output_tokens_returned"] == 1
    assert fin["finish_reason"] == "eos"


def test_length_hiding_returned_response_matches_default() -> None:
    seq = [_ARGMAX3, _ARGMAX3, _ARGMAX5]            # eos (=5) at step 2
    _, gen_default = _run_loop(False, 1.0, prompt_ids=[1], logits_seq=seq,
                               max_new_tokens=8, eos_ids=[5], stop_on_eos=True,
                               length_hide=False)
    p_hide, gen_hide = _run_loop(False, 1.0, prompt_ids=[1], logits_seq=seq,
                                 max_new_tokens=8, eos_ids=[5], stop_on_eos=True,
                                 length_hide=True)
    assert gen_hide == gen_default == [3, 3, 5]     # dummies do not change output
    fin = p_hide._examples_finish[0]
    assert fin["gpu_decode_rounds"] == 8            # filled to budget
    assert fin["dummy_decode_rounds"] == 8 - 3      # prefill+2 real = 3 rounds


def test_no_eos_both_modes_reach_budget() -> None:
    for hide in (False, True):
        p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits=_ARGMAX3,
                           max_new_tokens=5, eos_ids=[99], stop_on_eos=True,
                           length_hide=hide)
        assert gen == [3, 3, 3, 3, 3]
        fin = p._examples_finish[0]
        assert fin["gpu_decode_rounds"] == 5
        assert fin["dummy_decode_rounds"] == 0      # never hit EOS -> no dummies
        assert fin["finish_reason"] == "length"


def test_length_hiding_dummy_rounds_no_forbidden_fields() -> None:
    p, gen = _run_loop(True, 1.05, prompt_ids=[3, 4], logits=_ARGMAX3,
                       max_new_tokens=8, eos_ids=[3], stop_on_eos=True,
                       length_hide=True, dummy_token_id=99)
    # 1 prefill + 7 dummy decode requests were sent
    assert len(p._worker.sent) == 8
    for req in p._worker.sent:
        payload = encode_message(req)
        keys = _walk_keys(payload)
        assert forbidden_fields_in_payload(payload) == []
        for forbidden in ("input_ids", "generated_tokens", "token_ids",
                          "generated_token_history", "eos_decision",
                          "finish_reason", "recovered_logits", "plaintext_logits",
                          "dummy_token_id", "prompt"):
            assert forbidden not in keys
        # the trusted-only dummy token id (99) must never appear in a request
        assert 99 not in _walk_values(payload)


def test_true_tokens_vs_gpu_rounds_distinguished_in_records() -> None:
    seq = [_ARGMAX3, _ARGMAX5]                      # eos at step 1
    p, gen = _run_loop(False, 1.0, prompt_ids=[1], logits_seq=seq,
                       max_new_tokens=6, eos_ids=[5], stop_on_eos=True,
                       length_hide=True)
    fin = p._examples_finish[0]
    assert fin["true_generated_tokens"] == 2        # [3, 5]
    assert gen == [3, 5]
    assert fin["gpu_decode_rounds"] == 6            # filled to budget
    assert fin["dummy_decode_rounds"] == 4          # 6 - (prefill + 1 real)
    assert fin["true_generated_tokens"] != fin["gpu_decode_rounds"]


def test_length_hiding_requests_pass_extended_audit() -> None:
    # tie the non-measured length-hiding audit to the REAL decode path: every GPU
    # request emitted in strict mode must pass the extended forbidden-name audit.
    from pllo.security.length_hiding_audit import audit_gpu_request_payloads
    p, gen = _run_loop(True, 1.05, prompt_ids=[3, 4], logits=_ARGMAX3,
                       max_new_tokens=8, eos_ids=[3], stop_on_eos=True,
                       length_hide=True, dummy_token_id=99)
    payloads = [encode_message(req) for req in p._worker.sent]
    rep = audit_gpu_request_payloads(payloads)
    assert rep["fail"] is False
    assert rep["forbidden_fields_found"] == []
    assert len(payloads) == 8                       # prefill + 7 dummy rounds


def _walk_values(obj):
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_walk_values(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_walk_values(v))
    else:
        out.append(obj)
    return out
