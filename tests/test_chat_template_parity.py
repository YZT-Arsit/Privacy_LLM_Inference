"""Tests for trusted-side chat-template prompt formatting parity.

``--use-chat-template`` must ACTUALLY call ``tokenizer.apply_chat_template`` (not
just record a flag), via a SINGLE shared function used by both the plaintext and
folded-remote predictors -- so they decode the identical formatted string. The
formatting is trusted-side; nothing prompt-related is added to a GPU request.

Run: python -m pytest tests/test_chat_template_parity.py -q
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.benchmarks.real_predictors import (  # noqa: E402
    _PlaintextLocalPredictor,
    _RemoteMaskedPredictor,
    format_prompt_for_generation,
    prompt_format_info,
)


class _FakeTok:
    """Minimal tokenizer: Qwen-like chat template + word-count tokenization."""
    eos_token_id = 0
    pad_token_id = 0
    chat_template = "FAKE_QWEN_TEMPLATE"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        content = msgs[0]["content"]
        return ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You "
                "are a helpful assistant.<|im_end|>\n<|im_start|>user\n%s"
                "<|im_end|>\n<|im_start|>assistant\n" % content)

    def __call__(self, s, **kw):
        return {"input_ids": list(range(len(str(s).split())))}


# ---- shared formatting function --------------------------------------------

def test_format_applies_chat_template() -> None:
    out = format_prompt_for_generation("hello world", _FakeTok(), True)
    assert "<|im_start|>assistant" in out and "hello world" in out
    assert "<|im_start|>system" in out


def test_format_raw_when_disabled_or_no_tokenizer() -> None:
    assert format_prompt_for_generation("hi", _FakeTok(), False) == "hi"
    assert format_prompt_for_generation("hi", None, True) == "hi"


def test_prompt_format_info_counts_and_sha() -> None:
    raw = "make a tweet without capitals"
    info_chat = prompt_format_info(_FakeTok(), raw, True, seq_len=4096)
    info_raw = prompt_format_info(_FakeTok(), raw, False, seq_len=4096)
    assert info_chat["prompt_format"] == "chat"
    assert info_raw["prompt_format"] == "raw"
    # chat template adds scaffolding -> more tokens than the raw prompt
    assert info_chat["chat_prompt_token_count"] > info_chat["raw_prompt_token_count"]
    # chat info's used count equals the chat count (template applied)
    assert info_chat["prompt_token_count"] == info_chat["chat_prompt_token_count"]
    # raw info's used count equals the raw count
    assert info_raw["prompt_token_count"] == info_raw["raw_prompt_token_count"]
    # sha is over the FORMATTED string (differs chat vs raw)
    assert info_chat["formatted_prompt_sha256"] != \
        info_raw["formatted_prompt_sha256"]
    expect = hashlib.sha256(
        format_prompt_for_generation(raw, _FakeTok(), True).encode()).hexdigest()
    assert info_chat["formatted_prompt_sha256"] == expect


def test_plaintext_and_folded_formatting_parity() -> None:
    # both predictors delegate to the SAME function -> identical formatted sha
    raw = "do the thing"
    p = object.__new__(_PlaintextLocalPredictor)
    p._tok = _FakeTok(); p._use_chat_template = True; p.seq_len = 4096
    r = object.__new__(_RemoteMaskedPredictor)
    r._tok = _FakeTok(); r._use_chat_template = True; r.seq_len = 4096
    assert p.format_prompt(raw) == r.format_prompt(raw)
    assert (p.prompt_info(raw)["formatted_prompt_sha256"]
            == r.prompt_info(raw)["formatted_prompt_sha256"])
    assert p.prompt_info(raw)["prompt_format"] == "chat"


def test_chat_template_increases_token_count() -> None:
    r = object.__new__(_RemoteMaskedPredictor)
    r._tok = _FakeTok(); r._use_chat_template = True; r.seq_len = 4096
    info = r.prompt_info("short prompt here")
    assert info["chat_prompt_token_count"] > info["raw_prompt_token_count"]


def test_disabled_is_raw_for_both() -> None:
    raw = "keep raw"
    p = object.__new__(_PlaintextLocalPredictor)
    p._tok = _FakeTok(); p._use_chat_template = False; p.seq_len = 4096
    assert p.format_prompt(raw) == raw
    assert p.prompt_info(raw)["prompt_format"] == "raw"
