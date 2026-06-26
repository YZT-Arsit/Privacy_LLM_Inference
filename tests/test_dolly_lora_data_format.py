"""Tests for Dolly LoRA SFT data parsing + assistant-only loss formatting.

No 7B / torch: a fake tokenizer exercises the chat-template formatting + label
masking. Run:
    PYTHONPATH=$PWD/src pytest tests/test_dolly_lora_data_format.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_T = _load("train_dolly", "scripts/train_qwen7b_lora_dolly.py")


class _FakeTok:
    """Word-level tokenizer with a Qwen-ish chat template (marker tokens)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        parts = ["<sys>"]
        for m in msgs:
            parts += ["<%s>" % m["role"]] + m["content"].split() + ["<end>"]
        if add_generation_prompt:
            parts += ["<assistant>"]
        return " ".join(parts)

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": str(s).split()}


def test_build_prompt_variants() -> None:
    assert _T.build_prompt({"prompt": "  hi  "}) == "hi"
    assert _T.build_prompt({"instruction": "do x"}) == "do x"
    assert _T.build_prompt({"instruction": "sum", "context": "ctx"}) == "sum\n\nctx"


def test_load_dolly_jsonl_parsing_and_skips(tmp_path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join([
        json.dumps({"id": "a", "prompt": "p1", "response": "r1", "category": "qa"}),
        "",                                            # blank -> ignored
        "{bad json",                                   # malformed -> skipped
        json.dumps({"id": "b", "instruction": "i2", "response": "r2"}),
        json.dumps({"id": "c", "prompt": "p3"}),       # no response -> skipped
        json.dumps({"id": "d", "prompt": "p4", "response": "   "}),  # empty resp
    ]) + "\n")
    rows, skipped = _T.load_dolly_jsonl(p)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[1]["prompt"] == "i2"
    assert skipped == 3                                # bad json + c + d


def test_load_dolly_max_examples(tmp_path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(
        json.dumps({"id": str(i), "prompt": "p", "response": "r"})
        for i in range(5)) + "\n")
    rows, _ = _T.load_dolly_jsonl(p, max_examples=2)
    assert len(rows) == 2


def test_format_training_example_assistant_only_loss() -> None:
    tok = _FakeTok()
    feat = _T.format_training_example(tok, "tell me", "the answer here",
                                      max_seq_len=999)
    ids, labels = feat["input_ids"], feat["labels"]
    assert len(ids) == len(labels) == len(feat["attention_mask"])
    # the prefix (user turn + <assistant>) is masked; only response tokens learn
    prefix = tok.apply_chat_template([{"role": "user", "content": "tell me"}],
                                     add_generation_prompt=True).split()
    n_prefix = len(prefix)
    assert all(x == -100 for x in labels[:n_prefix])     # prompt masked
    assert any(x != -100 for x in labels[n_prefix:])     # response supervised
    # supervised labels equal the corresponding input ids
    for i in range(n_prefix, len(ids)):
        assert labels[i] == ids[i]
    # the response words appear in the supervised region
    supervised = [ids[i] for i in range(n_prefix, len(ids)) if labels[i] != -100]
    assert "answer" in supervised


def test_format_training_example_truncation() -> None:
    tok = _FakeTok()
    feat = _T.format_training_example(tok, "a b c", "d e f g h",
                                      max_seq_len=4)
    assert len(feat["input_ids"]) == 4
    assert len(feat["labels"]) == 4
