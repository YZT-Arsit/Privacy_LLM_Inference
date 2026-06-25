"""Security transcript recorder + scanner (Task D) -- numpy + stdlib only.

Run: python -m pytest tests/test_security_transcript_scanner.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.security import (  # noqa: E402
    TranscriptRecorder,
    record_message,
    scan_transcript,
)
from pllo.security.transcript_scanner import load_transcript_jsonl  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _main(mod, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return mod.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _no_leak_entries():
    return [
        {"seq": 0, "message_type": "BoundaryInitRequest",
         "direction": "boundary_to_worker",
         "public_metadata_keys": ["session_id", "hidden_size", "vocab_size",
                                  "num_layers", "dtype"],
         "tensor_specs": [], "byte_count": 0, "notes": ""},
        {"seq": 1, "message_type": "MaskedPrefillRequest",
         "direction": "boundary_to_worker",
         "public_metadata_keys": ["session_id", "positions", "batch_size",
                                  "seq_len"],
         "tensor_specs": [{"name": "masked_embeddings", "shape": [1, 12, 128],
                           "dtype": "float32"}],
         "byte_count": 6144, "notes": ""},
        {"seq": 2, "message_type": "MaskedPrefillResponse",
         "direction": "worker_to_boundary",
         "public_metadata_keys": ["session_id", "kv_cache_len"],
         "tensor_specs": [{"name": "masked_logits", "shape": [1, 2000],
                           "dtype": "float32"}],
         "byte_count": 8000, "notes": ""},
    ]


# ---------------------------------------------------------------------------
# scanner: no-leak
# ---------------------------------------------------------------------------


def test_no_leak_transcript_passes() -> None:
    rep = scan_transcript(_no_leak_entries())
    assert rep["stage"] == "security_transcript_scan"
    assert rep["fail"] is False
    assert rep["leak_count"] == 0
    assert rep["gpu_visible_entries"] == 3
    assert rep["forbidden_fields_found"] == []


def test_folded_tilde_tensors_allowed() -> None:
    entries = [
        {"seq": 0, "message_type": "MaskedPrefillRequest",
         "direction": "boundary_to_worker",
         "public_metadata_keys": ["session_id"],
         "tensor_specs": [
             {"name": "q_proj_lora_a_tilde", "shape": [8, 128],
              "dtype": "float32"},
             {"name": "wq_tilde", "shape": [128, 128], "dtype": "float32"},
             {"name": "a_tilde", "shape": [8, 64], "dtype": "float32"},
         ],
         "byte_count": 0, "notes": ""},
    ]
    rep = scan_transcript(entries)
    assert rep["fail"] is False
    assert rep["leak_count"] == 0


# ---------------------------------------------------------------------------
# scanner: leaks
# ---------------------------------------------------------------------------


def test_input_ids_tensor_is_a_leak() -> None:
    entries = _no_leak_entries()
    entries.append({"seq": 3, "message_type": "MaskedPrefillRequest",
                    "direction": "boundary_to_worker",
                    "public_metadata_keys": [],
                    "tensor_specs": [{"name": "input_ids", "shape": [1, 12],
                                      "dtype": "int64"}],
                    "byte_count": 0, "notes": ""})
    rep = scan_transcript(entries)
    assert rep["fail"] is True
    assert rep["leak_count"] == 1
    leak = rep["leaks"][0]
    assert leak["field"] == "input_ids"
    assert leak["kind"] == "tensor"
    assert leak["matched_forbidden"] == "input_ids"
    assert "input_ids" in rep["forbidden_fields_found"]


def test_mask_seed_metadata_is_a_leak() -> None:
    entries = [{"seq": 0, "message_type": "BoundaryInitRequest",
                "direction": "boundary_to_worker",
                "public_metadata_keys": ["hidden_size", "mask_seed"],
                "tensor_specs": [], "byte_count": 0, "notes": ""}]
    rep = scan_transcript(entries)
    assert rep["fail"] is True
    assert any(l["field"] == "mask_seed" and l["kind"] == "metadata"
               for l in rep["leaks"])


def test_raw_lora_tensor_is_a_leak() -> None:
    entries = [{"seq": 0, "message_type": "MaskedPrefillRequest",
                "direction": "boundary_to_worker",
                "public_metadata_keys": [],
                "tensor_specs": [{"name": "lora_A", "shape": [8, 128],
                                  "dtype": "float32"}],
                "byte_count": 0, "notes": ""}]
    rep = scan_transcript(entries)
    assert rep["fail"] is True
    assert any(l["matched_forbidden"] == "lora_a" for l in rep["leaks"])


def test_recovered_logits_outbound_is_a_leak() -> None:
    entries = [{"seq": 0, "message_type": "MaskedDecodeResponse",
                "direction": "worker_to_boundary",
                "public_metadata_keys": [],
                "tensor_specs": [{"name": "recovered_logits", "shape": [1, 2000],
                                  "dtype": "float32"}],
                "byte_count": 0, "notes": ""}]
    rep = scan_transcript(entries)
    assert rep["fail"] is True
    assert any(l["matched_forbidden"] == "recovered_logits"
               for l in rep["leaks"])


def test_trusted_side_direction_not_scanned() -> None:
    # A non-GPU-visible direction holding input_ids is NOT a leak (out of scope).
    entries = [{"seq": 0, "message_type": "RecoveredTokenResponse",
                "direction": "boundary_to_client",
                "public_metadata_keys": ["input_ids"],
                "tensor_specs": [{"name": "input_ids", "shape": [1, 12],
                                  "dtype": "int64"}],
                "byte_count": 0, "notes": ""}]
    rep = scan_transcript(entries)
    assert rep["fail"] is False
    assert rep["gpu_visible_entries"] == 0
    assert rep["scanned_entries"] == 1


def test_allowlist_suppresses_match() -> None:
    entries = [{"seq": 0, "message_type": "X",
                "direction": "boundary_to_worker",
                "public_metadata_keys": ["custom_grad_meter"],
                "tensor_specs": [], "byte_count": 0, "notes": ""}]
    assert scan_transcript(entries)["fail"] is True
    rep = scan_transcript(entries, allowlist=["custom_grad_meter"])
    assert rep["fail"] is False
    assert rep["allowlist_used"] == ["custom_grad_meter"]


# ---------------------------------------------------------------------------
# recorder: only shapes/dtypes/keys, no values; round-trip
# ---------------------------------------------------------------------------


def test_recorder_stores_only_metadata(tmp_path) -> None:
    rec = TranscriptRecorder()
    emb = np.zeros((1, 12, 128), dtype=np.float32)
    rec.record("MaskedPrefillRequest", "inbound",
               public_metadata={"session_id": "s0", "seq_len": 12,
                                "positions": [0, 1, 2]},
               tensors={"masked_embeddings": emb}, byte_count=int(emb.nbytes))
    logits = np.ones((1, 2000), dtype=np.float32)
    rec.record("MaskedPrefillResponse", "outbound",
               public_metadata={"session_id": "s0", "kv_cache_len": 12},
               tensors={"masked_logits": logits})

    assert len(rec.entries) == 2
    e0 = rec.entries[0]
    assert e0.seq == 0
    assert e0.direction == "boundary_to_worker"
    assert e0.public_metadata_keys == ["session_id", "seq_len", "positions"]
    assert e0.tensor_specs == [{"name": "masked_embeddings",
                                "shape": [1, 12, 128], "dtype": "float32"}]
    assert rec.entries[1].direction == "worker_to_boundary"

    # No tensor VALUES anywhere in the serialised transcript.
    blob = json.dumps(rec.to_list())
    assert "0.0" not in blob and "1.0" not in blob
    assert "s0" not in blob  # scalar values not stored, only keys

    # Round-trip through JSONL + scanner.
    p = rec.to_jsonl(tmp_path / "t.jsonl")
    loaded = load_transcript_jsonl(p)
    assert loaded == rec.to_list()
    assert scan_transcript(loaded)["fail"] is False


def test_record_message_introspects_dataclass_and_dict() -> None:
    from dataclasses import dataclass

    @dataclass
    class Msg:
        session_id: str
        masked_embeddings: object
        seq_len: int

    rec = TranscriptRecorder()
    emb = np.zeros((1, 4, 8), dtype=np.float32)
    record_message(rec, "inbound", Msg("s0", emb, 4))
    e = rec.entries[0]
    assert e.message_type == "Msg"
    assert e.tensor_specs == [{"name": "masked_embeddings", "shape": [1, 4, 8],
                               "dtype": "float32"}]
    assert set(e.public_metadata_keys) == {"session_id", "seq_len"}
    assert e.byte_count == int(emb.nbytes)

    # plain dict path
    rec2 = TranscriptRecorder()
    record_message(rec2, "outbound",
                   {"session_id": "s0", "masked_logits": np.ones((1, 5)),
                    "kv_cache_len": 4})
    e2 = rec2.entries[0]
    assert e2.direction == "worker_to_boundary"
    assert any(s["name"] == "masked_logits" for s in e2.tensor_specs)
    assert "kv_cache_len" in e2.public_metadata_keys


# ---------------------------------------------------------------------------
# script via importlib _main
# ---------------------------------------------------------------------------


def _write_jsonl(path, entries):
    with open(path, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e))
            fh.write("\n")


def test_script_leak_exits_1(tmp_path) -> None:
    entries = _no_leak_entries()
    entries.append({"seq": 3, "message_type": "MaskedPrefillRequest",
                    "direction": "boundary_to_worker",
                    "public_metadata_keys": [],
                    "tensor_specs": [{"name": "input_ids", "shape": [1, 12],
                                      "dtype": "int64"}],
                    "byte_count": 0, "notes": ""})
    tj = tmp_path / "leak.jsonl"
    _write_jsonl(tj, entries)
    mod = _load("scan_tx", "scripts/scan_security_transcript.py")
    oj = tmp_path / "scan.json"
    rc = _main(mod, ["x", "--transcript-jsonl", str(tj),
                     "--output-json", str(oj),
                     "--output-md", str(tmp_path / "scan.md"),
                     "--fail-on-leak", "true"])
    assert rc == 1
    rep = json.loads(oj.read_text())
    assert rep["fail"] is True
    assert rep["leak_count"] == 1


def test_script_no_leak_exits_0(tmp_path) -> None:
    tj = tmp_path / "clean.jsonl"
    _write_jsonl(tj, _no_leak_entries())
    mod = _load("scan_tx2", "scripts/scan_security_transcript.py")
    oj = tmp_path / "scan.json"
    md = tmp_path / "scan.md"
    rc = _main(mod, ["x", "--transcript-jsonl", str(tj),
                     "--output-json", str(oj), "--output-md", str(md),
                     "--fail-on-leak", "true"])
    assert rc == 0
    rep = json.loads(oj.read_text())
    assert rep["fail"] is False
    assert "No leaks" in md.read_text()
