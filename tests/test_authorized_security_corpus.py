from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_freeze_corpus_accepts_matched_ids_and_is_idempotent(tmp_path: Path) -> None:
    v0 = tmp_path / "v0.jsonl"
    v2 = tmp_path / "v2.jsonl"
    v0.write_text(json.dumps({"sample_id": "a", "output_text": "x"}) + "\n")
    v2.write_text(json.dumps({"sample_id": "a", "tensor_shapes": {"h": [1, 8]}}) + "\n")
    out = tmp_path / "manifest.json"
    cmd = [sys.executable, "scripts/authorized_security/freeze_corpus.py",
           "--input", f"V0={v0}", "--input", f"V2={v2}",
           "--task", "fixture", "--split", "test", "--output", str(out)]
    subprocess.run(cmd, check=True)
    first = out.read_bytes()
    subprocess.run(cmd, check=True)
    assert out.read_bytes() == first


def test_freeze_corpus_rejects_forbidden_field(tmp_path: Path) -> None:
    v0 = tmp_path / "v0.jsonl"
    v0.write_text(json.dumps({"sample_id": "a", "logits": [1.0]}) + "\n")
    proc = subprocess.run(
        [sys.executable, "scripts/authorized_security/freeze_corpus.py",
         "--input", f"V0={v0}", "--task", "fixture", "--split", "test",
         "--output", str(tmp_path / "manifest.json")], capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "outside its allowlist" in proc.stderr
