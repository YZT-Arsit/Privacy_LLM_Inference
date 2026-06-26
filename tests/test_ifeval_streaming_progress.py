"""run_ifeval_generation.py progress + streaming responses (mock path).

Run:
    PYTHONPATH=$PWD/src pytest tests/test_ifeval_streaming_progress.py -q
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


_R = _load("ifeval_runner_stream", "scripts/run_ifeval_generation.py")


def _prompts(tmp_path, n=2):
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(
        json.dumps({"id": "ex%d" % i, "prompt": "Q%d?" % i})
        for i in range(n)) + "\n", encoding="utf-8")
    return p


def _run(tmp_path, extra, n=2):
    inp = _prompts(tmp_path, n)
    rj, rep = tmp_path / "r.jsonl", tmp_path / "r.json"
    argv = ["x", "--input-jsonl", str(inp), "--backend", "folded_remote",
            "--mock-runtime", "--max-new-tokens", "4",
            "--output-response-jsonl", str(rj),
            "--output-report-json", str(rep)] + extra
    old = sys.argv
    try:
        sys.argv = argv
        rc = _R.main()
    finally:
        sys.argv = old
    report = json.loads(rep.read_text()) if rep.exists() else None
    lines = rj.read_text().splitlines() if rj.exists() else []
    return rc, report, lines


def test_progress_prints_example_i_of_n(tmp_path, capsys) -> None:
    rc, _report, _lines = _run(tmp_path, ["--progress", "--progress-every", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[ifeval] example 1/2" in out
    assert "[ifeval] example 2/2" in out
    assert "phase=start" in out
    assert "phase=generate_start" in out
    assert "phase=done" in out
    assert "tokens=" in out and "eta=" in out


def test_progress_every_throttles(tmp_path, capsys) -> None:
    # 4 examples, every=2 -> first, the 2nd, the 4th (last) print; not the 3rd
    rc, _report, _lines = _run(tmp_path, ["--progress", "--progress-every", "2"],
                               n=4)
    assert rc == 0
    out = capsys.readouterr().out
    assert "example 1/4" in out and "example 2/4" in out and "example 4/4" in out


def test_streaming_writes_one_line_per_example(tmp_path) -> None:
    rc, report, lines = _run(tmp_path, [])           # streaming ON by default
    assert rc == 0
    assert len(lines) == 2
    assert [json.loads(ln)["id"] for ln in lines] == ["ex0", "ex1"]
    assert report["responses_streamed"] is True
    assert report["progress_streaming_enabled"] is True
    assert report["completed_examples"] == 2
    assert report["generated_tokens"] == sum(
        json.loads(ln)["num_tokens"] for ln in lines)


def test_no_stream_still_writes_at_end(tmp_path) -> None:
    rc, report, lines = _run(tmp_path, ["--no-stream-responses"])
    assert rc == 0
    assert len(lines) == 2                            # written at the end
    assert report["responses_streamed"] is False
