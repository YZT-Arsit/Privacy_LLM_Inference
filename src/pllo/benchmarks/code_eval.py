"""Sandboxed code-generation evaluation (HumanEval / MBPP) -- pass@1.

Extracts the candidate code from a model completion and runs the dataset's tests
in an ISOLATED CPU subprocess with a hard timeout. This is a scoring step that is
fully separate from model inference and the TEE/GPU security claim: no model, no
GPU, no network, and the raw prompt is never logged by this module.

Honesty: untrusted model output is executed, so it MUST run sandboxed
(subprocess + timeout). Do not run this on a host where arbitrary code execution
is unacceptable; the default timeout is short and execution is per-example.

stdlib only (re / subprocess / tempfile).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

__all__ = [
    "extract_code",
    "build_humaneval_program",
    "evaluate_humaneval_example",
    "evaluate_mbpp_example",
    "pass_at_1",
]

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str | None, entry_point: str | None = None) -> str:
    """Extract code from a completion: the first fenced code block if present,
    else the raw completion. Trailing prose after a code block is dropped."""
    if not completion:
        return ""
    m = _CODE_BLOCK.search(completion)
    if m:
        return m.group(1).strip("\n")
    return completion.strip("\n")


def build_humaneval_program(prompt: str, completion: str, test: str,
                            entry_point: str) -> str:
    """Assemble a runnable HumanEval program.

    If the extracted code already defines ``entry_point``, use it as the full
    program (chat models return the whole function); otherwise treat the
    completion as a continuation of ``prompt`` (canonical HumanEval form)."""
    code = extract_code(completion, entry_point)
    if entry_point and re.search(r"def\s+%s\s*\(" % re.escape(entry_point), code):
        body = code
    else:
        body = (prompt or "") + "\n" + (completion or "")
    return "%s\n\n%s\n\ncheck(%s)\n" % (body, test, entry_point)


def _run_program(program: str, timeout: float) -> dict[str, Any]:
    """Run a Python program in an isolated subprocess; return pass/error."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "cand.py"
        f.write_text(program, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, str(f)], capture_output=True,
                               text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error_type": "timeout",
                    "error": "exceeded %.1fs" % timeout}
        except Exception as exc:                             # noqa: BLE001
            return {"passed": False, "error_type": type(exc).__name__,
                    "error": str(exc)[:200]}
    if p.returncode == 0:
        return {"passed": True, "error_type": None, "error": None}
    # keep only the last stderr line (no prompt content is in the program output)
    tail = (p.stderr or "").strip().splitlines()
    return {"passed": False, "error_type": "assertion_or_runtime",
            "error": (tail[-1] if tail else "nonzero exit")[:200]}


def evaluate_humaneval_example(*, prompt: str, completion: str, test: str,
                               entry_point: str, timeout: float = 10.0
                               ) -> dict[str, Any]:
    if not test or not entry_point:
        return {"passed": False, "error_type": "missing_test",
                "error": "no test/entry_point", "extracted_code":
                extract_code(completion, entry_point)}
    program = build_humaneval_program(prompt, completion, test, entry_point)
    res = _run_program(program, timeout)
    res["extracted_code"] = extract_code(completion, entry_point)
    return res


def evaluate_mbpp_example(*, completion: str, test_list: list[str],
                          timeout: float = 10.0) -> dict[str, Any]:
    code = extract_code(completion)
    if not test_list:
        return {"passed": False, "error_type": "missing_test",
                "error": "no test_list", "extracted_code": code}
    program = code + "\n\n" + "\n".join(str(t) for t in test_list) + "\n"
    res = _run_program(program, timeout)
    res["extracted_code"] = code
    return res


def pass_at_1(per_example: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_example)
    passed = sum(1 for e in per_example if e.get("passed"))
    return {"num": n, "num_passed": passed,
            "pass@1": (passed / n) if n else None,
            "failed_cases": [e.get("id") for e in per_example
                             if not e.get("passed")]}
