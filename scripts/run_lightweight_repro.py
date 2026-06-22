"""Stage 7.3 -- lightweight reproducibility entrypoint.

Runs a small set of fast checks only:

1. ``import pllo``
2. targeted tests (boundaries, skeleton, HF skeleton)
3. one tiny HF skeleton probe
4. output-size guard

It does NOT run the full test suite, the full Stage 7.6 scanner, or write
paper artifacts / large reports. CPU-only; transformers optional.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(label: str, cmd: list[str]) -> dict:
    print(f"\n=== {label} ===")
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True)
    tail = (proc.stdout or "").strip().splitlines()[-8:]
    for ln in tail:
        print(ln)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()[-6:]
        for ln in err:
            print(ln)
    return {"label": label, "returncode": proc.returncode}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the targeted pytest steps")
    ap.add_argument("--warn-mb", type=int, default=10)
    ap.add_argument("--fail-mb", type=int, default=100)
    args = ap.parse_args()

    results: list[dict] = []

    results.append(_run("import check", [PY, "-c", "import pllo; print('pllo import ok')"]))

    if not args.skip_tests:
        for t in ("tests/test_causal_lm_boundaries.py",
                  "tests/test_masked_causal_lm_skeleton.py",
                  "tests/test_hf_causal_lm_skeleton.py"):
            results.append(_run(f"pytest {t}", [PY, "-m", "pytest", t, "-q"]))

    results.append(_run(
        "tiny llama probe",
        [PY, "scripts/run_hf_causal_lm_skeleton_probe.py",
         "--model-family", "llama", "--max-layers", "1",
         "--prefill-seq-len", "3", "--decode-steps", "1",
         "--max-vocab-size", "128",
         "--output", "outputs/hf_causal_lm_skeleton_probe_llama.json"]))

    results.append(_run(
        "output size check",
        [PY, "scripts/check_output_sizes.py", "--output-dir", "outputs",
         "--warn-mb", str(args.warn_mb), "--fail-mb", str(args.fail_mb)]))

    print("\n=== summary ===")
    failed = [r for r in results if r["returncode"] != 0]
    for r in results:
        flag = "ok" if r["returncode"] == 0 else f"FAIL({r['returncode']})"
        print(f"  [{flag}] {r['label']}")
    if failed:
        print(f"{len(failed)} step(s) failed.")
        return 1
    print("all lightweight checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
