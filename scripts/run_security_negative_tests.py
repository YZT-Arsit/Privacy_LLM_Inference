"""Run the security NEGATIVE test battery (deliberately-broken inputs).

Each case feeds a malicious fixture to the real detector and asserts the failure
is caught. A case only ``pass`` when ``expected_failure`` is actually detected --
this proves the audit / package / hash / attestation / transcript guards work.
No H800 / TDX / CUDA / model required.

Example::

    python scripts/run_security_negative_tests.py \\
        --output-json outputs/security_negative_tests.json \\
        --output-md   outputs/security_negative_tests.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.security.negative_cases import run_all_negative_cases  # noqa: E402


def _render_md(rep: dict) -> str:
    L = ["# Security negative tests", "",
         "- cases: %d  passed: %d  all_passed: **%s**"
         % (rep["num_cases"], rep["num_pass"], rep["all_passed"]), "",
         "| negative_test_name | expected_failure | actually_failed | pass |",
         "| --- | --- | --- | --- |"]
    for c in rep["cases"]:
        L.append("| %s | %s | %s | %s |"
                 % (c["negative_test_name"], c["expected_failure"],
                    c["actually_failed"], "yes" if c["pass"] else "NO"))
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fail-on-incomplete", default="true",
                    help="exit non-zero unless every negative case is detected")
    ap.add_argument("--output-json", default="outputs/security_negative_tests.json")
    ap.add_argument("--output-md", default="outputs/security_negative_tests.md")
    args = ap.parse_args()

    rep = run_all_negative_cases()
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_render_md(rep), encoding="utf-8")

    print("=== security negative tests ===")
    for c in rep["cases"]:
        print("  [%s] %s" % ("PASS" if c["pass"] else "FAIL",
                             c["negative_test_name"]))
        if c.get("error"):
            print("      error: %s" % c["error"])
    print("\n%d/%d caught  all_passed=%s"
          % (rep["num_pass"], rep["num_cases"], rep["all_passed"]))
    fail_on = str(args.fail_on_incomplete).strip().lower() in {"1", "true", "yes"}
    return 0 if (rep["all_passed"] or not fail_on) else 1


if __name__ == "__main__":
    raise SystemExit(main())
