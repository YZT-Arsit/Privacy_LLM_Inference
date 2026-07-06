"""KPA known-plaintext (linear+permutation) — thin wrapper over run_all_attacks (attacks=kpa).

Runs this attack across --target-methods on the toy/tiny-Qwen source and writes
outputs/attacks/all/all_attacks.jsonl. See docs/attack_definitions.md.

Example:
    python scripts/attacks/run_kpa_known_plaintext.py
    python scripts/attacks/run_kpa_known_plaintext.py --dry-run
"""
from __future__ import annotations
import runpy
import sys

if __name__ == "__main__":
    if "--attacks" not in sys.argv:
        sys.argv += ["--attacks", "kpa"]
    sys.argv[0] = "run_all_attacks.py"
    runpy.run_path(__file__.rsplit("/", 1)[0] + "/run_all_attacks.py", run_name="__main__")
