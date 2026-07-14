#!/usr/bin/env python3
"""Execute the frozen legacy plus v2 local correctness suites."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "tests/obfuscatune_v2", "tests/obfuscatune", "tests/test_baselines_obfuscatune.py"]
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
