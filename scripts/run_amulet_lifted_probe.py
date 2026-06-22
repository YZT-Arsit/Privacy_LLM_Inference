"""Runner for the Amulet-style lifted nonlinear island probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.amulet_lifted_probe import (  # noqa: E402
    run_amulet_lifted_probe,
)


def main() -> None:
    report = run_amulet_lifted_probe()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
