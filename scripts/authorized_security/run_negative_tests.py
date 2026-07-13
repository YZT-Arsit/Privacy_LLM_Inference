#!/usr/bin/env python3
"""Run the view-boundary negative suite and emit auditable JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from view_contracts import ALLOWLISTS, View, ViewAccessError, ViewRecord


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    controls = []

    def rejected(name: str, view: View, field: str, value="forbidden") -> None:
        try:
            ViewRecord.build(view, {"sample_id": "negative-000", field: value})
        except ViewAccessError as exc:
            controls.append({"name": name, "view": view.value, "field": field,
                             "status": "rejected_ok", "error": str(exc)})
        else:
            controls.append({"name": name, "view": view.value, "field": field,
                             "status": "FAIL_ACCEPTED"})

    for view in (View.V0, View.V1, View.V2):
        for field in ("plaintext_weights", "plaintext_adapter", "reference", "labels", "model", "oracle"):
            rejected(f"{view.value}_rejects_{field}", view, field)
    for field in sorted(ALLOWLISTS[View.V2] - ALLOWLISTS[View.V0]):
        rejected(f"V0_rejects_{field}", View.V0, field, [])
    try:
        ViewRecord.build("AUTO", {"sample_id": "negative-000"})
    except ValueError as exc:
        controls.append({"name": "implicit_AUTO_view", "status": "rejected_ok", "error": str(exc)})
    else:
        controls.append({"name": "implicit_AUTO_view", "status": "FAIL_ACCEPTED"})

    result = {
        "schema": "authorized_security_view_negative_tests",
        "version": "1.0",
        "controls": controls,
        "passed": sum(c["status"] == "rejected_ok" for c in controls),
        "total": len(controls),
        "all_fail_closed": all(c["status"] == "rejected_ok" for c in controls),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{hashlib.sha256(text.encode()).hexdigest()}  {args.output.name}\n"
    )
    print(json.dumps({"passed": result["passed"], "total": result["total"]}))
    raise SystemExit(0 if result["all_fail_closed"] else 1)


if __name__ == "__main__":
    main()
