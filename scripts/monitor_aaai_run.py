"""Live progress monitor for an AAAI generation run (reads its status JSON).

Polls ``--status-json`` (written by ``run_aaai_generation_benchmark.py``) every
``--interval-sec`` and prints a one-line progress summary (completed / failed /
skipped / tokens / rate / ETA). ``--once`` prints a single snapshot and exits.
Read-only; never touches the run. stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return None


def _fmt(st):
    if not st:
        return "no status yet"
    total = st.get("total_examples") or 0
    done = st.get("completed_examples") or 0
    failed = st.get("failed_examples") or 0
    skipped = st.get("skipped_existing_examples") or 0
    processed = done + failed + skipped
    pct = (100.0 * processed / total) if total else 0.0
    elapsed = (st.get("update_time") or 0) - (st.get("start_time") or 0)
    rate = (done / elapsed) if elapsed > 0 else 0.0
    remaining = max(0, total - processed)
    eta = (remaining / rate) if rate > 0 else None
    alive = "DONE" if st.get("end_time") else "alive"
    return ("[%s] %s %d/%d (%.1f%%) done=%d failed=%d skipped=%d tokens=%d "
            "rate=%.2f/s eta=%s cur=%s/%s id=%s"
            % (alive, st.get("current_dataset"), processed, total, pct, done,
               failed, skipped, st.get("generated_tokens_total") or 0, rate,
               ("%.0fs" % eta) if eta is not None else "?",
               st.get("current_backend"), st.get("current_model"),
               st.get("current_example_id")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status-json", required=True)
    ap.add_argument("--interval-sec", type=float, default=5.0)
    ap.add_argument("--once", action="store_true", default=False)
    args = ap.parse_args()
    while True:
        st = _load(args.status_json)
        print(_fmt(st), flush=True)
        if args.once or (st and st.get("end_time")):
            return 0
        time.sleep(max(0.5, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
