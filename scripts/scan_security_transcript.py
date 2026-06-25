"""Scan a GPU-channel security transcript for forbidden (secret / plaintext) names.

Loads a metadata-only transcript JSONL (produced by a ``TranscriptRecorder``),
checks every GPU-visible tensor name + public-metadata key against the forbidden
set, and writes a JSON report + a Markdown leak table. Exits 1 if leaks are found
and ``--fail-on-leak`` is true (the default).

Example::

    python scripts/scan_security_transcript.py \\
        --transcript-jsonl outputs/tee_gpu_protocol_transcript.jsonl \\
        --output-json outputs/security_transcript_scan.json \\
        --output-md  outputs/security_transcript_scan.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.security.transcript_scanner import (  # noqa: E402
    load_transcript_jsonl,
    scan_transcript,
)


def _bool(s) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def _render_md(report: dict) -> str:
    L = ["# Security transcript scan", "",
         "- stage: %s" % report["stage"],
         "- scanned_entries: %s" % report["scanned_entries"],
         "- gpu_visible_entries: %s" % report["gpu_visible_entries"],
         "- leak_count: %s" % report["leak_count"],
         "- fail: %s" % report["fail"],
         "- allowlist_used: %s" % json.dumps(report["allowlist_used"]),
         "- forbidden_fields_found: %s"
         % json.dumps(report["forbidden_fields_found"]), ""]
    if not report["leaks"]:
        L += ["**No leaks found.** No forbidden field crossed a GPU-visible "
              "channel.", ""]
        return "\n".join(L)
    L += ["## Leaks", "",
          "| seq | message_type | direction | kind | field | matched_forbidden |",
          "| --- | --- | --- | --- | --- | --- |"]
    for lk in report["leaks"]:
        L.append("| %s | %s | %s | %s | %s | %s |"
                 % (lk["seq"], lk["message_type"], lk["direction"],
                    lk["kind"], lk["field"], lk["matched_forbidden"]))
    L += [""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transcript-jsonl", required=True,
                    help="metadata-only transcript JSONL to scan")
    ap.add_argument("--allowlist-json", default=None,
                    help="optional JSON list of allowed field substrings")
    ap.add_argument("--output-json",
                    default="outputs/security_transcript_scan.json")
    ap.add_argument("--output-md",
                    default="outputs/security_transcript_scan.md")
    ap.add_argument("--fail-on-leak", default="true",
                    help="exit 1 if leaks are found (default true)")
    args = ap.parse_args()

    allowlist = None
    if args.allowlist_json:
        allowlist = json.loads(Path(args.allowlist_json).read_text(
            encoding="utf-8"))
        if not isinstance(allowlist, list):
            ap.error("--allowlist-json must contain a JSON list of substrings")

    entries = load_transcript_jsonl(args.transcript_jsonl)
    report = scan_transcript(entries, allowlist=allowlist)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str),
                     encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_render_md(report), encoding="utf-8")

    print("=== security transcript scan ===")
    print("scanned_entries=%s gpu_visible_entries=%s leak_count=%s fail=%s"
          % (report["scanned_entries"], report["gpu_visible_entries"],
             report["leak_count"], report["fail"]))
    if report["leaks"]:
        print("forbidden_fields_found=%s"
              % ", ".join(report["forbidden_fields_found"]))
        for lk in report["leaks"]:
            print("  LEAK seq=%s %s/%s %s=%s (matched %s)"
                  % (lk["seq"], lk["message_type"], lk["direction"],
                     lk["kind"], lk["field"], lk["matched_forbidden"]))
    else:
        print("no leaks")

    if _bool(args.fail_on_leak) and report["fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
