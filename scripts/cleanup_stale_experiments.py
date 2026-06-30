"""Safe cleanup of stale experiment outputs (dry-run by default; archive, not rm).

Scans output dirs, classifies each report as stale / keep / skip, and (only with
``--execute``) MOVES stale files into ``outputs/archive_<timestamp>/`` preserving
their relative path. Nothing is ever ``rm -rf``'d. Model dirs, raw dataset dirs,
and folded-package dirs are PROTECTED (never touched) unless
``--allow-package-cleanup`` is given.

Stale criteria (a report is stale if any holds):
* ``dry_run == true`` or ``mock_runtime == true``;
* ``paper_ready == false``;
* under an AAAI dir, ``nonlinear_backend`` in {current, trusted_shortcut,
  amulet_secure_R};
* a paper-facing/AAAI report with ``max_new_tokens != 512`` or ``seq_len != 1024``;
* attestation evidence with ``simulated_unsigned == true`` (or ``paper_facing ==
  false``);
* ``passed == false`` (a failed validation report).

Always KEPT (even if old): the most recent ``attestation_evidence*.json``, every
folded-package ``manifest.json``, and dataset ``*_card.json``.

A cleanup report ``outputs/cleanup_report_<timestamp>.json`` records moved / kept /
deleted / skipped. stdlib only; importable ``classify`` + ``run_cleanup`` for tests.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_LEGACY_NONLINEAR = {"current", "trusted_shortcut", "amulet_secure_R"}
# backend labels that are NOT part of the AAAI experiment
_NON_AAAI_BACKEND_HINTS = ("amulet", "secure_r", "lora", "puretee", "pure_tee",
                           "full_tee")
# path components that mark a PROTECTED dir (never cleaned without an explicit flag)
_PROTECTED_PARTS = ("models", "model", "raw", "datasets_raw", "checkpoints",
                    "packages", "folded", "package")
_KEEP_NAME_HINTS = ("attestation_evidence", "manifest", "_card", "dataset_card")


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return None


def is_protected_path(path: Path, *, allow_package_cleanup: bool) -> bool:
    """True if ``path`` lives in a model / raw-dataset / folded-package dir."""
    if allow_package_cleanup:
        return False
    parts = {p.lower() for p in path.parts}
    if parts & set(_PROTECTED_PARTS):
        return True
    # a folded-package dir is identified by a sibling manifest.json
    for parent in [path] + list(path.parents):
        if (parent / "manifest.json").exists() and parent != path.parent.parent:
            # only protect if the manifest looks like a folded package
            m = _load_json(parent / "manifest.json")
            if isinstance(m, dict) and m.get("security_claim"):
                return True
    return False


def is_keep_file(path: Path) -> bool:
    name = path.name.lower()
    return any(h in name for h in _KEEP_NAME_HINTS)


def _is_aaai(path: Path) -> bool:
    return any("aaai" in p.lower() for p in path.parts)


def classify_report(report: dict[str, Any], path: Path) -> tuple[str, str]:
    """Return (verdict, reason): verdict in {stale, keep}."""
    if not isinstance(report, dict):
        return "keep", "not a report dict"
    if report.get("dry_run") is True:
        return "stale", "dry_run=true"
    if report.get("mock_runtime") is True:
        return "stale", "mock_runtime=true"
    if report.get("simulated_unsigned") is True:
        return "stale", "simulated_unsigned=true"
    if report.get("paper_facing") is False and report.get("tee"):
        return "stale", "attestation paper_facing=false"
    if report.get("paper_ready") is False:
        return "stale", "paper_ready=false"
    if report.get("passed") is False:
        return "stale", "validation passed=false"
    nb = report.get("nonlinear_backend")
    if _is_aaai(path) and nb in _LEGACY_NONLINEAR:
        return "stale", "legacy nonlinear_backend=%s under AAAI dir" % nb
    # non-AAAI backend (amulet / secure_R / LoRA / pure-TEE) under an AAAI dir
    be = str(report.get("backend") or "").lower()
    if _is_aaai(path) and any(h in be for h in _NON_AAAI_BACKEND_HINTS):
        return "stale", "non-AAAI backend=%s under AAAI dir" % report.get("backend")
    # claims staged but the no-secret audit did not pass / was absent
    if report.get("staged_schedule_used") is True and \
            report.get("staged_schedule_no_secret_audit_passed") is not True:
        return "stale", "claims staged schedule but no-secret audit not passed"
    # staged schedule that leaked a secret flag
    for flag in ("contains_raw_mask", "contains_raw_pad",
                 "gpu_staged_schedule_contains_raw_masks",
                 "gpu_staged_schedule_contains_raw_pad"):
        if report.get(flag) is True:
            return "stale", "%s=true (raw secret on GPU)" % flag
    pf = (report.get("paper_facing_aaai") or report.get("paper_facing")
          or _is_aaai(path))
    if pf:
        if report.get("max_new_tokens") not in (None, 512):
            return "stale", "max_new_tokens=%s != 512" % report.get(
                "max_new_tokens")
        if report.get("seq_len") not in (None, 1024):
            return "stale", "seq_len=%s != 1024" % report.get("seq_len")
        if report.get("stop_on_eos") is False:
            return "stale", "EOS disabled under paper-facing dir"
    return "keep", "valid/paper-facing"


def classify(roots, *, allow_package_cleanup: bool) -> dict[str, Any]:
    """Classify every *.json under ``roots`` into stale / kept / skipped."""
    stale: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for f in sorted(rp.rglob("*.json")):
            if "archive_" in str(f):
                continue
            if is_protected_path(f, allow_package_cleanup=allow_package_cleanup):
                skipped.append({"path": str(f), "reason": "protected dir"})
                continue
            if is_keep_file(f):
                kept.append({"path": str(f), "reason": "always-keep artifact"})
                continue
            rep = _load_json(f)
            verdict, reason = classify_report(rep, f)
            (stale if verdict == "stale" else kept).append(
                {"path": str(f), "reason": reason})
    return {"stale": stale, "kept": kept, "skipped": skipped}


def _sibling_responses(report_path: Path) -> list[Path]:
    """Best-effort: response JSONLs in the same dir as a stale report."""
    out = []
    for p in report_path.parent.glob("*.jsonl"):
        out.append(p)
    return out


def run_cleanup(roots, *, archive_root: Path, execute: bool,
                allow_package_cleanup: bool, move_siblings: bool = True
                ) -> dict[str, Any]:
    cls = classify(roots, allow_package_cleanup=allow_package_cleanup)
    moved: list[dict[str, Any]] = []
    for item in cls["stale"]:
        src = Path(item["path"])
        targets = [src]
        if move_siblings:
            targets += [p for p in _sibling_responses(src)
                        if not is_keep_file(p)]
        for t in targets:
            if not t.exists():
                continue
            # archive path preserves the absolute path under archive_root
            rel = str(t).lstrip("/")
            dst = archive_root / rel
            rec = {"src": str(t), "dst": str(dst), "reason": item["reason"]}
            if execute:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(t), str(dst))
                rec["moved"] = True
            else:
                rec["moved"] = False
            moved.append(rec)
    return {
        "stage": "cleanup_stale_experiments",
        "execute": execute,
        "allow_package_cleanup": allow_package_cleanup,
        "archive_root": str(archive_root),
        "num_stale": len(cls["stale"]), "num_kept": len(cls["kept"]),
        "num_skipped_protected": len(cls["skipped"]),
        "num_moved": sum(1 for m in moved if m["moved"]),
        "num_planned_moves": len(moved),
        "moved": moved, "kept": cls["kept"], "skipped": cls["skipped"],
        "deleted": [],          # this tool NEVER deletes; archive only
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["outputs"],
                    help="dirs to scan (default: outputs)")
    ap.add_argument("--execute", action="store_true", default=False,
                    help="actually MOVE stale files to archive (default: dry-run)")
    ap.add_argument("--allow-package-cleanup", action="store_true", default=False,
                    help="also consider model / raw-dataset / folded-package dirs "
                         "(DANGEROUS; off by default)")
    ap.add_argument("--archive-dir", default=None,
                    help="archive root (default: outputs/archive_<timestamp>)")
    ap.add_argument("--no-move-siblings", action="store_true", default=False)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    roots = args.roots or ["outputs"]
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    archive_root = Path(args.archive_dir or (Path(roots[0]) / ("archive_%s" % ts)))
    rep = run_cleanup(roots, archive_root=archive_root, execute=args.execute,
                      allow_package_cleanup=args.allow_package_cleanup,
                      move_siblings=not args.no_move_siblings)
    out_json = args.output_json or str(Path(roots[0])
                                       / ("cleanup_report_%s.json" % ts))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("=== cleanup (execute=%s) ===" % args.execute)
    print("stale=%d kept=%d skipped_protected=%d planned_moves=%d moved=%d"
          % (rep["num_stale"], rep["num_kept"], rep["num_skipped_protected"],
             rep["num_planned_moves"], rep["num_moved"]))
    print("report: %s" % out_json)
    if not args.execute and rep["num_planned_moves"]:
        print("(dry-run: nothing moved; re-run with --execute to archive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
