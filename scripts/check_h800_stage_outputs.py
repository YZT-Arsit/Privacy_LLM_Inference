"""H800 stage sanity checker -- is the H800 core matrix evidence complete?

Scans a results directory for the JSON reports produced during the H800 phase and
verifies coverage + honesty BEFORE moving to the TDX phase / final gate. It is
content-driven: reports are classified by their ``stage`` / fields and their
``nonlinear_backend`` tag, so it tolerates the exact output filenames you choose.

Pure parsing -- no model / GPU / network. Returns nonzero if any REQUIRED check
fails. Writes ``outputs/final/h800_stage_check.{json,md}``.

Example::

    python scripts/check_h800_stage_outputs.py \\
        --results-dir outputs --package-root /root/autodl-tmp/privacy_llm_packages \\
        --designs current,trusted_shortcut \\
        --datasets mmlu,gsm8k,boolq,ag_news \\
        --output-json outputs/final/h800_stage_check.json \\
        --output-md outputs/final/h800_stage_check.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_FORBIDDEN_DATA = ("tests/fixtures", "fixture", "tiny")
_E3_TOKENS = (1, 4, 8, 16)


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _norm_design(name):
    if not name:
        return None
    try:
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        return normalize_nonlinear_backend(name)
    except Exception:                                       # noqa: BLE001
        return str(name)


def _load_reports(results_dir: Path) -> List[Dict[str, Any]]:
    out = []
    for p in sorted(results_dir.rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            continue
        if isinstance(d, dict):
            out.append({"_path": str(p), "report": d,
                        "_design": _norm_design(d.get("nonlinear_backend"))})
    return out


def _norm_ds(name):
    if not name:
        return None
    low = str(name).lower().replace("-", "_")
    aliases = {"agnews": "ag_news", "ag_news_small": "ag_news",
               "mmlu_small": "mmlu", "gsm8k_small": "gsm8k",
               "boolq_small": "boolq", "sst2": "sst2"}
    return aliases.get(low, low)


def _data_looks_fixture(report) -> Optional[str]:
    """Return the offending value if the report points at fixture/tiny data."""
    for key in ("dataset", "dataset_path", "dataset_jsonl", "dataset_card",
                "dataset_card_json", "input_file", "output_jsonl"):
        v = report.get(key)
        if isinstance(v, str):
            low = v.replace("\\", "/").lower()
            for bad in _FORBIDDEN_DATA:
                if bad in low:
                    return "%s=%s" % (key, v)
    return None


def run_check(opts: dict) -> dict:
    results_dir = Path(opts.get("results_dir") or "outputs")
    pkg_root = Path(opts.get("package_root") or "outputs")
    designs = [_norm_design(d) for d in (opts.get("designs")
                                         or ["current", "trusted_shortcut"])]
    datasets = [_norm_ds(d) for d in (opts.get("datasets")
                                      or ["mmlu", "gsm8k", "boolq", "ag_news"])]
    require_claim_validator = bool(opts.get("require_claim_validator"))

    reports = _load_reports(results_dir) if results_dir.exists() else []
    checks: List[Dict[str, Any]] = []

    def chk(name, ok, detail, *, required=True):
        checks.append({"name": name, "ok": bool(ok), "detail": detail,
                       "required": bool(required)})
        return bool(ok)

    def by(pred):
        return [r for r in reports if pred(r["report"])]

    # ---- 1-2. folded package verify per design + manifest hashes differ ----
    verify_by_design = {}
    for r in reports:
        rep = r["report"]
        if "package_valid" in rep and "nonlinear_backend" in rep \
                and ("manifest_hash" in rep or "num_shards" in rep):
            d = r["_design"]
            if d in designs:
                verify_by_design.setdefault(d, rep)
    for d in designs:
        rep = verify_by_design.get(d)
        chk("folded_verify_exists[%s]" % d, rep is not None,
            "verify report for design %s %s" % (d, "found" if rep else "MISSING"))
        if rep is not None:
            chk("folded_package_valid[%s]" % d, rep.get("package_valid") is True,
                "package_valid=%s" % rep.get("package_valid"))
            chk("nonlinear_backend_ok[%s]" % d,
                rep.get("nonlinear_backend_ok") is True,
                "nonlinear_backend_ok=%s" % rep.get("nonlinear_backend_ok"))
    if len(designs) >= 2:
        hashes = {d: _g(verify_by_design.get(d, {}), "manifest_hash")
                  for d in designs}
        distinct = (all(hashes.get(d) for d in designs)
                    and len(set(hashes.values())) == len(designs))
        chk("manifest_hashes_differ", distinct,
            "per-design manifest_hash must differ: %s" % hashes)

    # ---- 3. boundary artifacts per design ----
    for d in designs:
        cand = [pkg_root / ("qwen7b_boundary_artifact_%s" % d),
                pkg_root / ("qwen7b_boundary_artifact_%s" % d) / "boundary_meta.json"]
        found = any(c.exists() for c in cand) or any(
            (Path(opts.get("boundary_%s" % d) or "/nonexistent")).exists()
            for _ in [0])
        chk("boundary_artifact_exists[%s]" % d, found,
            "expected qwen7b_boundary_artifact_%s under %s" % (d, pkg_root),
            required=bool(opts.get("require_boundary", True)))

    # ---- 4. local decode outputs (not dry_run) ----
    for d in designs:
        loc_reps = [r["report"] for r in reports
                    if "decode" in (_g(r["report"], "stage") or "")
                    and r["_design"] == d
                    and _g(r["report"], "package_backed_decode") is not None]
        non_dry = [x for x in loc_reps if x.get("dry_run") is not True]
        chk("local_decode_present[%s]" % d, bool(loc_reps),
            "%d local decode report(s)" % len(loc_reps), required=False)
        if loc_reps:
            chk("local_decode_not_dry_run[%s]" % d, bool(non_dry),
                "at least one non-dry local decode (found %d non-dry of %d)"
                % (len(non_dry), len(loc_reps)), required=False)

    # ---- 5-6. E3 short matrix per design with tokens 1,4,8,16 + fields ----
    for d in designs:
        e3 = [r["report"] for r in reports
              if (_g(r["report"], "stage") == "remote_package_decode_scaling"
                  or _g(r["report"], "experiment") == "E3")
              and r["_design"] == d]
        chk("e3_exists[%s]" % d, bool(e3),
            "E3 scaling report for design %s %s" % (d, "found" if e3 else "MISSING"))
        if e3:
            rows = _g(e3[0], "rows") or []
            toks = set()
            fields_ok = bool(rows)
            for row in rows:
                if isinstance(row, dict):
                    if row.get("max_new_tokens") is not None:
                        toks.add(int(row["max_new_tokens"]))
                    if not (("tokens_exact_match" in row
                             or "token_match_rate" in row)
                            and ("audit_passed" in row or "all_security_ok" in row
                                 or "security_ok" in row)):
                        fields_ok = False
            chk("e3_tokens_cover_1_4_8_16[%s]" % d,
                set(_E3_TOKENS).issubset(toks),
                "max_new_tokens present: %s" % sorted(toks))
            chk("e3_rows_have_token_audit_fields[%s]" % d, fields_ok,
                "each E3 row needs token-match + audit/security fields")

    # ---- 7-9. E9 plaintext + folded per dataset/design, paper-ready, real ----
    e9 = [r for r in reports
          if _g(r["report"], "stage") == "e9_task_utility_benchmark"]
    # plaintext baseline coverage (per dataset)
    for ds in datasets:
        plain = [r for r in e9
                 if _g(r["report"], "backend") == "plaintext_local"
                 and _norm_ds(_g(r["report"], "dataset")) == ds]
        chk("e9_plaintext_exists[%s]" % ds, bool(plain),
            "plaintext E9 for %s %s" % (ds, "found" if plain else "MISSING"))
    # folded coverage per design x dataset
    for d in designs:
        for ds in datasets:
            fold = [r for r in e9
                    if r["_design"] == d
                    and _g(r["report"], "backend") != "plaintext_local"
                    and _norm_ds(_g(r["report"], "dataset")) == ds]
            chk("e9_folded_exists[%s][%s]" % (d, ds), bool(fold),
                "folded E9 (%s,%s) %s" % (d, ds, "found" if fold else "MISSING"))
    # all E9 must be real (paper_ready, not dry_run) + not fixture/tiny data
    bad_real = []
    bad_data = []
    for r in e9:
        rep = r["report"]
        if rep.get("paper_ready") is not True or rep.get("dry_run") is True:
            bad_real.append(r["_path"])
        off = _data_looks_fixture(rep)
        if off:
            bad_data.append("%s (%s)" % (r["_path"], off))
    chk("e9_all_paper_ready_real", bool(e9) and not bad_real,
        "every E9 must have paper_ready=true & dry_run=false; offenders: %s"
        % (bad_real or "none"))
    chk("e9_no_fixture_or_tiny_data", not bad_data,
        "E9 dataset/cards must not be fixture/tiny: %s" % (bad_data or "none"))

    # ---- 10. pairwise aggregate per design ----
    for d in designs:
        agg = [r for r in reports
               if _g(r["report"], "stage") == "e9_aggregate_utility_preservation"
               and (r["_design"] == d or r["_design"] is None)]
        # prefer one explicitly tagged with this design
        tagged = [r for r in agg if r["_design"] == d]
        chk("pairwise_aggregate_exists[%s]" % d, bool(tagged or agg),
            "aggregate utility preservation for %s %s"
            % (d, "found" if (tagged or agg) else "MISSING"))

    # ---- 11. security transcript scans per design ----
    for d in designs:
        sc = [r for r in reports
              if _g(r["report"], "stage") == "security_transcript_scan"
              and (r["_design"] == d or r["_design"] is None)]
        chk("security_transcript_scan_exists[%s]" % d, bool(sc),
            "transcript scan for %s %s" % (d, "found" if sc else "MISSING"),
            required=False)

    # ---- 12. security negative tests ----
    neg = by(lambda x: _g(x, "stage") == "security_negative_tests")
    chk("security_negative_tests_exists", bool(neg),
        "security_negative_tests report %s" % ("found" if neg else "MISSING"))

    # ---- 13. latency reports ----
    lat = by(lambda x: _g(x, "stage") == "latency_baselines")
    chk("latency_report_exists", bool(lat),
        "latency_baselines report %s" % ("found" if lat else "MISSING"),
        required=False)

    # ---- 14. pre-TDX claim validator (optional) ----
    if require_claim_validator:
        cv = by(lambda x: _g(x, "stage") == "paper_claim_validation")
        chk("pre_tdx_claim_validator_exists", bool(cv),
            "paper_claim_validation report %s" % ("found" if cv else "MISSING"))

    required_fail = [c for c in checks if c["required"] and not c["ok"]]
    passed = not required_fail
    return {
        "stage": "h800_stage_check",
        "results_dir": str(results_dir), "package_root": str(pkg_root),
        "designs": designs, "datasets": datasets,
        "num_reports_scanned": len(reports),
        "passed": passed,
        "num_checks": len(checks),
        "num_required_failed": len(required_fail),
        "blockers": [c["name"] + ": " + c["detail"] for c in required_fail],
        "warnings": [c["name"] + ": " + c["detail"]
                     for c in checks if not c["required"] and not c["ok"]],
        "checks": checks,
    }


def render_md(r: dict) -> str:
    L = ["# H800 stage check", "",
         "- passed: **%s**" % r["passed"],
         "- reports scanned: %d" % r["num_reports_scanned"],
         "- required failed: %d / %d checks" % (r["num_required_failed"],
                                                r["num_checks"]),
         "- designs: %s  datasets: %s" % (", ".join(map(str, r["designs"])),
                                          ", ".join(map(str, r["datasets"]))),
         "", "## Checks", "", "| check | ok | required | detail |",
         "| --- | --- | --- | --- |"]
    for c in r["checks"]:
        L.append("| %s | %s | %s | %s |"
                 % (c["name"], "yes" if c["ok"] else "**NO**",
                    "yes" if c["required"] else "no", c["detail"]))
    if r["blockers"]:
        L += ["", "## Blockers", ""] + ["- %s" % b for b in r["blockers"]]
    if r["warnings"]:
        L += ["", "## Warnings", ""] + ["- %s" % w for w in r["warnings"]]
    L.append("")
    return "\n".join(L)


def _split(s):
    return [x.strip() for x in str(s).split(",") if x.strip()] if s else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="outputs")
    ap.add_argument("--package-root", default="outputs")
    ap.add_argument("--designs", default="A_rightmul,amulet_secure_R")
    ap.add_argument("--datasets", default="mmlu,gsm8k,boolq,ag_news")
    ap.add_argument("--require-claim-validator", action="store_true",
                    default=False)
    ap.add_argument("--no-require-boundary", dest="require_boundary",
                    action="store_false", default=True,
                    help="treat missing boundary artifacts as a warning, not a "
                         "blocker (e.g. boundary built later on the TDX guest)")
    ap.add_argument("--output-json", default="outputs/final/h800_stage_check.json")
    ap.add_argument("--output-md", default="outputs/final/h800_stage_check.md")
    args = ap.parse_args()

    report = run_check({
        "results_dir": args.results_dir, "package_root": args.package_root,
        "designs": _split(args.designs), "datasets": _split(args.datasets),
        "require_claim_validator": args.require_claim_validator,
        "require_boundary": args.require_boundary,
    })

    oj = Path(args.output_json)
    oj.parent.mkdir(parents=True, exist_ok=True)
    oj.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        om = Path(args.output_md)
        om.parent.mkdir(parents=True, exist_ok=True)
        om.write_text(render_md(report), encoding="utf-8")

    print("=== H800 stage check ===")
    print("passed=%s required_failed=%d (scanned %d reports)"
          % (report["passed"], report["num_required_failed"],
             report["num_reports_scanned"]))
    for b in report["blockers"]:
        print("  XX %s" % b)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
