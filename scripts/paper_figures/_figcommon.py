"""Shared loaders for the paper-figure scripts. Reads ONLY completed-experiment
artifacts; never fabricates numbers."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UTIL = REPO / "results/aaai_private_base/alicloud_a10_runs/utility_dataplane"
SEC = REPO / "results/aaai_private_base/security"
FIGDIR = REPO / "results/aaai_private_base/paper_materials/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def security_manifest():
    return load(SEC / "security_run_manifest.json")


def utility_summary():
    return load(UTIL / "PHASE7_SST2_UTILITY_SUMMARY.json")


def profiling_gate():
    return load(UTIL / "PHASE_profiling_gate.json")


def convergence_cell(tag: str):
    d = load(UTIL / f"{tag}.json")
    if not d:
        return None
    eh = d.get("eval", {}).get("eval_history", [])
    return [(e["step"], e["metric"]) for e in eh]
