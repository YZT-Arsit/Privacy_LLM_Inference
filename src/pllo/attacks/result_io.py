"""IO for AttackResult: JSONL append, CSV, Markdown. Failed/blocked records safe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import AttackResult, METRIC_KEYS


def write_jsonl(results: Iterable[AttackResult], path: str | Path, *, append: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode) as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[AttackResult]:
    path = Path(path)
    out: list[AttackResult] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(AttackResult.from_dict(json.loads(line)))
        except Exception:
            continue     # a malformed line never breaks a batch summary
    return out


def read_many_jsonl(paths: Iterable[str | Path]) -> list[AttackResult]:
    out: list[AttackResult] = []
    for p in paths:
        out.extend(read_jsonl(p))
    return out


_CSV_COLS = [
    "attack_id", "attack_name", "attack_family", "implementation_level",
    "target_method", "threat_model", "status",
] + list(METRIC_KEYS) + ["error", "notes"]


def _flat_get(d: dict, col: str):
    if col in d:
        return d[col]
    if col in d.get("metrics", {}):
        return d["metrics"][col]
    return ""


def results_to_csv(results: list[AttackResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(_CSV_COLS)]
    for r in results:
        d = r.to_dict()
        row = []
        for c in _CSV_COLS:
            v = _flat_get(d, c)
            s = "" if v is None else str(v)
            row.append(s.replace(",", ";").replace("\n", " "))
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def results_to_markdown(results: list[AttackResult], path: str | Path,
                        *, title: str = "Attack results") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["attack_id", "target_method", "implementation_level", "threat_model",
            "status", "attack_success_rate", "token_recovery_top1", "notes"]
    md = [f"# {title}", "", "| " + " | ".join(cols) + " |",
          "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in results:
        d = r.to_dict()
        cells = []
        for c in cols:
            v = _flat_get(d, c)
            cells.append("" if v is None else str(v).replace("\n", " ")[:80])
        md.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(md) + "\n")
    return path


def failed_result(attack_id: str, attack_name: str, attack_family: str,
                  target_method: str, threat_model: str, error: str, **kw) -> AttackResult:
    """A status='failed' record that preserves the error, never raising."""
    return AttackResult(
        attack_id=attack_id, attack_name=attack_name, attack_family=attack_family,
        target_method=target_method, threat_model=threat_model, status="failed",
        implementation_level=kw.pop("implementation_level", "full"),
        error=str(error)[:500], notes=f"FAILED: {error}"[:500], **kw)


__all__ = [
    "write_jsonl", "read_jsonl", "read_many_jsonl", "results_to_csv",
    "results_to_markdown", "failed_result",
]
