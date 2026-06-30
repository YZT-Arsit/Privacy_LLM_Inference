"""Resume / checkpoint / status / heartbeat helpers for long generation runs.

A memory-constrained GPU server may OOM or drop mid-run, so every AAAI generation
benchmark must be **crash-safe and resumable**:

* each response is appended to the output JSONL and **flushed + fsync'd**
  immediately, so an interrupted run keeps every completed example;
* re-running the same command **skips already-completed ids** (``--resume``);
* a periodically-written status file
  (``outputs/status/<run_id>.status.json``) records total / completed / failed /
  skipped counts, the last completed id, generated-token totals, timestamps, and
  the current dataset/backend/model/example; a heartbeat file
  (``outputs/status/<run_id>.heartbeat.json``) is updated frequently so a monitor
  can tell the run is alive.

stdlib only (json / os / time). No torch. Timestamps come from ``time.time`` via a
small indirection so tests can inject a clock.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "completed_ids_from_jsonl",
    "failed_ids_from_jsonl",
    "append_jsonl_record",
    "RunState",
    "plan_examples",
    "recount_status_from_jsonl",
]


def _read_jsonl_lenient(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, tolerating a truncated final line (crash mid-write)."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                # truncated/partial last line from an interrupted run -- drop it
                continue
    return rows


def completed_ids_from_jsonl(path: str | Path) -> set[str]:
    """Ids in ``path`` that completed successfully (status missing or == 'ok' and
    a non-empty response / a recorded token count)."""
    done: set[str] = set()
    for r in _read_jsonl_lenient(path):
        rid = r.get("id")
        if rid is None:
            continue
        status = r.get("status", "ok")
        if status in (None, "ok", "completed"):
            done.add(str(rid))
    return done


def failed_ids_from_jsonl(path: str | Path) -> set[str]:
    """Ids in ``path`` whose last record is a failure (so they may be retried)."""
    failed: set[str] = set()
    ok: set[str] = set()
    for r in _read_jsonl_lenient(path):
        rid = r.get("id")
        if rid is None:
            continue
        if r.get("status") == "failed":
            failed.add(str(rid))
        else:
            ok.add(str(rid))
    return failed - ok            # a later success clears an earlier failure


def recount_status_from_jsonl(path: str | Path, *, mt_bench: bool = False,
                              required_turns: dict[str, int] | None = None
                              ) -> dict[str, Any]:
    """Recompute the FULL completion state from a response JSONL (resume-aware).

    Uses the *latest* record per id (and per turn for MT-Bench), so an id that was
    ``failed`` and later succeeded counts as completed, never failed. ``skipped``
    records are ignored (neither ok nor failed).

    For MT-Bench a *question* is completed only when every required turn is ok
    (``required_turns[id]`` if given, else every turn seen for that id). Returns
    completed_total / failed_total / ok_ids / failed_ids / total_records (and
    turn_completed_total for MT-Bench)."""
    rows = _read_jsonl_lenient(path)
    total_records = len(rows)
    if mt_bench:
        # id -> {turn_index -> latest status}
        per: dict[str, dict[int, str]] = {}
        for r in rows:
            rid = r.get("id")
            if rid is None:
                continue
            status = r.get("status", "ok")
            if status == "skipped":
                continue
            ti = int(r.get("turn_index") or 0)
            per.setdefault(str(rid), {})[ti] = status
        ok_ids, failed_ids, turn_ok = [], [], 0
        for rid, turns in per.items():
            turn_ok += sum(1 for s in turns.values() if s == "ok")
            need = (range(int(required_turns[rid])) if required_turns
                    and rid in required_turns else turns.keys())
            complete = bool(turns) and all(turns.get(ti) == "ok" for ti in need)
            if complete:
                ok_ids.append(rid)
            elif any(s == "failed" for s in turns.values()):
                failed_ids.append(rid)
        return {"completed_total": len(ok_ids), "failed_total": len(failed_ids),
                "ok_ids": sorted(ok_ids), "failed_ids": sorted(failed_ids),
                "total_records": total_records, "turn_completed_total": turn_ok}
    # single-turn: latest status per id
    latest: dict[str, str] = {}
    for r in rows:
        rid = r.get("id")
        if rid is None:
            continue
        status = r.get("status", "ok")
        if status == "skipped":
            continue
        latest[str(rid)] = status
    ok_ids = sorted(k for k, v in latest.items() if v in (None, "ok", "completed"))
    failed_ids = sorted(k for k, v in latest.items() if v == "failed")
    return {"completed_total": len(ok_ids), "failed_total": len(failed_ids),
            "ok_ids": ok_ids, "failed_ids": failed_ids,
            "total_records": total_records}


def append_jsonl_record(fh, record: dict[str, Any]) -> None:
    """Append one JSON record + newline and force it to disk (flush + fsync)."""
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except (OSError, ValueError):                            # e.g. non-file handle
        pass


def _atomic_write_json(path: str | Path, obj: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)


class RunState:
    """Mutable run state with status + heartbeat checkpointing.

    The runner constructs one, calls :meth:`record_completed` /
    :meth:`record_failed` / :meth:`record_skipped` per example, and
    :meth:`checkpoint` every N examples (and once at the end via :meth:`finish`)."""

    def __init__(self, run_id: str, *, dataset: str | None = None,
                 backend: str | None = None, model: str | None = None,
                 total_examples: int = 0, status_json: str | None = None,
                 heartbeat_json: str | None = None,
                 resume_from_existing: bool = False,
                 nonlinear_backend: str | None = None,
                 output_response_jsonl: str | None = None,
                 paper_facing_generation: bool = False,
                 clock: Callable[[], float] | None = None) -> None:
        self.run_id = run_id
        self.status_json = status_json
        self.heartbeat_json = heartbeat_json
        self._clock = clock or time.time
        now = self._clock()
        self.total_examples = int(total_examples)
        self.completed_examples = 0
        self.failed_examples = 0
        self.skipped_existing_examples = 0
        self.generated_tokens_total = 0
        self.retries_total = 0
        self.reconnects_total = 0
        self.resume_from_existing = bool(resume_from_existing)
        self.last_completed_id: str | None = None
        self.current_dataset = dataset
        self.current_backend = backend
        self.current_model = model
        self.nonlinear_backend = nonlinear_backend
        self.output_response_jsonl = output_response_jsonl
        self.paper_facing_generation = bool(paper_facing_generation)
        self.current_example_id: str | None = None
        self.start_time = now
        self.update_time = now
        self.end_time: float | None = None
        self.failed: list[dict[str, Any]] = []
        try:
            self.pid = os.getpid()
        except Exception:                                       # noqa: BLE001
            self.pid = None
        try:
            self.hostname = socket.gethostname()
        except Exception:                                       # noqa: BLE001
            self.hostname = None

    @property
    def paper_ready_so_far(self) -> bool:
        """True while no example has failed (a single failure forces
        paper_ready=False at the end of the run)."""
        return self.failed_examples == 0

    # -- per-example transitions ---------------------------------------------
    def begin_example(self, example_id: str) -> None:
        self.current_example_id = str(example_id)
        self.update_time = self._clock()

    def record_completed(self, example_id: str, *, tokens: int = 0) -> None:
        self.completed_examples += 1
        self.generated_tokens_total += int(tokens)
        self.last_completed_id = str(example_id)
        self.update_time = self._clock()

    def record_failed(self, example_id: str, *, error_type: str = "",
                      error_message: str = "", retries: int = 0) -> None:
        self.failed_examples += 1
        self.failed.append({"id": str(example_id), "error_type": error_type,
                            "error_message": error_message, "retries": retries})
        self.update_time = self._clock()

    def record_skipped(self, example_id: str) -> None:
        self.skipped_existing_examples += 1
        self.update_time = self._clock()

    def record_robustness(self, *, retries: int = 0,
                          reconnects: int = 0) -> None:
        """Accumulate per-run retry / reconnect counters (worker robustness)."""
        self.retries_total += int(retries)
        self.reconnects_total += int(reconnects)
        self.update_time = self._clock()

    # -- snapshots ------------------------------------------------------------
    def to_status(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_examples": self.total_examples,
            "completed_examples": self.completed_examples,
            "failed_examples": self.failed_examples,
            "skipped_existing_examples": self.skipped_existing_examples,
            "resume_from_existing": self.resume_from_existing,
            "last_completed_id": self.last_completed_id,
            "generated_tokens_total": self.generated_tokens_total,
            "retries_total": self.retries_total,
            "reconnects_total": self.reconnects_total,
            "start_time": self.start_time,
            "update_time": self.update_time,
            "end_time": self.end_time,
            "dataset": self.current_dataset,
            "current_dataset": self.current_dataset,
            "backend": self.current_backend,
            "current_backend": self.current_backend,
            "model_name": self.current_model,
            "current_model": self.current_model,
            "nonlinear_backend": self.nonlinear_backend,
            "output_response_jsonl": self.output_response_jsonl,
            "paper_facing_generation": self.paper_facing_generation,
            "paper_ready_so_far": self.paper_ready_so_far,
            "current_example_id": self.current_example_id,
            "failed": self.failed,
        }

    def to_heartbeat(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "alive": self.end_time is None,
            "pid": self.pid,
            "hostname": self.hostname,
            "timestamp": self.update_time,
            "update_time": self.update_time,
            "elapsed_s": round((self.update_time or 0.0) - self.start_time, 6),
            "completed_examples": self.completed_examples,
            "failed_examples": self.failed_examples,
            "skipped_existing_examples": self.skipped_existing_examples,
            "total_examples": self.total_examples,
            "current_dataset": self.current_dataset,
            "current_backend": self.current_backend,
            "current_model": self.current_model,
            "current_example_id": self.current_example_id,
            "last_completed_id": self.last_completed_id,
        }

    def checkpoint(self) -> None:
        self.update_time = self._clock()
        if self.status_json:
            _atomic_write_json(self.status_json, self.to_status())
        self.heartbeat()

    def heartbeat(self) -> None:
        if self.heartbeat_json:
            _atomic_write_json(self.heartbeat_json, self.to_heartbeat())

    def finish(self) -> None:
        self.end_time = self._clock()
        self.update_time = self.end_time
        self.checkpoint()


def plan_examples(examples: Iterable[dict[str, Any]], completed: set[str], *,
                  resume: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Split ``examples`` into (to_run, skipped_ids) honouring ``resume``."""
    to_run: list[dict[str, Any]] = []
    skipped: list[str] = []
    for ex in examples:
        rid = str(ex.get("id"))
        if resume and rid in completed:
            skipped.append(rid)
        else:
            to_run.append(ex)
    return to_run, skipped
