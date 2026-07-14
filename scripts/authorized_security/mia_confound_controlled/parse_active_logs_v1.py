#!/usr/bin/env python3
"""Read-only SSH snapshot/parser for active T1/T2 protected-run logs."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


JOBS = (
    {"cell": "T1_qv", "host": "39.107.123.173", "pid": 59219,
     "log": "/root/pllo_pb/g2_e2e_T1_qv_s1234_protected.log", "targets": "q_proj,v_proj"},
    {"cell": "T2_qkvo", "host": "8.147.119.67", "pid": 16093,
     "log": "/root/pllo_pb/g2_e2e_T2_qkvo_s1234_protected.log", "targets": "q_proj,k_proj,v_proj,o_proj"},
)


def ssh(key: Path, host: str, command: str) -> str:
    result = subprocess.run([
        "ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new", f"root@{host}", command,
    ], check=True, capture_output=True, text=True, timeout=30)
    return result.stdout


def parse_log(text: str) -> dict:
    steps = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "gbi" in row and "ce" in row:
            steps.append(row)
    finite = all(bool(row.get("finite")) for row in steps)
    last = steps[-1] if steps else {}
    recent = steps[-20:]
    return {
        "parsed_steps": len(steps), "completed_steps": int(last.get("gbi", -1)) + 1,
        "last_gbi": last.get("gbi"), "last_ce": last.get("ce"),
        "last_step_wall_sec": last.get("wall"), "all_steps_finite": finite,
        "recent_mean_ce": (sum(float(row["ce"]) for row in recent) / len(recent) if recent else None),
        "recent_mean_step_wall_sec": (sum(float(row["wall"]) for row in recent) / len(recent) if recent else None),
        "terminal_marker": "G2_ALL_DONE" in text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite log snapshot directory: {args.output}")
    args.output.mkdir(parents=True)
    rows = []
    for job in JOBS:
        log_text = ssh(args.ssh_key, job["host"], f"cat {job['log']} 2>/dev/null || true")
        ps_text = ssh(args.ssh_key, job["host"], f"ps -p {job['pid']} -o user=,pid=,ppid=,lstart=,etime=,cmd= || true")
        gpu_text = ssh(args.ssh_key, job["host"],
                       "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,pstate --format=csv,noheader")
        stat_text = ssh(args.ssh_key, job["host"], f"stat -c '%y|%s' {job['log']} 2>/dev/null || true").strip()
        parsed = parse_log(log_text)
        modified, size = (stat_text.split("|", 1) if "|" in stat_text else ("", "0"))
        row = {
            "snapshot_time": datetime.now().astimezone().isoformat(), "cell": job["cell"],
            "host": job["host"], "remote_driver_pid": job["pid"], "targets": job["targets"],
            "driver_alive": bool(ps_text.strip()), "remote_os_owner": (ps_text.split()[0] if ps_text.strip() else ""),
            "log_modified": modified, "log_size_bytes": int(size or 0),
            "gpu_snapshot": gpu_text.strip(), "process_snapshot": re.sub(r"\s+", " ", ps_text.strip()),
            **parsed,
        }
        rows.append(row)
        (args.output / f"{job['cell']}_log_snapshot.txt").write_text(log_text)
    with (args.output / "active_job_log_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"schema": "read_only_active_target_log_snapshot", "version": "1.0",
              "jobs": rows, "remote_mutations": False, "signals_sent": False}
    (args.output / "active_job_log_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Active T1/T2 Read-Only Log Snapshot", "",
             "No process was signaled and no remote file was modified.", "",
             "| Cell | Alive | Steps | Last CE | Recent CE | Recent sec/step | Finite | GPU snapshot |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['cell']} | {row['driver_alive']} | {row['completed_steps']}/750 | "
                     f"{row['last_ce'] if row['last_ce'] is not None else 'N/A'} | "
                     f"{row['recent_mean_ce'] if row['recent_mean_ce'] is not None else 'N/A'} | "
                     f"{row['recent_mean_step_wall_sec'] if row['recent_mean_step_wall_sec'] is not None else 'N/A'} | "
                     f"{row['all_steps_finite']} | {row['gpu_snapshot']} |")
    (args.output / "active_job_snapshot.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
