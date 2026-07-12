#!/usr/bin/env python3
"""READ-ONLY health monitor for the active A10+TDX converged utility run.

Absolutely non-invasive: reads local task output + result JSONs, and issues
read-only SSH probes (nvidia-smi, ps, tail of nothing) over the EXISTING control
master (/tmp/cm-a10) so no new auth/connection load is placed on the training host.
Never writes, kills, restarts, or touches any training file/checkpoint/env.

Writes on every pass:
  results/aaai_private_base/security_progress_monitor/timestamped_status.json  (appended list)
  results/aaai_private_base/security_progress_monitor/progress_report.md         (latest human view)

Run:  python3 scripts/security/monitor_main_experiment.py --once      # single pass
      python3 scripts/security/monitor_main_experiment.py --loop 1800 # every 30 min
"""
from __future__ import annotations
import argparse, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UTIL = REPO / "results/aaai_private_base/alicloud_a10_runs/utility_dataplane"
OUT = REPO / "results/aaai_private_base/security_progress_monitor"
TASK_OUT = Path("/private/tmp/claude-501/-Users-Hoshino-Desktop-privacy-llm-obfuscation/"
                "03c8cc57-b3de-441b-b9fe-3686deb94965/tasks/b2iv851is.output")
A10 = "root@172.30.25.154"
CM = "/tmp/cm-a10"
SEEDS = [1234, 2025, 7]


def sh(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def ssh_ro(remote_cmd: str, timeout: int = 20) -> str:
    """Read-only SSH over the EXISTING master only (BatchMode; never opens a new master)."""
    rc, out = sh(["ssh", "-S", CM, "-o", "BatchMode=yes", A10, remote_cmd], timeout)
    if rc != 0 and "cm-a10" in out.lower() or "control socket" in out.lower():
        return "MASTER_UNAVAILABLE: " + out.strip()[:200]
    return out.strip()[:4000] if rc == 0 else f"RC{rc}: {out.strip()[:300]}"


def cell_status() -> dict:
    cells = {}
    for tag in ["sst2_conv_L0"] + [f"sst2_conv_{p}_s{s}" for s in SEEDS for p in ("L5", "L12")]:
        f = UTIL / f"{tag}.json"
        if not f.exists():
            cells[tag] = {"state": "pending"}
            continue
        try:
            d = json.loads(f.read_text())
            ev = d.get("eval") or {}
            cells[tag] = {"state": "present", "steps_run": d.get("steps_run"),
                          "acc": (ev.get("accuracy") if ev else None),
                          "nll": (ev.get("mean_nll") if ev else None),
                          "sel_step": ev.get("selected_checkpoint_step") if ev else None,
                          "mtime": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")}
        except Exception as e:  # noqa: BLE001
            cells[tag] = {"state": "unreadable", "err": str(e)[:120]}
    return cells


def one_pass() -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # local, always-safe signals
    driver = sh(["pgrep", "-f", "run_converged_sst2.py"])[1].strip()
    orch = sh(["pgrep", "-f", "gate0_a10_batch_orchestrator.py"])[1].strip()
    tail = TASK_OUT.read_text().splitlines()[-6:] if TASK_OUT.exists() else ["<no task output>"]
    # read-only remote GPU/process probes over existing master
    gpu = ssh_ro("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
                 "--format=csv,noheader 2>/dev/null || echo NO_SMI")
    procs = ssh_ro("ps -eo pid,etime,pcpu,pmem,comm | grep -E 'python|worker' | grep -v grep | head -6")
    ckpt = ssh_ro("ls -la --time-style=+%H:%M:%S /root/*.pt /root/gpustate_*.pt 2>/dev/null | tail -4 || echo none")
    cells = cell_status()
    present = [k for k, v in cells.items() if v.get("state") == "present"]
    healthy = bool(driver or orch)
    status = {
        "ts": now,
        "main_experiment_healthy": healthy,
        "driver_pids": driver.split() if driver else [],
        "orchestrator_pids": orch.split() if orch else [],
        "gpu": gpu,
        "remote_procs": procs,
        "remote_checkpoints": ckpt,
        "task_output_tail": tail,
        "cells_present": present,
        "cells": cells,
        "note": ("main experiment healthy, security experiments continue independently."
                 if healthy else "driver/orchestrator not detected locally; verify run state (read-only)."),
    }
    return status


def write(status: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hist_f = OUT / "timestamped_status.json"
    hist = []
    if hist_f.exists():
        try:
            hist = json.loads(hist_f.read_text())
        except Exception:  # noqa: BLE001
            hist = []
    hist.append(status)
    hist_f.write_text(json.dumps(hist[-200:], indent=2))
    # human report
    c = status["cells"]
    lines = [f"# Main-experiment health monitor (read-only)\n",
             f"_Last pass: {status['ts']} UTC_\n",
             f"**{status['note']}**\n",
             f"- driver pids: {status['driver_pids'] or 'none'} · orchestrator pids: {status['orchestrator_pids'] or 'none'}",
             f"- GPU: `{status['gpu']}`",
             f"- remote checkpoints: `{status['remote_checkpoints']}`\n",
             "| cell | state | steps | dev acc | nll | sel step | mtime |",
             "|---|---|---|---|---|---|---|"]
    for tag, v in c.items():
        lines.append(f"| {tag} | {v.get('state')} | {v.get('steps_run','')} | "
                     f"{v.get('acc','')} | {v.get('nll','')} | {v.get('sel_step','')} | {v.get('mtime','')} |")
    lines += ["\n## last task output", "```", *status["task_output_tail"], "```",
              "\n## remote processes (read-only)", "```", status["remote_procs"], "```"]
    (OUT / "progress_report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes")
    a = ap.parse_args()
    while True:
        st = one_pass()
        write(st)
        print(f"[monitor {st['ts']}] healthy={st['main_experiment_healthy']} "
              f"present={len(st['cells_present'])}/7 gpu={st['gpu'][:60]}", flush=True)
        if a.once or a.loop <= 0:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
