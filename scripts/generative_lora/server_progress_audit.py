"""Server-side audit for the long-running G2 generative-LoRA experiment.

Designed to run on the A10 from a systemd timer. It has no dependency on the
Mac control plane and appends durable JSONL records while also replacing a
human-readable latest snapshot atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def run(argv: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)


def proc_io(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            out[key] = int(value.strip())
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return out


def process_snapshot(pid: int) -> dict:
    p = run(["ps", "-p", str(pid), "-o", "pid=,ppid=,etimes=,state=,%cpu=,%mem=,rss=,cmd="])
    text = p.stdout.strip()
    return {"alive": bool(text), "ps": text}


def parse_log(path: Path, total_steps: int) -> dict:
    rows = []
    terminal = None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        if any(x in line for x in ("G2_ALL_DONE", "G2_TRAIN_FAILED", "G2_GEN_FAILED",
                                   "G2_COUNTER_FREEZE_FAILED")):
            terminal = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "gbi" in obj and "wall" in obj:
            rows.append(obj)
    completed = max((int(x["gbi"]) for x in rows), default=-1) + 1
    recent = rows[-20:]
    mean_wall = sum(float(x["wall"]) for x in recent) / len(recent) if recent else None
    eta_seconds = mean_wall * max(total_steps - completed, 0) if mean_wall else None
    losses = [float(x["ce"]) for x in recent if math.isfinite(float(x.get("ce", math.nan)))]
    return {
        "completed_steps": completed,
        "total_steps": total_steps,
        "progress_percent": round(100 * completed / total_steps, 3),
        "last_gbi": completed - 1 if completed else None,
        "last_ce": rows[-1].get("ce") if rows else None,
        "recent_mean_ce": round(sum(losses) / len(losses), 6) if losses else None,
        "recent_mean_step_seconds": round(mean_wall, 3) if mean_wall else None,
        "eta_seconds": round(eta_seconds) if eta_seconds is not None else None,
        "all_steps_finite": all(bool(x.get("finite")) for x in rows),
        "terminal_marker": terminal,
        "log_lines": len(lines),
    }


def gpu_samples(count: int, interval: float) -> dict:
    samples = []
    for i in range(count):
        p = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], timeout=10)
        try:
            u, mem, temp, power = [float(x.strip()) for x in p.stdout.strip().split(",")]
            samples.append({"util_percent": u, "memory_mib": mem, "temperature_c": temp,
                            "power_w": power})
        except (ValueError, TypeError):
            pass
        if i + 1 < count:
            time.sleep(interval)
    utils = [x["util_percent"] for x in samples]
    return {
        "samples": len(samples),
        "util_avg_percent": round(sum(utils) / len(utils), 2) if utils else None,
        "util_max_percent": max(utils) if utils else None,
        "active_sample_percent": round(100 * sum(u > 0 for u in utils) / len(utils), 1) if utils else None,
        "memory_mib_last": samples[-1]["memory_mib"] if samples else None,
        "temperature_c_last": samples[-1]["temperature_c"] if samples else None,
        "power_w_last": samples[-1]["power_w"] if samples else None,
    }


def tdx_snapshot(private_ip: str) -> dict:
    # The A10->TDX key is intentionally restricted to a forced trusted-service command. Never use
    # it for a diagnostic SSH command: doing so would spawn a second service. The established data
    # channel owned by the training worker is the correct non-invasive liveness signal on the A10.
    p = run(["ss", "-tpn"], timeout=10)
    lines = [line.strip() for line in p.stdout.splitlines()
             if private_ip in line and "ESTAB" in line and "ssh" in line]
    return {"data_channel_established": bool(lines), "connections": lines}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_fixed_tdx_counter(private_ip: str, key: str, remote: str, local: Path) -> dict:
    command = f"cat {remote}"
    p = subprocess.run(
        ["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         f"root@{private_ip}", command],
        capture_output=True, timeout=30, check=False)
    if p.returncode != 0:
        return {"collected": False, "error": p.stderr.decode("utf-8", "replace")[-500:]}
    try:
        json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        return {"collected": False, "error": f"invalid_json:{exc}"}
    local.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=local.name + ".", dir=local.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(p.stdout)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, local)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return {"collected": True, "path": str(local), "sha256": sha256_file(local),
            "size_bytes": local.stat().st_size}


def collect_completion(repo: Path, out: Path, private_ip: str, key: str) -> dict:
    gl = repo / "results/aaai_private_base/generative_lora"
    counter_dir = gl / "protected_runs"
    remote_root = "/root/privacy_llm_obfuscation/results/generative_lora_counters/e2e_L12_s1234_full"
    counter_results = {
        "training": fetch_fixed_tdx_counter(
            private_ip, key, remote_root + "/training_tdx_counters.json",
            counter_dir / "e2e_L12_s1234_full.training_tdx_counters.json"),
        "generation": fetch_fixed_tdx_counter(
            private_ip, key, remote_root + "/generation_tdx_counters.json",
            counter_dir / "e2e_L12_s1234_full.generation_tdx_counters.json"),
    }
    expected = [
        repo / "g2_e2e_L12_s1234_full.log",
        repo / "g2_driver_e2e_L12_s1234_full.sh",
        gl / "protected_runs/e2e_L12_s1234_full.json",
        gl / "protected_runs/e2e_L12_s1234_full.adapter.pt",
        gl / "protected_runs/e2e_L12_s1234_full.training_tdx_counters.json",
        gl / "protected_runs/e2e_L12_s1234_full.generation_tdx_counters.json",
        gl / "generations/g2_protected_e2e_L12_s1234_full.jsonl",
        gl / "generations/g2_protected_e2e_L12_s1234_full.jsonl.profile.json",
    ]
    files = {}
    for path in expected:
        rel = str(path.relative_to(repo))
        files[rel] = ({"exists": True, "size_bytes": path.stat().st_size,
                       "sha256": sha256_file(path)} if path.is_file() else {"exists": False})
    gen = gl / "generations/g2_protected_e2e_L12_s1234_full.jsonl"
    generation_rows = None
    generation_json_valid = False
    if gen.is_file():
        try:
            generation_rows = sum(1 for line in gen.open() if json.loads(line))
            generation_json_valid = True
        except (json.JSONDecodeError, OSError):
            pass
    manifest = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_tag": "e2e_L12_s1234_full",
        "tdx_counter_collection": counter_results,
        "files": files,
        "generation_rows": generation_rows,
        "generation_json_valid": generation_json_valid,
        "all_expected_files_present": all(v["exists"] for v in files.values()),
    }
    target = out / "completion_manifest.json"
    atomic_json(target, manifest)
    (out / "completion_manifest.json.sha256").write_text(
        f"{sha256_file(target)}  completion_manifest.json\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver-pid", type=int, default=8058)
    ap.add_argument("--worker-pid", type=int, default=8060)
    ap.add_argument("--total-steps", type=int, default=750)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tdx-private-ip", default="172.30.25.153")
    ap.add_argument("--tdx-key", default="/root/.ssh/a10_to_tdx")
    ap.add_argument("--gpu-samples", type=int, default=12)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    args = ap.parse_args()

    started = time.monotonic()
    io0 = proc_io(args.worker_pid)
    gpu = gpu_samples(args.gpu_samples, args.sample_interval)
    elapsed = max(time.monotonic() - started, 1e-6)
    io1 = proc_io(args.worker_pid)
    transport = {}
    for key in ("rchar", "wchar"):
        if key in io0 and key in io1:
            transport[key + "_mib_per_sec"] = round((io1[key] - io0[key]) / elapsed / 2**20, 3)

    record = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_tag": "e2e_L12_s1234_full",
        "driver": process_snapshot(args.driver_pid),
        "worker": process_snapshot(args.worker_pid),
        "progress": parse_log(Path(args.log), args.total_steps),
        "gpu": gpu,
        "process_io_rate": transport,
        "tdx": tdx_snapshot(args.tdx_private_ip),
    }
    if record["progress"]["terminal_marker"]:
        record["health"] = "terminal"
    elif not record["driver"]["alive"] or not record["worker"]["alive"]:
        record["health"] = "process_missing"
    elif (not record["progress"]["all_steps_finite"] or
          not record["tdx"]["data_channel_established"]):
        record["health"] = "unhealthy"
    else:
        record["health"] = "running"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if record["progress"]["terminal_marker"] == "G2_ALL_DONE":
        record["completion_collection"] = collect_completion(
            Path.cwd(), out, args.tdx_private_ip, args.tdx_key)
    atomic_json(out / "latest.json", record)
    with (out / "audit.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
