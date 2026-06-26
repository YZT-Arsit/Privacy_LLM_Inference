"""TDX -> H800 bridge preflight: verify the trusted TDX guest can reach the
untrusted H800 GPU worker BEFORE launching the (expensive) real benchmark.

Topology:  trusted TDX guest (boundary client)  --tunnel-->  H800 GPU worker.

Checks (all best-effort + non-fatal individually; the aggregate decides):
  1. local TDX guest identity (hostname / uname / TDX device / cpu flags),
  2. the H800 SSH alias is reachable,
  3. the worker /health endpoint is reachable,
  4. worker status == "ok",
  5. worker tee_used_on_gpu == false (it is the UNTRUSTED side),
  6. the IFEval input JSONL exists,
  7. the boundary embedding artifact exists,
  8. tokenizer/config small files exist (NO full 7B weights needed TDX-side),
  9. the TDX measurement-coverage log shows an OK marker.

Pure helpers (no network/subprocess) are unit-tested; main() wires the real
fetchers. Writes JSON + markdown. No model weights are ever loaded.

Example (on the TDX guest)::

    python scripts/preflight_tdx_h800_bridge.py \\
      --gpu-worker-url http://127.0.0.1:18082 \\
      --h800-worker-ssh-alias h800-new \\
      --input-jsonl $IFEVAL --embedding-path $EMBED \\
      --model-meta-path $MODEL_META \\
      --tdx-measurement-log outputs/tdx_evidence/tdx_measurement_coverage.log \\
      --output-json outputs/tdx_evidence/preflight.json \\
      --output-md   outputs/tdx_evidence/preflight.md
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
from pathlib import Path

# tokenizer/config small files the trusted boundary needs (NOT the weights)
_META_FILES = ("config.json", "generation_config.json", "tokenizer_config.json")
_TOKENIZER_ANY = ("tokenizer.json", "vocab.json", "merges.txt",
                  "tokenizer.model")


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def check_local_tdx() -> dict:
    """Local TDX-guest identity. Pure-ish (reads /proc + /dev, never network)."""
    flags = []
    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for ln in txt.splitlines():
            if ln.startswith("flags") or ln.startswith("Features"):
                flags = ln.split(":", 1)[1].split() if ":" in ln else []
                break
    except Exception:                                        # noqa: BLE001
        flags = []
    tdx_flag = any(f in flags for f in ("tdx", "tdx_guest"))
    tdx_dev = any(Path(p).exists() for p in ("/dev/tdx_guest", "/dev/tdx-guest",
                                             "/dev/tdx_attest"))
    return {
        "hostname": socket.gethostname(),
        "uname": " ".join(platform.uname()),
        "tdx_cpu_flag": bool(tdx_flag),
        "tdx_device_present": bool(tdx_dev),
    }


def check_worker_health(health) -> dict:
    """Pure: classify a worker /health dict (None when unreachable)."""
    ok = isinstance(health, dict)
    status_ok = bool(ok and health.get("status") == "ok")
    tee = health.get("tee_used_on_gpu") if ok else None
    tee_off = (tee is False)
    backend = health.get("gpu_backend") if ok else None
    resident = (health.get("resident_status") if ok else None) or {}
    return {
        "reachable": ok,
        "status_ok": status_ok,
        "tee_used_on_gpu": tee,
        "tee_used_on_gpu_is_false": tee_off,
        "gpu_backend": backend,
        "resident_folded_weights": resident.get("resident_folded_weights"),
        "resident_cache_active": resident.get("resident_cache_active"),
    }


def check_files(paths: dict) -> dict:
    """Pure: existence of each {label: path}."""
    return {k: bool(v and Path(v).exists()) for k, v in paths.items()}


def check_model_meta(model_meta_path) -> dict:
    """Pure: the trusted boundary needs tokenizer/config small files, NOT the
    full safetensors weights. Report which small files exist + flag if any heavy
    weight shard is present (informational only)."""
    base = Path(model_meta_path) if model_meta_path else None
    present = {}
    if base and base.exists():
        for f in _META_FILES:
            present[f] = (base / f).exists()
        present["tokenizer_any"] = any((base / f).exists()
                                       for f in _TOKENIZER_ANY)
        has_weights = any(base.glob("*.safetensors")) or any(base.glob("*.bin"))
    else:
        for f in _META_FILES:
            present[f] = False
        present["tokenizer_any"] = False
        has_weights = False
    present["config_and_tokenizer_present"] = bool(
        present.get("config.json") and present.get("tokenizer_any"))
    present["full_weights_present"] = bool(has_weights)
    return present


def check_measurement_coverage(path) -> dict:
    """Pure: parse the TDX measurement-coverage log for an OK marker."""
    if not path:
        return {"present": False, "coverage_ok": None}
    p = Path(path)
    if not p.exists():
        return {"present": False, "coverage_ok": False}
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception:                                        # noqa: BLE001
        return {"present": True, "coverage_ok": False}
    ok = ("TDX MEASUREMENT COVERAGE: OK" in txt
          or "exit_code=0" in txt or "exit_code: 0" in txt)
    return {"present": True, "coverage_ok": bool(ok)}


def build_preflight(*, local, ssh_reachable, worker, files, meta,
                    coverage) -> dict:
    """Pure: assemble the preflight report + a single pass/fail decision."""
    passed = bool(
        ssh_reachable and worker.get("reachable") and worker.get("status_ok")
        and worker.get("tee_used_on_gpu_is_false")
        and files.get("input_jsonl") and files.get("embedding_path")
        and meta.get("config_and_tokenizer_present")
        and coverage.get("coverage_ok") is not False)
    return {
        "stage": "tdx_h800_bridge_preflight",
        "preflight_passed": passed,
        "tdx_host": local.get("hostname"),
        "tdx_uname": local.get("uname"),
        "tdx_cpu_flag": local.get("tdx_cpu_flag"),
        "tdx_device_present": local.get("tdx_device_present"),
        "h800_ssh_reachable": bool(ssh_reachable),
        "h800_worker_reachable": worker.get("reachable"),
        "h800_worker_status_ok": worker.get("status_ok"),
        "h800_worker_tee_used_on_gpu": worker.get("tee_used_on_gpu"),
        "h800_worker_gpu_backend": worker.get("gpu_backend"),
        "h800_worker_resident_folded_weights":
            worker.get("resident_folded_weights"),
        "h800_worker_health": worker,
        "input_jsonl_present": files.get("input_jsonl"),
        "embedding_path_present": files.get("embedding_path"),
        "model_meta": meta,
        "tokenizer_config_present": meta.get("config_and_tokenizer_present"),
        "tdx_measurement_log_present": coverage.get("present"),
        "tdx_measurement_coverage_ok": coverage.get("coverage_ok"),
    }


# ---------------------------------------------------------------------------
# real fetchers (mockable)
# ---------------------------------------------------------------------------


def ssh_reachable(alias, *, timeout=10, runner=None) -> bool:
    """True if `ssh <alias> true` succeeds. runner is injectable for tests."""
    if not alias:
        return False
    if runner is None:
        def runner(cmd):
            return subprocess.run(cmd, timeout=timeout,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL).returncode
    try:
        rc = runner(["ssh", "-o", "BatchMode=yes",
                     "-o", "ConnectTimeout=%d" % int(timeout), alias, "true"])
        return rc == 0
    except Exception:                                        # noqa: BLE001
        return False


def fetch_worker_health(url, *, timeout=5, fetcher=None):
    """GET <url>/health -> dict (None on failure). fetcher injectable for tests."""
    if not url:
        return None
    if fetcher is None:
        def fetcher(u):
            import urllib.request
            with urllib.request.urlopen(u, timeout=timeout) as fh:
                return fh.read().decode("utf-8")
    try:
        return json.loads(fetcher(url.rstrip("/") + "/health"))
    except Exception:                                        # noqa: BLE001
        return None


def _markdown(rep) -> str:
    L = ["# TDX -> H800 bridge preflight", "",
         "**preflight_passed: %s**" % rep["preflight_passed"], "",
         "| check | value |", "|---|---|"]
    for k in ("tdx_host", "tdx_cpu_flag", "tdx_device_present",
              "h800_ssh_reachable", "h800_worker_reachable",
              "h800_worker_status_ok", "h800_worker_tee_used_on_gpu",
              "h800_worker_gpu_backend", "h800_worker_resident_folded_weights",
              "input_jsonl_present", "embedding_path_present",
              "tokenizer_config_present", "tdx_measurement_log_present",
              "tdx_measurement_coverage_ok"):
        L.append("| %s | %s |" % (k, rep.get(k)))
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu-worker-url", default="http://127.0.0.1:18082")
    ap.add_argument("--h800-worker-ssh-alias", default="h800-new")
    ap.add_argument("--input-jsonl", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--model-meta-path", default=None)
    ap.add_argument("--tdx-measurement-log", default=None)
    ap.add_argument("--output-json", default="outputs/tdx_evidence/preflight.json")
    ap.add_argument("--output-md", default="outputs/tdx_evidence/preflight.md")
    args = ap.parse_args()

    local = check_local_tdx()
    ssh_ok = ssh_reachable(args.h800_worker_ssh_alias)
    health = fetch_worker_health(args.gpu_worker_url)
    worker = check_worker_health(health)
    files = check_files({"input_jsonl": args.input_jsonl,
                         "embedding_path": args.embedding_path})
    meta = check_model_meta(args.model_meta_path)
    coverage = check_measurement_coverage(args.tdx_measurement_log)
    rep = build_preflight(local=local, ssh_reachable=ssh_ok, worker=worker,
                          files=files, meta=meta, coverage=coverage)

    jp = Path(args.output_json)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    mp = Path(args.output_md)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(_markdown(rep), encoding="utf-8")

    print("=== TDX -> H800 preflight ===")
    print("preflight_passed=%s tdx_host=%s ssh=%s worker_ok=%s tee_off=%s "
          "coverage_ok=%s"
          % (rep["preflight_passed"], rep["tdx_host"], rep["h800_ssh_reachable"],
             rep["h800_worker_status_ok"],
             worker.get("tee_used_on_gpu_is_false"),
             rep["tdx_measurement_coverage_ok"]))
    print("report -> %s" % jp)
    return 0 if rep["preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
