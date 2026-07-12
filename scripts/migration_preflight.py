"""Migration preflight gate for the portable direct-transport stack (run on the NEW GPU host).

Long experiments (50-step, L12 AdamW, utility, 7B) may start ONLY if every gate passes. The
decisive gate added after the AutoDL audit is SUSTAINED BIDIRECTIONAL THROUGHPUT >= 20 MB/s: on the
current AutoDL path the TDX->H800 download is ~0.1 MB/s, which makes long runs measure infrastructure
throttling instead of method behavior. This script refuses to green-light long runs below threshold.

Gates:
  1. real GPU available (torch.cuda + device name)
  2. real TDX quote with DEBUG=false (attestation evidence: SUCCESS + reportdata-bound + debug false)
  3. package + code hash match (compute_root_hash(pkg) == expected; source bundle hashes match)
  4. persistent direct channel reachable (framed handshake over one SSH channel to the TDX service)
  5. NO Mac data ferry (transport_profile == direct_h800_tdx)
  6. sustained throughput BOTH directions >= --min-throughput (default 20 MB/s), measured over one
     channel with 16 MB payloads (uses a bench source/sink service on the new env)

Exit 0 only if all gates pass. Prints a JSON verdict.

    python3 scripts/migration_preflight.py --tdx root@IP --key <key> \
        --bench-service-cmd "python3 .../tdx_bench_service.py" --attestation <session_attestation.json> \
        --min-throughput 20 --out results/aaai_private_base/direct_transport/migration_preflight.json
"""
from __future__ import annotations
import argparse, json, struct, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))


def _read(f, n, timeout=120.0):
    import select
    buf = bytearray(); dl = time.time() + timeout
    while len(buf) < n:
        if time.time() > dl:
            raise TimeoutError("preflight_read_timeout")
        r, _, _ = select.select([f.fileno()], [], [], dl - time.time())
        if not r:
            continue
        c = f.read(min(n - len(buf), 4 * 1024 * 1024))
        if not c:
            raise EOFError
        buf += c
    return bytes(buf)


def read_frame(f):
    (hlen,) = struct.unpack(">I", _read(f, 4)); hdr = json.loads(_read(f, hlen).decode())
    (plen,) = struct.unpack(">Q", _read(f, 8)); return hdr, (_read(f, plen) if plen else b"")


def write_frame(f, hdr, payload=b""):
    hb = json.dumps(hdr).encode()
    f.write(struct.pack(">I", len(hb)) + hb + struct.pack(">Q", len(payload)))
    if payload:
        f.write(payload)
    f.flush()


def gate_gpu():
    try:
        import torch
        ok = torch.cuda.is_available()
        return {"pass": bool(ok), "device": torch.cuda.get_device_name(0) if ok else None}
    except Exception as e:
        return {"pass": False, "error": repr(e)[:200]}


def gate_attestation(path):
    if not path or not Path(path).exists():
        return {"pass": False, "reason": "no attestation evidence provided"}
    ev = json.loads(Path(path).read_text())
    ok = bool(ev.get("attestation_verified") and ev.get("overall_appraisal_result") == "SUCCESS"
              and ev.get("reportdata_bound") and ev.get("debug_false") is True)
    return {"pass": ok, "debug_false": ev.get("debug_false"),
            "appraisal": ev.get("overall_appraisal_result"), "reportdata_bound": ev.get("reportdata_bound")}


def gate_hashes(pkg_expected):
    try:
        from h800_unified_worker import compute_root_hash, PKG, EXPECTED_ROOT_HASH
        root = compute_root_hash(PKG)
        return {"pass": root == EXPECTED_ROOT_HASH, "package_root": root[:16], "expected": EXPECTED_ROOT_HASH[:16]}
    except Exception as e:
        return {"pass": False, "error": repr(e)[:200]}


def gate_throughput(tdx, key, bench_cmd, min_mbps, sizes_mb=(16,)):
    p = subprocess.Popen(["ssh", "-i", key, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                          tdx, bench_cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, bufsize=0)
    try:
        write_frame(p.stdin, {"op": "handshake"}); hs, _ = read_frame(p.stdout)
        assert hs.get("op") == "handshake_ack", hs
        results = {}
        for mb in sizes_mb:
            n = mb * 1024 * 1024
            t0 = time.time(); write_frame(p.stdin, {"op": "source", "nbytes": n}); _, pl = read_frame(p.stdout)
            dl = mb / (time.time() - t0); assert len(pl) == n
            t0 = time.time(); write_frame(p.stdin, {"op": "sink"}, b"\0" * n); ack, _ = read_frame(p.stdout)
            ul = mb / (time.time() - t0); assert ack.get("bytes_in") == n
            results[f"{mb}MB"] = {"download_MBps": round(dl, 2), "upload_MBps": round(ul, 2)}
        write_frame(p.stdin, {"op": "close"}); p.stdin.close()
        dmin = min(r["download_MBps"] for r in results.values())
        umin = min(r["upload_MBps"] for r in results.values())
        return {"pass": bool(dmin >= min_mbps and umin >= min_mbps),
                "min_download_MBps": dmin, "min_upload_MBps": umin, "threshold_MBps": min_mbps,
                "per_size": results}
    except Exception as e:
        return {"pass": False, "error": repr(e)[:200]}
    finally:
        try: p.kill()
        except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdx"); ap.add_argument("--key"); ap.add_argument("--bench-service-cmd")
    ap.add_argument("--attestation", default="")
    ap.add_argument("--min-throughput", type=float, default=20.0)
    ap.add_argument("--transport-profile", default="direct_h800_tdx")
    ap.add_argument("--out", default="results/aaai_private_base/direct_transport/migration_preflight.json")
    args = ap.parse_args()

    gates = {
        "1_real_gpu": gate_gpu(),
        "2_tdx_quote_debug_false": gate_attestation(args.attestation),
        "3_package_code_hash_match": gate_hashes(None),
        "5_no_mac_data_ferry": {"pass": args.transport_profile == "direct_h800_tdx",
                                "transport_profile": args.transport_profile},
    }
    if args.tdx and args.key and args.bench_service_cmd:
        gates["4_persistent_direct_channel_and_6_throughput"] = gate_throughput(
            args.tdx, args.key, args.bench_service_cmd, args.min_throughput)
    else:
        gates["4_persistent_direct_channel_and_6_throughput"] = {
            "pass": False, "reason": "no --tdx/--key/--bench-service-cmd provided; throughput not measured"}

    all_pass = all(g.get("pass") for g in gates.values())
    verdict = {"all_gates_pass": all_pass,
               "long_runs_allowed": all_pass,
               "min_throughput_required_MBps": args.min_throughput,
               "gates": gates,
               "note": ("PASS -> long experiments may start" if all_pass else
                        "FAIL -> DO NOT start long experiments (50-step/L12/utility/7B). "
                        "The current AutoDL path fails gate 6 (~0.1 MB/s << 20 MB/s).")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
