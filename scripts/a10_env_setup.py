"""A10 environment validation + checkpoint fetch + deterministic package rebuild (runs ON A10).

PHASE 5 + PHASE 4(package): records hardware/software, runs CUDA/bf16/fwd-bwd smoke, downloads the
Qwen2.5-0.5B checkpoint via modelscope and verifies its sha == pinned, then deterministically REBUILDS
the immutable private package and compares its root hash to the pinned bfd578b8 (a documented,
hash-verified reproduction justified by the ~0.4 MB/s Mac-upload portability limit + H800 offline).
Emits PHASE 5 artifacts. Does NOT accept a non-matching package.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path("/root/pllo_pb")
OUT = REPO / "results/aaai_private_base/alicloud_a10_migration/a10_environment"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = Path("/root/qwen25_05b")
PINNED_CKPT_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
PINNED_PKG_ROOT = "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1"
REBUILD_DIR = REPO / "results/aaai_private_base/private_package/gpu_package_rebuilt"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def smoke():
    import torch
    dev = torch.device("cuda")
    r = {"cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0),
         "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
         "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    # bf16 matmul + fwd/bwd smoke
    a = torch.randn(512, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
    y = (a @ b).float().sum(); y.backward()
    torch.cuda.synchronize()
    r["bf16_matmul_ok"] = bool(torch.isfinite(y).item())
    r["fwd_bwd_ok"] = bool(a.grad is not None and torch.isfinite(a.grad).all().item())
    r["vram_alloc_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
    r["not_cpu_only_build"] = torch.cuda.is_available()
    return r


def main():
    import torch
    t0 = time.time()
    # hardware/software
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    hardware = {"gpu_csv": gpu, "cpu_count": os.cpu_count(),
                "mem_kb": int(subprocess.run(["awk", "/MemTotal/{print $2}", "/proc/meminfo"],
                              capture_output=True, text=True).stdout.strip() or 0),
                "disk": subprocess.run(["df", "-h", "/root"], capture_output=True, text=True).stdout.splitlines()[-1]}
    (OUT / "hardware.json").write_text(json.dumps(hardware, indent=2))
    software = {"python": sys.version.split()[0], "torch": torch.__version__,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    try:
        import transformers, safetensors
        software["transformers"] = transformers.__version__; software["safetensors"] = safetensors.__version__
    except Exception as e:
        software["deps_error"] = repr(e)[:100]
    (OUT / "software.json").write_text(json.dumps(software, indent=2))
    sm = smoke()
    (OUT / "cuda_smoke.json").write_text(json.dumps({k: sm[k] for k in ("cuda_available", "device", "capability", "torch", "cuda_runtime")}, indent=2))
    (OUT / "bf16_smoke.json").write_text(json.dumps({k: sm[k] for k in ("bf16_matmul_ok", "fwd_bwd_ok", "vram_alloc_mb", "not_cpu_only_build")}, indent=2))
    (OUT / "storage_inventory.json").write_text(json.dumps({"disk": hardware["disk"],
        "ckpt_present": (CKPT / "model.safetensors").exists()}, indent=2))

    # checkpoint present + sha
    ck = {"path": str(CKPT), "present": (CKPT / "model.safetensors").exists()}
    if ck["present"]:
        ck["sha256"] = sha256_file(CKPT / "model.safetensors")
        ck["matches_pinned"] = ck["sha256"] == PINNED_CKPT_SHA
    (OUT / "checkpoint_verify.json").write_text(json.dumps(ck, indent=2))

    # package: prefer an already-synced exact gpu_package if complete; else rebuild
    pkg = REPO / "results/aaai_private_base/private_package/gpu_package"
    sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
    from h800_unified_worker import compute_root_hash
    result = {"env_smoke": sm, "checkpoint": ck}
    synced_ok = pkg.exists() and len(list(pkg.glob("*.pt"))) >= 245
    if synced_ok:
        root = compute_root_hash(pkg)
        result["package_source"] = "synced_gpu_package"
        result["package_root_hash"] = root
        result["package_root_matches_pinned"] = root == PINNED_PKG_ROOT
    elif ck.get("matches_pinned"):
        os.environ["PB_CKPT_DIR"] = str(CKPT); os.environ["PB_PKG_DIR"] = str(REBUILD_DIR)
        rc = subprocess.run([sys.executable, str(REPO / "scripts/gate0_build_private_package.py")],
                            capture_output=True, text=True, env={**os.environ})
        (OUT / "rebuild_stdout.txt").write_text(rc.stdout[-4000:] + "\n---STDERR---\n" + rc.stderr[-2000:])
        if REBUILD_DIR.exists():
            root = compute_root_hash(REBUILD_DIR)
            result["package_source"] = "deterministic_rebuild"
            result["package_root_hash"] = root
            result["package_root_matches_pinned"] = root == PINNED_PKG_ROOT
            result["rebuild_rc"] = rc.returncode
    else:
        result["package_source"] = "unavailable"; result["package_root_matches_pinned"] = False
    result["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "environment_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ("package_source", "package_root_hash",
          "package_root_matches_pinned")}, indent=2))
    print("smoke:", json.dumps(sm))


if __name__ == "__main__":
    main()
