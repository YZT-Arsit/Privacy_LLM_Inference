"""PHASE 5 — efficiency / system-overhead comparison from EXISTING measured artifacts.
No new experiments. Separates prototype transport latency from algorithmic overhead.
Writes results/aaai_private_base/system_comparison/{system_comparison.md, manifest.json}.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/aaai_private_base/system_comparison"
LAT = REPO / "results/baselines/obfuscatune_latency_h800.json"
PROF = REPO / "results/aaai_private_base/alicloud_a10_runs/utility_dataplane/PHASE_profiling_gate.json"


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lat = json.loads(LAT.read_text())
    prof = json.loads(PROF.read_text()) if PROF.exists() else {}
    manifest = {
        "schema": "system_comparison", "no_new_experiments": True,
        "sources": {"inference_latency": {"path": str(LAT.relative_to(REPO)), "sha256": sha(LAT),
                                           "gpu": lat.get("gpu"), "dtype": lat.get("dtype"), "dims": lat.get("dims")},
                    "training_profile": {"path": str(PROF.relative_to(REPO)), "sha256": sha(PROF)}},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    L = []; A = L.append
    A("# System overhead comparison (private-base) — existing measured artifacts only\n")
    A(f"Sources: inference latency `{LAT.relative_to(REPO)}` (real {lat.get('gpu')}, {lat.get('dtype')}, "
      f"dims {lat.get('dims')}); training profile `PHASE_profiling_gate.json` (real A10 + Intel TDX). "
      "SHA256 in `manifest.json`. **No new experiments were run for this table.**\n")

    A("## A. Algorithmic inference overhead (compute + TEE boundary crossings)\n")
    A("This is the *algorithmic* cost (TEE compute + boundary crossings), independent of the "
      "prototype network transport. STIP/ObfuscaTune/Amulet are inference-only baselines "
      "(`baseline_manifest.json`).\n")
    A("| Setting | prefill 512 (ms) | slowdown | prefill 1024 (ms) | slowdown | TEE boundary crossings |")
    A("|---|---|---|---|---|---|")
    for m, label in [("plaintext", "plaintext (no protection)"), ("ours_gpu", "**ours** (GPU-TEE)"),
                     ("ours_cpu", "ours (CPU-TEE)"), ("obfuscatune_gpu", "ObfuscaTune (GPU-TEE)"),
                     ("obfuscatune_cpu", "ObfuscaTune (CPU-TEE)")]:
        p5 = lat["prefill"]["512"][m]; p10 = lat["prefill"]["1024"][m]
        A(f"| {label} | {p5['latency_ms']:.1f} | {p5['slowdown_vs_plaintext']:.3f}× | "
          f"{p10['latency_ms']:.1f} | {p10['slowdown_vs_plaintext']:.3f}× | {p5['tee_boundary_crossings']} |")
    A("\n**Key algorithmic result:** ours uses **2** TEE boundary crossings (single trusted "
      "session, zero nonlinear crossings) → **1.00–1.07× slowdown on a GPU-TEE**, versus "
      "ObfuscaTune's **224** crossings. On a CPU-only TEE the crossing count dominates "
      "(ours 2.4–3.0× vs ObfuscaTune 82–114×). STIP/Amulet: inference-only, no LoRA training.\n")

    if prof:
        sw = prof.get("sweep_characterization_only", {}).get("16", {})
        A("## B. Protected LoRA-training overhead (real A10 + Intel TDX, batch 16)\n")
        A("| Component | value | kind |")
        A("|---|---|---|")
        A(f"| sec / batch (16 ex) | {sw.get('sec_per_batch')} | measured |")
        A(f"| TDX AdamW compute / opt step | {sw.get('adamw_tdx_ms')} ms | **algorithmic** (CPU-only enclave, fixed) |")
        A(f"| CE network transfer / opt step | {sw.get('ce_net_ms')} ms | **prototype transport** |")
        A(f"| GPU forward / backward | {sw.get('gpu_fwd_ms')} / {sw.get('gpu_bwd_ms')} ms | GPU workload |")
        A(f"| trusted calls / opt step | 2 | algorithmic |")
        A(f"| comm / example (each way) | {sw.get('bytes_per_example')/1e6:.2f} MB | prototype transport |")

    A("\n## C. Prototype transport vs algorithmic overhead (must not be conflated)\n")
    A("- **Algorithmic overhead** (the paper's claim): 2 TEE crossings, GPU-TEE inference "
      "slowdown **≤1.07×**; fixed per-opt-step enclave AdamW (~114 ms on a CPU-only TDX guest).\n"
      "- **Prototype transport** (implementation, not algorithmic): the per-token decode overhead "
      "measured in the SSH-tunnelled A10↔TDX prototype (~9× per token; ~71 ms tunnel RTT + fp32-logit "
      "payload) is a **deployment artifact of the prototype network path**, not the algorithm; a "
      "co-located enclave / GPU-TEE removes it (cf. the GPU-TEE row above at ≤1.07×). Cross-machine "
      "data-plane throughput on the private VPC ≈170 MB/s (`alicloud_a10_migration/**`).\n")
    A("_Honest separation: we report the algorithmic overhead as the contribution, and label the "
      "prototype network latency as an implementation artifact of the current two-machine testbed._")

    (OUT / "system_comparison.md").write_text("\n".join(L))
    print("wrote system_comparison.md + manifest.json")


if __name__ == "__main__":
    main()
