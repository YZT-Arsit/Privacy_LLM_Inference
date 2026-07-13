"""PHASE 0 — audit + freeze for the Real Generative LoRA stage.

Records the REAL hardware/runtime baseline (A10 GPU + Intel TDX, both captured live over SSH by the
orchestrator and passed in via --remote-facts), hashes the local source files that define every
paper-facing code path, and writes the freeze manifests. No experiment is run here.

Every path is explicitly categorized (real GPU+real TDX / real GPU+local trusted diagnostic /
CPU simulation / synthetic-correctness-only / paper-facing-real) so categories are never mixed.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys, time
from pathlib import Path

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
OUT = REPO / "results/aaai_private_base/generative_lora"

# Source files that define the paper-facing code paths (local repo, authoritative post-commit).
SRC = [
    "scripts/a10_batch_runner.py",
    "scripts/tdx_persistent_service.py",
    "scripts/batch_dataplane.py",
    "scripts/secure_tensor_frame.py",
    "scripts/gate0_a10_batch_orchestrator.py",
    "scripts/gate0_a10_l12_orchestrator.py",
    "scripts/native_profile/adapter_handoff.py",
    "scripts/native_profile/profile_n_gate.py",
]


def sha(p: Path) -> str:
    if p.is_dir():
        return "dir"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def main():
    remote_facts = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {}
    OUT.mkdir(parents=True, exist_ok=True)

    src_hashes = {}
    lines = []
    for rel in SRC:
        p = REPO / rel
        h = sha(p) if p.exists() else "MISSING"
        src_hashes[rel] = h
        lines.append(f"{h}  {rel}")
    (OUT / "source_hashes_before.sha256").write_text("\n".join(lines) + "\n")

    manifest = {
        "stage": "real_generative_lora",
        "phase": "0_audit_and_freeze",
        "created_unix": time.time(),
        "git": {
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": git("rev-parse", "HEAD"),
            "dirty_files": [l for l in git("status", "--porcelain").splitlines() if l.strip()],
        },
        "control_plane_runtime": {  # the Mac driving the experiment (NOT a compute path)
            "role": "control_plane_only",
            "python": sys.version.split()[0],
            "note": "The Mac orchestrates over SSH and computes text-metrics offline; it performs "
                    "NO model forward/backward and holds NO mask secrets. Not a paper-facing compute path.",
        },
        "hardware": remote_facts.get("hardware", "NOT_CAPTURED"),
        "model": remote_facts.get("model", {}),
        "transformed_base_package": remote_facts.get("package", {}),
        "current_optimizer_profiles": {
            "E": "EXACT_REFERENCE_L12 (FP32 master + trusted m/v in TDX; plaintext-equivalent)",
            "N": "NATIVE_TRANSFORMED_ADAMW (transformed-coordinate m/v on GPU; NOT stepwise-plaintext-equivalent)",
            "note": "see results/aaai_private_base/native_profile_stage/profiles_registry.yaml",
        },
        "current_lora_target_set": remote_facts.get("lora_targets", "see a10_batch_runner (q/k/v/o/gate/up/down split)"),
        "current_adapter_format": "scripts/native_profile/adapter_handoff.py :: transformed_adapter_manifest v1.0 "
                                  "(safetensors A.l.proj/B.l.proj + manifest + per-tensor sha256 + provenance)",
        "source_hashes_before": src_hashes,
        "code_sync_state": remote_facts.get("code_sync_state", {}),
    }
    (OUT / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("[phase0] wrote baseline_manifest.json + source_hashes_before.sha256")
    print("[phase0] HEAD", manifest["git"]["head"][:12], "dirty", len(manifest["git"]["dirty_files"]))


if __name__ == "__main__":
    main()
