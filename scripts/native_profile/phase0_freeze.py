"""PHASE 0 — freeze the current baseline before any code change.

Records branch/HEAD/dirty, hashes protocol source files, the transformed package, the
Qwen2.5-0.5B checkpoint metadata, the L12 results, and archives existing paper tables.
Does NOT overwrite prior experiment artifacts. Read-only except writing into
results/aaai_private_base/native_profile_stage/.
"""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/aaai_private_base/native_profile_stage"
ARCH = OUT / "archived_paper_tables"

SRC_FILES = [
    "scripts/tdx_persistent_service.py", "scripts/a10_batch_runner.py",
    "scripts/batch_dataplane.py", "scripts/h800_unified_worker.py",
    "scripts/h800_direct_runner.py", "scripts/gate0_a10_batch_orchestrator.py",
    "src/pllo/deployment/private_base_package.py",
    "scripts/gate0_build_private_package.py",
]
PKG = REPO / "results/aaai_private_base/private_package/gpu_package"
CKPT_META = ["/Users/Hoshino/privacy_llm_data/checkpoints/Qwen2.5-0.5B/config.json",
             "/Users/Hoshino/privacy_llm_data/checkpoints/Qwen2.5-0.5B/tokenizer_config.json"]
L12_RESULTS = list((REPO / "results/aaai_private_base/alicloud_a10_runs/utility_dataplane").glob("sst2_conv_L12_*.json"))
PAPER_TABLES = REPO / "results/aaai_private_base/paper_materials"


def sha(p: Path):
    if not p.exists():
        return "MISSING"
    if p.is_dir():
        return "dir"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ARCH.mkdir(parents=True, exist_ok=True)
    src = {f: sha(REPO / f) for f in SRC_FILES}
    pkg = {}
    if PKG.exists():
        for p in sorted(PKG.glob("*")):
            if p.is_file():
                pkg[p.name] = sha(p)
    ckpt = {Path(c).name: sha(Path(c)) for c in CKPT_META}
    # checkpoint weights: hash metadata only (size + first bytes) to avoid re-hashing 988MB unnecessarily
    wf = Path("/Users/Hoshino/privacy_llm_data/checkpoints/Qwen2.5-0.5B/model.safetensors")
    ckpt["model.safetensors.size_bytes"] = wf.stat().st_size if wf.exists() else -1
    l12 = {p.name: sha(p) for p in L12_RESULTS}

    # archive paper tables (copy, do not move/overwrite originals)
    archived = []
    for md in sorted(PAPER_TABLES.glob("*.md")):
        dst = ARCH / md.name
        if not dst.exists():
            dst.write_bytes(md.read_bytes())
        archived.append(md.name)

    manifest = {
        "schema": "native_profile_baseline_freeze", "stage": "native_profile_stage",
        "git": {"branch": sh(["git", "branch", "--show-current"]),
                "head": sh(["git", "rev-parse", "HEAD"]),
                "dirty_file_count": len([l for l in sh(["git", "status", "--porcelain"]).splitlines() if l]),
                "dirty": sh(["git", "status", "--porcelain"]).splitlines()[:20]},
        "source_files_before": src,
        "transformed_package": {"dir": str(PKG.relative_to(REPO)), "artifact_count": len(pkg),
                                "artifact_hashes": pkg},
        "checkpoint_metadata": ckpt,
        "l12_results": l12,
        "archived_paper_tables": archived,
        "files_to_be_changed_this_stage": "recorded in stage_registry.yaml after each edit",
        "constraints": ["do not commit", "do not modify/invalidate frozen artifacts",
                        "do not rerun full completed matrix unless a new gate requires it"],
    }
    (OUT / "baseline_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    # source_hashes_before.sha256 (plain format)
    lines = [f"{h}  {f}" for f, h in src.items()]
    (OUT / "source_hashes_before.sha256").write_text("\n".join(lines) + "\n")
    print("PHASE 0 freeze written:", OUT)
    print("  branch", manifest["git"]["branch"], "head", manifest["git"]["head"][:12],
          "dirty", manifest["git"]["dirty_file_count"])
    print("  src files hashed:", len(src), "| pkg artifacts:", len(pkg), "| L12 results:", len(l12),
          "| archived tables:", len(archived))


if __name__ == "__main__":
    main()
