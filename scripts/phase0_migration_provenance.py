"""PHASE 0 — freeze migration provenance before touching either host.

Records HEAD, a fresh worktree source-bundle hash over the exact validated executable scripts, the
immutable private-package root hash (verified == pinned), checkpoint/tokenizer pins, dataset manifest
hashes, worker/service/correction-bundle hashes, and the experiment registry version. Emits the
migration manifest set. Local only; does not rebuild the package.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results/aaai_private_base/alicloud_a10_migration"
OUT.mkdir(parents=True, exist_ok=True)

# exact validated executable set to migrate (data plane + control plane + tests + attestation)
MIGRATION_SCRIPTS = [
    "h800_unified_worker.py", "h800_d4_worker.py", "tdx_persistent_service.py",
    "h800_direct_runner.py", "gate0_direct_orchestrator.py", "d4_trusted_verifier.py",
    "gate0_d4_attestation.py", "build_correction_bundle.py", "compare_direct_vs_ferried.py",
    "test_direct_transport_faults.py", "migration_preflight.py", "build_step_timeline.py",
    "tdx_grad_correction_service.py", "gate0_l10_orchestrator.py",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from h800_unified_worker import compute_root_hash, EXPECTED_ROOT_HASH, EXPECTED_CKPT_SHA, PKG

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    gitstatus = subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True, text=True).stdout

    # source-bundle hashes (fresh, over the exact validated scripts)
    src_hashes = {}
    for name in MIGRATION_SCRIPTS:
        p = REPO / "scripts" / name
        if p.exists():
            src_hashes[f"scripts/{name}"] = sha256_file(p)
    (OUT / "source_file_hashes.sha256").write_text(
        "".join(f"{h}  {k}\n" for k, h in sorted(src_hashes.items())))
    source_bundle_hash = hashlib.sha256(
        json.dumps(src_hashes, sort_keys=True).encode()).hexdigest()

    # immutable package root hash (verify == pinned) — reads every package file
    pkg_root = compute_root_hash(PKG)
    (OUT / "package_hash.txt").write_text(
        f"package_root_hash={pkg_root}\npinned_EXPECTED_ROOT_HASH={EXPECTED_ROOT_HASH}\n"
        f"matches_pinned={pkg_root == EXPECTED_ROOT_HASH}\npackage_dir={PKG}\n")

    # correction bundle + registry + dataset manifests
    cb = REPO / "results/aaai_private_base/correction_bundle/o1c_gamma_bundle.pt"
    reg = REPO / "results/aaai_private_base/empirical_matrix_closure/registry.json"
    ids = REPO / "results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json"
    registry = json.loads(reg.read_text()) if reg.exists() else {}
    ds_manifests = {}
    for name, p in {"dry_run_input_ids": ids}.items():
        if p.exists():
            ds_manifests[name] = sha256_file(p)

    manifest = {
        "phase": "PHASE_0_migration_provenance",
        "head": head,
        "committed_by_me": False,
        "git_status_dirty": bool(gitstatus.strip()),
        "source_bundle_hash": source_bundle_hash,
        "source_file_count": len(src_hashes),
        "package_root_hash": pkg_root,
        "package_root_matches_pinned": pkg_root == EXPECTED_ROOT_HASH,
        "checkpoint_sha_pinned": EXPECTED_CKPT_SHA,
        "tokenizer_hash": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "dataset_manifest_hashes": ds_manifests,
        "h800_unified_worker_hash": src_hashes.get("scripts/h800_unified_worker.py"),
        "h800_d4_worker_hash": src_hashes.get("scripts/h800_d4_worker.py"),
        "tdx_service_hash": src_hashes.get("scripts/tdx_persistent_service.py"),
        "direct_runner_hash": src_hashes.get("scripts/h800_direct_runner.py"),
        "correction_bundle_hash": sha256_file(cb) if cb.exists() else None,
        "experiment_registry_version": registry.get("version") or registry.get("registry_version") or "unversioned",
        "experiment_registry_keys": sorted(registry.keys()) if isinstance(registry, dict) else None,
        "migration_target": {"a10_private_ip": "172.30.25.154", "tdx_private_ip": "172.30.25.153",
                             "gpu": "NVIDIA A10 24GB", "same_vpc_vswitch": True,
                             "data_plane": "private_ip_only", "mac_role": "control_plane_only"},
    }
    (OUT / "migration_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "code_binding.json").write_text(json.dumps({
        "head": head, "source_bundle_hash": source_bundle_hash, "source_file_hashes": src_hashes,
        "package_root_hash": pkg_root, "package_root_matches_pinned": pkg_root == EXPECTED_ROOT_HASH,
        "note": "exact validated code + immutable package pinned for migration to A10"}, indent=2))
    (REPO / "results/aaai_private_base/alicloud_a10_migration/git_status_before.txt").write_text(gitstatus)
    print(json.dumps({"head": head, "source_bundle_hash": source_bundle_hash,
                      "package_root_hash": pkg_root[:16], "matches_pinned": pkg_root == EXPECTED_ROOT_HASH,
                      "scripts": len(src_hashes), "correction_bundle": manifest["correction_bundle_hash"][:16]}, indent=2))


if __name__ == "__main__":
    main()
