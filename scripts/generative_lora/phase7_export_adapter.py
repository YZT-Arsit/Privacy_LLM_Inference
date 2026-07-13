"""PHASE 7 (A10 side) — export the trained MASKED adapter (.adapter.pt) as a verified transformed
adapter package, and run the fail-closed negative controls. Runs on the A10. Exports ONLY masked
runtime A/B (never plaintext master / m / v). The fresh-process protected generation is launched
separately (protected_generate --adapter-package) so no training-process object is imported.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_handoff import build_adapter_package, load_adapter_for_generation, AdapterError, TRANSFORM_ALGEBRA_VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-pt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-root", required=True); ap.add_argument("--model-cfg", required=True)
    ap.add_argument("--tokenizer-hash", required=True)
    ap.add_argument("--targets", required=True); ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--adapter-id", required=True); ap.add_argument("--run-id", required=True)
    ap.add_argument("--steps", type=int, default=750); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--neg-out", required=True)
    a = ap.parse_args()

    adp = torch.load(a.adapter_pt, map_location="cpu", weights_only=False)
    targets = a.targets.split(",")
    # regroup {f"{l}.{p}": (A,B)} -> {(l,proj): (A,B)}, keep only target projections
    masked = {}
    for k, (A, B) in adp.items():
        l, proj = k.split("."); l = int(l)
        if proj in targets:
            masked[(l, proj)] = (A, B)
    manifest = build_adapter_package(
        a.out_dir, masked, adapter_id=a.adapter_id, base_package_root_hash=a.base_root,
        model_config_hash=a.model_cfg, tokenizer_hash=a.tokenizer_hash, targets=targets,
        rank=a.rank, alpha=a.alpha, dtype="float32", optimization_profile="EXACT_REFERENCE_L12",
        feature_mask_profile="orthogonal_signed_perm", rank_basis_profile="orthogonal",
        vocabulary_mask_profile="monomial_perm_only", optimizer_state_version=1,
        final_training_step=a.steps, checkpoint_sequence=1, source_run_id=a.run_id, timestamp=float(a.seed))
    print("[export] wrote package", a.out_dir, "factors", manifest["tensor_count"])

    good = dict(expected_base_root_hash=a.base_root, expected_model_config_hash=a.model_cfg,
                expected_targets=targets, expected_rank=a.rank,
                expected_transform_version=TRANSFORM_ALGEBRA_VERSION)
    # positive
    out = load_adapter_for_generation(a.out_dir, **good)
    results = {"positive_load": True, "factors": len(out["tensors"]),
               "no_plaintext": not out["manifest"]["contains_plaintext_adapter"],
               "no_optimizer": not out["manifest"]["contains_optimizer_state"], "controls": {}}

    def rejects(name, **over):
        try:
            load_adapter_for_generation(a.out_dir, **{**good, **over}); results["controls"][name] = "FAIL_ACCEPTED"
        except AdapterError:
            results["controls"][name] = "rejected_ok"
        except Exception as e:  # noqa: BLE001
            results["controls"][name] = f"wrong_exc:{type(e).__name__}"

    rejects("wrong_base_package", expected_base_root_hash="deadbeef" * 8)
    rejects("wrong_target_set", expected_targets=["q_proj", "k_proj"])
    rejects("wrong_rank", expected_rank=999)
    rejects("wrong_transform_version", expected_transform_version="v0.0")
    rejects("stale_version", min_optimizer_state_version=99)

    # tamper controls (modify tensor / manifest)
    from safetensors.torch import load_file, save_file
    import shutil, hashlib
    d2 = a.out_dir + "_modtensor"; shutil.rmtree(d2, ignore_errors=True); shutil.copytree(a.out_dir, d2)
    t = load_file(str(Path(d2) / "adapter_tensors.safetensors"))
    kk = sorted(t)[0]; t[kk] = t[kk] + 1.0; save_file(t, str(Path(d2) / "adapter_tensors.safetensors"))
    try:
        load_adapter_for_generation(d2, **good); results["controls"]["modified_tensor"] = "FAIL_ACCEPTED"
    except AdapterError:
        results["controls"]["modified_tensor"] = "rejected_ok"

    d3 = a.out_dir + "_modmanifest"; shutil.rmtree(d3, ignore_errors=True); shutil.copytree(a.out_dir, d3)
    mp = Path(d3) / "adapter_manifest.json"; mm = json.loads(mp.read_text()); mm["rank"] = 123
    mp.write_text(json.dumps(mm, indent=2, sort_keys=True))
    try:
        load_adapter_for_generation(d3, **good); results["controls"]["modified_manifest"] = "FAIL_ACCEPTED"
    except AdapterError:
        results["controls"]["modified_manifest"] = "rejected_ok"

    # missing tensor
    d4 = a.out_dir + "_missing"; shutil.rmtree(d4, ignore_errors=True); shutil.copytree(a.out_dir, d4)
    t = load_file(str(Path(d4) / "adapter_tensors.safetensors")); t.pop(sorted(t)[0])
    save_file(t, str(Path(d4) / "adapter_tensors.safetensors"))
    try:
        load_adapter_for_generation(d4, **good); results["controls"]["missing_tensor"] = "FAIL_ACCEPTED"
    except AdapterError:
        results["controls"]["missing_tensor"] = "rejected_ok"

    # unexpected optimizer tensor: simulate a package whose blob and outer blob hash were
    # recomputed after injecting optimizer state, while the signed manifest tensor registry
    # remains unchanged. The loader must reject the unexpected tensor name/content.
    d5 = a.out_dir + "_optimizer"; shutil.rmtree(d5, ignore_errors=True); shutil.copytree(a.out_dir, d5)
    t = load_file(str(Path(d5) / "adapter_tensors.safetensors"))
    t["optimizer.m"] = torch.zeros(1, dtype=torch.float32)
    save_file(t, str(Path(d5) / "adapter_tensors.safetensors"))
    hp = Path(d5) / "artifact_hashes.sha256"
    blob_hash = hashlib.sha256((Path(d5) / "adapter_tensors.safetensors").read_bytes()).hexdigest()
    lines = hp.read_text().splitlines()
    hp.write_text("\n".join(
        f"{blob_hash}  adapter_tensors.safetensors" if line.endswith("  adapter_tensors.safetensors") else line
        for line in lines) + "\n")
    try:
        load_adapter_for_generation(d5, **good); results["controls"]["unexpected_optimizer_tensor"] = "FAIL_ACCEPTED"
    except AdapterError:
        results["controls"]["unexpected_optimizer_tensor"] = "rejected_ok"

    results["all_controls_fail_closed"] = all(v == "rejected_ok" for v in results["controls"].values())
    Path(a.neg_out).write_text(json.dumps(results, indent=2))
    print("[export] negative controls:", json.dumps(results["controls"]))
    print("[export] all_fail_closed", results["all_controls_fail_closed"])


if __name__ == "__main__":
    main()
