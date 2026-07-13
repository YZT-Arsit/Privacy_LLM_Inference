"""PHASE 4 negative controls — transformed-adapter handoff must fail closed.

Builds an adapter, loads it (positive), then runs the 7 required negative controls:
wrong package, wrong rank, wrong target map, modified tensor, modified manifest, wrong transform
version, stale adapter version. Run: python3 scripts/native_profile/test_adapter_handoff.py
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_handoff import (build_adapter_package, load_adapter_for_generation, AdapterError,  # noqa: E402
                             TRANSFORM_ALGEBRA_VERSION)

P = F = 0


def ok(name, cond):
    global P, F
    if cond:
        P += 1; print(f"  ok   {name}")
    else:
        F += 1; print(f"  FAIL {name}")


def rejects(name, fn):
    try:
        fn(); ok(name, False)
    except AdapterError:
        ok(name, True)
    except Exception as e:  # noqa: BLE001
        ok(name + f" (wrong exc {type(e).__name__})", False)


ROOT = "bfd578b8" * 8
CFG = "479dcf0c" * 8
TARGETS = ["q_proj", "v_proj"]
RANK = 16
GOOD = dict(expected_base_root_hash=ROOT, expected_model_config_hash=CFG,
            expected_targets=TARGETS, expected_rank=RANK,
            expected_transform_version=TRANSFORM_ALGEBRA_VERSION)


def build(d, **over):
    masked = {(0, "q_proj"): (torch.randn(16, 32), torch.randn(16, 8)),
              (0, "v_proj"): (torch.randn(8, 32), torch.randn(16, 4))}
    kw = dict(adapter_id="adp1", base_package_root_hash=ROOT, model_config_hash=CFG,
              tokenizer_hash="tok", targets=TARGETS, rank=RANK, alpha=32, dtype="float32",
              optimization_profile="NATIVE_TRANSFORMED_ADAMW", feature_mask_profile="orthogonal_signed_perm",
              rank_basis_profile="orthogonal", vocabulary_mask_profile="monomial_perm_only",
              optimizer_state_version=5, final_training_step=50, checkpoint_sequence=3,
              source_run_id="run1", timestamp=1234.0)
    kw.update(over)
    return build_adapter_package(d, masked, **kw)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "adapter_package"
        build(d)
        # positive
        out = load_adapter_for_generation(d, **GOOD)
        ok("positive load succeeds", set(out["tensors"]) == {"A.0.q_proj", "B.0.q_proj", "A.0.v_proj", "B.0.v_proj"})
        ok("no plaintext/optimizer state in manifest",
           not out["manifest"]["contains_plaintext_adapter"] and not out["manifest"]["contains_optimizer_state"])

        # 1 wrong package
        rejects("1 wrong base package hash", lambda: load_adapter_for_generation(d, **{**GOOD, "expected_base_root_hash": "deadbeef" * 8}))
        # 2 wrong rank
        rejects("2 wrong rank", lambda: load_adapter_for_generation(d, **{**GOOD, "expected_rank": 8}))
        # 3 wrong target map
        rejects("3 wrong target map", lambda: load_adapter_for_generation(d, **{**GOOD, "expected_targets": ["q_proj", "k_proj"]}))
        # 6 wrong transform version
        rejects("6 wrong transform version", lambda: load_adapter_for_generation(d, **{**GOOD, "expected_transform_version": "v0.9"}))
        # 7 stale adapter version
        rejects("7 stale adapter version", lambda: load_adapter_for_generation(d, **GOOD, min_optimizer_state_version=99))

        # 4 modified tensor (rewrite the safetensors blob)
        d2 = Path(tmp) / "mod_tensor"; build(d2)
        from safetensors.torch import save_file, load_file
        t = load_file(str(d2 / "adapter_tensors.safetensors"))
        t["A.0.q_proj"] = t["A.0.q_proj"] + 1.0
        save_file(t, str(d2 / "adapter_tensors.safetensors"))
        rejects("4 modified tensor", lambda: load_adapter_for_generation(d2, **GOOD))

        # 5 modified manifest (flip a field WITHOUT updating artifact_hashes)
        d3 = Path(tmp) / "mod_manifest"; build(d3)
        mp = d3 / "adapter_manifest.json"
        mm = json.loads(mp.read_text()); mm["rank"] = 999; mp.write_text(json.dumps(mm, indent=2, sort_keys=True))
        rejects("5 modified manifest (hash mismatch)", lambda: load_adapter_for_generation(d3, **GOOD))

        # bonus: plaintext-flagged adapter refused
        d4 = Path(tmp) / "plain"; build(d4)
        mp4 = d4 / "adapter_manifest.json"; mm4 = json.loads(mp4.read_text())
        mm4["contains_plaintext_adapter"] = True
        mp4.write_text(json.dumps(mm4, indent=2, sort_keys=True))
        # rewrite hash so we reach the plaintext check (simulate a fully re-signed malicious pkg)
        import hashlib
        hp = d4 / "artifact_hashes.sha256"
        lines = hp.read_text().splitlines()
        newh = hashlib.sha256(mp4.read_bytes()).hexdigest()
        lines = [f"{newh}  adapter_manifest.json" if "adapter_manifest.json" in l else l for l in lines]
        hp.write_text("\n".join(lines) + "\n")
        rejects("bonus plaintext-flagged adapter refused", lambda: load_adapter_for_generation(d4, **GOOD))

    print(f"\n[test_adapter_handoff] PASS={P} FAIL={F}")
    sys.exit(1 if F else 0)


if __name__ == "__main__":
    main()
