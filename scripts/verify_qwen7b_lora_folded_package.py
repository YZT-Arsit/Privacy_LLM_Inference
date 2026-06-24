"""Verify a private folded-LoRA package.

Checks shard/manifest integrity, the absence of forbidden / raw-LoRA tensor names
(no raw A/B, optimizer state, training data, mask secrets), target-module
coverage, and (optional) compatibility with the base folded package manifest hash.
Exit 0 iff the package is valid.

Example::

    python scripts/verify_qwen7b_lora_folded_package.py \\
        --lora-folded-package-path /root/.../qwen7b_lora_folded \\
        --base-folded-package-path /root/.../qwen7b_folded_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.deployment.lora_folded_package import verify_lora_folded_package  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora-folded-package-path", required=True)
    ap.add_argument("--base-folded-package-path", default=None,
                    help="if given, require base_package_manifest_hash to match")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    base_hash = None
    if args.base_folded_package_path:
        try:
            from pllo.deployment import compute_manifest_hash, load_manifest
            base_hash = compute_manifest_hash(
                load_manifest(args.base_folded_package_path))
        except Exception as exc:                            # noqa: BLE001
            print("WARNING: could not read base manifest hash: %s" % exc)

    rep = verify_lora_folded_package(args.lora_folded_package_path,
                                     base_manifest_hash=base_hash)
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    print("=== folded-LoRA package verification ===")
    print("lora_package_valid=%s shard_integrity_valid=%s num_shards=%s"
          % (rep["lora_package_valid"], rep["shard_integrity_valid"],
             rep["num_shards"]))
    print("rank=%s alpha=%s target_modules=%s" % (rep["rank"], rep["alpha"],
                                                  rep["target_modules"]))
    print("target_modules_covered=%s missing=%s"
          % (rep["target_modules_covered"],
             rep["target_modules_missing_coverage"]))
    print("forbidden_fields_found=%s raw_lora_tensor_names_found=%s"
          % (rep["forbidden_fields_found"] or "[]",
             rep["raw_lora_tensor_names_found"] or "[]"))
    print("contains_raw_lora=%s contains_optimizer_state=%s "
          "contains_training_data=%s contains_mask_secrets=%s"
          % (rep["contains_raw_lora"], rep["contains_optimizer_state"],
             rep["contains_training_data"], rep["contains_mask_secrets"]))
    if rep.get("base_manifest_match") is not None:
        print("base_manifest_match=%s" % rep["base_manifest_match"])
    print("\nLoRA PACKAGE %s" % ("VALID" if rep["lora_package_valid"]
                                 else "INVALID"))
    return 0 if rep["lora_package_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
