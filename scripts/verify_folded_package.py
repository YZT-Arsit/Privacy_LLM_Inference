"""Verify a folded weight package on disk (manifest + shard hashes + secrets).

Checks the manifest is structurally + security valid, every shard exists and its
sha256 matches the manifest, and no forbidden (mask / plaintext / raw-LoRA /
optimizer) tensor names appear in any shard. Exit code 0 iff ``package_valid``.

Example::

    python scripts/verify_folded_package.py --package-dir packages/qwen7b_folded \\
        --check-manifest true --check-hashes true --check-no-secret-fields true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.deployment import verify_package  # noqa: E402


def _bool(s) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--check-manifest", default="true")
    ap.add_argument("--check-hashes", default="true")
    ap.add_argument("--check-no-secret-fields", default="true")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    rep = verify_package(
        args.package_dir, check_manifest=_bool(args.check_manifest),
        check_hashes=_bool(args.check_hashes),
        check_no_secret_fields=_bool(args.check_no_secret_fields))

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"=== verify folded package ({rep['package_dir']}) ===")
    print(f"package_valid={rep['package_valid']}")
    print(f"manifest_hash={rep['manifest_hash']}")
    print(f"package_size_gb={rep['package_size_gb']} num_shards={rep['num_shards']}")
    print(f"missing_shards={rep['missing_shards']}")
    print(f"hash_mismatches={len(rep['hash_mismatches'])}")
    print(f"forbidden_fields_found={rep['forbidden_fields_found']}")
    print(f"contains_mask_secrets={rep['contains_mask_secrets']} "
          f"contains_plaintext_inputs={rep['contains_plaintext_inputs']} "
          f"contains_raw_lora={rep['contains_raw_lora']} "
          f"contains_optimizer_state={rep['contains_optimizer_state']}")
    if rep["manifest_problems"]:
        print(f"manifest_problems={rep['manifest_problems']}")
    return 0 if rep["package_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
