#!/usr/bin/env python3
"""Hash-close the already completed q+v plaintext target ablation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/aaai_private_base/generative_lora"
OUT = ROOT / "target_ablation"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = [
        OUT / "qv_s1234_train.json", OUT / "qv_s1234_test.jsonl",
        OUT / "qv_s1234_test.jsonl.profile.json",
        OUT / "qv_s1234_adapter/adapter_config.json",
        OUT / "qv_s1234_adapter/adapter_model.safetensors",
        OUT / "metrics/aggregate_metrics.csv", OUT / "metrics/statistical_summary.json",
        OUT / "metrics/per_example_metrics.jsonl",
    ]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    train = json.loads(files[0].read_text())
    profile = json.loads(files[2].read_text())
    if train.get("steps") != 750 or profile.get("n") != 500:
        raise RuntimeError("q+v completion gates failed")
    manifest = {
        "schema": "minimal_target_ablation_manifest", "version": "1.0",
        "seed": 1234, "comparison": {"T1": "q_proj+v_proj", "T4": "all_seven_targets"},
        "all_seven_source": "frozen G1_s1234; not rerun",
        "qv_run": "750-step plaintext training plus 500 direct generations",
        "training_complete": True, "generation_complete": True, "metrics_complete": True,
        "source_hashes": {str(p.relative_to(REPO)): digest(p) for p in files},
    }
    path = OUT / "manifest.json"
    if path.exists() and path.read_text() != json.dumps(manifest, indent=2) + "\n":
        raise RuntimeError(f"refusing to overwrite non-identical manifest: {path}")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
