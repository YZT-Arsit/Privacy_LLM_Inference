#!/usr/bin/env python3
"""Create a fresh, adapter-bound inference-only A10/TDX session pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import time
from pathlib import Path

PACKAGE = "bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1"
MODEL = "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"
TOKENIZER = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
SCHEMA = "959c1a5e1ced8582eb9e43322d42ca596b4f4a79a6ad02269b4ff518c1e69058"
QUERIES = "cfcd454e2a394bbeef046b997c4fe8db7978e3227e30f9097e7f9599381f2826"
SAMPLE_IDS = "d077509179a93d8873bb5254993040ad336b08f48ed0704a5c9de0d1e59b845c"
SERVICE = "c2b32d13d778747cc5054294a5563d56c02150b4148cb00723293ddbe68e796d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--source-adapter-sha256", required=True)
    ap.add_argument("--masked-adapter-sha256", required=True)
    ap.add_argument("--membership-sha256", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    a10_path, tdx_path = args.output_dir / "a10_session.json", args.output_dir / "tdx_session.json"
    if a10_path.exists() or tdx_path.exists():
        raise RuntimeError("refusing to overwrite session files")
    key, nonce = secrets.token_bytes(32), secrets.token_hex(16)
    run_id = f"real-tdx-three-shadow-v1-s{args.shadow}-{int(time.time())}-{secrets.token_hex(4)}"
    commitment = hashlib.sha256(key + bytes.fromhex(nonce)).hexdigest()
    binding = {
        "d4_run_id": run_id, "package_root_hash": PACKAGE,
        "model_config_hash": MODEL, "tokenizer_hash": TOKENIZER,
        "validated_feature_schema_sha256": SCHEMA, "queries_sha256": QUERIES,
        "sample_ids_sha256": SAMPLE_IDS, "source_adapter_sha256": args.source_adapter_sha256,
        "masked_adapter_sha256": args.masked_adapter_sha256,
        "membership_map_sha256_evaluator_only": args.membership_sha256,
        "execution_profile": "paper_safe_inference_only", "optimizer_profile": "none",
        "collection_mode": "REAL_TDX_BACKED_VIEW", "collection_scale": "REAL_TDX_THREE_SHADOW_FULL",
        "feature_columns": 610, "data_plane_ip": "172.30.25.153",
        "nonce": nonce, "hmac_key_commitment": commitment, "debug_false_required": True,
    }
    tdx = {
        "session_key_hex": key.hex(), "run_id": run_id, "labels": [0],
        "gamma_bundle": "/tmp/o1c_bundle.pt", "vocab_seed": 8000,
        "attest": True,
        "attest_out": f"/root/real_tdx_three_shadow_v1/shadow_{args.shadow}/attestation",
        "binding_manifest": binding,
        "counters_out": f"/root/real_tdx_three_shadow_v1/shadow_{args.shadow}/session_counters.json",
        "package_root_hash": PACKAGE, "ckpt_dir": f"/tmp/mia3_inference_s{args.shadow}_ckpt",
        "model_config_hash": MODEL, "service_hash": SERVICE,
        "dataset_id": "frozen_mia_candidate_pool", "task": "clm",
        "template_hash": "not_applicable_token_ids_frozen", "label_schema_hash": "not_used_in_inference",
        "tokenizer_hash": TOKENIZER, "batch_label_tables": {}, "batch_schedules": {},
        "model_cfg": {"num_attention_heads": 14, "num_key_value_heads": 2,
                      "hidden_size": 896, "intermediate_size": 4864},
    }
    a10_path.write_text(json.dumps({"session_key_hex": key.hex(), "run_id": run_id}, indent=2) + "\n")
    tdx_path.write_text(json.dumps(tdx, indent=2) + "\n")
    print(json.dumps({"run_id": run_id, "shadow": args.shadow,
                      "hmac_key_commitment": commitment, "binding_manifest": binding}))


if __name__ == "__main__":
    main()
