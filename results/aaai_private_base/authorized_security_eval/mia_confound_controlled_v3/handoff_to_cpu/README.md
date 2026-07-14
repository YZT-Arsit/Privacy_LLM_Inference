# MIA v3 CPU handoff

Status: **PASS**

This handoff contains three independently trained shadow adapters and three matched 1,000-sample V0/V2 collections. V2 is a **SIMULATED_PROTOCOL_VIEW**, not real TDX execution. Paper-facing final MIA was not run in this GPU session.

## Frozen evidence

- 611 allowed invariant features; 96 positional-argmax features excluded.
- Each shadow has 500 members and 500 nonmembers.
- V0 and V2 share one ordered 1,000-sample path per shadow.
- Membership maps are separate from all feature matrices.
- All three training, collection, and validity terminal manifests are PASS.

## Exact GPU commands

```bash
CUDA_VISIBLE_DEVICES=0 python3 /root/mia_confound_controlled_v3/scripts/train_shadow.py --shadow-config /root/mia_confound_controlled_v3/inputs/shadow_s{SEED}/shadow_config.json --train-members /root/mia_confound_controlled_v3/inputs/shadow_s{SEED}/train_members.jsonl --base-model /root/qwen25_05b --output-dir /root/mia_confound_controlled_v3/shadow_{N}/training
PYTHONPATH=/root/mia_confound_controlled_v3/python_deps CUDA_VISIBLE_DEVICES=0 python3 /root/mia_confound_controlled_v3/scripts/collect_shadow.py --adapter /root/mia_confound_controlled_v3/shadow_{N}/training/adapter --queries /root/mia_confound_controlled_v3/inputs/shadow_s{SEED}/evaluation_queries.jsonl --membership /root/mia_confound_controlled_v3/inputs/shadow_s{SEED}/evaluator_membership_metadata.jsonl --schema /root/mia_confound_controlled_v3/inputs/validated_feature_schema.json --base-model /root/qwen25_05b --output-dir /root/mia_confound_controlled_v3/shadow_{N}/collection --max-new-tokens 96
```

Seeds map as N=1→7, N=2→1234, N=3→2025. See `handoff_manifest.json` for exact hashes and `collection_validation.md` for control results.
