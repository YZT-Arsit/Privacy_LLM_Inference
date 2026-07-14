#!/usr/bin/env python3
"""Build a read-only handoff for a later owner-provisioned real-TDX validation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from fixed_length_view_audit import descriptors

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v2',type=Path,required=True)
    ap.add_argument('--profile',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in a.v2.read_text().splitlines() if x.strip()]
    profile=json.loads(a.profile.read_text()); schema=[]
    for d in descriptors(rows[0]):
        if d['family'] != 'tensor_shapes':
            schema.append({k:d[k] for k in ('name','family','layer','statistic','invariant','runtime_metadata')})
    result={
      'schema':'real_tdx_fidelity_validation_handoff_v1',
      'status':'PREPARED_NOT_EXECUTED',
      'collection_mode_target':'REAL_TDX_BACKED_VIEW',
      'reference_mode':'SIMULATED_PROTOCOL_VIEW',
      'new_tdx_requested_or_configured':False,
      'frozen_sample_ids':[r['sample_id'] for r in rows],
      'sample_count':len(rows), 'feature_schema':schema,
      'simulated_artifacts':{'v2_sha256':sha(a.v2),'profile_sha256':sha(a.profile),
        'package_root_hash':profile['base_package_root_hash'],'adapter_sha256':profile['adapter_sha256']},
      'expected_capture':{'tensor_shapes':rows[0]['tensor_shapes'],'dtype':profile['dtype'],
        'prefill_per_sample':profile['prefill_per_sample'],'feature_count':len(schema),
        'forbidden_fields':['labels','loss','gradients','dlogits','optimizer_state','session_id','run_id','step','trusted_secrets']},
      'comparison_procedure':[
        'verify package, adapter, frozen-ID, capture-source, and schema hashes before collection',
        'collect each frozen ID once with identical dtype, batch size, padding, collection point, and mask policy',
        'remove raw shape and all run/session/source metadata before evaluator release',
        'require exact ID order, field names, tensor shapes, dtypes, missingness, and feature count',
        'compute paired per-feature absolute/relative errors, Spearman correlation, and distributional KS',
        'rerun the frozen output-property metric and report paired REAL_TDX minus SIMULATED differences with bootstrap CIs',
        'label every real result REAL_TDX_BACKED_VIEW and retain simulation results separately'
      ],
      'acceptance_policy':{'no_silent_fallback':True,'metadata_leakage':'zero forbidden fields',
        'decision':'a later evaluator must preregister numerical tolerances from implementation precision before collection'}
    }
    (a.out/'real_tdx_fidelity_handoff.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.out/'real_tdx_fidelity_handoff.md').write_text(
      '# Later real-TDX fidelity-validation handoff\n\n'
      '**Status: PREPARED_NOT_EXECUTED. No TDX resource was requested or configured.**\n\n'
      f"Frozen samples: {len(rows)}; metadata-free features: {len(schema)}; dtype: `{profile['dtype']}`. "
      'The JSON freezes IDs, hashes, schema, expected shapes/dtypes, forbidden fields, and the paired comparison procedure. '
      'A future authorized owner must preregister precision tolerances before collecting a REAL_TDX_BACKED_VIEW.\n')
if __name__=='__main__': main()
