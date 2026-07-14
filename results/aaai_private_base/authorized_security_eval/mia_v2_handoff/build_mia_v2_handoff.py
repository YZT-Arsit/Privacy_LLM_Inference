#!/usr/bin/env python3
"""Prepare, but do not execute, the corrected MIA handoff."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--schema',type=Path,required=True); ap.add_argument('--validation',type=Path,required=True)
    ap.add_argument('--pool',type=Path,required=True); ap.add_argument('--pool-manifest',type=Path,required=True)
    ap.add_argument('--assignments',type=Path,required=True); ap.add_argument('--shadow-plan',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    targets=['validated_feature_schema.json','frozen_sample_ids.json','member_nonmember_assignment.jsonl','shadow_adapter_requirements.json','prohibited_feature_list.json','collection_manifest.json','README.md']
    if any((a.output/x).exists() for x in targets): raise RuntimeError('refusing to overwrite MIA handoff')
    schema=json.loads(a.schema.read_text()); validation=json.loads(a.validation.read_text()); plan=json.loads(a.shadow_plan.read_text())
    pool=[json.loads(x) for x in a.pool.read_text().splitlines() if x.strip()]; assignments=[json.loads(x) for x in a.assignments.read_text().splitlines() if x.strip()]
    if [x['sample_id'] for x in pool] != [x['sample_id'] for x in assignments]: raise RuntimeError('pool/assignment order mismatch')
    (a.output/'validated_feature_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    frozen={'schema':'mia_v2_frozen_sample_ids','records':len(pool),'ordered_sample_ids':[x['sample_id'] for x in pool],
            'normalized_input_hashes':[x['normalized_input_hash'] for x in pool],'duplicate_group_ids':[x['duplicate_group_id'] for x in pool],
            'pool_sha256':sha(a.pool),'order_frozen':True}
    (a.output/'frozen_sample_ids.json').write_text(json.dumps(frozen,indent=2)+'\n')
    (a.output/'member_nonmember_assignment.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in assignments))
    req={'schema':'mia_v2_shadow_adapter_requirements','status':'PREPARED_NOT_LAUNCHED','shadow_seeds':[7,1234,2025],
         'members_per_shadow':500,'evaluation_samples_per_shadow':1000,'rank':8,'alpha':16,
         'targets':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
         'requirements':['one independently trained adapter per shadow seed','same frozen 1000-sample pool for member and nonmember collection','same capture binary/config/batch/dtype/padding/collection point for both labels','fresh transform policy identical across labels within each shadow','adapter/package/schema hashes pinned before capture','membership labels joined evaluator-side only','leave-one-shadow-run-out evaluation'],
         'existing_plan':plan['shadows'],'jobs_launched_by_this_handoff':False}
    (a.output/'shadow_adapter_requirements.json').write_text(json.dumps(req,indent=2)+'\n')
    (a.output/'prohibited_feature_list.json').write_text(json.dumps({'schema':'mia_v2_prohibited_features','features':schema['prohibited_fields'],
        'additional':['membership_by_shadow','source_population','original_sample_id','duplicate_group_id as model input','normalized_input_hash as model input'],
        'enforcement':'fail audit if any prohibited name or derived path/presence flag enters the feature matrix'},indent=2)+'\n')
    blocked=any(x['status']!='COMPLETE' for x in plan['shadows'])
    manifest={'schema':'mia_v2_corrected_collection_manifest','status':'BLOCKED' if blocked else 'READY',
              'blocked_reasons':['three shadow adapters are PREPARED_NOT_LAUNCHED','matched 1000-sample V0/V2 captures do not exist'] if blocked else [],
              'feature_validation_status':validation['status'],'feature_source_auc':validation['source_auc'],
              'validated_feature_count':schema['allowed_feature_count'],'raw_metadata_feature_count':0,
              'artifacts':{'schema_sha256':sha(a.schema),'validation_sha256':sha(a.validation),'pool_sha256':sha(a.pool),
                           'pool_manifest_sha256':sha(a.pool_manifest),'assignments_sha256':sha(a.assignments),'shadow_plan_sha256':sha(a.shadow_plan)},
              'collection_contract':{'same_ordered_sample_ids':True,'same_collection_code_path':True,'same_batch_construction':True,
                  'same_dtype':'fp32','same_padding_and_fixed_length_buckets':True,'identical_feature_extraction':True,
                  'metadata_stripped_before_model':True,'labels_evaluator_only':True,'source_classifier_must_pass_per_shadow':True},
              'final_mia_executed':False,'reason':'handoff preparation only'}
    (a.output/'collection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.output/'README.md').write_text('# Corrected MIA v2 handoff\n\n**Status: BLOCKED. Final MIA was not run.** The corrected 611-feature schema passed paired collection-source validation, but the three shadow adapters and matched 1,000-sample captures remain unexecuted. This directory freezes sample order, membership assignments, prohibited fields, adapter requirements, hashes, and the collection contract.\n')
    print(json.dumps({'status':manifest['status'],'samples':len(pool),'features':schema['allowed_feature_count']}))
if __name__=='__main__': main()
