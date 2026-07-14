#!/usr/bin/env python3
"""Supplement and verify a paired collection with a complete terminal manifest."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def loadl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--collection-dir',type=Path,required=True); ap.add_argument('--terminal-training-manifest',type=Path,required=True); ap.add_argument('--collector-source',type=Path,required=True); ap.add_argument('--schema',type=Path,required=True); ap.add_argument('--prohibited',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 c=json.loads((a.collection_dir/'collection_manifest.json').read_text()); t=json.loads(a.terminal_training_manifest.read_text()); schema=json.loads(a.schema.read_text()); order=json.loads((a.collection_dir/'sample_order.json').read_text()); v0=loadl(a.collection_dir/'v0_outputs.jsonl'); join=loadl(a.collection_dir/'membership_join.jsonl'); meta=loadl(a.collection_dir/'evaluator_metadata.jsonl')
 with (a.collection_dir/'v2_features.csv').open() as f:
  r=csv.reader(f); header=next(r); x=np.asarray([[float(v) for v in z] for z in r],float)
 expected=[z['feature_name'] for z in schema['features'] if z['status']=='ALLOWED']; lengths=[z['prompt_length'] for z in meta]
 gates={'training_terminal_pass':t['status']=='PASS','collector_pass':c['status']=='PASS','records_1000':len(order)==len(v0)==len(join)==len(meta)==1000,'unique_ids':len(set(order))==1000,'identical_order':order==[z['sample_id'] for z in v0]==[z['sample_id'] for z in join]==[z['sample_id'] for z in meta],'matrix_1000x611':x.shape==(1000,611),'exact_schema':header==expected,'finite':bool(np.isfinite(x).all()),'balanced_ground_truth':sum(z['member'] for z in join)==500,'same_path':c['same_forward_path_for_v0_v2'],'fresh_transform':c['per_prompt_fresh_transform']}
 report={'schema':'mia_v3_collection_terminal_manifest','status':'PASS' if all(gates.values()) else 'FAIL','run_id':t['run_id']+'_collection','shadow_seed':t['shadow_seed'],'views':['V0_MATCHED_EXTERNAL_OUTPUT','V2_SIMULATED_PROTOCOL_VIEW'],'feature_schema_version':schema['version'],'feature_schema_sha256':sha(a.schema),'prohibited_feature_list_sha256':sha(a.prohibited),'collector_source_sha256':sha(a.collector_source),'base_package_root_hash':t['base_package_root_hash'],'adapter_sha256':t['adapter_sha256'],'dtype':'fp32','batch_size':1,'padding_policy':'batch-size-one; no padding','effective_length_handling':{'source':'separate evaluator metadata only','min':min(lengths),'max':max(lengths),'excluded_from_feature_matrix':True},'outputs':{n:sha(a.collection_dir/n) for n in ['v0_outputs.jsonl','v2_features.csv','sample_order.json','membership_join.jsonl','evaluator_metadata.jsonl','transform_commitments.jsonl','transform_invariance_validation.json','collection_manifest.json']},'gates':gates}
 if a.output.exists(): raise RuntimeError(f'refusing overwrite {a.output}')
 a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
 if report['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
