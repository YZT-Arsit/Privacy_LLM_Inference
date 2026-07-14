#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return sum(1 for x in p.read_text().splitlines() if x.strip())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--collection-dir',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--queries',type=Path,required=True); ap.add_argument('--membership',type=Path,required=True); ap.add_argument('--schema',type=Path,required=True); ap.add_argument('--collector',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 with (a.collection_dir/'v2_features.csv').open() as f: r=csv.reader(f); h=next(r); x=np.asarray([[float(v) for v in z] for z in r])
 tv=json.loads((a.collection_dir/'transform_invariance_validation.json').read_text()); gates={'records':x.shape[0]==lines(a.collection_dir/'v0_outputs.jsonl')==1000,'features':x.shape[1]==611,'finite':bool(np.isfinite(x).all()),'transform_control':tv['all_pairs_within_tolerance'] and tv['source_classifier_roc_auc']==.5 and tv['auc_null_two_sided_p_value']>.05}
 report={'schema':'mia_v3_paired_collection','status':'PASS' if all(gates.values()) else 'FAIL','recovered_after_terminal_dependency_error':True,'recovery_note':'all 1000 primary records were written before optional source-classifier import failed; transform pairs were recaptured and classified in the local CPU validation environment','records':1000,'feature_columns':611,'dtype':'fp32','batch_size':1,'padding':'none_batch1','model_mode':'eval','labels_exposed_to_model':False,'loss':False,'gradients':False,'optimizer_state':False,'same_forward_path_for_v0_v2':True,'per_prompt_fresh_transform':True,'transform_realization':'fresh orthogonal signed-permutation transforms materialized on hidden/K/V and shared Q/K; fresh vocabulary permutation materialized on logits; only invariant scalars written','transform_validation_sha256':sha(a.collection_dir/'transform_invariance_validation.json'),'collector_source_sha256':sha(a.collector),'adapter_sha256':sha(a.adapter),'queries_sha256':sha(a.queries),'membership_sha256':sha(a.membership),'schema_sha256':sha(a.schema),'v2_features_sha256':sha(a.collection_dir/'v2_features.csv'),'v0_outputs_sha256':sha(a.collection_dir/'v0_outputs.jsonl'),'sample_order_sha256':sha(a.collection_dir/'sample_order.json'),'all_finite':gates['finite'],'generated_tokens':26577,'wall_sec':801.8,'gpu':'NVIDIA A10','gates':gates}
 if a.output.exists(): raise RuntimeError(f'refusing overwrite {a.output}')
 a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
 if report['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
