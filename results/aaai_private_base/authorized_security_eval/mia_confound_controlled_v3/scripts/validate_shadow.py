#!/usr/bin/env python3
"""Fail-closed structural, leakage, and consistency gates for one shadow."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def loadl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--training-dir',type=Path,required=True); ap.add_argument('--collection-dir',type=Path,required=True); ap.add_argument('--schema',type=Path,required=True); ap.add_argument('--prohibited',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 schema=json.loads(a.schema.read_text()); expected=[x['feature_name'] for x in schema['features'] if x['status']=='ALLOWED']; forbidden_schema=[x['feature_name'] for x in schema['features'] if x['status']!='ALLOWED']; prohibited=json.loads(a.prohibited.read_text())['features']+json.loads(a.prohibited.read_text()).get('additional',[])
 with (a.collection_dir/'v2_features.csv').open() as f:
  r=csv.reader(f); header=next(r); matrix=np.asarray([[float(x) for x in row] for row in r],dtype=np.float64)
 order=json.loads((a.collection_dir/'sample_order.json').read_text()); membership=loadl(a.collection_dir/'membership_join.jsonl'); evaluator=loadl(a.collection_dir/'evaluator_metadata.jsonl'); v0=loadl(a.collection_dir/'v0_outputs.jsonl'); commits=loadl(a.collection_dir/'transform_commitments.jsonl'); train=json.loads((a.training_dir/'training_manifest.json').read_text()); collect=json.loads((a.collection_dir/'collection_manifest.json').read_text()); transform=json.loads((a.collection_dir/'transform_invariance_validation.json').read_text())
 lower=[x.lower() for x in header]
 leakage=[]
 for name in header:
  n=name.lower()
  if name in forbidden_schema or any(str(p).lower() in n for p in prohibited if ' ' not in str(p)): leakage.append(name)
 gates={
  'training_pass':train['status']=='PASS' and train['adapter']['all_finite'] and not train['warm_started'],
  'collection_pass':collect['status']=='PASS' and collect['same_forward_path_for_v0_v2'],
  'shape_1000x611':matrix.shape==(1000,611),
  'exact_allowed_header':header==expected,
  'all_finite':bool(np.isfinite(matrix).all()),
  'no_prohibited_feature':not leakage,
  'one_common_order':len(order)==1000 and len(set(order))==1000 and order==[x['sample_id'] for x in membership]==[x['sample_id'] for x in evaluator]==[x['sample_id'] for x in v0]==[x['sample_id'] for x in commits],
  'balanced_membership':sum(bool(x['member']) for x in membership)==500 and sum(not bool(x['member']) for x in membership)==500,
  'fresh_transform_commitments':len({x['transform_commitment'] for x in commits})==1000,
  'fresh_transform_invariance':transform['all_pairs_within_tolerance'] and abs(transform['source_classifier_roc_auc']-0.5)<=0.1 and transform['auc_null_two_sided_p_value']>0.05,
  'labels_evaluator_only':all(set(x)=={'sample_id','member'} for x in membership) and 'member' not in header,
  'v0_text_only':all(set(x)=={'sample_id','generated_text'} for x in v0),
 }
 report={'schema':'mia_v3_shadow_validity_gates','status':'PASS' if all(gates.values()) else 'FAIL','gates':gates,'matrix_shape':list(matrix.shape),'leakage_columns':leakage,'matrix_sha256':sha(a.collection_dir/'v2_features.csv'),'adapter_sha256':train['adapter']['sha256'],'schema_sha256':sha(a.schema),'source_transform_control':transform}
 if a.output.exists(): raise RuntimeError(f'refusing overwrite {a.output}')
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
 if report['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
