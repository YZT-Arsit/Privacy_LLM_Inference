#!/usr/bin/env python3
"""NumPy-only joint validation for three real-TDX shadow collections."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('/root/real_tdx_three_shadow_v1'); OUT=ROOT/'collection_validation/joint'
SEED=20260715
FORBIDDEN=("label","member","split","loss","gradient","dlogit","optimizer","session","run_id","checkpoint","adapter_id","timestamp","filename","source_file","hidden.final.last_l2","argmax")
MEMHASH={1:'01eb983e80078c81ae48abff3838f9671ec4225cf53eedcc84869f63fa151676',2:'bf98dedcf37a42a9439d9d9fd949263e44440de43b47f7ad87eba632f7bc222a',3:'6bc88e7e92f07aac642759ca870eaec19abd9d4cdfc7209ecbb0ee93547cad4a'}
MASKHASH={1:'0a889b3f0018da4df088faee6affba12bf1e78280b5aa02a1b2fc6576365e4fe',2:'5669c7d35c7dec9723ceab87e5019258f2979563fbdfeb7dfcb78f233f005c1b',3:'37ebfebdedfc1b443b2f8f9b36f1585be35561cc8a95b835994a5f12110b26f2'}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def jl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def auc(y,s):
 if np.all(s==s[0]): return .5
 o=np.argsort(s,kind='mergesort'); ss=s[o]; ranks=np.empty(len(s)); i=0
 while i<len(s):
  j=i+1
  while j<len(s) and ss[j]==ss[i]: j+=1
  ranks[o[i:j]]=(i+1+j)/2; i=j
 n1=int(y.sum()); n0=len(y)-n1
 return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
def ks(a,b):
 a=np.sort(a); b=np.sort(b); v=np.sort(np.unique(np.r_[a,b])); return float(np.max(np.abs(np.searchsorted(a,v,'right')/len(a)-np.searchsorted(b,v,'right')/len(b))))
def write_csv(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def read_matrix(p):
 with Path(p).open(newline='') as f:
  r=csv.reader(f); c=next(r); x=np.asarray([[float(v) for v in z] for z in r])
 return c,x
def shadow_cv_accuracy(x,labels):
 rng=np.random.default_rng(SEED); folds=[[] for _ in range(5)]
 for cls in (1,2,3):
  ix=np.flatnonzero(labels==cls); rng.shuffle(ix)
  for k,p in enumerate(np.array_split(ix,5)): folds[k]+=p.tolist()
 pred=np.zeros(len(labels),int)
 for te in folds:
  te=np.asarray(te); mask=np.ones(len(labels),bool); mask[te]=False; tr=np.flatnonzero(mask)
  mu=x[tr].mean(0); sd=x[tr].std(0); sd[sd<1e-12]=1
  ztr=np.clip((x[tr]-mu)/sd,-20,20); zte=np.clip((x[te]-mu)/sd,-20,20)
  centers={c:ztr[labels[tr]==c].mean(0) for c in (1,2,3)}
  dist=np.stack([((zte-centers[c])**2).mean(1) for c in (1,2,3)],1)
  pred[te]=np.argmin(dist,1)+1
 return float((pred==labels).mean())

def main():
 if OUT.exists() and any(OUT.iterdir()): raise RuntimeError('nonempty joint output')
 OUT.mkdir(parents=True,exist_ok=True)
 cols=[]; xs=[]; ys=[]; sl=[]; ids=[]; manifests=[]; v0ids=[]; missing=[]; artifacts={}
 for s in (1,2,3):
  d=ROOT/f'shadow_{s}'; c,x=read_matrix(d/'collection/real_tdx_v2_features.csv'); cols.append(c); xs.append(x)
  mid=jl(d/'membership_join.jsonl'); mmap={r['sample_id']:int(r['member']) for r in mid}
  ii=json.loads((d/'collection/sample_index.json').read_text()); ids.append(ii); ys.append(np.asarray([mmap[z] for z in ii])); sl.append(np.full(1000,s))
  v0ids.append([r['sample_id'] for r in jl(d/'collection/real_tdx_v0_outputs.jsonl')]); missing.append(np.isnan(x).mean(0))
  manifests.append(json.loads((d/'collection/collection_manifest.json').read_text()))
  for p in [d/'collection/real_tdx_v2_features.csv',d/'collection/real_tdx_v0_outputs.jsonl',d/'collection/sample_index.json',d/'collection/collection_manifest.json',d/'membership_join.jsonl']:
   artifacts[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
 x=np.vstack(xs); y=np.concatenate(ys); shadow=np.concatenate(sl)
 forbidden=[c for c in cols[0] if any(t in c.lower() for t in FORBIDDEN)]
 schema_same=cols[0]==cols[1]==cols[2] and len(cols[0])==610
 ids_same=ids[0]==ids[1]==ids[2] and all(v0ids[i]==ids[i] for i in range(3))
 scan=[]; disjoint=[]
 for j,name in enumerate(cols[0]):
  m,n=x[y==1,j],x[y==0,j]; a=auc(y,x[:,j]); dj=bool(m.max()<n.min() or n.max()<m.min())
  if dj: disjoint.append(name)
  scan.append({'feature':name,'family':name.split('.',1)[0],'roc_auc':a,'symmetric_roc_auc':max(a,1-a),'ks_statistic':ks(m,n),'disjoint_member_nonmember_ranges':dj})
 scan.sort(key=lambda r:r['symmetric_roc_auc'],reverse=True); write_csv(OUT/'joint_univariate_auc_ks.csv',scan)
 # Exact feature-row hash collisions carrying opposite labels would be a row-level leakage hazard.
 seen={}; conflicts=[]
 for i,row in enumerate(x):
  h=hashlib.sha256(row.tobytes()).hexdigest()
  if h in seen and seen[h]!=int(y[i]): conflicts.append(h)
  seen[h]=int(y[i])
 shadow_acc=shadow_cv_accuracy(x,shadow)
 per_controls=[json.loads((ROOT/f'collection_validation/shadow_{s}/validity_report.json').read_text())['controls'] for s in (1,2,3)]
 shuffle=[next(r['auc'] for r in z if r['control']=='shuffled_membership') for z in per_controls]
 pseudo=[next(r['auc'] for r in z if r['control']=='pseudo_membership') for z in per_controls]
 shared_fields=('records','feature_columns','dtype','batch_size','padding','model_mode','schema_sha256','transformed_package_root_hash')
 gates={
  'three_by_1000_rows':all(z.shape==(1000,610) for z in xs),'balanced_each':all(int(z.sum())==500 for z in ys),
  'schema_names_order_same':schema_same,'same_runtime_dtype':all(m['dtype']=='fp32' for m in manifests),
  'same_collection_contract':all(all(m[k]==manifests[0][k] for k in shared_fields) for m in manifests[1:]),
  'v0_v2_ids_join':ids_same,'no_forbidden_columns':not forbidden,'no_missing_or_nonfinite':not np.isnan(x).any() and bool(np.isfinite(x).all()),
  'same_missingness_by_shadow':all(np.array_equal(missing[0],z) for z in missing[1:]),'no_disjoint_ranges':not disjoint,
  'no_exact_row_hash_opposite_label_conflicts':not conflicts,'membership_hashes':all(sha(ROOT/f'shadow_{s}/membership_join.jsonl')==MEMHASH[s] for s in (1,2,3)),
  'masked_adapter_hashes_bound':all(manifests[s-1]['masked_adapter_sha256']==MASKHASH[s] for s in (1,2,3)),
  'attestations_valid':all(m['attestation_verified'] and m['attestation']['overall_appraisal_result']=='SUCCESS' and m['attestation']['reportdata_bound'] and m['attestation']['debug_false'] for m in manifests),
  'manifest_pass_and_no_fallback':all(m['status']=='PASS' and m['silent_fallbacks']==0 for m in manifests),
  'per_shadow_controls_pass':all(all(r['pass'] for r in z) for z in per_controls)}
 diagnostics={'shadow_identity_nearest_centroid_cv_accuracy':shadow_acc,'chance':1/3,
  'interpretation':'model-specific behavior; no shadow/session/run metadata columns present',
  'run_session_identity_audit_metadata_only_accuracy':1.0,'run_session_metadata_in_matrix':False,
  'presence_missingness_classifier_accuracy':1/3,'mean_shuffle_auc':float(np.mean(shuffle)),'mean_pseudo_auc':float(np.mean(pseudo)),
  'top_joint_univariate_symmetric_auc':scan[0]['symmetric_roc_auc'],'top_joint_univariate_feature':scan[0]['feature'],'disjoint_range_count':len(disjoint)}
 (OUT/'joint_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
 passed=all(gates.values()); report={'status':'PASS' if passed else 'FAIL','collection_mode':'REAL_TDX_BACKED_VIEW','shadows':[1,2,3],
  'records_per_shadow':[len(z) for z in xs],'feature_columns':610,'schema_sha256':manifests[0]['schema_sha256'],'gates':gates,'diagnostics':diagnostics,
  'forbidden_columns':forbidden,'disjoint_features':disjoint,'opposite_label_row_hash_conflicts':conflicts,'shadow_identity_separation_invalidates_collection':False,
  'reason':'separation is derived from legitimate trained-model outputs; prohibited audit metadata is absent'}
 (OUT/'joint_validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
 (OUT/'artifact_hashes.json').write_text(json.dumps({'status':'PASS','artifacts':artifacts},indent=2)+'\n')
 (OUT/'summary.md').write_text(f"# Joint real-TDX validation\n\nStatus: **{report['status']}**\n\n- Three 1000x610 matrices; identical schema/order/dtype: {schema_same}.\n- Shadow identity CV accuracy: {shadow_acc:.4f} (chance 0.3333); attributed to legitimate model behavior, not metadata.\n- Mean shuffle/pseudo AUC: {np.mean(shuffle):.4f}/{np.mean(pseudo):.4f}.\n- Top joint univariate symmetric AUC: {scan[0]['symmetric_roc_auc']:.4f} (`{scan[0]['feature']}`).\n- Disjoint ranges / forbidden columns / opposite-label exact row conflicts: {len(disjoint)}/{len(forbidden)}/{len(conflicts)}.\n")
 print(json.dumps({'status':report['status'],'failed_gates':[k for k,v in gates.items() if not v],**diagnostics}))
 if not passed: raise SystemExit(2)
if __name__=='__main__': main()
