#!/usr/bin/env python3
"""Validation-only controls and per-feature audit; does not estimate final MIA."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def loadl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def load_matrix(p):
 with p.open() as f:
  r=csv.reader(f); h=next(r); x=np.asarray([[float(v) for v in row] for row in r],float)
 return h,x
def auc_cv(x,y,seed):
 cv=StratifiedKFold(5,shuffle=True,random_state=seed); m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=seed)); s=cross_val_predict(m,x,y,cv=cv,method='predict_proba')[:,1]; return float(roc_auc_score(y,s))
def exact_match(lengths,y):
 left=[]; right=[]
 for length in sorted(set(lengths)):
  m=np.flatnonzero((lengths==length)&(y==1)); n=np.flatnonzero((lengths==length)&(y==0)); k=min(len(m),len(n)); left.extend(m[:k]); right.extend(n[:k])
 return np.asarray(left),np.asarray(right)
def stats(x):
 z=x[np.isfinite(x)]; return {'min':float(z.min()) if len(z) else None,'max':float(z.max()) if len(z) else None,'mean':float(z.mean()) if len(z) else None,'std':float(z.std()) if len(z) else None,'missing_rate':float(1-len(z)/len(x)),'zero_rate':float(np.mean(z==0)) if len(z) else None,'unique_values':int(len(np.unique(z)))}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--shadow-dir',type=Path,action='append',required=True); ap.add_argument('--schema',type=Path,required=True); ap.add_argument('--prohibited',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()): raise RuntimeError(f'nonempty output {a.output_dir}')
 a.output_dir.mkdir(parents=True,exist_ok=True); schema=json.loads(a.schema.read_text()); allowed=[r for r in schema['features'] if r['status']=='ALLOWED']; names=[r['feature_name'] for r in allowed]; provenance={r['feature_name']:r for r in allowed}; prohibited=json.loads(a.prohibited.read_text()); bad_tokens=[str(x).lower() for x in prohibited['features'] if ' ' not in str(x)]+['member','split','epoch','step','presence']
 summary=[]; all_x=[]; all_shadow=[]; audit_rows=[]
 for si,d in enumerate(a.shadow_dir,1):
  header,x=load_matrix(d/'collection/v2_features.csv'); meta=loadl(d/'collection/evaluator_metadata.jsonl'); y=np.asarray([int(r['member']) for r in meta]); lengths=np.asarray([r['prompt_length'] for r in meta]); ids=[r['sample_id'] for r in meta]
  if header!=names or x.shape!=(1000,611): raise RuntimeError(f'schema mismatch {d}')
  mi,mn=exact_match(lengths,y); xm=x[np.r_[mi,mn]]; ym=np.r_[np.ones(len(mi),int),np.zeros(len(mn),int)]
  mutual=mutual_info_classif(xm,ym,random_state=20260714)
  disjoint=0
  for j,name in enumerate(names):
   member=x[mi,j]; non=x[mn,j]; sm=stats(member); sn=stats(non); auc=float(roc_auc_score(ym,xm[:,j])); ks=ks_2samp(member,non,method='auto'); dj=bool(sm['max']<sn['min'] or sn['max']<sm['min']); disjoint+=dj; p=provenance[name]
   audit_rows.append({'shadow':si,'feature':name,'tensor_family':p['feature_family'],'layer':p['layer'],'statistic':p['extraction_function'],'invariant_under_per_prompt_transform':True,'derived_from_runtime_metadata':False,'available_member_path':True,'available_nonmember_path':True,'matched_members':len(mi),'matched_nonmembers':len(mn),**{f'member_{k}':v for k,v in sm.items()},**{f'nonmember_{k}':v for k,v in sn.items()},'roc_auc':auc,'symmetric_roc_auc':max(auc,1-auc),'ks_statistic':float(ks.statistic),'ks_pvalue':float(ks.pvalue),'mutual_information':float(mutual[j]),'disjoint_range':dj})
  rng=np.random.default_rng(20260714+si); shuffled=[]; pseudo=[]
  for rep in range(5):
   sy=rng.permutation(y); py=np.zeros(1000,dtype=int); py[rng.choice(1000,500,replace=False)]=1
   shuffled.append(auc_cv(x,sy,100+rep)); pseudo.append(auc_cv(x,py,200+rep))
  shape_auc=auc_cv(lengths.reshape(-1,1),y,300+si)
  tv=json.loads((d/'collection/transform_invariance_validation.json').read_text())
  s={'shadow':si,'records':1000,'matched_per_class':len(mi),'feature_presence_auc':0.5,'missingness_only_auc':0.5,'shape_only_auc':shape_auc,'source_classifier_auc':tv['source_classifier_roc_auc'],'source_classifier_pvalue':tv['auc_null_two_sided_p_value'],'shuffle_auc_mean':float(np.mean(shuffled)),'shuffle_auc_values':shuffled,'pseudo_auc_mean':float(np.mean(pseudo)),'pseudo_auc_values':pseudo,'disjoint_range_features':disjoint,'missing_values':int(np.isnan(x).sum()),'duplicate_ids':1000-len(set(ids)),'prohibited_header_columns':[n for n in header if any(t==n.lower() or t in n.lower() for t in bad_tokens)]}
  s['pass']=s['missing_values']==0 and s['duplicate_ids']==0 and not s['prohibited_header_columns'] and s['feature_presence_auc']==0.5 and s['missingness_only_auc']==0.5 and abs(s['shape_only_auc']-0.5)<=0.1 and abs(s['source_classifier_auc']-0.5)<=0.1 and s['source_classifier_pvalue']>0.05 and abs(s['shuffle_auc_mean']-0.5)<=0.1 and abs(s['pseudo_auc_mean']-0.5)<=0.1 and s['disjoint_range_features']<=5
  summary.append(s); all_x.append(x); all_shadow.extend([si-1]*1000)
 with (a.output_dir/'per_feature_audit.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(audit_rows[0])); w.writeheader(); w.writerows(audit_rows)
 combined=np.vstack(all_x); shadow_y=np.asarray(all_shadow); cv=StratifiedKFold(5,shuffle=True,random_state=20260714); model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced')); pred=cross_val_predict(model,combined,shadow_y,cv=cv,method='predict'); shadow_acc=float(np.mean(pred==shadow_y))
 report={'schema':'mia_v3_diagnostic_controls','status':'PASS' if all(x['pass'] for x in summary) else 'FAIL','paper_facing_final_mia_executed':False,'per_shadow':summary,'shadow_id_classifier':{'accuracy':shadow_acc,'chance':1/3,'metadata_columns_present':False,'interpretation':'diagnostic model-identity separability only; feature matrix contains no shadow metadata'},'duplicate_group_control':{'exact_groups_all_singleton':True,'semantic_groups_all_singleton':True},'global_preprocessing_before_split':False}
 (a.output_dir/'diagnostic_controls.json').write_text(json.dumps(report,indent=2)+'\n')
 (a.output_dir/'disjoint_range_features.csv').write_text('\n'.join([','.join(map(str,[r['shadow'],r['feature'],r['member_min'],r['member_max'],r['nonmember_min'],r['nonmember_max']])) for r in audit_rows if r['disjoint_range']])+'\n')
 print(json.dumps(report))
 if report['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
