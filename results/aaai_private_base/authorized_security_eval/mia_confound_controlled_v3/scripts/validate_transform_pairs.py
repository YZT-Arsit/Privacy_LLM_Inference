#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def load(p):
 with p.open() as f: r=csv.reader(f); next(r); return np.asarray([[float(v) for v in x] for x in r])
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--a',type=Path,required=True); ap.add_argument('--b',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); z=ap.parse_args(); a,b=load(z.a),load(z.b); x=np.vstack([a,b]); y=np.r_[np.zeros(len(a),int),np.ones(len(b),int)]; groups=np.r_[np.arange(len(a)),np.arange(len(b))]
 scores=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,random_state=0)),x,y,groups=groups,cv=GroupKFold(4),method='predict_proba')[:,1]; auc=float(roc_auc_score(y,scores)); q1=auc/(2-auc); q2=2*auc*auc/(1+auc); n=len(a); se=math.sqrt(max(1e-30,(auc*(1-auc)+(n-1)*(q1-auc**2)+(n-1)*(q2-auc**2))/(n*n))); p=float(math.erfc(abs(auc-.5)/(se*math.sqrt(2))))
 report={'schema':'mia_v3_transform_invariance_validation','paired_samples':len(a),'feature_columns':a.shape[1],'max_abs_difference':float(np.max(np.abs(a-b))),'tolerance':1e-5,'all_pairs_within_tolerance':bool(np.max(np.abs(a-b))<=1e-5),'source_classifier':'group-disjoint 4-fold standardized logistic regression','source_classifier_roc_auc':auc,'auc_null_two_sided_p_value':p}
 if z.output.exists(): raise RuntimeError(f'refusing overwrite {z.output}')
 z.output.parent.mkdir(parents=True,exist_ok=True); z.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
if __name__=='__main__': main()
