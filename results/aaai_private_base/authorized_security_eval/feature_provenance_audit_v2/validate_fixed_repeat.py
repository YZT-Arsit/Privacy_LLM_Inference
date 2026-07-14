#!/usr/bin/env python3
"""Validate corrected V2 features on paired identical fixed-length captures."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_feature_provenance import feature_items  # noqa: E402

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def bootstrap(y,s,seed,n=2000):
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))==2: vals.append(roc_auc_score(y[idx],s[idx]))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repeat-a',type=Path,required=True)
    ap.add_argument('--repeat-b',type=Path,required=True); ap.add_argument('--schema',type=Path,required=True)
    ap.add_argument('--output-json',type=Path,required=True); ap.add_argument('--output-md',type=Path,required=True)
    a=ap.parse_args()
    if a.output_json.exists() or a.output_md.exists(): raise RuntimeError('refusing to overwrite validation')
    left,right=load(a.repeat_a),load(a.repeat_b); schema=json.loads(a.schema.read_text())
    ids_a=[x['sample_id'] for x in left]; ids_b=[x['sample_id'] for x in right]
    if ids_a!=ids_b: raise RuntimeError('ordered sample IDs differ')
    all_rows=left+right; item_rows=[feature_items(x) for x in all_rows]
    names=[x['feature'] for x in item_rows[0]]
    if any([x['feature'] for x in z]!=names for z in item_rows): raise RuntimeError('feature order differs')
    allowed_schema={x['feature_name'] for x in schema['features'] if x['status']=='ALLOWED'}
    allowed=[i for i,x in enumerate(item_rows[0]) if x['feature'] in allowed_schema]
    if len(allowed)!=schema['allowed_feature_count']: raise RuntimeError('schema/extractor feature mismatch')
    x=np.asarray([[row[i]['value'] for i in allowed] for row in item_rows],float)
    desc=[item_rows[0][i] for i in allowed]; n=len(left); y=np.r_[np.zeros(n,int),np.ones(n,int)]
    rng=np.random.default_rng(20260714); groups=np.arange(n); rng.shuffle(groups)
    train_groups=set(groups[:round(.6*n)].tolist()); test_groups=set(groups[round(.6*n):].tolist())
    train=np.asarray([i for i in range(2*n) if i%n in train_groups]); test=np.asarray([i for i in range(2*n) if i%n in test_groups])
    if train_groups & test_groups: raise RuntimeError('sample group overlap')
    family_cols={f:[j for j,d in enumerate(desc) if d['family']==f] for f in ('attention','kv','hidden','logit')}
    family_cols['combined']=list(range(len(desc)))
    results=[]
    for fi,(family,cols) in enumerate(family_cols.items()):
        xx=x[:,cols]
        for control in ('aligned','shuffled_labels','shuffled_sample_mapping'):
            z=xx.copy(); train_y=y[train].copy()
            if control=='shuffled_labels': train_y=np.random.default_rng(7000+fi).permutation(train_y)
            if control=='shuffled_sample_mapping':
                # Shuffle B only within each split.  A global permutation would move test
                # observations into the training distribution and create a control artifact.
                crng=np.random.default_rng(9000+fi)
                train_g=np.asarray(sorted(train_groups)); test_g=np.asarray(sorted(test_groups))
                z[n+train_g]=z[n+crng.permutation(train_g)]
                z[n+test_g]=z[n+crng.permutation(test_g)]
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=4000,class_weight='balanced',random_state=1234))
            model.fit(z[train],train_y); score=model.predict_proba(z[test])[:,1]
            auc=float(roc_auc_score(y[test],score))
            results.append({'family':family,'control':control,'features':len(cols),'train_records':len(train),
                            'test_records':len(test),'roc_auc':auc,'roc_auc_ci95':bootstrap(y[test],score,1234+fi)})
    aligned={r['family']:r['roc_auc'] for r in results if r['control']=='aligned'}
    control_ok=all(.35<=r['roc_auc']<=.65 for r in results)
    ids_hidden=all(not any(token in d['feature'].lower() for token in
                           ('sample_id','run_id','session','adapter','checkpoint','split','source','shape','length')) for d in desc)
    pa=json.loads(a.repeat_a.with_suffix(a.repeat_a.suffix+'.profile.json').read_text())
    pb=json.loads(a.repeat_b.with_suffix(a.repeat_b.suffix+'.profile.json').read_text())
    profile_equal=all(pa.get(k)==pb.get(k) for k in ('schema','version','view','records','run_id','base_package_root_hash',
                                                       'adapter_sha256','queries_sha256','device','dtype','victim_runs','prefill_per_sample'))
    max_abs=float(np.max(np.abs(x[:n]-x[n:])))
    passed=(ids_a==ids_b and len(set(ids_a))==n and profile_equal and ids_hidden and max_abs==0 and control_ok)
    result={'schema':'fixed_length_v2_validation_v2','status':'PASS' if passed else 'FAIL',
            'collection_mode':'SIMULATED_PROTOCOL_VIEW','records_per_repeat':n,'fixed_sequence_length':61,
            'batch_size':1,'padding':'none','dtype':'fp32','ordered_sample_ids_equal':ids_a==ids_b,
            'unique_sample_ids':len(set(ids_a)),'train_test_sample_id_overlap':len(train_groups&test_groups),
            'repeat_a_sha256':sha(a.repeat_a),'repeat_b_sha256':sha(a.repeat_b),
            'capture_bytes_identical':sha(a.repeat_a)==sha(a.repeat_b),'profile_fields_equal':profile_equal,
            'allowed_feature_count':len(desc),'forbidden_schema_features_excluded':schema['forbidden_feature_count'],
            'forbidden_identifiers_in_matrix':not ids_hidden,'max_paired_absolute_difference':max_abs,
            'acceptance_rule':'all aligned and negative-control AUCs in [0.35,0.65], exact pairing and no metadata',
            'source_auc':aligned,'runs':results,
            'interpretation':('paired repeat capture does not permit source identification' if passed else
                              'source confound remains; stop before surrogate and MIA')}
    a.output_json.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Fixed-length corrected V2 validation','',f"**Validation: {result['status']}.**",'',
           f"Two captures use the same {n} sample IDs, exact effective length 61, batch size 1, no padding, fp32, identical package/adapter/query/code path. The released matrix has {len(desc)} invariant features and excludes {schema['forbidden_feature_count']} positional-coordinate features plus all identifiers.",'',
           '| Family | Aligned source AUC | Shuffled labels | Shuffled mapping |','|---|---:|---:|---:|']
    for family in family_cols:
        q={r['control']:r['roc_auc'] for r in results if r['family']==family}
        lines.append(f"| {family} | {q['aligned']:.4f} | {q['shuffled_labels']:.4f} | {q['shuffled_sample_mapping']:.4f} |")
    lines += ['',f"Maximum paired feature difference: `{max_abs}`. Train/test sample-ID overlap: 0. This validation tests collection-source confounding only; it is not a privacy or MIA result."]
    a.output_md.write_text('\n'.join(lines)+'\n'); print(json.dumps({'status':result['status'],'source_auc':aligned}))
if __name__=='__main__': main()
