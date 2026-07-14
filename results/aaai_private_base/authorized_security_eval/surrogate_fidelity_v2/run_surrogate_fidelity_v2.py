#!/usr/bin/env python3
"""Validated-schema surrogate fidelity evaluation (not model extraction)."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import torch

BUDGETS=[.01,.05,.10,.20]; SEEDS=[7,1234,2025]; DIM=1024; HIDDEN=64; STEPS=300
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def acc(y,p): return float(np.mean(np.asarray(y)==np.asarray(p)))
def f1(y,p):
    z=[]
    for c in range(3):
        tp=np.sum((y==c)&(p==c)); fp=np.sum((y!=c)&(p==c)); fn=np.sum((y==c)&(p!=c))
        z.append(float(2*tp/max(1,2*tp+fp+fn)))
    return float(np.mean(z))
def ce(y,p): return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())
def red(v):
    x=np.asarray(v,float); return [float(x.mean()),float(x.std()),float(x.min()),float(x.max())]
def v2_features(r):
    out=[]; h=r['transformed_hidden']; out.append(float(h['final_last_l2']))
    out += [float(x['last_l2']) for x in h['per_layer']]
    for field in ('masked_k','masked_v'):
        for x in r[field]: out += red(x['last_l2_by_head'])
    for x in r['attention_scores']:
        for key in ('last_entropy_by_head','last_max_by_head','true_score_last_mean_by_head','true_score_last_std_by_head'):
            out += red(x[key])
    lg=r['masked_logits']; out += [float(lg[k]) for k in ('mean','std','l2','min','max')]
    out += red(lg['top_values']); out.append(float(lg['top_values'][0]-lg['top_values'][1]))
    if len(out)!=611: raise RuntimeError(f'expected 611 validated features, got {len(out)}')
    return np.asarray(out,np.float32)
def hash_text(t,dim=256):
    x=np.zeros(dim,np.float32); t=' '.join(t.lower().split())
    for n in (2,3,4):
        for i in range(max(0,len(t)-n+1)):
            h=int.from_bytes(hashlib.blake2b(t[i:i+n].encode(),digest_size=8).digest(),'little')
            x[h%dim] += 1 if h>>63 else -1
    d=np.linalg.norm(x); return x/d if d else x
def pad(x):
    if len(x)>DIM: raise RuntimeError('feature dimension overflow')
    return np.pad(np.asarray(x,np.float32),(0,DIM-len(x)))
def label(r):
    n=len(r['output_token_ids']); return 0 if n<=27 else 1 if n<=34 else 2
def split_groups(rows):
    norm=[' '.join(r['output_text'].lower().split()) for r in rows]; grams=[]
    for t in norm:
        w=t.split(); grams.append(set(zip(w,w[1:])))
    parent=list(range(len(rows)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for i in range(len(rows)):
        for j in range(i):
            u=len(grams[i]|grams[j]); sim=len(grams[i]&grams[j])/u if u else 0
            if norm[i]==norm[j] or sim>=.90: union(i,j)
    g={}
    for i in range(len(rows)): g.setdefault(find(i),[]).append(i)
    groups=list(g.values()); np.random.default_rng(20260714).shuffle(groups)
    test=[]; val=[]; pool=[]
    for q in groups: (test if len(test)<180 else val if len(val)<40 else pool).extend(q)
    return np.asarray(test),np.asarray(val),np.asarray(pool),len(groups)
def prefix(idx,y,n,seed):
    rng=np.random.default_rng(seed); by=[]
    for c in range(3):
        z=idx[y[idx]==c].copy(); rng.shuffle(z); by.append(list(z))
    out=[]
    while len(out)<n:
        for c in range(3):
            if by[c] and len(out)<n: out.append(by[c].pop())
    return np.asarray(out)
class MLP(torch.nn.Module):
    def __init__(self): super().__init__(); self.net=torch.nn.Sequential(torch.nn.Linear(DIM,HIDDEN),torch.nn.ReLU(),torch.nn.Linear(HIDDEN,3))
    def forward(self,x): return self.net(x)
def fit(x,y,tr,te,seed,device,train_labels=None):
    torch.manual_seed(seed); mean=x[tr].mean(0); std=x[tr].std(0); std[std<1e-6]=1
    xt=torch.tensor((x[tr]-mean)/std,device=device); xe=torch.tensor((x[te]-mean)/std,device=device)
    yy=torch.tensor(y[tr] if train_labels is None else train_labels,dtype=torch.long,device=device)
    m=MLP().to(device); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-4)
    for _ in range(STEPS): opt.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(m(xt),yy); loss.backward(); opt.step()
    with torch.no_grad(): return torch.softmax(m(xe),-1).cpu().numpy()
def boot_acc(y,p,seed,n=2000):
    rng=np.random.default_rng(seed); z=[]
    for _ in range(n):
        q=rng.integers(0,len(y),len(y)); z.append(acc(y[q],p[q]))
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]
def boot_mean(v,seed,n=5000):
    v=np.asarray(v,float); rng=np.random.default_rng(seed); z=[rng.choice(v,len(v),replace=True).mean() for _ in range(n)]
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--v0',type=Path,required=True); ap.add_argument('--v2',type=Path,required=True)
    ap.add_argument('--schema',type=Path,required=True); ap.add_argument('--validation',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    validation=json.loads(a.validation.read_text()); schema=json.loads(a.schema.read_text())
    if validation['status']!='PASS': raise RuntimeError('validated feature schema gate did not pass')
    if schema['allowed_feature_count']!=611: raise RuntimeError('unexpected schema feature count')
    v0,v2=load(a.v0),load(a.v2); ids0=[r['sample_id'] for r in v0]; ids2=[r['sample_id'] for r in v2]
    if ids0!=ids2 or len(set(ids0))!=len(ids0): raise RuntimeError('pool alignment/duplicate failure')
    y=np.asarray([label(r) for r in v0]); xv0=np.asarray([pad(hash_text(r['output_text'])) for r in v0]); xv2=np.asarray([pad(v2_features(r)) for r in v2])
    views={'V0':xv0,'V2':xv2,'V0_plus_V2':np.asarray([pad(np.r_[x[:256],z[:611]]) for x,z in zip(xv0,xv2)]),
           'TRUSTED_POSITIVE_CONTROL':np.asarray([pad(np.eye(3,dtype=np.float32)[c]) for c in y])}
    test,val,pool,groups=split_groups(v0); device='cuda' if torch.cuda.is_available() else 'cpu'; runs=[]
    random=np.random.default_rng(20260714).normal(size=(len(y),DIM)).astype(np.float32)
    mapped=xv2[np.random.default_rng(20260715).permutation(len(y))]
    control_views={'RANDOM_FEATURES':random,'RANDOM_MAPPING':mapped}
    for budget in BUDGETS:
        nq=max(3,math.ceil(len(y)*budget))
        for seed in SEEDS:
            tr=prefix(pool,y,nq,seed)
            for view,x in {**views,**control_views}.items():
                prob=fit(x,y,tr,test,seed,device); pred=prob.argmax(1)
                runs.append({'budget':budget,'queries':len(tr),'seed':seed,'view':view,'control':'aligned',
                             'target_agreement':acc(y[test],pred),'task_accuracy':acc(y[test],pred),
                             'output_agreement':acc(y[test],pred),'macro_f1':f1(y[test],pred),'cross_entropy':ce(y[test],prob),
                             'agreement_ci95':json.dumps(boot_acc(y[test],pred,seed))})
                shuffled=np.random.default_rng(seed+10000).permutation(y[tr]); prob=fit(x,y,tr,test,seed,device,shuffled); pred=prob.argmax(1)
                runs.append({'budget':budget,'queries':len(tr),'seed':seed,'view':view,'control':'SHUFFLED_OUTPUTS',
                             'target_agreement':acc(y[test],pred),'task_accuracy':acc(y[test],pred),
                             'output_agreement':acc(y[test],pred),'macro_f1':f1(y[test],pred),'cross_entropy':ce(y[test],prob),
                             'agreement_ci95':json.dumps(boot_acc(y[test],pred,seed+1))})
    with (a.output/'metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(runs[0])); w.writeheader(); w.writerows(runs)
    agg=[]
    for budget in BUDGETS:
        for view in {**views,**control_views}:
            z=[r for r in runs if r['budget']==budget and r['view']==view and r['control']=='aligned']
            vals=[r['target_agreement'] for r in z]; agg.append({'budget':budget,'queries':z[0]['queries'],'view':view,'agreement_mean':float(np.mean(vals)),'agreement_ci95':boot_mean(vals,20260714)})
    gains=[]
    for budget in BUDGETS:
        d=[]
        for seed in SEEDS:
            q=lambda v:next(r['target_agreement'] for r in runs if r['budget']==budget and r['seed']==seed and r['view']==v and r['control']=='aligned')
            d.append(q('V2')-q('V0'))
        gains.append({'budget':budget,'v2_minus_v0_mean':float(np.mean(d)),'ci95':boot_mean(d,20260714)})
    conclusion='does not provide consistent measurable additional surrogate utility' if any(g['v2_minus_v0_mean']<=0 or g['ci95'][0]<=0 for g in gains) else 'provides measurable additional surrogate utility'
    cfg={'schema':'surrogate_fidelity_v2_config','collection_mode':'SIMULATED_PROTOCOL_VIEW','target':'coarse output-token-length class','budgets':BUDGETS,'seeds':SEEDS,
         'architecture':f'{DIM}->{HIDDEN}->3','optimizer':'AdamW(lr=0.003,weight_decay=0.0001)','steps':STEPS,'validated_v2_features':611,
         'v0_sha256':sha(a.v0),'v2_sha256':sha(a.v2),'schema_sha256':sha(a.schema),'validation_sha256':sha(a.validation),
         'same_pool':True,'test_records':len(test),'validation_records':len(val),'query_pool_records':len(pool),'near_duplicate_groups':groups,'device':device}
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (a.output/'confidence_intervals.json').write_text(json.dumps({'aggregate':agg,'v2_minus_v0':gains},indent=2)+'\n')
    lines=['# Surrogate fidelity evaluation v2','',f"**Under evaluated views and budgets, V2 {conclusion}.**",'',
           '**SIMULATED_PROTOCOL_VIEW. This is not model extraction and does not establish black-box security.**','',
           '| Budget | Queries | V0 | V2 | V0+V2 | V2−V0 |','|---:|---:|---:|---:|---:|---:|']
    for budget in BUDGETS:
        q=lambda v:next(x for x in agg if x['budget']==budget and x['view']==v)
        g=next(x for x in gains if x['budget']==budget)
        lines.append(f"| {budget:.0%} | {q('V0')['queries']} | {q('V0')['agreement_mean']:.4f} | {q('V2')['agreement_mean']:.4f} | {q('V0_plus_V2')['agreement_mean']:.4f} | {g['v2_minus_v0_mean']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] |")
    lines += ['','Target agreement, task accuracy, and output agreement all denote exact agreement on the preregistered coarse output-length class; exact generated-text reconstruction and KL are not applicable. All views use the same model, 500-sample pool, split, random-init architecture, optimizer, and steps. Preprocessing is fit on attack-training records only. Shuffled outputs, Gaussian features, random mapping, and the trusted plaintext label positive control are included in `metrics.csv`.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n'); print(json.dumps({'conclusion':conclusion,'device':device,'gains':gains}))
if __name__=='__main__': main()
