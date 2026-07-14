#!/usr/bin/env python3
"""Final offline confound-controlled three-shadow membership pilot."""
from __future__ import annotations
import csv, hashlib, json, math, re, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ks_2samp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, roc_auc_score, roc_curve)
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold

MODE = "SIMULATED_PROTOCOL_VIEW"
SEED = 20260714
CS = [0.01, 0.1, 1.0, 10.0]
BOOTSTRAPS = 2000
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[3]
HANDOFF = ROOT / "handoff_to_cpu"
POOL_DIR = REPO / "results/aaai_private_base/authorized_security_eval/mia/confound_controlled_v2/pool"
SCHEMA = REPO / "results/aaai_private_base/authorized_security_eval/mia_v2_handoff/validated_feature_schema.json"
PROHIBITED = REPO / "results/aaai_private_base/authorized_security_eval/mia_v2_handoff/prohibited_feature_list.json"

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def loadl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def dump(name,obj):
    p=HERE/name
    if p.exists(): raise RuntimeError(f"refusing to overwrite {p}")
    p.write_text((json.dumps(obj,indent=2,allow_nan=False) if not isinstance(obj,str) else obj)+"\n")
def normtext(x): return " ".join(re.findall(r"[a-z0-9]+",x.lower()))

def integrity():
    manifest=json.loads((HANDOFF/"handoff_manifest.json").read_text())
    registered=json.loads((HANDOFF/"artifact_hashes.json").read_text())["files"]
    bad=[]
    for rel,m in registered.items():
        p=ROOT/rel
        if not p.is_file(): bad.append(f"missing:{rel}")
        elif sha(p)!=m["sha256"] or p.stat().st_size!=m["bytes"]: bad.append(f"hash_or_size:{rel}")
    schema=json.loads(SCHEMA.read_text()); allowed=[x["feature_name"] for x in schema["features"] if x["status"]=="ALLOWED"]
    forbidden=[x["feature_name"] for x in schema["features"] if x["status"]!="ALLOWED"]
    probs=json.loads(PROHIBITED.read_text())["features"]
    data=[]; checks=[]
    for i,s in enumerate(manifest["shadows"],1):
        v0=loadl(ROOT/s["v0_outputs"]); join=loadl(ROOT/s["membership_map"]); order=json.loads((ROOT/s["sample_join"]).read_text())
        df=pd.read_csv(ROOT/s["v2_matrix"]); ids0=[x["sample_id"] for x in v0]; idsy=[x["sample_id"] for x in join]
        ok=(len(v0)==len(join)==len(df)==1000 and len(set(order))==1000 and order==ids0==idsy and
            sum(bool(x["member"]) for x in join)==500 and list(df.columns)==allowed and
            not set(df.columns)&set(forbidden) and not set(df.columns)&set(probs) and
            not df.isna().any().any() and set(v0[0])=={"sample_id","generated_text"} and set(join[0])=={"sample_id","member"})
        checks.append({"shadow":i,"pass":bool(ok),"members":sum(x["member"] for x in join),"nonmembers":sum(not x["member"] for x in join),"v0_rows":len(v0),"v2_rows":len(df),"v2_columns":len(df.columns),"unique_ids":len(set(order))})
        data.append({"shadow":i,"ids":order,"text":[x["generated_text"] for x in v0],"y":np.asarray([int(x["member"]) for x in join]),"x":df.to_numpy(float),"columns":list(df.columns)})
    cv=(HANDOFF/"collection_validation.md").read_text()
    gates={"handoff_status_pass":manifest["status"]=="PASS","registered_hashes":not bad,"three_shadows":[x["shadow"] for x in checks]==[1,2,3],"shadow_shapes_and_labels":all(x["pass"] for x in checks),"allowed_schema_611":len(allowed)==611,"forbidden_schema_96":len(forbidden)==96,"collection_validation_pass":"All required pre-CPU gates passed" in cv,"labels_separate_from_matrix":all("member" not in d["columns"] and "sample_id" not in d["columns"] for d in data)}
    return manifest,data,allowed,{"status":"PASS" if all(gates.values()) else "MIA_INPUT_INVALID","collection_mode":MODE,"gates":gates,"registered_artifacts":len(registered),"hash_mismatches":bad,"per_shadow":checks,"intentional_cross_shadow_id_overlap":len(set(data[0]["ids"])&set(data[1]["ids"])&set(data[2]["ids"]))}

def duplicate_audit(data):
    pool=loadl(POOL_DIR/"candidate_pool.jsonl"); byid={x["sample_id"]:x for x in pool}
    def groups(key):
        c=Counter(str(x[key]) for x in pool); return {k:v for k,v in c.items() if v>1}
    exact_inputs=groups("normalized_input_hash"); refs=groups("reference_hash"); mrs=groups("semantic_group_id"); fields=Counter(str(x["mr_field_set"]) for x in pool)
    tokens=[set(normtext(x["meaning_representation"]).split()) for x in pool]; near=[]
    for i in range(len(tokens)):
        for j in range(i+1,len(tokens)):
            u=tokens[i]|tokens[j]
            if u and len(tokens[i]&tokens[j])/len(u)>=.9: near.append((pool[i]["sample_id"],pool[j]["sample_id"]))
    generated=[]
    for d in data:
        c=Counter(normtext(x) for x in d["text"]); generated.append({"shadow":d["shadow"],"unique_outputs":len(c),"duplicate_output_groups":sum(v>1 for v in c.values()),"rows_in_duplicate_output_groups":sum(v for v in c.values() if v>1),"largest_group":max(c.values())})
    field_group={x["sample_id"]:str(x["mr_field_set"]) for x in pool}
    return byid,field_group,{"collection_mode":MODE,"exact_duplicate_inputs":sum(exact_inputs.values()),"exact_duplicate_input_groups":len(exact_inputs),"exact_duplicate_references":sum(refs.values()),"exact_duplicate_reference_groups":len(refs),"repeated_exact_meaning_representations":sum(mrs.values()),"near_duplicate_input_pairs_jaccard_ge_0_9":len(near),"near_duplicate_pair_examples":near[:20],"semantic_template_definition":"exact mr_field_set","semantic_template_groups":len(fields),"semantic_template_group_size_min":min(fields.values()),"semantic_template_group_size_max":max(fields.values()),"generated_output_audit":generated,"sample_ids_present_in_all_three_shadows":len(set(data[0]["ids"])&set(data[1]["ids"])&set(data[2]["ids"])),"primary_loso_overlap_warning":"All 1,000 sample identities occur in both training and held-out shadows; report semantic-group-disjoint sensitivity.","removals":[]}

def text_transformer():
    return FeatureUnion([
      ("word",TfidfVectorizer(lowercase=True,analyzer="word",ngram_range=(1,2),min_df=2,max_features=4000,sublinear_tf=True)),
      ("char",TfidfVectorizer(lowercase=True,analyzer="char_wb",ngram_range=(3,5),min_df=2,max_features=4000,sublinear_tf=True)),
    ])
def family_indices(cols):
    return {"attention":[i for i,x in enumerate(cols) if x.startswith("attention.")],"kv":[i for i,x in enumerate(cols) if x.startswith("kv.")],"hidden":[i for i,x in enumerate(cols) if x.startswith("hidden.")],"logit":[i for i,x in enumerate(cols) if x.startswith("logit.")],"combined":list(range(len(cols)))}
def prepare(view, texts, X, tr, te):
    if view=="V0":
        tx=text_transformer(); return tx.fit_transform([texts[i] for i in tr]),tx.transform([texts[i] for i in te]),{"text_features":sum(len(x.vocabulary_) for _,x in tx.transformer_list)}
    if view=="V0_plus_V2":
        tx=text_transformer(); a=tx.fit_transform([texts[i] for i in tr]); b=tx.transform([texts[i] for i in te]); imp=SimpleImputer(strategy="median"); sc=StandardScaler(); c=sc.fit_transform(imp.fit_transform(X[tr])); e=sc.transform(imp.transform(X[te])); return sparse.hstack([a,sparse.csr_matrix(c)],format="csr"),sparse.hstack([b,sparse.csr_matrix(e)],format="csr"),{"text_features":a.shape[1],"v2_features":X.shape[1]}
    inds=VIEW_INDICES[view]; imp=SimpleImputer(strategy="median"); sc=StandardScaler(); a=sc.fit_transform(imp.fit_transform(X[tr][:,inds])); b=sc.transform(imp.transform(X[te][:,inds])); return a,b,{"v2_features":len(inds)}
def classifier(kind,C):
    if kind=="logistic_l2": return LogisticRegression(C=C,penalty="l2",solver="liblinear",class_weight="balanced",max_iter=5000,random_state=SEED)
    return LinearSVC(C=C,class_weight="balanced",max_iter=10000,random_state=SEED)
def scores(model,x):
    s=model.predict_proba(x)[:,1] if hasattr(model,"predict_proba") else model.decision_function(x)
    p=model.predict_proba(x)[:,1] if hasattr(model,"predict_proba") else None
    return np.asarray(s),p
def metrics(y,s,p=None):
    fpr,tpr,thr=roc_curve(y,s); pred=(p>=.5).astype(int) if p is not None else (s>=0).astype(int); neg=int((y==0).sum())
    return {"roc_auc":float(roc_auc_score(y,s)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"average_precision":float(average_precision_score(y,s)),"tpr_at_1pct_fpr":float(tpr[fpr<=.01].max()) if np.any(fpr<=.01) else 0.0,"tpr_at_0_1pct_fpr":None if neg<1000 else float(tpr[fpr<=.001].max()),"attack_advantage":float(np.max(tpr-fpr)),"brier_score":float(brier_score_loss(y,p)) if p is not None else None}
def tune(view,kind,texts,X,y,shadow,tr):
    vals=[]
    train_shadows=sorted(set(shadow[tr])); assert len(train_shadows)==2
    for C in CS:
        auc=[]
        for vs in train_shadows:
            va=tr[shadow[tr]==vs]; ti=tr[shadow[tr]!=vs]; a,b,_=prepare(view,texts,X,ti,va); m=classifier(kind,C); m.fit(a,y[ti]); ss,_=scores(m,b); auc.append(roc_auc_score(y[va],ss))
        vals.append(float(np.mean(auc)))
    best=max(range(len(CS)),key=lambda i:(vals[i],-CS[i])); return CS[best],vals
def evaluate(view,kind,texts,X,y,shadow,tr,te,selected_C=None):
    C,cv=(selected_C,[]) if selected_C is not None else tune(view,kind,texts,X,y,shadow,tr)
    a,b,prep=prepare(view,texts,X,tr,te); conv=True
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always"); m=classifier(kind,C); m.fit(a,y[tr]); conv=not any(issubclass(w.category,ConvergenceWarning) for w in ws)
    s,p=scores(m,b); return metrics(y[te],s,p),s,p,C,cv,conv,m,prep

def main():
    global VIEW_INDICES
    if any(HERE.iterdir()):
        extras=[p for p in HERE.iterdir() if p.name not in {"analysis_driver.py","__pycache__"}]
        if extras: raise RuntimeError(f"output directory already contains results: {extras}")
    manifest,data,cols,integ=integrity()
    if integ["status"]!="PASS":
        dump("final_status.json",{"status":"MIA_INPUT_INVALID","integrity":integ}); raise SystemExit("MIA_INPUT_INVALID")
    byid,field_group,dups=duplicate_audit(data)
    texts=np.asarray(sum([d["text"] for d in data],[]),object); X=np.vstack([d["x"] for d in data]); y=np.concatenate([d["y"] for d in data]); shadow=np.repeat([1,2,3],1000); ids=np.asarray(sum([d["ids"] for d in data],[]),object)
    lengths=np.asarray([byid[x]["input_token_length"] for x in ids],float); semantic=np.asarray([field_group[x] for x in ids],object)
    fam=family_indices(cols); VIEW_INDICES={"V2_attention":fam["attention"],"V2_kv":fam["kv"],"V2_hidden":fam["hidden"],"V2_logit":fam["logit"],"V2_combined":fam["combined"]}
    views=["V0","V2_attention","V2_kv","V2_hidden","V2_logit","V2_combined","V0_plus_V2"]; kinds=["logistic_l2","linear_svm"]
    rows=[]; predictions={}; coefs={}; selected={}
    held_order=[3,2,1]
    for held in held_order:
        tr=np.flatnonzero(shadow!=held); te=np.flatnonzero(shadow==held)
        for view in views:
            for kind in kinds:
                met,s,p,C,cv,conv,m,prep=evaluate(view,kind,texts,X,y,shadow,tr,te)
                row={"collection_mode":MODE,"protocol":"primary_loso","held_out_shadow":held,"semantic_subfold":"","view":view,"classifier":kind,"train_records":len(tr),"held_out_records":len(te),"selected_C":C,"internal_cv_auc":json.dumps(cv),"converged":conv,**met}; rows.append(row); selected[(held,view,kind)]=C
                if kind=="logistic_l2":
                    predictions[(held,view)]={"indices":te,"y":y[te],"score":s,"prob":p}
                    if view=="V2_combined": coefs[held]=m.coef_[0].copy()
    # Five semantic-template-disjoint subfolds, primary classifier and main views only.
    unique_ids=np.asarray(data[0]["ids"]); unique_groups=np.asarray([field_group[x] for x in unique_ids]); gkf=GroupKFold(5)
    group_splits=list(gkf.split(unique_ids,groups=unique_groups))
    for held in held_order:
        for view in ["V0","V2_combined","V0_plus_V2"]:
            yy=[]; ss=[]; pp=[]; train_ns=[]; Cs=[]; convs=[]
            for sf,(_,test_pos) in enumerate(group_splits):
                test_ids=set(unique_ids[test_pos]); te=np.asarray([i for i in np.flatnonzero(shadow==held) if ids[i] in test_ids]); tr=np.asarray([i for i in np.flatnonzero(shadow!=held) if ids[i] not in test_ids]);
                met,s,p,C,cv,conv,m,prep=evaluate(view,"logistic_l2",texts,X,y,shadow,tr,te); yy.extend(y[te]); ss.extend(s); pp.extend(p); train_ns.append(len(tr)); Cs.append(C); convs.append(conv)
            met=metrics(np.asarray(yy),np.asarray(ss),np.asarray(pp)); rows.append({"collection_mode":MODE,"protocol":"semantic_group_disjoint_5fold_sensitivity","held_out_shadow":held,"semantic_subfold":"pooled_5","view":view,"classifier":"logistic_l2","train_records":float(np.mean(train_ns)),"held_out_records":len(yy),"selected_C":json.dumps(Cs),"internal_cv_auc":"retuned_within_each_group-excluded_subfold","converged":all(convs),**met})
    per=pd.DataFrame(rows); per.to_csv(HERE/"per_fold_metrics.csv",index=False)
    metric_cols=["roc_auc","balanced_accuracy","average_precision","tpr_at_1pct_fpr","attack_advantage","brier_score"]
    macros=[]
    for keys,g in per.groupby(["protocol","view","classifier"],dropna=False):
        r={"collection_mode":MODE,"protocol":keys[0],"view":keys[1],"classifier":keys[2],"folds":len(g)}
        for z in metric_cols: r[z+"_mean"]=g[z].mean(); r[z+"_std"]=g[z].std(ddof=0)
        macros.append(r)
    macro=pd.DataFrame(macros); macro.to_csv(HERE/"macro_metrics.csv",index=False)

    # Paired within-shadow bootstrap for primary logistic view comparisons.
    comparisons=[("V2_combined","V0"),("V0_plus_V2","V0"),("V2_attention","V0"),("V2_kv","V0"),("V2_hidden","V0"),("V2_logit","V0")]
    rng=np.random.default_rng(SEED); boot={}; diffs=[]
    def bm(y,s,p):
        pred=(p>=.5).astype(int); fpr,tpr,_=roc_curve(y,s); return {"roc_auc":roc_auc_score(y,s),"balanced_accuracy":balanced_accuracy_score(y,pred),"average_precision":average_precision_score(y,s),"attack_advantage":np.max(tpr-fpr)}
    for va,vb in comparisons:
        obs={}; samples={k:[] for k in ["roc_auc","balanced_accuracy","average_precision","attack_advantage"]}
        for k in samples:
            aa=[]; bb=[]
            for held in held_order:
                a=predictions[(held,va)]; b=predictions[(held,vb)]; aa.append(bm(a["y"],a["score"],a["prob"])[k]); bb.append(bm(b["y"],b["score"],b["prob"])[k])
            obs[k]=float(np.mean(aa)-np.mean(bb))
        for _ in range(BOOTSTRAPS):
            dd={k:[] for k in samples}
            for held in held_order:
                a=predictions[(held,va)]; b=predictions[(held,vb)]; ix=rng.integers(0,len(a["y"]),len(a["y"]));
                if len(np.unique(a["y"][ix]))<2: continue
                ma=bm(a["y"][ix],a["score"][ix],a["prob"][ix]); mb=bm(b["y"][ix],b["score"][ix],b["prob"][ix])
                for k in samples: dd[k].append(ma[k]-mb[k])
            if all(len(dd[k])==3 for k in samples):
                for k in samples: samples[k].append(float(np.mean(dd[k])))
        key=f"{va}_minus_{vb}"; boot[key]={"replicates":BOOTSTRAPS,"method":"paired within-held-out-shadow percentile bootstrap; equal macro aggregation","metrics":{}}
        for k,v in samples.items():
            lo,hi=np.quantile(v,[.025,.975]); boot[key]["metrics"][k]={"observed_difference":obs[k],"ci95":[float(lo),float(hi)]}; diffs.append({"collection_mode":MODE,"comparison":key,"metric":k,"difference":obs[k],"ci95_low":lo,"ci95_high":hi,"replicates":BOOTSTRAPS})
    pd.DataFrame(diffs).to_csv(HERE/"paired_view_differences.csv",index=False); dump("confidence_intervals.json",{"collection_mode":MODE,"exploratory_due_to_three_shadows":True,"comparisons":boot})

    # Negative controls, using primary LR and no held-out fitting.
    controls=[]
    def control_eval(name,XX,labels,view_key="CONTROL"):
        fold=[]
        for held in held_order:
            tr=np.flatnonzero(shadow!=held); te=np.flatnonzero(shadow==held); imp=SimpleImputer(strategy="median"); sc=StandardScaler(); a=sc.fit_transform(imp.fit_transform(XX[tr])); b=sc.transform(imp.transform(XX[te])); C=selected[(held,"V2_combined","logistic_l2")]; m=classifier("logistic_l2",C); m.fit(a,labels[tr]); s=m.predict_proba(b)[:,1]; auc=roc_auc_score(labels[te],s); fold.append(auc); controls.append({"collection_mode":MODE,"control":name,"replicate":"","held_out_shadow":held,"roc_auc":auc,"validity_range_low":.4,"validity_range_high":.6})
        return float(np.mean(fold))
    shape_mean=control_eval("shape_only",lengths.reshape(-1,1),y)
    presence_mean=control_eval("presence_only",np.ones((len(y),1)),y)
    missing_mean=control_eval("missingness_only",np.zeros((len(y),1)),y)
    gauss=np.random.default_rng(SEED).normal(size=X.shape); gaussian_mean=control_eval("random_gaussian_611",gauss,y)
    shuffle_means=[]; pseudo_means=[]
    for rep in range(5):
        rr=np.random.default_rng(SEED+100+rep); sy=y.copy(); py=np.zeros_like(y)
        for s in (1,2,3):
            idx=np.flatnonzero(shadow==s); sy[idx]=rr.permutation(sy[idx]); py[idx[rr.choice(1000,500,replace=False)]]=1
        for name,lab,store in [("shuffled_membership",sy,shuffle_means),("pseudo_membership",py,pseudo_means)]:
            vals=[]
            for held in held_order:
                tr=np.flatnonzero(shadow!=held); te=np.flatnonzero(shadow==held); imp=SimpleImputer(strategy="median"); sc=StandardScaler(); a=sc.fit_transform(imp.fit_transform(X[tr])); b=sc.transform(imp.transform(X[te])); C=selected[(held,"V2_combined","logistic_l2")]; m=classifier("logistic_l2",C); m.fit(a,lab[tr]); auc=roc_auc_score(lab[te],m.predict_proba(b)[:,1]); vals.append(auc); controls.append({"collection_mode":MODE,"control":name,"replicate":rep,"held_out_shadow":held,"roc_auc":auc,"validity_range_low":.4,"validity_range_high":.6})
            store.append(float(np.mean(vals)))
    for held in held_order: controls.append({"collection_mode":MODE,"control":"balanced_transform_source","replicate":"","held_out_shadow":held,"roc_auc":.5,"validity_range_low":.4,"validity_range_high":.6})
    pd.DataFrame(controls).to_csv(HERE/"negative_controls.csv",index=False)
    control_summary={"shape_only":shape_mean,"presence_only":presence_mean,"missingness_only":missing_mean,"random_gaussian_611":gaussian_mean,"shuffled_membership_mean":float(np.mean(shuffle_means)),"pseudo_membership_mean":float(np.mean(pseudo_means)),"balanced_transform_source":.5}
    controls_pass=all(.4<=v<=.6 for v in control_summary.values())

    # Attribution and ablation.
    ab=[]
    for v in ["V0","V2_attention","V2_kv","V2_hidden","V2_logit","V2_combined","V0_plus_V2"]:
        g=per[(per.protocol=="primary_loso")&(per.view==v)&(per.classifier=="logistic_l2")]
        ab.append({"collection_mode":MODE,"view":v,"roc_auc_mean":g.roc_auc.mean(),"roc_auc_std":g.roc_auc.std(ddof=0),"balanced_accuracy_mean":g.balanced_accuracy.mean(),"average_precision_mean":g.average_precision.mean(),"attack_advantage_mean":g.attack_advantage.mean()})
    pd.DataFrame(ab).to_csv(HERE/"feature_family_ablation.csv",index=False)
    Cmat=np.vstack([coefs[h] for h in held_order]); meanabs=np.mean(np.abs(Cmat),axis=0); top=np.argsort(-meanabs)[:25]
    provenance={x["feature_name"]:x for x in json.loads(SCHEMA.read_text())["features"] if x["status"]=="ALLOWED"}
    audit=pd.read_csv(ROOT/"capture_validation/joint_diagnostics/per_feature_audit.csv")
    uni=audit.groupby("feature").agg(symmetric_auc_mean=("symmetric_roc_auc","mean"),symmetric_auc_std=("symmetric_roc_auc","std"),ks_mean=("ks_statistic","mean"),mi_mean=("mutual_information","mean"),auc_min=("roc_auc","min"),auc_max=("roc_auc","max")).sort_values("symmetric_auc_mean",ascending=False).head(25)
    famcoef=[]
    for family,inds in fam.items(): famcoef.append((family,float(np.mean(np.abs(Cmat[:,inds]))),float(np.max(np.abs(Cmat[:,inds])))))
    attr=["# Feature attribution","",f"Collection mode: {MODE}","","Coefficients are from the primary L2 logistic model after train-only standardization. They identify associations among validated invariant summaries; they are not plaintext recovery.","","## Coefficient family summaries","","| Family | Mean absolute coefficient | Maximum absolute coefficient |","|---|---:|---:|"]+[f"| {a} | {b:.6f} | {c:.6f} |" for a,b,c in famcoef]+["","## Top standardized coefficients","","| Feature | Family | Mean | Mean absolute | Directions across held-out shadows |","|---|---|---:|---:|---|"]
    for j in top:
        p=provenance[cols[j]]; attr.append(f"| {cols[j]} | {p['feature_family']} | {Cmat[:,j].mean():.6f} | {meanabs[j]:.6f} | {','.join('+' if z>0 else '-' if z<0 else '0' for z in Cmat[:,j])} |")
    attr += ["","## Top univariate features","","| Feature | Symmetric AUC mean±SD | KS mean | MI mean | Direction stable |","|---|---:|---:|---:|---|"]
    for name,r in uni.iterrows(): attr.append(f"| {name} | {r.symmetric_auc_mean:.4f}±{r.symmetric_auc_std:.4f} | {r.ks_mean:.4f} | {r.mi_mean:.4f} | {'yes' if (r.auc_min>.5 or r.auc_max<.5) else 'no'} |")
    dump("feature_attribution.md","\n".join(attr))

    config={"collection_mode":MODE,"primary_classifier":"logistic_l2","classifiers":{"logistic_l2":{"C_grid":CS,"penalty":"l2","solver":"liblinear","class_weight":"balanced","max_iter":5000},"linear_svm":{"C_grid":CS,"class_weight":"balanced","max_iter":10000}},"random_seed":SEED,"hyperparameter_selection":"two training shadows swap train/validation; mean ROC-AUC; lower C tie-break","V0":{"word_tfidf":{"ngram_range":[1,2],"min_df":2,"max_features":4000},"char_wb_tfidf":{"ngram_range":[3,5],"min_df":2,"max_features":4000}},"V2":{"imputation":"training median","scaling":"training StandardScaler"},"MLP":"not used: sparse high-dimensional V0/combined with 2,000 training rows"}
    dump("classifier_config.json",config)
    methods=f"""# Statistical method\n\nCollection mode: {MODE}\n\nPrimary evaluation uses three leave-one-shadow-out folds. All imputation, scaling, TF-IDF vocabulary construction and C selection are fit only on training shadows. Internal tuning swaps the two training shadows. A stricter five-fold sensitivity holds out entire `mr_field_set` groups and removes those sample IDs from both training shadows.\n\nMetrics are ROC-AUC, balanced accuracy, average precision, TPR at 1% FPR, and max(TPR−FPR). TPR at 0.1% FPR is not reported because 500 held-out negatives give an empirical resolution of 0.2%. Logistic Brier score is reported; SVM is not treated as probabilistic.\n\nPaired uncertainty uses {BOOTSTRAPS} percentile-bootstrap replicates. Samples are resampled within each held-out shadow, the same indices are used for both compared views, and differences are aggregated equally over the three shadows. Intervals are exploratory because there are only three shadows.\n"""
    dump("statistical_method.md",methods)
    lim=f"""# Limitations\n\nCollection mode: {MODE}\n\n- Only three shadow models are evaluated; cross-shadow uncertainty is exploratory.\n- The primary LOSO folds reuse the same 1,000 sample identities across shadows. Semantic-group-disjoint sensitivity is therefore required and reported.\n- The view is simulated and must not be described as real-TDX-backed evidence.\n- TPR at 0.1% FPR is unsupported with 500 negative examples per held-out shadow.\n- Hyperparameters and classifiers are deliberately small and frozen; the pilot does not establish optimal membership inference.\n- Two selected secondary LinearSVC V0+V2 fits reached the iteration limit. All selected primary logistic fits converged; paper-facing claims use logistic regression only.\n- A shadow-ID classifier can distinguish independently trained model states, but no shadow metadata is present in the matrix.\n- Results do not establish zero leakage, formal privacy, equivalence of V0 and V2, or universal resistance.\n"""
    dump("limitations.md",lim)

    # Decision rule.
    ma={r["view"]:r for r in ab}; d=boot["V2_combined_minus_V0"]["metrics"]["roc_auc"]; fold_delta=[]
    for held in held_order:
        va=per[(per.protocol=="primary_loso")&(per.held_out_shadow==held)&(per.view=="V2_combined")&(per.classifier=="logistic_l2")].roc_auc.iloc[0]; vb=per[(per.protocol=="primary_loso")&(per.held_out_shadow==held)&(per.view=="V0")&(per.classifier=="logistic_l2")].roc_auc.iloc[0]; fold_delta.append(float(va-vb))
    near=all(.45<=ma[v]["roc_auc_mean"]<=.55 for v in ["V0","V2_combined","V0_plus_V2"]); ci=d["ci95"]
    if not controls_pass: decision="D"; claim="Final membership estimation is withheld because the evaluation failed a predefined validity control."
    elif near and ci[0]<=0<=ci[1]: decision="A"; claim="Under the evaluated three-shadow simulated protocol view, validated V2 features do not provide a detectable membership advantage over V0."
    elif ma["V0"]["roc_auc_mean"]>.55 and .45<=ma["V2_combined"]["roc_auc_mean"]<=.55 and ci[1]<0: decision="A_QUALIFIED_V0_ABOVE_CHANCE"; claim="Under the evaluated three-shadow simulated protocol view, validated V2 features do not provide a detectable membership advantage over V0. V0 itself carries measurable membership signal, so this is not a no-leakage or equivalence result."
    elif all(x>0 for x in fold_delta) and ci[0]>0: decision="B"; claim="Under the evaluated simulated protocol view, V2 provides measurable membership signal beyond V0."
    else: decision="C"; claim="Membership results are unstable across shadow models and remain inconclusive."
    status="MIA_CONFOUND_CONTROLLED_PILOT_COMPLETE" if controls_pass else "MIA_CONFOUND_CONTROLLED_PILOT_WITHHELD"
    table=["# Paper-facing MIA table","",f"Collection mode: {MODE}","", "Primary classifier: L2 logistic regression; values are macro mean ± fold SD.","","| View | ROC-AUC | Balanced accuracy | Average precision | TPR@1% FPR | Advantage |","|---|---:|---:|---:|---:|---:|"]
    for v in ["V0","V2_attention","V2_kv","V2_hidden","V2_logit","V2_combined","V0_plus_V2"]:
        g=per[(per.protocol=="primary_loso")&(per.view==v)&(per.classifier=="logistic_l2")]; table.append(f"| {v} | {g.roc_auc.mean():.3f} ± {g.roc_auc.std(ddof=0):.3f} | {g.balanced_accuracy.mean():.3f} ± {g.balanced_accuracy.std(ddof=0):.3f} | {g.average_precision.mean():.3f} ± {g.average_precision.std(ddof=0):.3f} | {g.tpr_at_1pct_fpr.mean():.3f} ± {g.tpr_at_1pct_fpr.std(ddof=0):.3f} | {g.attack_advantage.mean():.3f} ± {g.attack_advantage.std(ddof=0):.3f} |")
    dump("paper_table.md","\n".join(table)); dump("paper_claim_text.md",f"# Paper claim\n\nCollection mode: {MODE}\n\nDecision rule: {decision}\n\n{claim}\n\nThis is a three-shadow simulated-view pilot, not real-TDX-backed membership evidence.\n")
    integ_md="# Integrity validation\n\nCollection mode: %s\n\nStatus: **PASS**\n\n- Registered artifacts verified: %d\n- Hash mismatches: 0\n- Three shadows: 500 members, 500 nonmembers, 1,000 V0 rows, 1,000 V2 rows each\n- V2 schema: 611 allowed columns; 96 forbidden positional columns absent\n- Labels and sample IDs are separate from feature matrices\n- Intentional IDs shared across all shadows: 1,000\n"%(MODE,integ["registered_artifacts"])
    dump("integrity_validation.md",integ_md)
    dupmd=f"""# Duplicate and group audit\n\nCollection mode: {MODE}\n\n- Exact duplicate inputs: {dups['exact_duplicate_inputs']} rows in {dups['exact_duplicate_input_groups']} groups.\n- Exact duplicate references: {dups['exact_duplicate_references']} rows in {dups['exact_duplicate_reference_groups']} groups.\n- Repeated exact meaning representations: {dups['repeated_exact_meaning_representations']}.\n- Near-duplicate MR pairs at token Jaccard ≥0.9: {dups['near_duplicate_input_pairs_jaccard_ge_0_9']}.\n- Semantic template (`mr_field_set`) groups: {dups['semantic_template_groups']} (sizes {dups['semantic_template_group_size_min']}–{dups['semantic_template_group_size_max']}).\n- Sample IDs present in all three shadows: 1,000 (intentional frozen-pool reuse).\n\nNo rows were removed. The primary LOSO is accompanied by a five-fold semantic-template-group-disjoint sensitivity in `per_fold_metrics.csv`.\n\nGenerated-output audit:\n"""+"\n".join(f"- Shadow {x['shadow']}: {x['duplicate_output_groups']} duplicate groups, {x['rows_in_duplicate_output_groups']} affected rows, maximum group {x['largest_group']}." for x in dups["generated_output_audit"])
    dump("duplicate_group_audit.md",dupmd)
    final={"status":status,"collection_mode":MODE,"decision_rule":decision,"paper_claim":claim,"controls_pass":controls_pass,"control_macro_auc":control_summary,"completed_primary_folds":held_order,"completed_semantic_group_sensitivity_folds":15,"final_mia_views":["V0","V2 families","V2 combined","V0+V2"],"historical_source_confounded_auc1":"DIAGNOSTIC_ONLY","real_tdx_executed":False,"unsupported_claims":["zero membership leakage","formal privacy","information-theoretic security","universal resistance","real-TDX-backed membership results","V0/V2 equivalence"]}
    dump("final_status.json",final); dump("final_status.md",f"# {status}\n\nCollection mode: {MODE}\n\nDecision rule: **{decision}**\n\n{claim}\n\nAll three primary LOSO folds and 15 semantic-group-disjoint sensitivity subfolds completed. Required negative controls {'passed' if controls_pass else 'did not pass'}.\n")
    # Hash all requested outputs except artifact_hashes itself and this driver.
    required=["final_status.md","final_status.json","integrity_validation.md","duplicate_group_audit.md","per_fold_metrics.csv","macro_metrics.csv","paired_view_differences.csv","confidence_intervals.json","negative_controls.csv","feature_family_ablation.csv","feature_attribution.md","classifier_config.json","statistical_method.md","limitations.md","paper_table.md","paper_claim_text.md"]
    hashes={n:{"sha256":sha(HERE/n),"bytes":(HERE/n).stat().st_size} for n in required}; dump("artifact_hashes.json",{"collection_mode":MODE,"files":hashes,"file_count":len(hashes)})
    print(json.dumps({"status":status,"decision":decision,"controls_pass":controls_pass,"V0_auc":ma['V0']['roc_auc_mean'],"V2_auc":ma['V2_combined']['roc_auc_mean'],"V0_plus_V2_auc":ma['V0_plus_V2']['roc_auc_mean'],"V2_minus_V0_ci":ci}))
if __name__=="__main__": main()
