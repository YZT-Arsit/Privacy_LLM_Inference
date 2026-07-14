#!/usr/bin/env python3
"""Collect paired V0/V2 from one identical greedy forward path per frozen query."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, secrets, time
from pathlib import Path
import numpy as np
import torch

def sha256(p: Path) -> str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def loadl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write_new(p:Path,s:str):
 if p.exists(): raise RuntimeError(f'refusing to overwrite {p}')
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s)
def reductions(v):
 x=np.asarray(v,dtype=np.float64); return dict(zip(('mean','std','min','max'),(x.mean(),x.std(),x.min(),x.max())))

class ProjectionCapture:
 def __init__(self,model):
  self.data={}; self.handles=[]
  for layer,block in enumerate(model.base_model.model.model.layers):
   for name in ('q_proj','k_proj','v_proj'):
    def hook(_m,_i,out,layer=layer,name=name): self.data[(layer,name)]=out.detach()
    self.handles.append(getattr(block.self_attn,name).register_forward_hook(hook))
 def clear(self): self.data.clear()
 def close(self):
  for h in self.handles: h.remove()

def extract(model,out,captured,names,transform_seed):
 from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
 gen=torch.Generator(device='cpu'); gen.manual_seed(int(transform_seed)%(2**63-1))
 def orthogonal_signed_permutation(x):
  d=x.shape[-1]; perm=torch.randperm(d,generator=gen).to(x.device); signs=(torch.randint(0,2,(d,),generator=gen,dtype=torch.int8)*2-1).to(x.device,x.dtype)
  return x.index_select(-1,perm)*signs
 def canonical_norm(x):
  z=x.float().abs().sort(dim=-1).values; return (z.square().sum(dim=-1)).sqrt()
 vals={}; hidden=out.hidden_states
 vals['hidden.final.last_l2']=float(canonical_norm(orthogonal_signed_permutation(hidden[-1][0,-1])).cpu())
 for layer in range(24): vals[f'hidden.layer_{layer:02d}.last_l2']=float(canonical_norm(orthogonal_signed_permutation(hidden[layer][0,-1])).cpu())
 position_ids=torch.arange(hidden[0].shape[1],device=hidden[0].device).unsqueeze(0)
 for layer,block in enumerate(model.base_model.model.model.layers):
  att=block.self_attn; q=captured[(layer,'q_proj')]; k=captured[(layer,'k_proj')]; v=captured[(layer,'v_proj')]
  b,t,_=q.shape
  q=q.view(b,t,att.num_heads,att.head_dim).transpose(1,2)
  k=k.view(b,t,att.num_key_value_heads,att.head_dim).transpose(1,2)
  v=v.view(b,t,att.num_key_value_heads,att.head_dim).transpose(1,2)
  cos,sin=att.rotary_emb(v,position_ids); q,k=apply_rotary_pos_emb(q,k,cos,sin)
  k=repeat_kv(k,att.num_key_value_groups); v=repeat_kv(v,att.num_key_value_groups)
  # Materialize a fresh orthogonal signed-permutation mask. Q and K share the
  # same mask (so scores are invariant); V receives an independent mask.
  d=att.head_dim; perm=torch.randperm(d,generator=gen).to(q.device); signs=(torch.randint(0,2,(d,),generator=gen,dtype=torch.int8)*2-1).to(q.device,q.dtype)
  q=q.index_select(-1,perm)*signs; k=k.index_select(-1,perm)*signs
  v=orthogonal_signed_permutation(v)
  for kind,tensor in [('k',k),('v',v)]:
   st=reductions(canonical_norm(tensor[0,:,-1,:]).cpu().numpy())
   for red,x in st.items(): vals[f'kv.{kind}.layer_{layer:02d}.last_l2_by_head.{red}']=float(x)
  products=q[0,:,-1,None,:].float()*k[0].float()
  last_scores=products.sort(dim=-1).values.sum(dim=-1)/math.sqrt(att.head_dim); probs=torch.softmax(last_scores,dim=-1)
  metrics={
   'last_entropy_by_head':-(probs*probs.clamp_min(1e-30).log()).sum(-1),
   'last_max_by_head':probs.max(-1).values,
   'true_score_last_mean_by_head':last_scores.mean(-1),
   'true_score_last_std_by_head':last_scores.std(-1),
  }
  for key,x in metrics.items():
   for red,y in reductions(x.cpu().numpy()).items(): vals[f'attention.layer_{layer:02d}.{key}.{red}']=float(y)
 # The vocabulary transform is permutation-only.
 logits=out.logits[0,-1].float().index_select(0,torch.randperm(out.logits.shape[-1],generator=gen).to(out.logits.device))
 canonical_logits=logits.sort().values; top=torch.topk(logits,k=32).values
 vals.update({'logit.mean':float(canonical_logits.mean().cpu()),'logit.std':float(canonical_logits.std().cpu()),
              'logit.l2':float(canonical_norm(canonical_logits).cpu()),'logit.min':float(canonical_logits[0].cpu()),
              'logit.max':float(canonical_logits[-1].cpu())})
 for red,x in reductions(top.cpu().numpy()).items(): vals[f'logit.top_values.{red}']=float(x)
 vals['logit.top1_top2_margin']=float((top[0]-top[1]).cpu())
 missing=[x for x in names if x not in vals]; extra=[x for x in vals if x not in names]
 if missing or extra: raise RuntimeError(f'schema mismatch missing={missing[:3]} extra={extra[:3]}')
 row=np.asarray([vals[x] for x in names],dtype=np.float64)
 if not np.isfinite(row).all(): raise RuntimeError('non-finite V2 feature')
 return row

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--queries',type=Path,required=True); ap.add_argument('--membership',type=Path,required=True); ap.add_argument('--schema',type=Path,required=True); ap.add_argument('--base-model',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--max-new-tokens',type=int,default=96); a=ap.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()): raise RuntimeError(f'nonempty output {a.output_dir}')
 a.output_dir.mkdir(parents=True,exist_ok=True)
 queries=loadl(a.queries); membership=loadl(a.membership)
 if len(queries)!=1000 or [x['sample_id'] for x in queries]!=[x['sample_id'] for x in membership]: raise RuntimeError('query/membership order mismatch')
 schema=json.loads(a.schema.read_text()); allowed=[x for x in schema['features'] if x['status']=='ALLOWED']; names=[x['feature_name'] for x in allowed]
 if len(names)!=611 or len(set(names))!=611: raise RuntimeError('allowed schema is not 611 unique columns')
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from peft import PeftModel
 base=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=torch.float32,local_files_only=True).cuda()
 model=PeftModel.from_pretrained(base,a.adapter,is_trainable=False).eval(); tok=AutoTokenizer.from_pretrained(a.base_model,local_files_only=True)
 cap=ProjectionCapture(model); eos=tok.eos_token_id; matrix=[]; v0=[]; commitments=[]; transform_diffs=[]; transform_a=[]; transform_b=[]; master=secrets.token_bytes(32); t0=time.time(); tokens=0
 with torch.inference_mode():
  for index,qry in enumerate(queries):
   cap.clear(); ids=torch.tensor([qry['prompt_ids']],device='cuda')
   out=model(input_ids=ids,use_cache=True,output_hidden_states=True,return_dict=True)
   digest=hashlib.sha256(master+qry['sample_id'].encode()).hexdigest(); commitments.append({'sample_id':qry['sample_id'],'transform_commitment':digest})
   feature_row=extract(model,out,cap.data,names,int(digest[:16],16)); matrix.append(feature_row)
   if index<32:
    repeat=extract(model,out,cap.data,names,int(digest[16:32],16))
    transform_diffs.append(float(np.max(np.abs(feature_row-repeat)))); transform_a.append(feature_row); transform_b.append(repeat)
   generated=[]; nxt=int(out.logits[0,-1].argmax()); past=out.past_key_values
   for _ in range(a.max_new_tokens):
    if nxt==eos: break
    generated.append(nxt); tokens+=1
    step=model(input_ids=torch.tensor([[nxt]],device='cuda'),past_key_values=past,use_cache=True,return_dict=True)
    past=step.past_key_values; nxt=int(step.logits[0,-1].argmax())
   v0.append({'sample_id':qry['sample_id'],'generated_text':tok.decode(generated,skip_special_tokens=True).strip()})
   if index%20==0 or index==999: print(json.dumps({'captured':index+1,'total':1000,'generated_tokens':tokens,'wall_sec':round(time.time()-t0,1)}),flush=True)
 cap.close(); matrix=np.stack(matrix)
 csvpath=a.output_dir/'v2_features.csv'
 if csvpath.exists(): raise RuntimeError('refusing overwrite')
 with csvpath.open('w',newline='') as f:
  w=csv.writer(f); w.writerow(names); w.writerows(matrix.tolist())
 write_new(a.output_dir/'sample_order.json',json.dumps([x['sample_id'] for x in queries],indent=2)+'\n')
 write_new(a.output_dir/'membership_join.jsonl',''.join(json.dumps(x,sort_keys=True)+'\n' for x in membership))
 evaluator=[{'sample_id':q['sample_id'],'member':bool(m['member']),'prompt_length':len(q['prompt_ids'])} for q,m in zip(queries,membership)]
 write_new(a.output_dir/'evaluator_metadata.jsonl',''.join(json.dumps(x,sort_keys=True)+'\n' for x in evaluator))
 write_new(a.output_dir/'v0_outputs.jsonl',''.join(json.dumps(x,sort_keys=True)+'\n' for x in v0))
 write_new(a.output_dir/'transform_commitments.jsonl',''.join(json.dumps(x,sort_keys=True)+'\n' for x in commitments))
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import roc_auc_score
 from sklearn.model_selection import GroupKFold,cross_val_predict
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 tx=np.vstack([transform_a,transform_b]); ty=np.asarray([0]*32+[1]*32); groups=np.asarray(list(range(32))*2)
 clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,random_state=0))
 scores=cross_val_predict(clf,tx,ty,groups=groups,cv=GroupKFold(4),method='predict_proba')[:,1]; source_auc=float(roc_auc_score(ty,scores))
 q1=source_auc/(2-source_auc); q2=2*source_auc*source_auc/(1+source_auc)
 se=math.sqrt(max(1e-30,(source_auc*(1-source_auc)+31*(q1-source_auc**2)+31*(q2-source_auc**2))/(32*32)))
 pvalue=float(math.erfc(abs(source_auc-0.5)/(se*math.sqrt(2))))
 transform_validation={'schema':'mia_v3_transform_invariance_validation','paired_samples':32,'feature_columns':611,'max_abs_difference':max(transform_diffs),'tolerance':1e-5,'all_pairs_within_tolerance':max(transform_diffs)<=1e-5,'source_classifier':'group-disjoint 4-fold standardized logistic regression','source_classifier_roc_auc':source_auc,'auc_null_two_sided_p_value':pvalue}
 write_new(a.output_dir/'transform_invariance_validation.json',json.dumps(transform_validation,indent=2)+'\n')
 if not transform_validation['all_pairs_within_tolerance']: raise RuntimeError('fresh-transform invariance validation failed')
 manifest={'schema':'mia_v3_paired_collection','status':'PASS','records':1000,'feature_columns':611,'dtype':'fp32','batch_size':1,'padding':'none_batch1','model_mode':'eval','labels_exposed_to_model':False,'loss':False,'gradients':False,'optimizer_state':False,'same_forward_path_for_v0_v2':True,'per_prompt_fresh_transform':True,'transform_realization':'fresh orthogonal signed-permutation transforms materialized on hidden/K/V and shared Q/K; fresh vocabulary permutation materialized on logits; only invariant scalars written','transform_validation_sha256':sha256(a.output_dir/'transform_invariance_validation.json'),'adapter_sha256':sha256(a.adapter/'adapter_model.safetensors'),'queries_sha256':sha256(a.queries),'membership_sha256':sha256(a.membership),'schema_sha256':sha256(a.schema),'v2_features_sha256':sha256(csvpath),'v0_outputs_sha256':sha256(a.output_dir/'v0_outputs.jsonl'),'sample_order_sha256':sha256(a.output_dir/'sample_order.json'),'all_finite':bool(np.isfinite(matrix).all()),'generated_tokens':tokens,'wall_sec':round(time.time()-t0,1),'gpu':torch.cuda.get_device_name(0)}
 write_new(a.output_dir/'collection_manifest.json',json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest),flush=True)
if __name__=='__main__': main()
