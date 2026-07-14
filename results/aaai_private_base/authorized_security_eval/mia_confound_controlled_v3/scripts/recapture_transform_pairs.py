#!/usr/bin/env python3
"""Recover the 32-pair fresh-transform validation after collection completion."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--collector',type=Path,required=True); ap.add_argument('--adapter',type=Path,required=True); ap.add_argument('--queries',type=Path,required=True); ap.add_argument('--schema',type=Path,required=True); ap.add_argument('--base-model',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()): raise RuntimeError(f'nonempty output {a.output_dir}')
 a.output_dir.mkdir(parents=True,exist_ok=True); spec=importlib.util.spec_from_file_location('collector',a.collector); c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
 from transformers import AutoModelForCausalLM
 from peft import PeftModel
 base=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=torch.float32,local_files_only=True).cuda(); model=PeftModel.from_pretrained(base,a.adapter,is_trainable=False).eval(); cap=c.ProjectionCapture(model)
 queries=c.loadl(a.queries)[:32]; names=[x['feature_name'] for x in json.loads(a.schema.read_text())['features'] if x['status']=='ALLOWED']; left=[]; right=[]
 with torch.inference_mode():
  for i,q in enumerate(queries):
   cap.clear(); out=model(input_ids=torch.tensor([q['prompt_ids']],device='cuda'),use_cache=False,output_hidden_states=True,return_dict=True)
   digest=hashlib.sha256(f'mia-v3-recapture:{q["sample_id"]}'.encode()).hexdigest(); left.append(c.extract(model,out,cap.data,names,int(digest[:16],16))); right.append(c.extract(model,out,cap.data,names,int(digest[16:32],16)))
   if i%8==0: print(json.dumps({'paired':i+1,'total':32}),flush=True)
 cap.close(); left=np.stack(left); right=np.stack(right)
 for name,x in [('transform_a.csv',left),('transform_b.csv',right)]:
  with (a.output_dir/name).open('w',newline='') as f: w=csv.writer(f); w.writerow(names); w.writerows(x.tolist())
 report={'schema':'mia_v3_transform_pair_recapture','paired_samples':32,'features':611,'max_abs_difference':float(np.max(np.abs(left-right))),'all_finite':bool(np.isfinite(left).all() and np.isfinite(right).all()),'collector_sha256':hashlib.sha256(a.collector.read_bytes()).hexdigest(),'adapter_sha256':hashlib.sha256((a.adapter/'adapter_model.safetensors').read_bytes()).hexdigest()}
 (a.output_dir/'recapture_manifest.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
if __name__=='__main__': main()
