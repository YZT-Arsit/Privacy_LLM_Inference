#!/usr/bin/env python3
"""Freeze the largest exact-length test cohort for paired repeat capture."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--queries',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if a.out.exists(): raise RuntimeError(f'refusing to overwrite {a.out}')
    q=json.loads(a.queries.read_text()); counts=collections.Counter(len(x['prompt_ids']) for x in q)
    length=max(counts, key=lambda k:(counts[k],-k)); selected=[x for x in q if len(x['prompt_ids'])==length]
    a.out.write_text(json.dumps(selected,indent=2)+'\n')
    manifest={'schema':'fixed_length_repeat_capture_plan_v1','source_queries_sha256':sha(a.queries),
              'fixed_queries_sha256':sha(a.out),'fixed_length':length,'records':len(selected),
              'ordered_sample_ids':[f"e2e_nlg:test:{int(x['sample_id']):06d}" for x in selected],
              'batch_size':1,'padding':'none; every effective sequence length is exactly fixed_length',
              'dtype':'fp32','capture_repeats':2,'same_capture_code_and_run_id':True,'seed':'deterministic_forward'}
    mp=a.out.with_suffix('.manifest.json')
    if mp.exists(): raise RuntimeError(f'refusing to overwrite {mp}')
    mp.write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps({'length':length,'records':len(selected)}))
if __name__=='__main__': main()
