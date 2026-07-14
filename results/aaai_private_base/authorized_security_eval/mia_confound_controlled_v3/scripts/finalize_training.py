#!/usr/bin/env python3
"""Verify a completed shadow adapter and emit its append-only terminal manifest."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import torch

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ids_sha(ids): return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
def loadl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--shadow-config',type=Path,required=True); ap.add_argument('--train-members',type=Path,required=True); ap.add_argument('--membership',type=Path,required=True); ap.add_argument('--training-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 cfg=json.loads(a.shadow_config.read_text()); train=loadl(a.train_members); membership=loadl(a.membership); tm=json.loads((a.training_dir/'training_manifest.json').read_text()); members=[x['sample_id'] for x in membership if x['member']]; nonmembers=[x['sample_id'] for x in membership if not x['member']]
 from safetensors.torch import load_file
 adapter=a.training_dir/'adapter/adapter_model.safetensors'; tensors=load_file(str(adapter),device='cpu')
 dims={'q_proj':(896,896),'k_proj':(896,128),'v_proj':(896,128),'o_proj':(896,896),'gate_proj':(896,4864),'up_proj':(896,4864),'down_proj':(4864,896)}
 shape_errors=[]
 for key,t in tensors.items():
  proj=next((x for x in dims if f'.{x}.' in key),None); kind='A' if 'lora_A' in key else 'B' if 'lora_B' in key else None
  if proj is None or kind is None: shape_errors.append({'key':key,'reason':'unexpected key'}); continue
  inp,out=dims[proj]; want=[8,inp] if kind=='A' else [out,8]
  if list(t.shape)!=want: shape_errors.append({'key':key,'actual':list(t.shape),'expected':want})
 gates={'training_manifest_pass':tm['status']=='PASS','members_500':len(members)==500,'nonmembers_500':len(nonmembers)==500,'no_overlap':not(set(members)&set(nonmembers)),'exact_training_members':[x['sample_id'] for x in train]==members,'membership_hash':sha(a.membership)==cfg['membership_sha256'],'adapter_hash':sha(adapter)==tm['adapter']['sha256'],'tensor_count_336':len(tensors)==336,'tensor_shapes':not shape_errors,'all_finite':all(torch.isfinite(x).all() for x in tensors.values()),'not_warm_started':not tm['warm_started']}
 report={'schema':'mia_v3_shadow_terminal_manifest','status':'PASS' if all(gates.values()) else 'FAIL','run_id':f"mia_v3_shadow_seed{cfg['shadow_seed']}_{sha(adapter)[:12]}",'shadow_seed':cfg['shadow_seed'],'member_count':len(members),'nonmember_count':len(nonmembers),'member_ids_sha256':ids_sha(members),'nonmember_ids_sha256':ids_sha(nonmembers),'membership_file_sha256':sha(a.membership),'train_members_sha256':sha(a.train_members),'shadow_config_sha256':sha(a.shadow_config),'base_package_root_hash':cfg['base_package_root_hash'],'adapter_path':str(adapter),'adapter_sha256':sha(adapter),'training_manifest_sha256':sha(a.training_dir/'training_manifest.json'),'optimizer':tm['optimizer'],'dtype':tm['training']['dtype'],'steps':tm['training']['steps'],'gates':gates,'shape_errors':shape_errors,'tensor_shapes':tm['adapter']['tensor_shapes']}
 if a.output.exists(): raise RuntimeError(f'refusing overwrite {a.output}')
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))
 if report['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
