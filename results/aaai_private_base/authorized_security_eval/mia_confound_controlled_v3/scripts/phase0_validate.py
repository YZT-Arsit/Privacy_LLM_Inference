#!/usr/bin/env python3
"""Fail-closed validation of the frozen pool, handoff, and shadow assignments."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path

PINS = {
 "pool/candidate_pool.jsonl": "b65e8b4a65566c438845c9d2a43a3a77dcec7327f8f7ae32452d37d800cebcae",
 "handoff/README.md": "a24db382a29fc2fcbdbe4ba51b0ed64afe5da8e71865ec93249fa6fd6a5727af",
 "handoff/build_mia_v2_handoff.py": "e95d9a5973d083867f70d35fc9d8ef2a06eb19af4abf02f399a2bd28beb6dcb7",
 "handoff/collection_manifest.json": "ba2ebcad7746bc649cd74ae9a81d0e773fb026a18f92834f07a2522550310d2b",
 "handoff/frozen_sample_ids.json": "0f7776fa4d22e29e9fce398589e7af40adb9adce729afa645262b089653bfaa7",
 "handoff/member_nonmember_assignment.jsonl": "70635e2c587a77b75f98fba33783e293a2508a797c2c0e97ccf7fd98d52feebd",
 "handoff/prohibited_feature_list.json": "2c492e5573e8a32ce443d6e6ea79d55cd475751e1dc7b878a665cbdeab491bf8",
 "handoff/shadow_adapter_requirements.json": "18a974db5de60ccf897c67b20811dc0b39dfc83010a5ba9398a078a6d223f4cd",
 "handoff/validated_feature_schema.json": "7be799cbebf59d383c805e2195cf855bf22de3ade04c511dd2dd68069486406e",
}
MEMBERSHIP = {7:"01eb983e80078c81ae48abff3838f9671ec4225cf53eedcc84869f63fa151676",
              1234:"bf98dedcf37a42a9439d9d9fd949263e44440de43b47f7ad87eba632f7bc222a",
              2025:"6bc88e7e92f07aac642759ca870eaec19abd9d4cdfc7209ecbb0ee93547cad4a"}
TRAIN = {7:"ff94bf56ea522f61229bd0e9f84b7aaab1c7f13a80704ddcce0cf8b27441fac6",
         1234:"3878308db4ede7420d7e51ddb991d8e58e351d94d2d028cfa75fc984330be4b9",
         2025:"060dad0382a2d89a35386b12cd60091546edfd09ae69411c72ad0b9e5b718c3a"}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def loadl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 h=a.root/'results/aaai_private_base/authorized_security_eval/mia_v2_handoff'
 b=a.root/'results/aaai_private_base/authorized_security_eval/mia/confound_controlled_v2'
 paths={"pool/candidate_pool.jsonl":b/'pool/candidate_pool.jsonl'}
 paths.update({f"handoff/{x.name}":x for x in h.iterdir() if x.is_file()})
 checks=[]
 for key,want in PINS.items():
  got=sha(paths[key]); checks.append({"artifact":key,"expected":want,"actual":got,"pass":got==want})
 pool=loadl(paths['pool/candidate_pool.jsonl']); ids=[x['sample_id'] for x in pool]
 if len(ids)!=1000 or len(set(ids))!=1000: raise RuntimeError('pool is not 1000 unique IDs')
 freq=collections.Counter(); shadows=[]
 for seed in (7,1234,2025):
  d=b/f'shadow_runs/shadow_s{seed}'; m=d/'evaluator_membership_metadata.jsonl'; t=d/'train_members.jsonl'; q=d/'evaluation_queries.jsonl'
  mr=loadl(m); tr=loadl(t); qr=loadl(q); mem=[x['sample_id'] for x in mr if x['member']]; non=[x['sample_id'] for x in mr if not x['member']]
  ok=(sha(m)==MEMBERSHIP[seed] and sha(t)==TRAIN[seed] and len(mem)==len(non)==500 and
      len(set(mem)&set(non))==0 and set(mem)|set(non)==set(ids) and
      [x['sample_id'] for x in tr]==mem and [x['sample_id'] for x in qr]==ids)
  shadows.append({"seed":seed,"members":len(mem),"nonmembers":len(non),"membership_sha256":sha(m),"train_sha256":sha(t),"pass":ok})
  freq.update(mem)
 schema=json.loads((h/'validated_feature_schema.json').read_text()); allowed=[x for x in schema['features'] if x['status']=='ALLOWED']
 gates={"all_hashes":all(x['pass'] for x in checks),"cross_shadow":set(freq.values())=={1,2} and collections.Counter(freq.values())=={1:500,2:500},"shadows":all(x['pass'] for x in shadows),"allowed_features_611":len(allowed)==611,"forbidden_features_96":sum(x['status']=='FORBIDDEN' for x in schema['features'])==96}
 report={"schema":"mia_v3_phase0_validation","status":"PASS" if all(gates.values()) else "FAIL","gates":gates,"hashes":checks,"shadows":shadows,"pool_records":1000,"allowed_feature_count":len(allowed)}
 if a.output.exists(): raise RuntimeError(f'refusing to overwrite {a.output}')
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
 if report['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
