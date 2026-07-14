#!/usr/bin/env python3
"""Generate and independently verify a quote bound to the v2 worker hash."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_alibaba_tdx_quote_evidence import generate_quote_alibaba, run_preflight

VERIFIER = "/opt/alibaba/tdx-quote-verification-sample/verifier"


def b64url(value): return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def verify(path: Path):
    proc=subprocess.run([VERIFIER,"-quote",str(path)],cwd=str(Path(VERIFIER).parent),
                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
    output=proc.stdout.decode("utf-8","replace")
    token=next((part for part in output.split() if part.startswith("eyJ")),None)
    if not token: raise RuntimeError(f"standalone verifier produced no JWT: rc={proc.returncode}")
    payload=json.loads(b64url(token.split(".")[1])); appraisal=payload["appraisal_result"]
    if isinstance(appraisal,str): appraisal=json.loads(appraisal)
    result=(appraisal[0] if isinstance(appraisal,list) else appraisal)["result"]
    measurement={}
    for report in result.get("appraised_reports",[]):
        candidate=report.get("report",{}).get("measurement",{})
        if "tdx_reportdata" in candidate: measurement=candidate; break
    attributes=measurement.get("tdx_attributes",""); debug=None if not attributes else bool(int(attributes,16)&1)
    return {"overall_appraisal_result":"SUCCESS" if str(result.get("overall_appraisal_result"))=="1" else "FAIL",
            "tdx_reportdata":measurement.get("tdx_reportdata","").lower().removeprefix("0x"),
            "mr_td":measurement.get("tdx_mrtd","").lower(),"debug":debug,"verifier_rc":proc.returncode}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--runtime-hash",required=True); parser.add_argument("--package-hash",required=True)
    args=parser.parse_args()
    if args.output_dir.exists(): raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    if len(args.runtime_hash)!=128: raise ValueError("runtime hash must be SHA-512 hex")
    binding={"paper_facing_name":"OBFUSCATUNE_STYLE_ADAPTED_LORA_BASELINE",
             "contract_id":"OBFUSCATUNE_LORA_V2_TRANSFORMED_EXTERNAL_ADAMW_V1",
             "runtime_hash":args.runtime_hash,"package_hash":args.package_hash,"nonce":__import__('secrets').token_hex(32)}
    report_data=hashlib.sha512(json.dumps(binding,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    (args.output_dir/"binding_manifest.json").write_text(json.dumps(binding,indent=2,sort_keys=True)+"\n")
    preflight=run_preflight(); (args.output_dir/"preflight.json").write_text(json.dumps(preflight,indent=2)+"\n")
    if not preflight.get("all_ok"): raise RuntimeError("TDX preflight failed")
    started=time.time(); quote=generate_quote_alibaba(report_data,args.output_dir,quote_out_name="td_quote.dat")
    parsed=verify(quote); bound=parsed["tdx_reportdata"]==report_data
    verified=parsed["overall_appraisal_result"]=="SUCCESS" and bound and parsed["debug"] is False
    evidence={"schema":"obfuscatune_v2_attestation_v1","attestation_verified":verified,
              "paper_facing":verified,"quote_generated":True,"quote_sha256":hashlib.sha256(quote.read_bytes()).hexdigest(),
              "runtime_hash":args.runtime_hash,"package_hash":args.package_hash,"report_data_expected":report_data,
              "reportdata_bound":bound,**parsed,"wall_sec":time.time()-started}
    (args.output_dir/"evidence.json").write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
    print(json.dumps(evidence,indent=2,sort_keys=True))
    if not verified: raise SystemExit(2)


if __name__=="__main__": main()
