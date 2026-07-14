#!/usr/bin/env python3
"""Real A10 client for the baseline-specific Intel TDX worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from pllo.baselines.obfuscatune_lora_v2.artifacts import (apply_adapter, export_adapter, load_adapter,
                                                          load_training_checkpoint, save_training_checkpoint)
from pllo.baselines.obfuscatune_lora_v2.optimizer import AdamWHyperparameters, TransformedAdamW
from pllo.baselines.obfuscatune_lora_v2.remote_model import RemoteAdaptedQwenCausalLM
from pllo.baselines.obfuscatune_lora_v2.remote_runtime import RemoteTrustedRuntime
from pllo.baselines.obfuscatune_lora_v2.rpc_protocol import RPCClient


def build_batch(rows, device):
    length=max(len(row["input_ids"]) for row in rows); ids=torch.zeros(len(rows),length,dtype=torch.long)
    labels=torch.full_like(ids,-100)
    for i,row in enumerate(rows):
        n=len(row["input_ids"]); ids[i,:n]=torch.tensor(row["input_ids"])
        labels[i,row["sup_start"]+1:n]=torch.tensor(row["input_ids"][row["sup_start"]+1:n])
    return ids.to(device),labels.to(device)


def binding(args):
    return {"method":"OBFUSCATUNE_STYLE_ADAPTED_LORA_BASELINE","seed":args.seed,"rank":8,"alpha":16,
            "package_manifest_sha256":hashlib.sha256((args.package/"package_manifest.json").read_bytes()).hexdigest()}


def connect(args):
    client=RPCClient(args.tdx_host,args.tdx_port,args.secret_hex); health=client.call("health")
    if health["status"]!="ok" or health["cached_calls"]!=0: raise RuntimeError("TDX worker not clean")
    return client,RemoteTrustedRuntime(client)


def train(args):
    output=args.output_dir.resolve()
    if output.exists() and not args.resume: raise RuntimeError(f"refusing to overwrite {output}")
    output.mkdir(parents=True,exist_ok=args.resume); client,runtime=connect(args)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_config={"schema":"obfuscatune_style_adapted_lora_run_v1","paper_facing_name":"OBFUSCATUNE_STYLE_ADAPTED_LORA_BASELINE",
                "seed":args.seed,"steps":args.steps,"batch_size":args.batch_size,"rank":8,"alpha":16,"dropout":0,
                "targets":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                "optimizer":{"name":"transformed_coordinate_adamw","lr":2e-4,"betas":[.9,.999],"eps":1e-8,"weight_decay":.01},
                "package":str(args.package),"train_data":str(args.train_data),"schedule":str(args.schedule),
                "tdx_host":args.tdx_host,"tdx_port":args.tdx_port,"secret_commitment":hashlib.sha256(bytes.fromhex(args.secret_hex)).hexdigest(),
                "source_sha256":source_hash,"checkpoint_every":args.checkpoint_every}
    config_path=output/"run_config.json"
    if not args.resume: config_path.write_text(json.dumps(run_config,indent=2,sort_keys=True)+"\n")
    elif json.loads(config_path.read_text())!=run_config: raise RuntimeError("resume run config mismatch")
    pid_path=output/f"pid_{os.getpid()}.json"; pid_path.write_text(json.dumps({"pid":os.getpid(),"host":os.uname().nodename,
            "started_at":time.time(),"resume":bool(args.resume),"source_sha256":source_hash},indent=2)+"\n")
    model=RemoteAdaptedQwenCausalLM(args.package,runtime,device="cuda").cuda(); parameters=model.transformed_parameters()
    optimizer=TransformedAdamW(parameters,AdamWHyperparameters(lr=2e-4,weight_decay=.01))
    data=torch.load(args.train_data,map_location="cpu",weights_only=False); by_id={int(x["sample_id"]):x for x in data}
    schedule=json.loads(args.schedule.read_text())["schedule"]; trajectory=[]; start=time.time(); start_step=0
    if args.resume:
        checkpoints=sorted(output.glob("checkpoint_step_*.pt"),key=lambda p:int(p.stem.rsplit("_",1)[1]))
        if not checkpoints: raise RuntimeError("resume requested but no versioned checkpoint exists")
        payload=load_training_checkpoint(checkpoints[-1],parameters=parameters,optimizer=optimizer,expected_binding=binding(args))
        start_step=int(payload["step"])
        rows={}
        for path in sorted(output.glob("trajectory_segment_*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    row=json.loads(line)
                    if int(row["step"])<=start_step: rows[int(row["step"])]=row
        trajectory=[rows[index] for index in range(1,start_step+1) if index in rows]
        if len(trajectory)!=start_step: raise RuntimeError("resume trajectory/checkpoint coverage mismatch")
    trajectory_path=output/f"trajectory_segment_{os.getpid()}.jsonl"
    torch.cuda.reset_peak_memory_stats()
    for index in range(start_step,args.steps):
        entry=schedule[index]; rows=[by_id[int(s)] for s in entry["sample_ids"][:args.batch_size]]
        ids,labels=build_batch(rows,"cuda"); step_start=time.time(); _,_,loss=model(ids,labels=labels)
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss step {index}")
        loss.backward(); gradients={}
        for name,value in parameters.items():
            if value.grad is None or not torch.isfinite(value.grad).all(): raise FloatingPointError(f"invalid gradient {name}")
            gradients[name]=value.grad.detach().clone()
        optimizer.step(gradients); runtime.counters.optimizer_updates+=1
        for value in parameters.values(): value.grad=None
        row={"step":index+1,"loss":float(loss.detach()),"finite":True,"wall_sec":time.time()-step_start}
        trajectory.append(row)
        with trajectory_path.open("a") as handle: handle.write(json.dumps(row,sort_keys=True)+"\n")
        (output/"heartbeat.json").write_text(json.dumps({"completed_steps":index+1,"total_steps":args.steps,"pid":os.getpid(),"time":time.time()}))
        if args.checkpoint_every and ((index+1)%args.checkpoint_every==0 or index+1==args.steps):
            versioned=output/f"checkpoint_step_{index+1}.pt"
            if versioned.exists(): raise RuntimeError(f"refusing to overwrite {versioned}")
            save_training_checkpoint(versioned,parameters=parameters,optimizer=optimizer,binding=binding(args),step=index+1,
                                     counters=runtime.counters.to_dict())
    checkpoint=output/"checkpoint.pt"
    if checkpoint.exists(): raise RuntimeError(f"refusing to overwrite {checkpoint}")
    checkpoint_hash=save_training_checkpoint(checkpoint,parameters=parameters,optimizer=optimizer,
            binding=binding(args),step=args.steps,counters=runtime.counters.to_dict())
    export_adapter(output/"adapter",parameters=parameters,binding=binding(args),final_step=args.steps)
    report={"schema":"obfuscatune_real_a10_tdx_smoke_v1","status":"TRAINING_COMPLETE_PENDING_FRESH_PROCESS",
            "steps":args.steps,"finite_steps":len(trajectory),"trajectory":trajectory,"wall_sec":time.time()-start,
            "checkpoint_sha256":checkpoint_hash,"runtime_counters":runtime.counters.to_dict(),
            "rpc":{"round_trips":client.round_trips,"bytes_sent":client.bytes_sent,"bytes_received":client.bytes_received,
                   "transport_ns":client.transport_ns},"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(),
            "gpu_name":torch.cuda.get_device_name(),"gpu_uuid":os.environ.get("OBF_GPU_UUID"),"tdx_host":args.tdx_host}
    (output/"training_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    client.close(); print(json.dumps(report,indent=2))


def verify(args):
    client,runtime=connect(args); model=RemoteAdaptedQwenCausalLM(args.package,runtime,device="cuda").cuda()
    tensors=load_adapter(args.output_dir/"adapter",expected_binding=binding(args)); apply_adapter(model.transformed_parameters(),tensors)
    prompt=torch.tensor([[1,2,3,4]],device="cuda")
    with torch.no_grad():
        logits,caches,_=model(prompt); token=logits[:,-1].argmax(-1,keepdim=True); values=[token]
        for _ in range(args.generation_tokens-1):
            logits,caches,_=model(token,caches=caches); token=logits[:,-1].argmax(-1,keepdim=True); values.append(token)
    tokens=torch.cat(values,1).cpu(); digest=hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    result={"pid":os.getpid(),"tokens":tokens.tolist(),"token_sha256":digest,"rpc_round_trips":client.round_trips,
            "bytes_sent":client.bytes_sent,"bytes_received":client.bytes_received}
    path=args.output_dir/f"fresh_process_{args.verify_index}.json"
    if path.exists(): raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); client.close(); print(json.dumps(result))


def repetition(tokens):
    if len(tokens)<2:return 0.0
    pairs=list(zip(tokens,tokens[1:])); return 1-len(set(pairs))/len(pairs)


def generate(args):
    from transformers import AutoTokenizer
    output=args.output_dir.resolve(); adapter_dir=output/"adapter"
    if not adapter_dir.exists(): raise RuntimeError("formal adapter missing")
    generation_path=output/"generations.jsonl"; completed=[]
    if generation_path.exists():
        completed=[json.loads(line) for line in generation_path.read_text().splitlines() if line.strip()]
        if not args.resume_generation: raise RuntimeError("generation output exists; use --resume-generation")
    client,runtime=connect(args); model=RemoteAdaptedQwenCausalLM(args.package,runtime,device="cuda").cuda()
    apply_adapter(model.transformed_parameters(),load_adapter(adapter_dir,expected_binding=binding(args)))
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True); eos=tokenizer.eos_token_id
    prompts=json.loads(args.generation_input.read_text())[:args.generation_max]
    if len(prompts)!=args.generation_max: raise RuntimeError("generation subset length mismatch")
    done={row["sample_id"] for row in completed}; start=time.time(); total_tokens=sum(r["generated_token_count"] for r in completed)
    for index,prompt in enumerate(prompts):
        if prompt["sample_id"] in done: continue
        ids=torch.tensor([prompt["prompt_ids"]],device="cuda"); caches=None; tokens=[]
        with torch.no_grad():
            logits,caches,_=model(ids)
            for _ in range(args.max_new):
                token=int(logits[:,-1].argmax(-1).item())
                if token==eos: break
                tokens.append(token)
                logits,caches,_=model(torch.tensor([[token]],device="cuda"),caches=caches)
        text=tokenizer.decode(tokens,skip_special_tokens=True).strip(); total_tokens+=len(tokens)
        row={"sample_id":prompt["sample_id"],"meaning_representation":prompt.get("meaning_representation",""),
             "references":prompt.get("references",[]),"generated_text":text,"token_ids":tokens,
             "generated_token_count":len(tokens),"invalid_output":not bool(text),
             "repetition_bigram_frac":repetition(tokens),"cell":"OBFUSCATUNE_STYLE_ADAPTED_LORA_BASELINE","seed":args.seed}
        with generation_path.open("a") as handle: handle.write(json.dumps(row,sort_keys=True)+"\n")
        (output/"generation_heartbeat.json").write_text(json.dumps({"completed":index+1,"total":len(prompts),"time":time.time()}))
    rows=[json.loads(line) for line in generation_path.read_text().splitlines() if line.strip()]
    if len(rows)!=args.generation_max or len({r["sample_id"] for r in rows})!=args.generation_max: raise RuntimeError("generation completeness gate failed")
    report={"records":len(rows),"unique_sample_ids":len({r["sample_id"] for r in rows}),"total_tokens":total_tokens,
            "wall_sec_this_process":time.time()-start,"rpc_round_trips":client.round_trips,"bytes_sent":client.bytes_sent,
            "bytes_received":client.bytes_received,"generation_sha256":hashlib.sha256(generation_path.read_bytes()).hexdigest()}
    final=output/"generation_report.json"
    if final.exists(): raise RuntimeError(f"refusing to overwrite {final}")
    final.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); client.close(); print(json.dumps(report,indent=2))


def parse_args():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="mode",required=True)
    for mode in ("train","verify","generate"):
        q=sub.add_parser(mode); q.add_argument("--package",type=Path,required=True); q.add_argument("--output-dir",type=Path,required=True)
        q.add_argument("--tdx-host",required=True); q.add_argument("--tdx-port",type=int,required=True); q.add_argument("--secret-hex",required=True)
        q.add_argument("--seed",type=int,default=1234)
        if mode=="train":
            q.add_argument("--train-data",type=Path,required=True); q.add_argument("--schedule",type=Path,required=True)
            q.add_argument("--steps",type=int,default=3); q.add_argument("--batch-size",type=int,default=1)
            q.add_argument("--checkpoint-every",type=int,default=0); q.add_argument("--resume",action="store_true")
        elif mode=="verify": q.add_argument("--generation-tokens",type=int,default=4); q.add_argument("--verify-index",type=int,required=True)
        else:
            q.add_argument("--generation-input",type=Path,required=True); q.add_argument("--tokenizer",type=Path,required=True)
            q.add_argument("--generation-max",type=int,default=500); q.add_argument("--max-new",type=int,default=96)
            q.add_argument("--resume-generation",action="store_true")
    return p.parse_args()


if __name__=="__main__":
    a=parse_args(); train(a) if a.mode=="train" else (verify(a) if a.mode=="verify" else generate(a))
