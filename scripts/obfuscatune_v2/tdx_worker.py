#!/usr/bin/env python3
"""Baseline-specific trusted operator and autograd worker for Intel TDX."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import socket
import traceback
import uuid

import torch
from transformers import AutoModelForCausalLM

from pllo.baselines.obfuscatune_lora_v2.rpc_protocol import receive_frame, send_frame
from pllo.baselines.obfuscatune_lora_v2.transforms import orthogonal_matrix
from prepare_transformed_package import TARGETS, projections, rotation_seed


class State:
    def __init__(self, model_path, seed):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                          torch_dtype=torch.float32).cpu().eval()
        self.seed = seed; self.rotations = {}; self.biases = {}; self.cache = {}
        for layer_index, layer in enumerate(self.model.model.layers):
            for target_index, (name, module) in enumerate(projections(layer).items()):
                weight = module.weight.detach().transpose(0, 1)
                direction = "output" if name in {"o_proj", "down_proj"} else "input"
                dim = weight.shape[1] if direction == "output" else weight.shape[0]
                key = f"layers.{layer_index}.{name}"
                self.rotations[key] = orthogonal_matrix(dim, rotation_seed(seed, layer_index, target_index), dtype=torch.float32)
                self.biases[key] = None if module.bias is None else module.bias.detach().cpu()

    def trim_if_idle(self):
        """Return freed RPC/autograd arenas after a complete backward pass."""
        if self.cache: return
        gc.collect()
        try: ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception: pass

    def health(self):
        try: rss_bytes=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError,IndexError,ValueError): rss_bytes=None
        return {"status":"ok","cached_calls":len(self.cache),"rss_bytes":rss_bytes}

    def operation(self, name, attrs, values):
        if name == "transform_input": return [values[0] @ self.rotations[attrs["key"]]]
        if name == "restore_output":
            out = values[0] @ self.rotations[attrs["key"]].T
            bias = self.biases[attrs["key"]]
            return [out if bias is None else out + bias]
        if name == "add_bias":
            bias = self.biases[attrs["key"]]; return [values[0] if bias is None else values[0] + bias]
        if name == "rmsnorm_key":
            key = attrs["key"]
            if key == "final": weight = self.model.model.norm.weight
            else:
                layer, which = attrs["layer"], attrs["which"]
                weight = getattr(self.model.model.layers[layer], which).weight
            x = values[0]; x32 = x.float()
            return [(x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + attrs["eps"])).to(x.dtype) * weight]
        if name == "rope":
            q, k, cos, sin = values
            def half(x):
                n=x.shape[-1]//2; return torch.cat((-x[..., n:], x[..., :n]), -1)
            while cos.ndim < q.ndim: cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            return [q*cos+half(q)*sin, k*cos+half(k)*sin]
        if name == "softmax": return [torch.softmax(values[0].float(), dim=attrs["dim"]).to(values[0].dtype)]
        if name == "swiglu": return [torch.nn.functional.silu(values[0]) * values[1]]
        if name == "residual": return [values[0] + values[1]]
        if name == "embedding": return [torch.nn.functional.embedding(values[0], self.model.model.embed_tokens.weight)]
        if name == "output": return [values[0] @ self.model.lm_head.weight.T]
        if name == "cross_entropy":
            logits, labels = values
            return [torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                       labels[:, 1:].reshape(-1), ignore_index=attrs["ignore_index"])]
        raise ValueError(f"unknown trusted operation {name}")

    def forward(self, args):
        if args["operation"]=="output_loss": return self.output_loss(args)
        training = bool(args["training"]); values=[]; differentiable=[]
        for index, value in enumerate(args["tensors"]):
            value = value.detach().cpu()
            if training and value.is_floating_point(): value.requires_grad_(True); differentiable.append(index)
            values.append(value)
        outputs = self.operation(args["operation"], args["attributes"], values)
        call_id = None
        if training and any(out.requires_grad for out in outputs):
            call_id = uuid.uuid4().hex; self.cache[call_id] = (values, differentiable, outputs)
        return {"call_id": call_id, "outputs": [out.detach() for out in outputs]}

    def output_loss(self,args):
        hidden=args["tensors"][0].detach().cpu(); labels=args["tensors"][1].detach().cpu()
        ignore=args["attributes"]["ignore_index"]; chunk=max(1,args["attributes"]["chunk_tokens"])
        shifted=hidden[:,:-1].reshape(-1,hidden.shape[-1]); targets=labels[:,1:].reshape(-1)
        valid=targets.ne(ignore); indices=valid.nonzero(as_tuple=False).flatten()
        if not len(indices): raise ValueError("output_loss has no supervised tokens")
        selected=shifted.index_select(0,indices); selected_targets=targets.index_select(0,indices)
        gradient_selected=torch.empty_like(selected); total=torch.zeros((),dtype=torch.float64)
        weight=self.model.lm_head.weight.detach(); count=len(indices)
        with torch.no_grad():
            for start in range(0,count,chunk):
                end=min(start+chunk,count); logits=selected[start:end]@weight.T
                total+=torch.nn.functional.cross_entropy(logits,selected_targets[start:end],reduction="sum").double()
                probabilities=torch.softmax(logits.float(),-1)
                probabilities[torch.arange(end-start),selected_targets[start:end]]-=1
                gradient_selected[start:end]=(probabilities@weight).to(hidden.dtype)/count
        gradient=torch.zeros_like(shifted); gradient.index_copy_(0,indices,gradient_selected)
        gradient=torch.cat((gradient.view(hidden.shape[0],hidden.shape[1]-1,hidden.shape[2]),
                            torch.zeros_like(hidden[:,:1])),dim=1)
        call_id=uuid.uuid4().hex
        self.cache[call_id]={"kind":"precomputed_output_loss","input_gradients":[gradient,None]}
        return {"call_id":call_id,"outputs":[(total/count).to(hidden.dtype)]}

    def backward(self, args):
        cached=self.cache.pop(args["call_id"])
        if isinstance(cached,dict) and cached.get("kind")=="precomputed_output_loss":
            scale=args["gradients"][0]
            result={"input_gradients":[None if value is None else value*scale for value in cached["input_gradients"]]}
            self.trim_if_idle(); return result
        values, differentiable, outputs = cached
        active_outputs=[]; active_gradients=[]
        for output, gradient in zip(outputs, args["gradients"]):
            if output.requires_grad and gradient is not None:
                active_outputs.append(output); active_gradients.append(gradient)
        inputs=[values[i] for i in differentiable]
        grads=torch.autograd.grad(active_outputs, inputs, active_gradients, allow_unused=True)
        result=[None]*len(values)
        for index, gradient in zip(differentiable, grads): result[index]=gradient
        response={"input_gradients": result}; self.trim_if_idle(); return response


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--host", default="0.0.0.0"); parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--secret-hex", required=True); parser.add_argument("--status", required=True)
    args=parser.parse_args(); secret=bytes.fromhex(args.secret_hex)
    if len(secret)!=32: raise ValueError("secret must be 32 bytes")
    state=State(args.model,args.seed)
    stop=False
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); server.bind((args.host,args.port)); server.listen(1)
        open(args.status,"w").write(json.dumps({"status":"ready","port":args.port,"seed":args.seed,"pid":__import__('os').getpid()}))
        while not stop:
            connection,_=server.accept()
            with connection:
              while True:
                try: request,_=receive_frame(connection,secret)
                except EOFError: break
                try:
                    method=request["method"]
                    if method=="forward": result=state.forward(request["args"])
                    elif method=="backward": result=state.backward(request["args"])
                    elif method=="health": result=state.health()
                    elif method=="shutdown_connection":
                        send_frame(connection,{"id":request["id"],"ok":True,"result":{}},secret); break
                    elif method=="shutdown_worker":
                        send_frame(connection,{"id":request["id"],"ok":True,"result":{}},secret); stop=True; break
                    else: raise ValueError(f"unknown method {method}")
                    response={"id":request["id"],"ok":True,"result":result}
                except Exception as exc:
                    response={"id":request.get("id"),"ok":False,"error":repr(exc),"traceback":traceback.format_exc()}
                send_frame(connection,response,secret)


if __name__=="__main__": main()
