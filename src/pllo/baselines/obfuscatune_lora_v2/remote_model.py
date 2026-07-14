"""A10-side model containing only transformed base and LoRA tensors."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from .qwen_ops import KVCache, causal_mask, repeat_kv
from .remote_runtime import RemoteTrustedRuntime

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class RemoteExternalLinear(nn.Module):
    def __init__(self, *, key: str, direction: str, weight_star: torch.Tensor,
                 a_star: torch.Tensor, b_star: torch.Tensor, alpha: float,
                 runtime: RemoteTrustedRuntime):
        super().__init__(); self.key=key; self.direction=direction; self.runtime=runtime
        self.register_buffer("weight_star",weight_star); self.a_star=nn.Parameter(a_star); self.b_star=nn.Parameter(b_star)
        self.scale=float(alpha)/a_star.shape[1]

    def forward(self,x):
        self.runtime.counters.transformed_linear_calls += 1
        merged=self.weight_star+self.scale*(self.a_star@self.b_star)
        if self.direction=="input":
            return self.runtime.add_bias(self.key,self.runtime.transform_input(self.key,x)@merged)
        return self.runtime.restore_output(self.key,x@merged)


class RemoteBlock(nn.Module):
    def __init__(self,index,config,tensors,manifest,runtime):
        super().__init__(); self.index=index; self.config=config; self.runtime=runtime
        for name in TARGETS:
            key=f"layers.{index}.{name}"; direction=manifest["inventory"][key]["direction"]
            setattr(self,name,RemoteExternalLinear(key=key,direction=direction,weight_star=tensors[f"{key}.weight_star"],
                    a_star=tensors[f"{key}.a_star"],b_star=tensors[f"{key}.b_star"],alpha=manifest["alpha"],runtime=runtime))

    def forward(self,hidden,cos,sin,cache):
        h=self.runtime.rmsnorm_key(hidden,key=f"layer.{self.index}.input",layer=self.index,
                                   which="input_layernorm",eps=self.config.rms_norm_eps)
        b,s,_=h.shape; hd=self.config.hidden_size//self.config.num_attention_heads
        q=self.q_proj(h).view(b,s,self.config.num_attention_heads,hd).transpose(1,2)
        k=self.k_proj(h).view(b,s,self.config.num_key_value_heads,hd).transpose(1,2)
        v=self.v_proj(h).view(b,s,self.config.num_key_value_heads,hd).transpose(1,2)
        q,k=self.runtime.rope(q,k,cos,sin); offset=cache.length; cache.append(k,v)
        kr=repeat_kv(cache.key,self.config.num_attention_heads//self.config.num_key_value_heads)
        vr=repeat_kv(cache.value,self.config.num_attention_heads//self.config.num_key_value_heads)
        scores=q@kr.transpose(-1,-2)/(hd**.5)
        scores=scores.masked_fill(~causal_mask(s,cache.length,offset=offset,device=hidden.device),torch.finfo(scores.dtype).min)
        probs=self.runtime.softmax(scores); context=(probs@vr).transpose(1,2).reshape(b,s,self.config.hidden_size)
        hidden=self.runtime.residual(hidden,self.o_proj(context))
        h=self.runtime.rmsnorm_key(hidden,key=f"layer.{self.index}.post",layer=self.index,
                                   which="post_attention_layernorm",eps=self.config.rms_norm_eps)
        mlp=self.down_proj(self.runtime.swiglu(self.gate_proj(h),self.up_proj(h)))
        return self.runtime.residual(hidden,mlp),cache


class RemoteAdaptedQwenCausalLM(nn.Module):
    def __init__(self,package: str|Path,runtime: RemoteTrustedRuntime,device="cuda"):
        super().__init__(); package=Path(package); manifest=json.loads((package/"package_manifest.json").read_text())
        if manifest["paper_facing_name"]!="OBFUSCATUNE_STYLE_ADAPTED_LORA_BASELINE": raise ValueError("package method mismatch")
        self.config=Qwen2Config.from_dict(manifest["model_config"]); self.runtime=runtime
        tensors={k:v.to(device) for k,v in load_file(package/"transformed_package.safetensors").items()}
        self.blocks=nn.ModuleList([RemoteBlock(i,self.config,tensors,manifest,runtime) for i in range(self.config.num_hidden_layers)])
        self.rotary=Qwen2RotaryEmbedding(self.config).to(device)

    def forward(self,input_ids,caches=None,labels=None):
        hidden=self.runtime.embedding(input_ids,None); offset=0 if not caches else caches[0].length
        positions=torch.arange(offset,offset+input_ids.shape[1],device=input_ids.device).unsqueeze(0)
        cos,sin=self.rotary(hidden,positions); caches=caches or [KVCache() for _ in self.blocks]
        for i,block in enumerate(self.blocks): hidden,caches[i]=block(hidden,cos,sin,caches[i])
        hidden=self.runtime.rmsnorm_key(hidden,key="final",eps=self.config.rms_norm_eps)
        logits=self.runtime.output(hidden,None); loss=None if labels is None else self.runtime.cross_entropy(logits,labels)
        return logits,caches,loss

    def transformed_parameters(self):
        return {f"layers.{i}.{name}.{factor}":getattr(getattr(block,name),factor)
                for i,block in enumerate(self.blocks) for name in TARGETS for factor in ("a_star","b_star")}
