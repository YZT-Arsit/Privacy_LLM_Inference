#!/usr/bin/env python
"""Measure the compute cost of the k=1 layer-0 TEE-relocation guardrail.

The guardrail relocates ONE attention block (layer-0 self_attn: RMSNorm + q/k/v/o
+ SDPA + residual) into the TEE at the existing single handoff (no extra
round-trip). Cost = the wall time of that one attention block.

We time (CUDA events, warmup + averaged):
  * full 28-layer forward (prefill S; decode 1 token with KV cache)
  * layer-0 self_attn block within that forward (hook-timed)
  -> absolute ms + fraction of total.
GPU number represents a confidential-GPU TEE (best case). We also give a CPU proxy
(the layer-0 q/k/v/o projections + SDPA on CPU, fp32) representing a TDX CPU enclave.
"""
from __future__ import annotations
import argparse, json, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefill", default="512,1024")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=dev).eval()
    cfg = model.config
    d = cfg.hidden_size
    attn0 = model.model.layers[0].self_attn

    ev_a0 = torch.cuda.Event(enable_timing=True)
    ev_a1 = torch.cuda.Event(enable_timing=True)
    def pre(m, a, k): ev_a0.record()
    def post(m, i, o): ev_a1.record()
    attn0.register_forward_pre_hook(lambda m, a, k: ev_a0.record(), with_kwargs=True)
    attn0.register_forward_hook(lambda m, i, o: ev_a1.record())

    def timed(input_ids, past=None):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1), ev_a0.elapsed_time(ev_a1), out.past_key_values

    res = {"model": args.model, "hidden": d, "iters": args.iters, "prefill": {}}
    g = torch.Generator().manual_seed(0)
    for S in [int(x) for x in args.prefill.split(",")]:
        ids = torch.randint(0, cfg.vocab_size, (1, S), generator=g).to(dev)
        for _ in range(3):
            timed(ids)                                  # warmup
        fulls, attns = [], []
        for _ in range(args.iters):
            f, a, _ = timed(ids)
            fulls.append(f); attns.append(a)
        # decode: build cache then time 1-token step
        _, _, pkv = timed(ids)
        nxt = torch.randint(0, cfg.vocab_size, (1, 1), generator=g).to(dev)
        for _ in range(3):
            from copy import deepcopy
            timed(nxt, past=pkv)
        dfulls, dattns = [], []
        for _ in range(args.iters):
            f, a, _ = timed(nxt, past=pkv)
            dfulls.append(f); dattns.append(a)
        import statistics as st
        res["prefill"][f"S={S}"] = {
            "full_forward_ms": round(st.median(fulls), 3),
            "layer0_attn_block_ms": round(st.median(attns), 3),
            "attn_block_fraction_pct": round(100 * st.median(attns) / st.median(fulls), 3),
            "decode_step_full_ms": round(st.median(dfulls), 3),
            "decode_layer0_attn_ms": round(st.median(dattns), 4),
            "decode_attn_fraction_pct": round(100 * st.median(dattns) / st.median(dfulls), 3),
        }
        print(f"S={S}: {res['prefill'][f'S={S}']}", flush=True)

    # ---- CPU proxy (TDX enclave): layer-0 projections + SDPA on CPU, fp32 -----
    import torch.nn.functional as F
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or d // n_q
    Wq = attn0.q_proj.weight.detach().float().cpu()
    Wk = attn0.k_proj.weight.detach().float().cpu()
    Wv = attn0.v_proj.weight.detach().float().cpu()
    Wo = attn0.o_proj.weight.detach().float().cpu()
    res["cpu_proxy_ms"] = {}
    for S in [int(x) for x in args.prefill.split(",")]:
        x = torch.randn(S, d)
        for _ in range(2):
            q = (x @ Wq.T).view(S, n_q, hd).transpose(0, 1)
            k = (x @ Wk.T).view(S, n_kv, hd).transpose(0, 1)
            v = (x @ Wv.T).view(S, n_kv, hd).transpose(0, 1)
            k = k.repeat_interleave(n_q // n_kv, 0)
            v = v.repeat_interleave(n_q // n_kv, 0)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            _ = o.transpose(0, 1).reshape(S, -1) @ Wo.T
        t0 = time.time()
        for _ in range(5):
            q = (x @ Wq.T).view(S, n_q, hd).transpose(0, 1)
            k = (x @ Wk.T).view(S, n_kv, hd).transpose(0, 1)
            v = (x @ Wv.T).view(S, n_kv, hd).transpose(0, 1)
            k = k.repeat_interleave(n_q // n_kv, 0)
            v = v.repeat_interleave(n_q // n_kv, 0)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            _ = o.transpose(0, 1).reshape(S, -1) @ Wo.T
        res["cpu_proxy_ms"][f"S={S}"] = round((time.time() - t0) / 5 * 1000, 2)
        print(f"CPU proxy S={S}: {res['cpu_proxy_ms'][f'S={S}']} ms", flush=True)

    print(json.dumps(res, indent=2), flush=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
