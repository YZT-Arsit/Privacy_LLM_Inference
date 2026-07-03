#!/usr/bin/env python
"""ObfuscaTune (arXiv:2407.02960) latency harness at Qwen2.5-7B dimensions.

ObfuscaTune runs EVERY non-linearity (RMSNorm x2, softmax, SiLU) inside the TEE
on de-obfuscated values, so every block round-trips the TEE<->GPU boundary
(8 crossings/layer => 224 for 28 layers). Our A_rightmul runs all non-linearities
on the untrusted GPU with a single entry/exit (2 crossings) plus the k=1 layer-0
guardrail (one attention block in the TEE).

This measures the wall-clock consequence on ONE box (self-contained: it times a
plaintext floor and our scheme on the same hardware, so numbers are internally
comparable and the ratio is hardware-portable). Configs, all Qwen-7B shaped:

  plaintext          all ops on GPU, 0 crossings                (privacy floor)
  ours_a_rightmul    all nonlinear on GPU; 2 crossings (embed-in / logits-out)
                     via TEE + 1 layer-0 attn block in TEE (k=1 guardrail)
  obfuscatune_gpu    nonlinear "in TEE" = same GPU (best case: 224 device-local
                     crossings, +Ra obfuscation matmuls, ~0 transfer cost)
  obfuscatune_cpu    nonlinear in CPU enclave proxy: 224 real GPU<->CPU transfers
                     + fp32 CPU nonlinear (realistic TDX-CPU cost)

Reports prefill (S=512, 1024) and decode (1 token, ctx=512) latency + slowdown.
"""
from __future__ import annotations

import argparse
import json

import torch

# Qwen2.5-7B-Instruct
D = 3584
N_HEADS = 28
N_KV = 4
HEAD_DIM = 128
FFN = 18944
N_LAYERS = 28
EPS = 1e-6


def _cuda_time(fn, iters, warmup, dev):
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        st = torch.cuda.Event(True); en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        return st.elapsed_time(en) / iters
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _weights(gpu, dtype):
    g = torch.Generator(device="cpu").manual_seed(0)
    def r(a, b):
        return (torch.randn(a, b, generator=g) * 0.02).to(gpu, dtype)
    return {
        "Wq": r(D, D), "Wk": r(D, N_KV * HEAD_DIM), "Wv": r(D, N_KV * HEAD_DIM),
        "Wo": r(D, D), "Wg": r(D, FFN), "Wu": r(D, FFN), "Wd": r(FFN, D),
        "gamma1": torch.ones(D, device=gpu, dtype=dtype),
        "gamma2": torch.ones(D, device=gpu, dtype=dtype),
    }


def rmsnorm(x, gamma):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + EPS)).to(x.dtype) * gamma


def _attn(q, k, v, S):
    q = q.view(S, N_HEADS, HEAD_DIM).transpose(0, 1)
    k = k.view(-1, N_KV, HEAD_DIM).transpose(0, 1)
    v = v.view(-1, N_KV, HEAD_DIM).transpose(0, 1)
    rep = N_HEADS // N_KV
    k = k.repeat_interleave(rep, 0)
    v = v.repeat_interleave(rep, 0)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(S > 1))
    return o.transpose(0, 1).reshape(q.shape[1], D)


def layer(x, w, S, tee_dev, *, attn_in_tee, mlp_in_tee, obf):
    """One decoder layer with independent attn/mlp TEE placement + obfuscation.

    tee_dev == x.device -> device-local (confidential-GPU TEE / plaintext).
    tee_dev == cpu      -> real GPU<->CPU crossings (CPU enclave proxy).
    """
    gpu = x.device
    to_tee = (lambda t: t.to(tee_dev)) if tee_dev != gpu else (lambda t: t)
    to_gpu = (lambda t: t.to(gpu)) if tee_dev != gpu else (lambda t: t)

    # --- attention ---
    if attn_in_tee:
        h = to_gpu(rmsnorm(to_tee(x), w["gamma1"].to(tee_dev)))     # RMSNorm in TEE
    else:
        h = rmsnorm(x, w["gamma1"])
    if obf:  # ObfuscaTune X* = X Ra (extra TEE matmul); W* precomputed at deploy
        h = h @ w["_Ra"]
    q, k, v = h @ w["Wq"], h @ w["Wk"], h @ w["Wv"]                 # GPU matmuls
    if attn_in_tee:                                                 # softmax in TEE
        o = to_gpu(_attn(to_tee(q), to_tee(k), to_tee(v), S))
    else:
        o = _attn(q, k, v, S)
    x = x + o @ w["Wo"]

    # --- MLP ---
    if mlp_in_tee:
        h2 = to_gpu(rmsnorm(to_tee(x), w["gamma2"].to(tee_dev)))    # RMSNorm in TEE
    else:
        h2 = rmsnorm(x, w["gamma2"])
    if obf:
        h2 = h2 @ w["_Ra"]
    g_, u_ = h2 @ w["Wg"], h2 @ w["Wu"]
    if mlp_in_tee:                                                  # SiLU in TEE
        inter = to_gpu(torch.nn.functional.silu(to_tee(g_)) * to_tee(u_))
    else:
        inter = torch.nn.functional.silu(g_) * u_
    x = x + inter @ w["Wd"]
    return x


# tee: which TEE the scheme's trusted ops use. guardrail: # early layers whose
# ATTENTION block runs in the TEE (ours k=1). all_nonlinear: every nonlinear in
# the TEE (ObfuscaTune). obf: ObfuscaTune Ra masking matmuls. crossings for
# reporting (ObfuscaTune 8/layer; ours = 2 boundary transfers).
CONFIGS = {
    "plaintext":       {"tee": "gpu", "guardrail": 0, "all_nonlinear": False, "obf": False, "crossings": 0},
    "ours_gpu":        {"tee": "gpu", "guardrail": 1, "all_nonlinear": False, "obf": False, "crossings": 2},
    "ours_cpu":        {"tee": "cpu", "guardrail": 1, "all_nonlinear": False, "obf": False, "crossings": 2},
    "obfuscatune_gpu": {"tee": "gpu", "guardrail": 0, "all_nonlinear": True,  "obf": True,  "crossings": 8 * N_LAYERS},
    "obfuscatune_cpu": {"tee": "cpu", "guardrail": 0, "all_nonlinear": True,  "obf": True,  "crossings": 8 * N_LAYERS},
}


def run_config(name, w, gpu, dtype, S, iters, warmup):
    cfg = CONFIGS[name]
    tee_dev = gpu if cfg["tee"] == "gpu" else torch.device("cpu")
    if cfg["obf"] and "_Ra" not in w:
        g = torch.Generator(device="cpu").manual_seed(1)
        a = torch.randn(D, D, generator=g)
        Ra, _ = torch.linalg.qr(a)
        w["_Ra"] = Ra.to(gpu, dtype)

    x0 = torch.randn(S, D, device=gpu, dtype=dtype) * 0.1

    def full():
        x = x0
        # ours: single boundary entry (embed masked in TEE -> GPU)
        if cfg["guardrail"] and tee_dev != gpu:
            x = x.to(tee_dev).to(gpu)
        for li in range(N_LAYERS):
            if cfg["all_nonlinear"]:                 # ObfuscaTune: every nonlinear in TEE
                a_tee = m_tee = True
            else:                                    # ours: only layer<guardrail attn in TEE
                a_tee = li < cfg["guardrail"]
                m_tee = False
            x = layer(x, w, S, tee_dev, attn_in_tee=a_tee, mlp_in_tee=m_tee, obf=cfg["obf"])
        # ours: single boundary exit (logits recovered in TEE)
        if cfg["guardrail"] and tee_dev != gpu:
            x = x.to(tee_dev).to(gpu)
        return x

    ms = _cuda_time(full, iters, warmup, gpu)
    return {"latency_ms": ms, "tee_type": cfg["tee"],
            "tee_boundary_crossings": cfg["crossings"],
            "guardrail_attn_blocks_in_tee": cfg["guardrail"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prefill", default="512,1024")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    w = _weights(gpu, dtype)
    gpu_name = torch.cuda.get_device_name(0) if gpu.type == "cuda" else "cpu"
    print(f"[latency] {gpu_name} dtype={a.dtype} Qwen7B-dims x{N_LAYERS} layers", flush=True)

    result = {"gpu": gpu_name, "dtype": a.dtype, "dims": {"D": D, "FFN": FFN, "layers": N_LAYERS},
              "prefill": {}, "decode": {}}
    seqs = [int(s) for s in a.prefill.split(",")]
    for S in seqs:
        result["prefill"][str(S)] = {}
        for name in CONFIGS:
            r = run_config(name, w, gpu, dtype, S, a.iters, a.warmup)
            result["prefill"][str(S)][name] = r
            print(f"  prefill S={S:>5} {name:<16} {r['latency_ms']:8.2f} ms  "
                  f"crossings={r['tee_boundary_crossings']}", flush=True)
    # decode: 1 new token, but layer() attends S=1 causal; approximate step cost
    for name in CONFIGS:
        r = run_config(name, w, gpu, dtype, 1, max(a.iters, 20), a.warmup)
        result["decode"][name] = r
        print(f"  decode  S=    1 {name:<16} {r['latency_ms']:8.3f} ms  "
              f"crossings={r['tee_boundary_crossings']}", flush=True)

    # slowdowns vs plaintext
    for S, d in result["prefill"].items():
        base = d["plaintext"]["latency_ms"]
        for name, r in d.items():
            r["slowdown_vs_plaintext"] = round(r["latency_ms"] / base, 3)
    base = result["decode"]["plaintext"]["latency_ms"]
    for name, r in result["decode"].items():
        r["slowdown_vs_plaintext"] = round(r["latency_ms"] / base, 3)

    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[latency] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
