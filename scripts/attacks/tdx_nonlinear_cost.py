#!/usr/bin/env python
"""Definitive TEE-side (Intel TDX enclave) cost of the non-linear ops.

Run this ON the TDX guest VM (the whole guest is the trust domain, so this
measures the real in-enclave CPU cost including TDX memory-encryption overhead).
It replaces the host-CPU proxy used in obfuscatune_latency.py's CPU-enclave
column with the actual TDX-enclave compute cost.

Times, at Qwen2.5-7B dims, the non-linear "TEE work" each scheme puts in the
enclave PER forward:
  ObfuscaTune : every layer runs {RMSNorm, softmax-attention, RMSNorm, SiLU} in
                the TEE  ->  28 x that.
  Ours (k=1)  : only the layer-0 attention block {RMSNorm, softmax-attention}
                runs in the TEE  ->  1 x that.

Reports per-op and per-forward enclave-compute ms for prefill (S=512,1024) and
decode (1 new token over a KV context). This is the TEE-COMPUTE component; the
end-to-end CPU-enclave latency = GPU linear cost (H800) + this + network transfer.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import torch

D = 3584
N_HEADS = 28
N_KV = 4
HEAD_DIM = 128
FFN = 18944
N_LAYERS = 28
EPS = 1e-6


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / iters


def rmsnorm(x, g):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + EPS) * g


def attention(q, k, v, S):
    q = q.view(S, N_HEADS, HEAD_DIM).transpose(0, 1)
    k = k.view(-1, N_KV, HEAD_DIM).transpose(0, 1).repeat_interleave(N_HEADS // N_KV, 0)
    v = v.view(-1, N_KV, HEAD_DIM).transpose(0, 1).repeat_interleave(N_HEADS // N_KV, 0)
    att = torch.softmax(q @ k.transpose(-1, -2) / (HEAD_DIM ** 0.5), dim=-1)
    return (att @ v).transpose(0, 1).reshape(S, D)


def measure(S, ctx, iters, warmup, dtype):
    """S = query length; ctx = key/value length (== S for prefill, > S for decode)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(S, D, generator=g, dtype=dtype)
    gamma = torch.ones(D, dtype=dtype)
    q = torch.randn(S, N_HEADS * HEAD_DIM, generator=g, dtype=dtype)
    k = torch.randn(ctx, N_KV * HEAD_DIM, generator=g, dtype=dtype)
    v = torch.randn(ctx, N_KV * HEAD_DIM, generator=g, dtype=dtype)
    inter = torch.randn(S, FFN, generator=g, dtype=dtype)
    up = torch.randn(S, FFN, generator=g, dtype=dtype)

    def _attn():
        qh = q.view(S, N_HEADS, HEAD_DIM).transpose(0, 1)
        kh = k.view(ctx, N_KV, HEAD_DIM).transpose(0, 1).repeat_interleave(N_HEADS // N_KV, 0)
        vh = v.view(ctx, N_KV, HEAD_DIM).transpose(0, 1).repeat_interleave(N_HEADS // N_KV, 0)
        att = torch.softmax(qh @ kh.transpose(-1, -2) / (HEAD_DIM ** 0.5), dim=-1)
        return (att @ vh).transpose(0, 1).reshape(S, D)

    t_rms = _time(lambda: rmsnorm(x, gamma), iters, warmup)
    t_attn = _time(_attn, iters, warmup)
    t_silu = _time(lambda: torch.nn.functional.silu(inter) * up, iters, warmup)

    per_layer_obf = 2 * t_rms + t_attn + t_silu           # 2 norms + softmax + SiLU
    per_layer_ours = t_rms + t_attn                        # layer-0 attention block only
    return {
        "seq_q": S, "seq_ctx": ctx,
        "rmsnorm_ms": t_rms, "attention_ms": t_attn, "silu_ms": t_silu,
        "obfuscatune_tee_per_layer_ms": per_layer_obf,
        "obfuscatune_tee_total_ms": per_layer_obf * N_LAYERS,
        "ours_k1_tee_total_ms": per_layer_ours,
        "tee_compute_ratio_obf_over_ours": round(per_layer_obf * N_LAYERS / per_layer_ours, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--decode-ctx", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16

    is_tdx = False
    try:
        with open("/proc/cpuinfo") as f:
            is_tdx = "tdx_guest" in f.read()
    except Exception:
        pass

    res = {
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "tdx_guest_detected": is_tdx,
        "torch_threads": torch.get_num_threads(),
        "dtype": a.dtype,
        "dims": {"D": D, "FFN": FFN, "layers": N_LAYERS,
                 "heads": N_HEADS, "kv_heads": N_KV, "head_dim": HEAD_DIM},
        "prefill": {},
        "decode": {},
    }
    print(f"[tdx] host={res['host']} tdx_guest={is_tdx} threads={res['torch_threads']} "
          f"dtype={a.dtype}", flush=True)
    for S in (512, 1024):
        r = measure(S, S, a.iters, a.warmup, dtype)
        res["prefill"][str(S)] = r
        print(f"  prefill S={S}: rms={r['rmsnorm_ms']:.3f} attn={r['attention_ms']:.3f} "
              f"silu={r['silu_ms']:.3f} | ObfTEE_total={r['obfuscatune_tee_total_ms']:.1f}ms "
              f"ours_k1={r['ours_k1_tee_total_ms']:.3f}ms "
              f"ratio={r['tee_compute_ratio_obf_over_ours']}x", flush=True)
    rd = measure(1, a.decode_ctx, max(a.iters, 50), a.warmup, dtype)
    res["decode"] = rd
    print(f"  decode  S=1 ctx={a.decode_ctx}: ObfTEE_total={rd['obfuscatune_tee_total_ms']:.2f}ms "
          f"ours_k1={rd['ours_k1_tee_total_ms']:.3f}ms "
          f"ratio={rd['tee_compute_ratio_obf_over_ours']}x", flush=True)

    if not is_tdx:
        res["WARNING"] = ("tdx_guest NOT detected in /proc/cpuinfo -- this is a "
                          "plain-CPU proxy, not a real TDX enclave measurement.")
        print("  [WARN] tdx_guest not detected; numbers are a plain-CPU proxy.", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[tdx] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
