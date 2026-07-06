"""Realistic hybrid throughput: Qwen on GPU + TDX mask ops on CPU.

The deployment splits work: the heavy Qwen-7B forward runs on the GPU, and only
the trusted mask operations run in the CPU TDX enclave. So the honest per-scheme
overhead is the CPU-TEE mask cost measured AGAINST a real GPU decode step — not
mask-op-vs-mask-op.

Measures:
  * base_gpu_decode_ms  — a real Qwen-7B single-token decode on the GPU (with a
    populated KV cache), the shared cost every scheme pays.
  * per-scheme CPU-TEE mask cost per token (mask generation in the enclave):
      static (signed-perm / ObfuscaTune fixed R): sampled once -> ~0 amortized
      fresh_signed_perm: O(D) randperm+signs each token
      fresh_pad:         O(D^3) orthogonal QR each token
      non_isometric:     2x O(D^3) QR each token
  * ObfuscaTune's 224 in-TEE nonlinearities/token: a CPU proxy (RMSNorm/softmax/
    SiLU on the decode token across 28 layers) — its dominant cost.
Then end_to_end_ms = base_gpu_decode + cpu_tee_overhead, and tokens/s.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _median(fn, repeats, warmup, sync=None):
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cond", type=float, default=5.0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "baselines" / "e2e_tput"
    if args.dry_run:
        print(f"[dry-run] e2e throughput model={args.model_name_or_path} ctx={args.context} -> {out}")
        return

    import torch
    from pllo.baselines.obfuscatune.qwen_config import load_qwen
    from pllo.baselines.obfuscatune.random_matrices import (matrix_with_condition_number,
                                                            orthogonal_matrix)
    assert torch.cuda.is_available(), "need a GPU for the Qwen side"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = load_qwen(model_name_or_path=args.model_name_or_path, seed=0, dtype=dtype).to("cuda").eval()
    D = model.config.hidden_size

    # --- real GPU decode step (populate KV cache, then time 1-token decode) ---
    ids = torch.randint(0, model.config.vocab_size, (1, args.context), device="cuda")
    with torch.no_grad():
        out0 = model(ids, use_cache=True)
        past = out0.past_key_values
        nxt = out0.logits[:, -1:].argmax(-1)

    def decode_step():
        with torch.no_grad():
            o = model(nxt, past_key_values=past, use_cache=True)
        return o.logits

    base = _median(decode_step, args.repeats, args.warmup, sync=torch.cuda.synchronize)

    # --- CPU-TEE mask generation cost per token ---
    def gen_signed_perm():
        perm = torch.randperm(D)
        _ = torch.where(torch.rand(D) < 0.5, torch.tensor(-1.0), torch.tensor(1.0))
        return perm

    def gen_fresh_pad():
        return orthogonal_matrix(D, seed=int(torch.randint(1, 10**8, (1,)).item()),
                                 dtype=torch.float32, device="cpu")

    def gen_non_iso():
        return matrix_with_condition_number(D, cond=args.cond,
                                            seed=int(torch.randint(1, 10**8, (1,)).item()),
                                            dtype=torch.float32, device="cpu")

    g_fresh_sp = _median(gen_signed_perm, args.repeats, args.warmup)
    g_fresh_pad = _median(gen_fresh_pad, max(4, args.repeats // 3), 1)
    g_non_iso = _median(gen_non_iso, max(4, args.repeats // 3), 1)

    # --- ObfuscaTune 224 in-TEE nonlinearities/token (CPU proxy) ---
    nl_reps = model.config.num_hidden_layers
    xh = torch.randn(1, D, dtype=torch.float32)
    xf = torch.randn(1, model.config.intermediate_size, dtype=torch.float32)
    att = torch.randn(model.config.num_attention_heads, 1, args.context, dtype=torch.float32)

    def obf_tee_nonlin():
        for _ in range(nl_reps):
            _ = torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + 1e-6) * xh   # RMSNorm x1
            _ = torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + 1e-6) * xh   # RMSNorm x2
            _ = torch.softmax(att, dim=-1)                                  # softmax
            _ = torch.nn.functional.silu(xf)                                # SiLU
    obf_nonlin = _median(obf_tee_nonlin, max(4, args.repeats // 3), 1)

    # static masks: sampled once -> ~0 amortized per token
    schemes = {
        "plaintext":              {"cpu_tee_ms": 0.0,            "tee": "none",   "kpa": "-",     "gram": "-"},
        "stip_qwen":              {"cpu_tee_ms": 0.0,            "tee": "none",   "kpa": "broken","gram": "broken"},
        "obfuscatune_orth":       {"cpu_tee_ms": obf_nonlin,     "tee": "224x",   "kpa": "broken","gram": "resist"},
        "ours_signed_perm":       {"cpu_tee_ms": 0.0,            "tee": "2x",     "kpa": "broken","gram": "broken"},
        "ours_fresh_signed_perm": {"cpu_tee_ms": g_fresh_sp,     "tee": "2x",     "kpa": "RESIST","gram": "RESIST"},
        "ours_fresh_pad":         {"cpu_tee_ms": g_fresh_pad,    "tee": "2x",     "kpa": "RESIST","gram": "RESIST"},
        "ours_non_isometric":     {"cpu_tee_ms": g_non_iso,      "tee": "2x",     "kpa": "RESIST","gram": "RESIST"},
    }
    for name, s in schemes.items():
        e2e = base + s["cpu_tee_ms"]
        s["end_to_end_ms_per_token"] = e2e
        s["tokens_per_s"] = 1e3 / e2e
        s["overhead_pct_vs_plaintext"] = 100.0 * (e2e - base) / base

    report = {"model": args.model_name_or_path, "gpu": "cuda", "dtype": args.dtype,
              "hidden": D, "context": args.context,
              "base_gpu_decode_ms_per_token": base,
              "note": "Qwen forward on GPU; TDX mask ops on CPU. Static-mask schemes add ~0 "
                      "(mask folded/ sampled once); fresh-dense masks are dominated by the CPU QR; "
                      "ObfuscaTune is dominated by 224 in-TEE nonlinearities/token.",
              "schemes": schemes}
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2e_throughput.json").write_text(json.dumps(report, indent=2))
    md = [f"# Realistic hybrid throughput (Qwen on GPU + CPU TDX), {args.model_name_or_path}",
          "", f"base GPU decode = **{base:.2f} ms/token** ({1e3/base:.1f} tok/s), ctx={args.context}, {args.dtype}",
          "", "| scheme | CPU-TEE ms/tok | end-to-end ms/tok | tok/s | overhead | TEE cross | KPA | Gram |",
          "|---|---|---|---|---|---|---|---|"]
    for name, s in schemes.items():
        md.append(f"| {name} | {s['cpu_tee_ms']:.3f} | {s['end_to_end_ms_per_token']:.2f} | "
                  f"{s['tokens_per_s']:.1f} | {s['overhead_pct_vs_plaintext']:.1f}% | {s['tee']} | "
                  f"{s['kpa']} | {s['gram']} |")
    (out / "e2e_throughput.md").write_text("\n".join(md) + "\n")
    print(f"base GPU decode = {base:.2f} ms/token ({1e3/base:.1f} tok/s)")
    for name, s in schemes.items():
        print(f"  {name:24s} +{s['cpu_tee_ms']:9.3f} ms  = {s['end_to_end_ms_per_token']:9.2f} ms/tok  "
              f"{s['tokens_per_s']:8.1f} tok/s  (+{s['overhead_pct_vs_plaintext']:.1f}%)")
    print(f"wrote {out}/e2e_throughput.json/.md")


if __name__ == "__main__":
    main()
