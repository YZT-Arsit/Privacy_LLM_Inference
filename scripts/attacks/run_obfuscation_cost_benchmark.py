"""Obfuscation-cost microbenchmark at Qwen-7B dims: the security-vs-latency tradeoff.

The existing crossing-based harness (obfuscatune_latency.py) shows ObfuscaTune's
224 TEE crossings/layer vs ours' 2. It does NOT capture the OTHER cost axis: the
per-token MASK-GENERATION tax that the KPA-resistant fresh variants pay. This
isolates, at D=hidden (Qwen-7B 3584), the marginal obfuscation work each scheme
adds — mask setup (one-time) and per-token recurring (sample + apply) — on the
same box (GPU + a CPU-TEE proxy), so the numbers are internally comparable.

Schemes (residual-stream mask + weight fold):
  plaintext            no mask                                   (floor)
  stip_qwen            OFFLINE feature permutation baked into weights; runtime
                       relabel only (~free), NO TEE
  obfuscatune_orth     FIXED orthogonal R sampled ONCE; per-token apply H@R
  ours_signed_perm     FIXED signed-perm sampled once; per-token index+sign
  ours_fresh_pad       FRESH orthogonal per token (QR O(D^3)) + apply  <-- KPA tax
  ours_non_isometric   FRESH dense well-cond per token (2 QR) + apply + inverse

Security context (this round, measured on real Qwen2.5-7B): KPA breaks the fixed
masks (success 1.0), fresh masks resist (0.0); Gram/ArrowMatch break the
permutation masks (stip, signed-perm), the fresh/dense masks resist.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=3584, help="hidden size (Qwen2.5-7B=3584)")
    p.add_argument("--batch", type=int, default=1, help="tokens per step (1=decode)")
    p.add_argument("--prefill", type=int, default=512, help="a second batch size (prefill)")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cond", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "baselines" / "obf_cost"
    if args.dry_run:
        print(f"[dry-run] obf-cost D={args.dim} batches=[{args.batch},{args.prefill}] -> {out}")
        return

    import torch
    from pllo.baselines.obfuscatune.random_matrices import (matrix_with_condition_number,
                                                            orthogonal_matrix)

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    D = args.dim
    g = torch.Generator().manual_seed(args.seed)

    def _sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def _time(fn, repeats, warmup):
        import time
        for _ in range(warmup):
            fn()
        _sync()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            _sync()
            ts.append((time.perf_counter() - t0) * 1e3)   # ms
        return statistics.median(ts)

    # pre-sampled static masks (one-time setup)
    R_fixed, _ = orthogonal_matrix(D, seed=1, dtype=dtype, device=dev)
    perm_fixed = torch.randperm(D, generator=g).to(dev)
    signs_fixed = torch.where(torch.rand(D, generator=g) < 0.5,
                              torch.tensor(-1.0), torch.tensor(1.0)).to(dev).to(dtype)

    def per_token_ops(B):
        H = torch.randn(B, D, device=dev, dtype=dtype)

        def plaintext():
            return H.clone()

        def stip():                       # runtime relabel (baked into weights offline)
            return H[:, perm_fixed]

        def obfuscatune():                # fixed R apply
            return H @ R_fixed

        def signed_perm():                # fixed signed-perm apply
            return H[:, perm_fixed] * signs_fixed

        def fresh_signed_perm():          # FRESH signed-perm per step: O(D), no QR
            perm = torch.randperm(D, generator=g).to(dev)
            sg = torch.where(torch.rand(D, generator=g) < 0.5,
                             torch.tensor(-1.0), torch.tensor(1.0)).to(dev).to(dtype)
            return H[:, perm] * sg

        def fresh_pad():                  # FRESH orthogonal per token-step (QR) + apply
            Q, _ = orthogonal_matrix(D, seed=int(torch.randint(1, 10**8, (1,)).item()),
                                     dtype=dtype, device=dev)
            return H @ Q

        def non_isometric():              # FRESH dense well-cond (2 QR) + apply + inverse fold
            M, M_inv = matrix_with_condition_number(
                D, cond=args.cond, seed=int(torch.randint(1, 10**8, (1,)).item()),
                dtype=dtype, device=dev)
            y = H @ M
            _ = M_inv                     # inverse needed to fold into next linear
            return y

        return {"plaintext": plaintext, "stip_qwen": stip,
                "obfuscatune_orth": obfuscatune, "ours_signed_perm": signed_perm,
                "ours_fresh_signed_perm": fresh_signed_perm,
                "ours_fresh_pad": fresh_pad, "ours_non_isometric": non_isometric}

    # one-time setup cost (sampling a D x D mask once)
    setup = {
        "plaintext": 0.0, "stip_qwen": 0.0,
        "obfuscatune_orth": _time(lambda: orthogonal_matrix(D, seed=2, dtype=dtype, device=dev),
                                  max(3, args.repeats // 5), 1),
        "ours_signed_perm": 0.0, "ours_fresh_signed_perm": 0.0,
        "ours_fresh_pad": 0.0, "ours_non_isometric": 0.0,
    }

    report = {"gpu_or_cpu": dev, "dtype": args.dtype, "dim": D,
              "security_note": "fixed masks (obfuscatune/signed_perm/stip) broken by KPA/Gram; "
                               "fresh masks (fresh_pad/non_isometric) resist KPA + Gram (this round).",
              "one_time_setup_ms": setup, "per_token_step": {}}
    for label, B in (("decode_B1", args.batch), (f"prefill_B{args.prefill}", args.prefill)):
        ops = per_token_ops(B)
        row = {}
        base = None
        for name, fn in ops.items():
            ms = _time(fn, args.repeats, args.warmup)
            row[name] = {"latency_ms_per_step": ms, "us_per_token": ms * 1e3 / B,
                         "throughput_tok_per_s": B / (ms / 1e3) if ms > 0 else None}
            if name == "plaintext":
                base = ms
        for name in row:
            row[name]["overhead_vs_plaintext"] = (row[name]["latency_ms_per_step"] / base
                                                  if base and base > 0 else None)
        report["per_token_step"][label] = row
        print(f"--- {label} (D={D}, {dev}, {args.dtype}) ---")
        for name, v in row.items():
            print(f"  {name:20s} {v['latency_ms_per_step']:9.4f} ms/step  "
                  f"{v['throughput_tok_per_s']:12.1f} tok/s  x{v['overhead_vs_plaintext']:.2f}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "obfuscation_cost_benchmark.json").write_text(json.dumps(report, indent=2))
    # markdown
    md = [f"# Obfuscation-cost benchmark (D={D}, {dev}, {args.dtype})", "",
          f"> {report['security_note']}", "",
          f"One-time setup (sample D×D mask): obfuscatune={setup['obfuscatune_orth']:.3f} ms; "
          "fresh variants sample per-token (see below); stip/signed-perm/plaintext ~0.", ""]
    for label, row in report["per_token_step"].items():
        md += [f"## {label}", "",
               "| scheme | ms/step | tok/s | ×plaintext | TEE | KPA | Gram |",
               "|---|---|---|---|---|---|---|"]
        sec = {"plaintext": ("none", "n/a", "n/a"), "stip_qwen": ("none", "broken", "broken"),
               "obfuscatune_orth": ("224 cross/layer", "broken", "resist"),
               "ours_signed_perm": ("2 cross", "broken", "broken"),
               "ours_fresh_signed_perm": ("per-boundary mask", "RESIST", "RESIST"),
               "ours_fresh_pad": ("2 cross", "RESIST", "RESIST"),
               "ours_non_isometric": ("2 cross", "RESIST", "RESIST")}
        for name, v in row.items():
            tee, kpa, gram = sec[name]
            md.append(f"| {name} | {v['latency_ms_per_step']:.4f} | {v['throughput_tok_per_s']:.0f} "
                      f"| {v['overhead_vs_plaintext']:.2f} | {tee} | {kpa} | {gram} |")
        md.append("")
    (out / "obfuscation_cost_benchmark.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/obfuscation_cost_benchmark.json/.md")


if __name__ == "__main__":
    main()
