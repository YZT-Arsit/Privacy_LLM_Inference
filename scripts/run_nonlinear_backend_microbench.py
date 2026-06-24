"""Microbench for selectable nonlinear backends (Line A vs Line B).

Runs GELU / SiLU / Softmax / LayerNorm / RMSNorm through the ``current`` and/or
``amulet_migrated`` backend on identical inputs and reports per-op correctness
(vs a float64 reference) + efficiency + trust placement.

CLI::

    python scripts/run_nonlinear_backend_microbench.py \\
        --nonlinear-backend current|amulet_migrated|both \\
        --shapes 32x128,16x512,8x2048 --dtypes float32,float16 --iters 30

Reported per row: op_name, backend, input_shape, dtype, max_abs_error,
mean_abs_error, relative_l2_error, cosine_similarity, top1_match_rate (softmax),
trusted_calls, trusted_bytes, gpu_bytes, latency_ms, tee_used_on_gpu.

SECURITY: this measures correctness + efficiency only. The Amulet migration's
security is NOT formally claimed (status ``not_formally_claimed`` /
``under_discussion``). No security claim is made for either backend here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.nonlinear.backends import (  # noqa: E402
    OP_NAMES,
    reference_gelu,
    reference_layernorm,
    reference_rmsnorm,
    reference_silu,
    reference_softmax,
)
from pllo.nonlinear.registry import (  # noqa: E402
    available_backends,
    backend_security_claim_status,
    backend_security_status,
    make_nonlinear_backend,
)

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32, "float64": torch.float64}


def _parse_shapes(s: str) -> list[tuple[int, ...]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(tuple(int(d) for d in part.lower().split("x")))
    return out


def _reference(op, x, weight, bias, eps):
    if op == "gelu":
        return reference_gelu(x)
    if op == "silu":
        return reference_silu(x)
    if op == "softmax":
        return reference_softmax(x, dim=-1)
    if op == "layernorm":
        return reference_layernorm(x, weight, bias, eps)
    if op == "rmsnorm":
        return reference_rmsnorm(x, weight, eps)
    raise ValueError(op)


def _errors(out: torch.Tensor, ref64: torch.Tensor) -> dict:
    a = out.detach().to(torch.float64).reshape(-1)
    b = ref64.detach().to(torch.float64).reshape(-1)
    diff = a - b
    denom = float(torch.linalg.norm(b)) or 1.0
    cos_d = float(torch.linalg.norm(a) * torch.linalg.norm(b)) or 1.0
    return {
        "max_abs_error": float(diff.abs().max()),
        "mean_abs_error": float(diff.abs().mean()),
        "relative_l2_error": float(torch.linalg.norm(diff) / denom),
        "cosine_similarity": float((a @ b) / cos_d),
    }


def _bench_op(backend, op, x, weight, bias, eps, iters):
    kw = {}
    if op in ("layernorm", "rmsnorm"):
        kw = {"weight": weight, "eps": eps}
        if op == "layernorm":
            kw["bias"] = bias
    res = backend.run(op, x, **kw)
    # latency: average wall time over `iters` (after a small warmup)
    for _ in range(2):
        backend.run(op, x, **kw)
    t0 = time.perf_counter()
    for _ in range(iters):
        backend.run(op, x, **kw)
    latency_ms = (time.perf_counter() - t0) / iters * 1e3
    return res, latency_ms


def run(backend_names, shapes, dtypes, iters, seed, lift_k):
    rows = []
    for bname in backend_names:
        backend = make_nonlinear_backend(
            bname, **({"lift_k": lift_k, "seed": seed}
                      if bname == "amulet_migrated" else {}))
        for op in OP_NAMES:
            for shape in shapes:
                for dname in dtypes:
                    dt = _DTYPES[dname]
                    g = torch.Generator().manual_seed(seed)
                    x = torch.randn(*shape, generator=g).to(dt)
                    feat = shape[-1]
                    weight = torch.randn(feat, generator=g).to(dt)
                    bias = torch.randn(feat, generator=g).to(dt)
                    eps = 1e-5 if op == "layernorm" else 1e-6
                    res, latency_ms = _bench_op(
                        backend, op, x, weight, bias, eps, iters)
                    ref64 = _reference(op, x, weight, bias, eps)
                    err = _errors(res.output, ref64)
                    if op == "softmax":
                        t1 = float((res.output.to(torch.float64).argmax(-1) ==
                                    ref64.argmax(-1)).float().mean())
                    else:
                        t1 = None
                    rows.append({
                        "op_name": op, "backend": bname,
                        "input_shape": "x".join(str(d) for d in shape),
                        "dtype": dname,
                        "max_abs_error": err["max_abs_error"],
                        "mean_abs_error": err["mean_abs_error"],
                        "relative_l2_error": err["relative_l2_error"],
                        "cosine_similarity": err["cosine_similarity"],
                        "top1_match_rate": t1,
                        "trusted_calls": res.trusted_calls,
                        "trusted_bytes": res.trusted_bytes,
                        "gpu_bytes": res.gpu_bytes,
                        "latency_ms": latency_ms,
                        "tee_used_on_gpu": res.tee_used_on_gpu,
                        "security_status": backend.security_status,
                        "security_claim_status": backend.security_claim_status,
                    })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_md(path: Path, rows, sec, claim) -> None:
    L = ["# Nonlinear backend microbench (Line A vs Line B)", "",
         "Security claim status (NOT a proof; Amulet under discussion):",
         f"- current: status=`{sec['current']}` "
         f"claim=`{claim['current']}`",
         f"- amulet_migrated: status=`{sec['amulet_migrated']}` "
         f"claim=`{claim['amulet_migrated']}` "
         "(security **under_discussion / not proven**)", "",
         "| op | backend | shape | dtype | max_abs | rel_l2 | cosine | top1 | "
         "trusted_calls | trusted_bytes | gpu_bytes | latency_ms | tee_gpu |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t1 = "n/a" if r["top1_match_rate"] is None else f"{r['top1_match_rate']:.3f}"
        L.append(
            f"| {r['op_name']} | {r['backend']} | {r['input_shape']} | "
            f"{r['dtype']} | {r['max_abs_error']:.2e} | "
            f"{r['relative_l2_error']:.2e} | {r['cosine_similarity']:.5f} | "
            f"{t1} | {r['trusted_calls']} | {r['trusted_bytes']} | "
            f"{r['gpu_bytes']} | {r['latency_ms']:.4f} | {r['tee_used_on_gpu']} |")
    L += ["", "_current = nonlinear in the trusted boundary (trusted_calls>=1, "
          "gpu_bytes=0). amulet_migrated = nonlinear migrated to the untrusted "
          "accelerator (activations trusted_calls=0, gpu_bytes>0; norms/softmax "
          "keep a small trusted reduction shortcut). tee_used_on_gpu=False for "
          "both. Amulet security is not formally claimed._"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nonlinear-backend", default="both",
                    choices=["current", "amulet_migrated", "both"])
    ap.add_argument("--shapes", default="32x128,16x512,8x2048")
    ap.add_argument("--dtypes", default="float32,float16")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lift-k", type=int, default=2)
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()

    names = (available_backends() if args.nonlinear_backend == "both"
             else [args.nonlinear_backend])
    shapes = _parse_shapes(args.shapes)
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    rows = run(names, shapes, dtypes, args.iters, args.seed, args.lift_k)
    sec = backend_security_status()
    claim = backend_security_claim_status()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"stage": "nonlinear_backend_microbench",
               "backends": names, "security_status": sec,
               "security_claim_status": claim,
               "amulet_security_claim": "under_discussion (not proven)",
               "tee_used_on_gpu": False, "rows": rows}
    (out / "nonlinear_backend_microbench.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_csv(out / "nonlinear_backend_microbench.csv", rows)
    _write_md(out / "nonlinear_backend_microbench.md", rows, sec, claim)

    print("=== nonlinear backend microbench ===")
    print(f"backends={names} shapes={shapes} dtypes={dtypes} iters={args.iters}")
    print(f"security_status={sec}")
    print(f"security_claim_status={claim}")
    print(f"amulet_security_claim=under_discussion (not proven)")
    hdr = (f"{'op':10} {'backend':16} {'shape':10} {'dtype':8} {'max_abs':>10} "
           f"{'cos':>9} {'tcalls':>6} {'tbytes':>9} {'gpu_bytes':>10} {'ms':>8}")
    print(hdr)
    for r in rows:
        print(f"{r['op_name']:10} {r['backend']:16} {r['input_shape']:10} "
              f"{r['dtype']:8} {r['max_abs_error']:10.2e} "
              f"{r['cosine_similarity']:9.5f} {r['trusted_calls']:6d} "
              f"{r['trusted_bytes']:9d} {r['gpu_bytes']:10d} {r['latency_ms']:8.4f}")
    print(f"\nwrote {out}/nonlinear_backend_microbench.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
