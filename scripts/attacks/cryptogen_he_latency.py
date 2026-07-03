#!/usr/bin/env python
"""Real homomorphic-encryption latency for CryptoGen's kernels (BFV, SEAL).

CryptoGen (arXiv:2602.08798) runs Transformer linear layers as HE CT x PT and
attention as HE CT x CT, over a BFV context with poly_modulus_degree n=8192.
This measures the REAL wall-clock of those ciphertext operations using Microsoft
SEAL (via TenSEAL) and composes a GPT-2 decoder-block latency from CryptoGen's
own operation counts, to validate the paper's "seconds per block" (Table IV).

Measured (real HE, averaged): ct x pt multiply, ct x ct multiply, folding-sum
reduction (.sum, log-depth Galois rotations), and dot (inner product) at n=8192.
Composed GPT-2 block (d1=768, ffn=3072, 12 heads) per decode step:
  - linear layers Q/K/V/O + FFN1 + FFN2 as inner-product CT x PT dots;
  - attention over L cached keys as CT x CT score + aggregation.
Also runs a real end-to-end 64-output CT x PT matvec to validate the composition.

The MPC non-linear cost (EzPC GELU/LayerNorm/Softmax) is NOT executed here (it
needs a 2-party MPC network); the paper's Table IV shows it is a minority of the
block time, dominated by the CT x PT / CT x CT costs measured here.
"""
from __future__ import annotations

import argparse
import json
import time

import tenseal as ts


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192, help="poly_modulus_degree")
    ap.add_argument("--d1", type=int, default=768)
    ap.add_argument("--ffn", type=int, default=3072)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t_setup0 = time.perf_counter()
    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=a.n,
                     plain_modulus=1032193)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    setup_s = time.perf_counter() - t_setup0

    # full-width operands for the (reduction-free) element-wise multiplies
    full = [1] * a.n
    enc_full_a = ts.bfv_vector(ctx, full)
    enc_full_b = ts.bfv_vector(ctx, full)
    # linear inner-product operands (hidden dim d1); small values avoid BFV overflow
    lin_vec = [1] * a.d1
    enc_lin = ts.bfv_vector(ctx, lin_vec)
    # attention score operands (head dim d2)
    d2 = a.d1 // a.heads
    enc_q = ts.bfv_vector(ctx, [1] * d2)
    enc_k = ts.bfv_vector(ctx, [1] * d2)

    # real per-op HE costs
    t_ctpt = _time(lambda: enc_full_a * full, a.iters, a.warmup)          # ct x pt elementwise
    t_ctct = _time(lambda: enc_full_a * enc_full_b, a.iters, a.warmup)    # ct x ct elementwise
    t_sum = _time(lambda: enc_lin.sum(), a.iters, a.warmup)               # folding-sum over d1
    t_dot = _time(lambda: enc_lin.dot(lin_vec), max(a.iters // 2, 5), a.warmup)   # linear inner product
    t_ctct_score = _time(lambda: (enc_q * enc_k).sum(), max(a.iters // 2, 5), a.warmup)  # attention score

    print(f"[he] n={a.n} setup={setup_s:.2f}s | ctpt={t_ctpt*1e3:.3f}ms "
          f"ctct={t_ctct*1e3:.3f}ms sum={t_sum*1e3:.3f}ms dot={t_dot*1e3:.3f}ms "
          f"ctct_score={t_ctct_score*1e3:.3f}ms", flush=True)

    # compose a GPT-2 decode block from CryptoGen op structure, two ways.
    import math
    # (1) naive inner-product HE matvec: one dot (log-depth reduction) per output.
    #     This is the *unoptimized* HE matvec = an upper bound.
    lin_dots = 4 * a.d1 + a.ffn + a.d1        # Q,K,V,O + FFN1(d1->ffn) + FFN2(ffn->d1)
    lin_naive_s = lin_dots * t_dot
    # (2) CryptoGen's diagonal encoding: a dxd matvec = d ct-pt mults + d rotations
    #     (NOT d reductions). Derive a single-rotation cost from the folding-sum
    #     (folding-sum over d1 = ceil(log2 d1) rotate+add steps).
    t_rot = t_sum / max(1, math.ceil(math.log2(a.d1)))
    def diag_matvec_s(rows, cols):
        return cols * (t_ctpt + t_rot)         # cols ct-pt mults + cols rotations
    lin_diag_s = (4 * diag_matvec_s(a.d1, a.d1)          # Q,K,V,O
                  + diag_matvec_s(a.d1, a.ffn)           # FFN1
                  + diag_matvec_s(a.ffn, a.d1))          # FFN2
    block = {}
    for L in (64, 128, 256, 512):
        attn_s = 2 * L * t_ctct_score          # qK^T scores + aggregation over L keys
        block[str(L)] = {
            "linear_naive_innerproduct_s": round(lin_naive_s, 3),
            "linear_diagonal_encoding_s": round(lin_diag_s, 3),
            "attention_ctct_s": round(attn_s, 3),
            "block_total_diagonal_s": round(lin_diag_s + attn_s, 3),
            "block_total_naive_s": round(lin_naive_s + attn_s, 3),
            "paper_table_iv_cryptogen_s": {"64": 9.6, "128": 16.45,
                                           "256": 33.84, "512": 65.55}[str(L)],
        }
        print(f"  L={L:>4}: diag_linear={lin_diag_s:.1f}s attn={attn_s:.1f}s "
              f"block(diag)={lin_diag_s+attn_s:.1f}s [naive {lin_naive_s+attn_s:.0f}s] "
              f"(paper {block[str(L)]['paper_table_iv_cryptogen_s']}s)", flush=True)

    # real end-to-end validation: a 64-output CT x PT matvec (inner-product form)
    t_e2e0 = time.perf_counter()
    outs = [enc_lin.dot(lin_vec) for _ in range(64)]
    e2e_s = time.perf_counter() - t_e2e0
    print(f"  [validate] real 64-output CTxPT matvec = {e2e_s:.2f}s "
          f"(= {e2e_s/64*1e3:.3f}ms/output vs t_dot {t_dot*1e3:.3f}ms)", flush=True)

    result = {
        "library": "tenseal(SEAL) BFV",
        "n_poly_modulus_degree": a.n,
        "setup_seconds": setup_s,
        "per_op_ms": {"ctpt_mul": t_ctpt * 1e3, "ctct_mul": t_ctct * 1e3,
                      "sum_reduce": t_sum * 1e3, "dot": t_dot * 1e3,
                      "ctct_score": t_ctct_score * 1e3,
                      "single_rotation_derived": t_rot * 1e3},
        "gpt2_block": {"dims": {"d1": a.d1, "ffn": a.ffn, "heads": a.heads},
                       "linear_inner_product_dots": lin_dots, "by_seq_len": block},
        "e2e_validation_64out_matvec_s": e2e_s,
        "note": ("Linear/attention HE costs measured with real SEAL ciphertext ops;"
                 " MPC nonlinear (EzPC) not executed (minority of block time per Table IV)."),
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[he] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
