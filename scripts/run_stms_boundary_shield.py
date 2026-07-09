#!/usr/bin/env python
"""Run the Variant F (STMS boundary shield) design + verification and emit results.

Writes under results/attacks/stms_boundary_shield/:
    correctness.{json,md}       roundtrip + GPT-2 prefill integration
    forward_oracle.{json,md}    candidate / sensitive-token attacks per case
    bss_ica.{json,md}           ICA p95 cosine + shield / block / cond sweeps
    efficiency.{json,md}
    summary.md

Runs offline; uses a locally-cached GPT-2 as the public forward oracle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from pllo.experiments.public_weight_prompt_audit import Gpt2Oracle, qr_orthogonal  # noqa: E402
from pllo.experiments.stms_boundary_shield import (  # noqa: E402
    candidate_attack, efficiency, ica_bss_recovery, sample_shield_rows,
    stms_forward, stms_recover, stms_report_fields, token_mix_blocklocal,
    token_mix_nonorthogonal, token_mix_orthogonal,
)

OUT = Path(__file__).resolve().parents[1] / "results/attacks/stms_boundary_shield"


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(_clean(obj), indent=2))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return o


# ---------------------------------------------------------------------------


def run_correctness(oracle):
    d = oracle.d
    n = qr_orthogonal(d, seed=0)
    h = oracle.hidden("The password is gold and it is secret", 6)
    m = h.shape[0]
    cases = {}
    for name, a, shield in [
        ("baseline", None, None),
        ("orthogonal", token_mix_orthogonal(m, seed=1), None),
        ("nonorthogonal", token_mix_nonorthogonal(m, seed=1, cond=5.0), None),
        ("blocklocal", token_mix_blocklocal(m, block=4, seed=1), None),
        ("orthogonal_shield", token_mix_orthogonal(m + 3, seed=1),
         sample_shield_rows(3, d, scale=5.0, seed=2)),
    ]:
        fwd = stms_forward(h, n, a, shield=shield)
        rec = stms_recover(fwd["u"], a, num_real_tokens=m)
        err = float((rec - fwd["hn"]).abs().max())
        cases[name] = {"roundtrip_max_abs": err, "exact": err <= 1e-9,
                       "shield_rows_dropped": fwd["num_shield_rows"]}

    # prefill integration: shield H N at a boundary, recover, confirm the state fed
    # onward is bit-identical -> downstream logits unchanged; shield never enters KV.
    a = token_mix_orthogonal(m + 2, seed=9)
    shield = sample_shield_rows(2, d, scale=8.0, seed=3)
    fwd = stms_forward(h, n, a, shield=shield)
    rec = stms_recover(fwd["u"], a, num_real_tokens=m)
    integ = {
        "recovered_equals_HN_max_abs": float((rec - fwd["hn"]).abs().max()),
        "shield_rows_in_recovered_state": int(rec.shape[0] - m),   # must be 0
        "downstream_identical": bool((rec - fwd["hn"]).abs().max() <= 1e-9),
        "note": ("recovery is bit-exact (fp64), so the state fed into "
                 "attention/RMSNorm/RoPE/KV is identical to the un-shielded H N; "
                 "shield rows are dropped before compute and never enter KV/logits."),
    }
    fields = stms_report_fields(
        roundtrip_max_abs=max(c["roundtrip_max_abs"] for c in cases.values()),
        exact_lossless_tests_passed=all(c["exact"] for c in cases.values()))
    return {"cases": cases, "prefill_integration": integ, "report_fields": fields}


def _same_len(oracle, template, fillers):
    out, base = [], None
    for f in fillers:
        p = template.format(f)
        L = oracle.ids(p).shape[0]
        if base is None:
            base = L
        if L == base:
            out.append(p)
    return out


def run_forward_oracle(oracle):
    n = qr_orthogonal(oracle.d, seed=0)
    hof = lambda p: oracle.hidden(p, 6)  # noqa: E731
    words = [" gold", " fire", " moon", " tree", " book", " road", " lamp", " star",
             " snow", " rain", " wind", " leaf", " king", " wolf", " ship", " song"]
    cand = _same_len(oracle, "My secret word is{}", words)
    templates = ["The password is{}", "The diagnosis is{}", "My phone code is{}"]

    results = {"candidate_set": {}, "sensitive_token": {}}
    for case in ["baseline", "orthogonal", "nonorthogonal", "blocklocal"]:
        r = candidate_attack(hof, cand, 3, case=case, n=n, seed=1,
                             cond=8.0, block=4)
        results["candidate_set"][case] = r
        # sensitive-token top1 across templates
        hits = []
        for ti, tmpl in enumerate(templates):
            cc = _same_len(oracle, tmpl, words)
            if len(cc) < 4:
                continue
            rr = candidate_attack(hof, cc, 2 % len(cc), case=case, n=n,
                                  seed=10 + ti, cond=8.0, block=4)
            hits.append(rr["top1"])
        results["sensitive_token"][case] = {
            "top1": sum(hits) / max(1, len(hits)), "n_templates": len(hits)}
    return results


_LONG_TEXT = (
    "The quarterly report summarizes revenue growth across regions and product "
    "lines, highlighting the shift toward subscription income and the impact of "
    "currency fluctuations on international sales. Management expects margins to "
    "stabilize as supply chains normalize and logistics costs decline through the "
    "second half of the year. The board reviewed capital allocation, dividend "
    "policy, and the pace of share repurchases, while the audit committee examined "
    "internal controls and the treatment of deferred revenue. Analysts asked about "
    "customer retention, pricing power, and the competitive response from smaller "
    "entrants, and the team reiterated its guidance for the coming fiscal year "
    "despite macroeconomic uncertainty and softening demand in some markets."
)


def run_bss_ica(oracle):
    n = qr_orthogonal(oracle.d, seed=0)
    h = oracle.hidden(_LONG_TEXT, 6)             # ~120-140 tokens so block sweep is exercised
    m, d = h.shape
    hn = h @ n
    out = {"shield_sweep": [], "block_sweep": [], "cond_sweep": [], "case_baseline": {}}

    # orthogonal A, no shield (the baseline STMS ICA risk)
    a = token_mix_orthogonal(m, seed=1)
    u = stms_forward(h, n, a)["u"]
    out["case_baseline"] = ica_bss_recovery(u, hn, seed=0)

    # shield sweep (fraction of extra rows, scale)
    for frac in (0.0, 0.01, 0.03, 0.05, 0.10):
        for scale in (1.0, 2.5, 5.0, 10.0, 20.0):
            k = max(0, round(frac * m))
            a = token_mix_orthogonal(m + k, seed=1)
            shield = sample_shield_rows(k, d, scale=scale, seed=2) if k > 0 else None
            u = stms_forward(h, n, a, shield=shield)["u"]
            rec = ica_bss_recovery(u, hn, seed=0)
            out["shield_sweep"].append({"shield_fraction": frac, "shield_scale": scale,
                                        "shield_rows": k, **rec})

    # block-size sweep (orthogonal block-local)
    for b in (16, 32, 64, 128, 256):
        if b > m:
            continue
        a = token_mix_blocklocal(m, block=b, seed=1)
        u = stms_forward(h, n, a)["u"]
        out["block_sweep"].append({"block": b, **ica_bss_recovery(u, hn, seed=0)})

    # non-orthogonal condition-number sweep
    for c in (2.0, 5.0, 10.0, 50.0, 200.0):
        a = token_mix_nonorthogonal(m, seed=1, cond=c)
        u = stms_forward(h, n, a)["u"]
        rec = stms_recover(u, a, num_real_tokens=m)
        out["cond_sweep"].append({
            "cond": c, "unmix_max_abs": float((rec - hn).abs().max()),
            **ica_bss_recovery(u, hn, seed=0)})
    return out


def run_efficiency():
    rows = []
    for nseq in (64, 128, 256, 512, 1024):
        for d in (768,):
            rows.append(efficiency(nseq, d, block=128))
    # one larger d to gauge scaling if feasible
    rows.append(efficiency(256, 4096, block=128))
    return {"rows": rows}


def _md(v, title, extra=""):
    return f"# {title}\n\n```json\n{json.dumps(_clean(v), indent=2)[:4000]}\n```\n{extra}"


def main():
    print("[*] loading GPT-2 ...")
    oracle = Gpt2Oracle(dtype=torch.float64)

    print("[*] correctness ...")
    c = run_correctness(oracle)
    _dump("correctness", c)
    (OUT / "correctness.md").write_text(_md(c, "STMS correctness + prefill integration"))

    print("[*] forward oracle ...")
    fo = run_forward_oracle(oracle)
    _dump("forward_oracle", fo)
    ft = "\n".join(
        f"| {k} | {v['top1']} | #{v['true_rank']+1}/{v['num_candidates']} | "
        f"{v['true_residual']:.2e} | {v['residual_margin']:.2e} |"
        for k, v in fo["candidate_set"].items())
    st = "\n".join(f"| {k} | {v['top1']:.2f} |" for k, v in fo["sensitive_token"].items())
    (OUT / "forward_oracle.md").write_text(
        "# STMS forward-matching oracle\n\n## Candidate-set (16 same-length)\n\n"
        "| case | top1 | true_rank | true_resid | margin |\n|---|---|---|---|---|\n" + ft
        + "\n\n## Sensitive-token top1 (templates)\n\n| case | sensitive_top1 |\n|---|---|\n" + st
        + "\n\n- baseline uses the Procrustes/Gram oracle; STMS cases use the "
        "spectrum oracle (mask-agnostic).\n")

    print("[*] BSS/ICA ...")
    bss = run_bss_ica(oracle)
    _dump("bss_ica", bss)
    bl = "\n".join(f"| {r['block']} | {r.get('median_cosine',float('nan')):.3f} | "
                   f"{r.get('p95_cosine',float('nan')):.3f} |" for r in bss["block_sweep"])
    cs = "\n".join(f"| {r['cond']} | {r['unmix_max_abs']:.1e} | "
                   f"{r.get('median_cosine',float('nan')):.3f} | {r.get('p95_cosine',float('nan')):.3f} |"
                   for r in bss["cond_sweep"])
    # shield: best (lowest p95) per fraction
    by_frac = {}
    for r in bss["shield_sweep"]:
        by_frac.setdefault(r["shield_fraction"], []).append(r)
    sh = "\n".join(
        f"| {frac} | {min(x['p95_cosine'] for x in rs):.3f} | {max(x['p95_cosine'] for x in rs):.3f} |"
        for frac, rs in sorted(by_frac.items()))
    (OUT / "bss_ica.md").write_text(
        "# STMS BSS/ICA recovery risk\n\n"
        f"- orthogonal-A baseline ICA: median cos "
        f"{bss['case_baseline'].get('median_cosine',float('nan')):.3f}, "
        f"p95 cos {bss['case_baseline'].get('p95_cosine',float('nan')):.3f}\n\n"
        "## Block-size sweep (orthogonal block-local)\n\n"
        "| block | median_cos | p95_cos |\n|---|---|---|\n" + bl
        + "\n\n## Non-orthogonal condition-number sweep\n\n"
        "| cond | unmix_max_abs | median_cos | p95_cos |\n|---|---|---|---|\n" + cs
        + "\n\n## Shield sweep (p95 cosine range over scales)\n\n"
        "| shield_fraction | best_p95 | worst_p95 |\n|---|---|---|\n" + sh + "\n")

    print("[*] efficiency ...")
    eff = run_efficiency()
    _dump("efficiency", eff)
    et = "\n".join(
        f"| {r['n']} | {r['d']} | {r['baseline_HN_ms']:.3f} | {r['mix_ms']:.3f} | "
        f"{r['unmix_orth_ms']:.3f} | {r['unmix_solve_ms']:.3f} | {r['gen_A_orthogonal_ms']:.3f} |"
        for r in eff["rows"])
    (OUT / "efficiency.md").write_text(
        "# STMS efficiency\n\n| n | d | baseline HN | mix A·HN | unmix(orth) | "
        "unmix(solve) | gen A |\n|---|---|---|---|---|---|---|\n" + et + "\n")

    _summary(c, fo, bss, eff)
    print(f"[done] {OUT}")


def _summary(c, fo, bss, eff):
    base = fo["candidate_set"]["baseline"]
    orth = fo["candidate_set"]["orthogonal"]
    nonorth = fo["candidate_set"]["nonorthogonal"]
    md = f"""# Variant F — Selective Token-Mixing Shield (STMS): summary

## Threat model (STRICTLY WEAKER THAN V4 — read first)
STMS shields the observable boundary artifact `U = A (H N)` against an observer
who does **not** know `A`/shield rows and does **not** see the post-recovery state
`H N`. It is a **boundary shield**, not full-transformer left mixing: `A` never
enters attention/RMSNorm/RoPE/KV, and `H N = A^{{-1}} U` is recovered before any
real compute. **If the adversary sees `H N`** (the V4 / compute-offload model),
**STMS adds nothing** — V4 and the O(d^3) weight-mask recovery apply to `H N`
unchanged. `protects_compute_visible_HN = False`.

## What observation is protected
Only the transmitted/logged snapshot `U`. Not the compute state, not decode
single-token states, not KV.

## Correctness (roundtrip + prefill integration)
- roundtrip max_abs over all cases: **{max(x['roundtrip_max_abs'] for x in c['cases'].values()):.2e}** (fp64 exact).
- prefill: recovered state == `H N` to {c['prefill_integration']['recovered_equals_HN_max_abs']:.2e};
  shield rows in recovered state: **{c['prefill_integration']['shield_rows_in_recovered_state']}** (dropped before KV).

## Forward-matching oracle (candidate id, 16 same-length candidates)
| case | top1 | sensitive-token top1 |
|---|---|---|
| baseline (H N) | {base['top1']} | {fo['sensitive_token']['baseline']['top1']:.2f} |
| STMS orthogonal A | {orth['top1']} | {fo['sensitive_token']['orthogonal']['top1']:.2f} |
| STMS non-orthogonal A | {nonorth['top1']} | {fo['sensitive_token']['nonorthogonal']['top1']:.2f} |
| STMS block-local A | {fo['candidate_set']['blocklocal']['top1']} | {fo['sensitive_token']['blocklocal']['top1']:.2f} |

- **orthogonal A does NOT stop the oracle**: `U U^T = A (H H^T) A^T` is an
  orthogonal *similarity*, so the singular-value spectrum of `H` is invariant and
  a spectrum-matching oracle still ranks candidates.
- non-orthogonal A distorts the spectrum (helps), but a free-`A` oracle overfits,
  so security reduces to BSS/ICA hardness (below).

## BSS / ICA recovery risk (median AND p95 |cosine|)
- orthogonal-A ICA: median {bss['case_baseline'].get('median_cosine',float('nan')):.3f},
  **p95 {bss['case_baseline'].get('p95_cosine',float('nan')):.3f}** (residual risk if p95 high).
- shield / block / condition sweeps: see bss_ica.md.

## Efficiency
- token-mix adds an `[n,n] · [n,d]` matmul (`mix`) + an unmix (`A^T` for
  orthogonal, `solve` for non-orthogonal); both scale with sequence length `n`.
- see efficiency.md; non-orthogonal `solve` and shield rows add cost.

## Limitations / honesty
- **Not production-ready** (no full integration beyond bit-exact prefill roundtrip).
- **Does not protect decode single-token states** (a length-1 boundary has no token
  dimension to mix).
- **Does not cross attention/KV/RMSNorm** (by design) — so it cannot change what the
  compute engine sees.
- **No cryptographic privacy claim.**
- orthogonal A: forward oracle still succeeds (spectrum) → reported as failure of
  that configuration.
- shield rows cost throughput (extra rows through the boundary transform).
- if only p95 |cosine| stays high, that is residual BSS risk — reported, not hidden.
"""
    (OUT / "summary.md").write_text(md)


if __name__ == "__main__":
    main()
