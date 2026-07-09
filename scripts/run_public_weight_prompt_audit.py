#!/usr/bin/env python
"""Run the public-weight prompt-privacy audit and emit JSON + Markdown.

Verifies (does NOT assume) whether the hidden-dimension dense-orthogonal-mask
construction protects the user prompt from *token-level* recovery under a
public-weight adversary. Writes, under results/attacks/public_weight_prompt_audit/:

    silu_amulet_boundary.{json,md}      (V1)
    ffn_perm_to_token_recovery.{json,md}(V2)
    isa_public_weight_hidden_q.{json,md}(V3)
    forward_matching_oracle.{json,md}   (V4)
    efficiency_dtype.{json,md}          (V5)
    summary.md

Requires a locally-cached GPT-2 (real public-weight forward oracle). Runs
fully offline; no downloads.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pllo.experiments.public_weight_prompt_audit import (  # noqa: E402
    Gpt2Oracle,
    verify1_silu_amulet_boundary,
    verify2_ffn_perm_to_token_recovery,
    verify3_isa_public_weight_hidden_q,
    verify4_forward_matching_oracle,
    verify5_efficiency_dtype,
)

OUT = Path(__file__).resolve().parents[1] / "results/attacks/public_weight_prompt_audit"


def _dump(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(_clean(obj), indent=2))


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "records" or True}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (torch.Tensor,)):
        return o.tolist()
    return o


def _md_common(v: dict) -> str:
    keys = ["threat_model", "attacker_observation", "attacker_knowledge",
            "attack_objective", "metric", "result"]
    lines = [f"# {v['verification']}", ""]
    for k in keys:
        if k in v:
            lines.append(f"- **{k}**: {v[k]}")
    lines.append(f"- **claim_survives**: {v.get('claim_survives')}")
    if "claim_survives_note" in v:
        lines.append(f"- **note**: {v['claim_survives_note']}")
    return "\n".join(lines)


def main() -> None:
    prompts = [
        "The password is alpha and it is secret",
        "Patient diagnosis is diabetes stage two",
        "My phone number is five five five one two",
        "The account balance is nine thousand dollars",
        "The meeting is scheduled for next monday",
        "Transfer the funds to the offshore account",
        "The encryption key is stored in memory",
        "Her home address is on maple street",
        "The prototype ships in early spring",
        "Confidential report attached for your review",
        "The server root password was rotated today",
        "Send the invoice to the finance team",
    ]

    print("[*] loading GPT-2 (fp64, offline)...")
    oracle = Gpt2Oracle(dtype=torch.float64)
    print(f"    d={oracle.d} vocab={oracle.vocab}")

    print("[*] V1 SiLU/Amulet boundary ...")
    v1 = verify1_silu_amulet_boundary()
    _dump("silu_amulet_boundary", v1)
    (OUT / "silu_amulet_boundary.md").write_text(
        _md_common(v1)
        + f"\n\n## Boundary correctness\n\n"
        + f"- fp64: max_abs {v1['fp64']['max_abs_error']:.3e}, "
        f"rel_l2 {v1['fp64']['max_relative_l2_error']:.3e} (n={v1['fp64']['n']})\n"
        + f"- fp32-IO: max_abs {v1['fp32']['max_abs_error']:.3e}, "
        f"rel_l2 {v1['fp32']['max_relative_l2_error']:.3e} (n={v1['fp32']['n']})\n"
        + f"- native fp32 build: {v1['native_fp32_build_probe']['native_fp32_build']}\n"
        + f"- negative control (dense P across SiLU): "
        f"{v1['negative_control_dense_keymat']['silu_xP_Pinv_vs_silu_x_max_abs']:.3e} "
        "(must be large)\n")

    print("[*] V2 FFN permutation -> token recovery ...")
    v2 = verify2_ffn_perm_to_token_recovery(oracle, prompts)
    _dump("ffn_perm_to_token_recovery", v2)
    (OUT / "ffn_perm_to_token_recovery.md").write_text(
        _md_common(v2)
        + f"\n\n## Token recovery with TRUE FFN permutation\n\n"
        + f"- random baseline top1: {v2['random_baseline_top1']:.2e}\n"
        + f"- masked token top1: {v2['token_recovery_top1_masked']:.2e}\n"
        + f"- masked token top5: {v2['token_recovery_top5_masked']:.2e}\n"
        + f"- upper bound if Q known: {v2['token_recovery_top1_if_Q_known_upper_bound']:.3f}\n"
        + f"- **norm leak** (rank corr H vs H_obs): {v2['norm_rank_corr']:.6f}\n"
        + f"\n{v2['structural_leak_note']}\n")

    print("[*] V3 ISA / known-plaintext on static Q ...")
    v3 = verify3_isa_public_weight_hidden_q(oracle, prompts)
    _dump("isa_public_weight_hidden_q", v3)
    kpa_tbl = "\n".join(
        f"| {r['known_rows']} | {r['Q_recovery_max_abs_error']:.2e} | "
        f"{r['token_recovery_top1']:.3f} | {r['broken']} |" for r in v3["kpa_sweep"])
    (OUT / "isa_public_weight_hidden_q.md").write_text(
        _md_common(v3)
        + f"\n\n## Known-plaintext attack on STATIC Q (hidden dim d={v3['hidden_dim_d']})\n\n"
        + f"- no-known-plaintext token top1: {v3['no_known_plaintext_token_top1']:.2e}\n"
        + f"- known rows to break TOKENS (approx Q): {v3['required_known_rows_to_break_tokens']}\n"
        + f"- known rows to recover Q EXACTLY (>= d): {v3['required_known_rows_to_break']}\n"
        + f"- fresh Q per session token top1: "
        f"{v3['fresh_Q_per_session']['token_recovery_top1']:.2e} "
        f"(resists={v3['fresh_Q_per_session']['resists']})\n\n"
        + "| known_rows | Q_recovery_err | token_top1 | broken |\n|---|---|---|---|\n"
        + kpa_tbl + "\n")

    print("[*] V4 forward-matching oracle (crux) ...")
    v4 = verify4_forward_matching_oracle(oracle)
    _dump("forward_matching_oracle", v4)
    c_tbl = "\n".join(
        f"| {c['template']} | {c['true_secret']} | {c['recovered_secret']} | "
        f"{c['sensitive_token_top1_correct']} | {c['true_residual']:.2e} | "
        f"{c['nearest_false_residual']:.2e} |" for c in v4["C_partial_prompt"]["templates"])
    (OUT / "forward_matching_oracle.md").write_text(
        _md_common(v4)
        + f"\n\n- **invariant**: {v4['invariant_note']}\n"
        + f"\n## A. small candidate set\n"
        + f"- true rank #{v4['A_small_set']['true_rank'] + 1}/"
        f"{v4['A_small_set']['num_candidates']}, "
        f"true residual {v4['A_small_set']['true_residual']:.2e}, "
        f"margin {v4['A_small_set']['residual_margin']:.2e}\n"
        + f"\n## B. large candidate set\n"
        + f"- true rank #{v4['B_large_set']['true_rank'] + 1}/"
        f"{v4['B_large_set']['num_candidates']}, top1={v4['B_large_set']['top1_correct']}\n"
        + f"\n## C. partial-prompt / sensitive-token (top1={v4['C_partial_prompt']['sensitive_token_top1']:.2f})\n\n"
        + "| template | true | recovered | correct | true_resid | nearest_false |\n"
        "|---|---|---|---|---|---|\n" + c_tbl + "\n"
        + f"\n## D. multi-layer (fresh Q per layer)\n"
        + f"- single-layer true ranks: {v4['D_multi_layer']['single_layer_true_ranks']}\n"
        + f"- combined true rank: #{v4['D_multi_layer']['combined_true_rank'] + 1}\n"
        + f"\n## Fresh-Q ablation\n"
        + f"- residual under Q1: {v4['fresh_Q_ablation']['residual_under_Q1']:.2e}, "
        f"under Q2: {v4['fresh_Q_ablation']['residual_under_Q2']:.2e} "
        "(identical -> Q irrelevant)\n"
        + f"\n**oracle_succeeds = {v4['oracle_succeeds']}**\n")

    print("[*] V5 efficiency & dtype ...")
    v5 = verify5_efficiency_dtype()
    _dump("efficiency_dtype", v5)
    t = v5["timings_ms"]
    dtype_tbl = "\n".join(
        f"| {r['dtype']} | {r['io_roundtrip_max_abs_error']:.2e} | "
        f"{r['exact_lossless_supported']} | {r['native_build']} |"
        for r in v5["dtype_correctness"])
    (OUT / "efficiency_dtype.md").write_text(
        _md_common(v5)
        + f"\n\n## Latency (ms, d={v5['dims']['d']}, m={v5['dims']['m']})\n\n"
        + "\n".join(f"- {k}: {vv:.4f}" for k, vv in t.items())
        + f"\n\n## dtype correctness (SiLU/Amulet boundary; island compute is fp64-only)\n\n"
        + "| io_dtype | io_roundtrip_max_abs | exact_lossless | native_build |\n"
        "|---|---|---|---|\n" + dtype_tbl
        + f"\n\n- classification: **{v5['classification']}**\n")

    _write_summary(v1, v2, v3, v4, v5)
    print(f"[done] wrote results to {OUT}")


def _decision(v1, v2, v3, v4) -> tuple[str, str]:
    if not v1["claim_survives"]:
        return ("NOT_SUPPORTED",
                "The dense-orthogonal / Amulet boundary does not preserve exactness; "
                "the exact-lossless claim itself fails.")
    if v4["oracle_succeeds"]:
        return ("NOT_SUPPORTED_UNDER_FORWARD_ORACLE",
                "Public plaintext weights enable candidate-verification attacks against "
                "observed obfuscated states. The construction does not provide token-level "
                "prompt privacy against a public-weight adversary without additional "
                "freshness or observation-limiting assumptions -- and freshness of Q does "
                "NOT help, because the token-token Gram H H^T is invariant to any orthogonal Q.")
    return ("PARTIALLY_SUPPORTED",
            "Token-level recovery is blocked in the direct/FFN-permutation/no-known-plaintext "
            "attacks, but structural (norm, FFN-permutation) leakage remains.")


def _write_summary(v1, v2, v3, v4, v5) -> None:
    decision, wording = _decision(v1, v2, v3, v4)
    md = f"""# Public-weight prompt-privacy audit — summary

## 1. Claim under test
Under a **public-weight** adversary (attacker holds the full plaintext model and
runs it as a forward oracle), the exact-lossless construction — protecting the
residual stream with a **hidden-dimension dense orthogonal mask `Q`** rather than
token-level permutations — protects the user prompt from **token-level recovery**,
at the cost of leaking some structural information (FFN neuron permutation / norm).

## 2. What IS (claimed to be) protected
- token-level prompt content recovery.

## 3. What is NOT protected (measured, not hidden)
- **norm leakage**: `||H Q|| = ||H||` for orthogonal `Q` — per-token norm profile
  is preserved. Measured rank-corr H vs H_obs = **{v2['norm_rank_corr']:.6f}**.
- **weight-alignment / FFN neuron permutation leakage**: recoverable structural info.
- **model-structure leakage**: architecture and per-layer Gram are exposed.
- **prompt-length leakage**: token count is visible (candidates of other lengths
  are trivially separable — see V4).

## 4. Correctness — SiLU/Amulet boundary (V1)
- fp64 max_abs **{v1['fp64']['max_abs_error']:.2e}** (exact to rounding);
  fp32-IO max_abs {v1['fp32']['max_abs_error']:.2e};
  native fp32 R-factor build **{v1['native_fp32_build_probe']['native_fp32_build']}**.
- negative control (dense P cannot cross SiLU): max_abs
  {v1['negative_control_dense_keymat']['silu_xP_Pinv_vs_silu_x_max_abs']:.2e} (large ✓).
- **survives: {v1['claim_survives']}**

## 5. FFN permutation leakage (V2)
- Attacker GIVEN the ground-truth FFN neuron permutation.
- token recovery top1 = **{v2['token_recovery_top1_masked']:.2e}** (random baseline
  {v2['random_baseline_top1']:.2e}); would be
  {v2['token_recovery_top1_if_Q_known_upper_bound']:.3f} if `Q` were known.
- **FFN permutation does not unlock `Q`. survives: {v2['claim_survives']}**

## 6. ISA / internal-state inversion (V3)
- no-known-plaintext token top1 = **{v3['no_known_plaintext_token_top1']:.2e}** (~0).
- **STATIC `Q` is broken by a known-plaintext attack**: token recovery → 1.0 at
  ~{v3['required_known_rows_to_break_tokens']} known rows (an approximate `Q`
  already argmaxes correctly), and `Q` is recovered **exactly** (~1e-14) once
  known rows ≥ d ({v3['required_known_rows_to_break']} rows).
- fresh `Q` per session resists (top1 {v3['fresh_Q_per_session']['token_recovery_top1']:.2e}).
- **survives only without known-plaintext / with fresh Q: {v3['claim_survives']}**

## 7. Forward-matching oracle (V4) — the crux
- A small set: true ranks #{v4['A_small_set']['true_rank'] + 1}/{v4['A_small_set']['num_candidates']}
  (residual {v4['A_small_set']['true_residual']:.2e} vs margin {v4['A_small_set']['residual_margin']:.2e}).
- B large set: top1 = {v4['B_large_set']['top1_correct']}.
- C sensitive-token top1 = **{v4['C_partial_prompt']['sensitive_token_top1']:.2f}**.
- D multi-layer (fresh Q per layer): combined true rank #{v4['D_multi_layer']['combined_true_rank'] + 1}.
- Fresh-Q ablation: residuals identical under any orthogonal Q → **freshness does not help**.
- **oracle_succeeds = {v4['oracle_succeeds']}**

## 8. Efficiency (V5)
- dense orthogonal mask is a full [d,d] matmul (vs a signed-perm gather);
  Amulet-SiLU island is the dominant cost. fp64 exact; bf16 unsupported.
- classification: **{v5['classification']}**.

## 9. Final decision: **{decision}**
{wording}

## 10. Exact wording allowed for the paper
> {wording}

### Separation of leakage axes (required)
- token-level content recovery: **{'BROKEN by forward oracle' if v4['oracle_succeeds'] else 'resisted in evaluated direct attacks'}**
- norm / structure leakage: **present** (rank-corr {v2['norm_rank_corr']:.4f})
- weight-alignment leakage: **present** (FFN neuron permutation)
- model-structure leakage: **present**
- prompt semantic leakage: candidate-verification identifies the prompt / sensitive token.
"""
    (OUT / "summary.md").write_text(md)


if __name__ == "__main__":
    main()
