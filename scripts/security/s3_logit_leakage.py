"""S3 — Logit leakage: validate the monomial logit-masking design.

Compare three schemes on REAL Qwen2.5-0.5B logits:
  B0 plaintext logits            (full leak reference)
  B1 permutation-only masking    logits @ Pi        (D = I)  -> value multiset EXACT
  B2 proposed monomial masking   logits @ (D Pi)    (D>0, bounded condition)

Attacks (attacker does NOT know Pi):
  * value-multiset recovery : sorted logit values plaintext vs observed
  * ranking / confidence    : sorted softmax prob vector, entropy, max-prob correlation
  * KL divergence           : KL(sorted plaintext probs || sorted observed probs)

Expectation (and honest scoping): permutation-only leaks the EXACT value multiset,
so the full confidence/uncertainty profile is recoverable (only token identities are
hidden by the unknown permutation). The monomial D perturbs values, reducing this
distributional leakage at a bounded condition number. We do NOT claim the monomial
hides token identity better (both use the same permutation) — only that it reduces the
exactly-preserved multiset/confidence leak. CPU only. Writes security/S3_logit_leakage/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_harness import PrivateBaseOracle, DEFAULT_PROMPTS, stamp  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results/aaai_private_base/security/S3_logit_leakage"
torch.manual_seed(0)


def softmax_sorted(logits):
    p = torch.softmax(logits, dim=-1)
    return p.sort(dim=-1, descending=True).values


def entropy(logits):
    p = torch.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-30)).log()).sum(-1)


def kl_sorted(p_ref_sorted, p_obs_sorted):
    # both sorted descending -> compares distribution SHAPE (identity-agnostic)
    return (p_ref_sorted * (p_ref_sorted.clamp_min(1e-30).log()
                            - p_obs_sorted.clamp_min(1e-30).log())).sum(-1)


def scheme_metrics(o, logits_plain, scheme):
    obs = o.observe_logits(logits_plain, scheme).to(torch.float32)
    plain = logits_plain.to(torch.float32)
    # value multiset: sorted values
    vs_plain = plain.sort(dim=-1).values
    vs_obs = obs.sort(dim=-1).values
    multiset_gap = float((vs_plain - vs_obs).abs().max())
    # sorted prob shape + KL
    sp_plain = softmax_sorted(plain)
    sp_obs = softmax_sorted(obs)
    kl = kl_sorted(sp_plain, sp_obs)
    # confidence (max prob) + entropy
    conf_plain = torch.softmax(plain, -1).max(-1).values
    conf_obs = torch.softmax(obs, -1).max(-1).values
    # correlation of confidence across rows
    cp = conf_plain - conf_plain.mean(); co = conf_obs - conf_obs.mean()
    conf_corr = float((cp * co).sum() / (cp.norm() * co.norm() + 1e-30))
    ent_plain = entropy(plain); ent_obs = entropy(obs)
    return {
        "value_multiset_max_abs_gap": multiset_gap,
        "sorted_prob_KL_mean": float(kl.mean()),
        "sorted_prob_KL_max": float(kl.max()),
        "confidence_maxprob_correlation": conf_corr,
        "confidence_maxprob_mean_abs_diff": float((conf_plain - conf_obs).abs().mean()),
        "entropy_mean_abs_diff": float((ent_plain - ent_obs).abs().mean()),
        "top1_value_preserved_frac": float((plain.argmax(-1) == obs.argmax(-1)).float().mean()),
    }


def main():
    t0 = time.time()
    o = PrivateBaseOracle()
    fwd = o.real_forward(DEFAULT_PROMPTS, max_len=24)
    mask = fwd["attention_mask"].bool()
    logits = fwd["logits"][mask]                    # (Nvalid, V) plaintext
    # subsample rows to bound memory (V=151936)
    idx = torch.randperm(logits.shape[0])[:64]
    logits = logits[idx].to(torch.float64)
    results = {"experiment": "S3_logit_leakage", "model": "Qwen2.5-0.5B",
               "vocab": o.V, "n_rows": int(logits.shape[0]),
               "monomial_D_condition_number": float(o.vocab_D.max() / o.vocab_D.min()),
               "budget": {"device": "cpu", "seeds": [0], "note": "one real forward, closed-form metrics"},
               "schemes": {}, "interpretation": {}, "limitations": []}

    results["schemes"]["B0_plaintext"] = scheme_metrics(o, logits, "plaintext")
    results["schemes"]["B1_permutation_only"] = scheme_metrics(o, logits, "perm_only")
    results["schemes"]["B2_monomial"] = scheme_metrics(o, logits, "monomial")

    b1 = results["schemes"]["B1_permutation_only"]
    b2 = results["schemes"]["B2_monomial"]
    results["interpretation"] = {
        "permutation_only": "value multiset gap ~0 and confidence correlation ~1 => the EXACT confidence/"
                            "uncertainty profile leaks (only token identities hidden by the unknown permutation).",
        "monomial": "value multiset gap > 0, sorted-prob KL > 0, confidence correlation < 1 => the monomial D "
                    "perturbs the distribution, reducing the exactly-preserved multiset/confidence leak.",
        "leak_reduction": {
            "multiset_gap_B1_to_B2": [b1["value_multiset_max_abs_gap"], b2["value_multiset_max_abs_gap"]],
            "confidence_corr_B1_to_B2": [b1["confidence_maxprob_correlation"], b2["confidence_maxprob_correlation"]],
            "sorted_prob_KL_B1_to_B2": [b1["sorted_prob_KL_mean"], b2["sorted_prob_KL_mean"]]},
        "scope": "both B1 and B2 hide token IDENTITY equally (same permutation); the design improvement is "
                 "strictly in reducing distributional (multiset/confidence) leakage. We do NOT claim identity "
                 "confidentiality from the logit mask alone."}
    results["limitations"] = [
        "The frozen shipped package uses permutation-only (D=I); B2 evaluates the proposed monomial D as a "
        "design option, not the currently-built default.",
        "Monomial condition number is bounded (~7) to avoid bf16 overflow; larger D reduces leakage further "
        "but risks numerical range — a documented trade-off, not a free win.",
        "An attacker with known-plaintext logit pairs could estimate D per column (D>0 diagonal is identifiable "
        "from ratios); monomial resists identity-free multiset reading, not a full known-plaintext attacker.",
    ]
    results["wall_sec"] = round(time.time() - t0, 1)
    stamp(OUT, "s3_results.json", results)
    print("[S3] done %.1fs  B1 multiset_gap=%.2e conf_corr=%.4f | B2 multiset_gap=%.3f conf_corr=%.4f KL=%.4f" % (
        results["wall_sec"], b1["value_multiset_max_abs_gap"], b1["confidence_maxprob_correlation"],
        b2["value_multiset_max_abs_gap"], b2["confidence_maxprob_correlation"], b2["sorted_prob_KL_mean"]))


if __name__ == "__main__":
    main()
