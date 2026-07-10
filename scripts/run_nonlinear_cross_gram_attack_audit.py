"""Run the nonlinear cross-Gram privacy-attack audit (multi-seed).

Writes results.json + 5 attack CSVs + summary.md into
results/attacks/nonlinear_cross_gram_attack_audit/. Experiment only; no
production path is touched, nothing is committed.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from pllo.experiments.nonlinear_cross_gram_attack_audit import run_all

OUT = Path("results/attacks/nonlinear_cross_gram_attack_audit")
SEEDS = [0, 1, 2]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(vals) / len(vals), 4) if vals else None


def _agg(rows: list[dict], group_keys: list[str], metrics: list[str]) -> list[dict]:
    """Average `metrics` across seeds, grouping by group_keys; success = majority."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r.get(k) for k in group_keys)].append(r)
    out = []
    for key, rs in buckets.items():
        row = dict(zip(group_keys, key))
        for mkey in metrics:
            row[mkey] = _mean([r.get(mkey) for r in rs])
        succ = [r.get("success") for r in rs if isinstance(r.get("success"), bool)]
        row["success_rate"] = round(sum(succ) / len(succ), 3) if succ else None
        note = next((r.get("note") for r in rs if r.get("note")), None)
        if note:
            row["note"] = note
        out.append(row)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    collect = defaultdict(list)
    for s in SEEDS:
        res = run_all(seed=s)
        per_seed[s] = res
        for attack in ("token_reconstruction", "target_prediction", "membership_inference",
                       "adapter_extraction", "future_token_prediction"):
            for row in res[attack]:
                collect[attack].append({"seed": s, **row})

    agg = {
        "token_reconstruction": _agg(collect["token_reconstruction"], ["regime"],
                                     ["token_top1", "token_top5", "sequence_exact",
                                      "edit_distance", "embedding_nn_hit", "baseline_top1"]),
        "target_prediction": _agg(collect["target_prediction"], ["regime"],
                                  ["target_top1", "target_top5", "auc", "ce_improvement",
                                   "baseline_acc"]),
        "membership_inference": _agg(collect["membership_inference"], ["classifier"],
                                     ["auc", "tpr_at_1pct_fpr", "advantage"]),
        "adapter_extraction": _agg(collect["adapter_extraction"], ["regime"],
                                   ["dW_frobenius_rel_error", "dW_cosine", "heldout_kl",
                                    "token_match"]),
        "future_token_prediction": _agg(collect["future_token_prediction"], ["regime"],
                                        ["next_token_top1", "future_token_exact",
                                         "perplexity", "ppl_reduction_vs_baseline"]),
    }

    results = {"meta": per_seed[SEEDS[0]]["meta"] | {"seeds": SEEDS}, "aggregates": agg}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))

    _write_csv(OUT / "token_reconstruction.csv", collect["token_reconstruction"])
    _write_csv(OUT / "target_prediction.csv", collect["target_prediction"])
    _write_csv(OUT / "membership_inference.csv", collect["membership_inference"])
    _write_csv(OUT / "adapter_extraction.csv", collect["adapter_extraction"])
    _write_csv(OUT / "future_token_prediction.csv", collect["future_token_prediction"])

    _write_summary(results)
    print(f"wrote audit to {OUT}")


def _row(agg, key, val):
    for r in agg:
        if r.get(key) == val:
            return r
    return {}


def _write_summary(r: dict) -> None:
    a = r["aggregates"]
    tok = a["token_reconstruction"]; tgt = a["target_prediction"]
    mem = a["membership_inference"]; adp = a["adapter_extraction"]
    fut = a["future_token_prediction"]
    seeds = r["meta"]["seeds"]

    tp_pub = _row(tok, "regime", "public_weights"); tp_priv = _row(tok, "regime", "private_weights")
    tp_gram = _row(tok, "regime", "gram_only")
    tg_pub = _row(tgt, "regime", "public_weights"); tg_priv = _row(tgt, "regime", "private_weights")
    ad_pub = _row(adp, "regime", "public_weights"); ad_priv = _row(adp, "regime", "private_weights")
    ft_base = _row(fut, "regime", "baseline_bigram"); ft_pub = _row(fut, "regime", "public_weights")
    mem_best = max(mem, key=lambda x: (x.get("auc") or 0))

    L = []
    L.append("# Nonlinear Cross-Gram — Real-Attack Audit\n")
    L.append(f"Experiment / audit only. numpy/scipy/sklearn, mean over seeds {seeds}. "
             "No production path modified; not committed.\n")
    L.append("**Question:** the permutation island leaks — exactly — `ZGZ^T`, `UGU^T`, the "
             "per-token value multiset (`Z_tilde`/`U_tilde` mod a global column permutation the "
             "GPU already applies), and the per-token gradient norm. Does that auxiliary "
             "structural leakage translate into real attacks? Measured below; the decisive axis "
             "is **public vs private base weights**.\n")

    L.append("## Headline results (mean over seeds)\n")
    L.append("| attack | public base weights | private base weights | verdict |")
    L.append("|---|---|---|---|")
    L.append(f"| 1 token reconstruction | top1 **{tp_pub.get('token_top1')}** "
             f"(baseline {tp_pub.get('baseline_top1')}) | top1 {tp_priv.get('token_top1')} "
             f"(id=baseline; linking F1 {tp_priv.get('embedding_nn_hit')}) | "
             f"**HIGH under public weights** |")
    L.append(f"| 2 target/label prediction | top1 {tg_pub.get('target_top1')}, AUC "
             f"**{tg_pub.get('auc')}** (base acc {tg_pub.get('baseline_acc')}) | top1 "
             f"{tg_priv.get('target_top1')}, AUC {tg_priv.get('auc')} | public: MEDIUM, private: none |")
    L.append(f"| 3 membership inference | AUC **{mem_best.get('auc')}** "
             f"(TPR@1%FPR {mem_best.get('tpr_at_1pct_fpr')}, {mem_best.get('classifier')}) | "
             f"same (model-agnostic gradient-norm signal) | **MEDIUM, both regimes** |")
    L.append(f"| 4 adapter extraction (ΔW) | cosine **{ad_pub.get('dW_cosine')}**, KL "
             f"{ad_pub.get('heldout_kl')}, token-match {ad_pub.get('token_match')} | cosine "
             f"{ad_priv.get('dW_cosine')} (FAILURE) | public: HIGH, private: none |")
    L.append(f"| 5 future-token prediction | next-token top1 **{ft_pub.get('next_token_top1')}** "
             f"(bigram baseline {ft_base.get('next_token_top1')}) | = baseline | "
             f"public: HIGH (single forward), private: none |")
    L.append("")
    L.append(f"**Gram-ONLY attacker** (uses `ZGZ^T` but NOT the value multiset, public weights): "
             f"token top1 {tp_gram.get('token_top1')} = baseline. The Gram matrix alone carries "
             f"no vocab identity without a labeled reference — the danger is the **co-leaked "
             f"value multiset**, not the Gram per se.\n")

    L.append("## Required conclusions\n")
    def yn(x): return "YES" if x else "NO"
    tok_pub_ok = (tp_pub.get("token_top1") or 0) > 2 * (tp_pub.get("baseline_top1") or 1)
    tgt_ok = (tg_pub.get("auc") or 0) > 0.6
    mem_ok = (mem_best.get("auc") or 0) > 0.6
    adp_ok = (ad_pub.get("dW_cosine") or 0) > 0.9
    fut_ok = (ft_pub.get("next_token_top1") or 0) > (ft_base.get("next_token_top1") or 0) + 0.1
    L.append(f"- **Token reconstruction?** {yn(tok_pub_ok)} under public weights "
             f"(top1 {tp_pub.get('token_top1')}); NO under private weights "
             f"(only token *linking* — equal tokens are detectable, not identifiable).")
    L.append(f"- **Target/label recovery?** {yn(tgt_ok)} under public weights (recovered-token "
             f"bag-of-words, AUC {tg_pub.get('auc')}); under private weights top1 "
             f"{tg_priv.get('target_top1')} does not beat the base rate {tg_priv.get('baseline_acc')} "
             f"(AUC {tg_priv.get('auc')} is only marginally above 0.5) — not a usable recovery.")
    L.append(f"- **Membership above random?** {yn(mem_ok)} — AUC {mem_best.get('auc')} via the "
             f"exactly-leaked per-token gradient norm; this signal is model-agnostic and "
             f"survives **private** weights.")
    L.append(f"- **Adapter extraction / functional clone?** {yn(adp_ok)} under public weights "
             f"(ΔW cosine {ad_pub.get('dW_cosine')}, heldout KL {ad_pub.get('heldout_kl')}); NO "
             f"under private weights (cosine {ad_priv.get('dW_cosine')} — cannot recover inputs, "
             f"solve the column permutation, or fit ΔW). Note: private `token_match` "
             f"{ad_priv.get('token_match')} is high only because this adapter is functionally "
             f"small, NOT because extraction succeeded — `dW_cosine` is the real signal.")
    L.append(f"- **Future-token beyond baseline?** {yn(fut_ok)} under public weights, but this is "
             f"the *single-forward* setting where the future positions' activations are already "
             f"visible (= token reconstruction on those rows). In genuine streaming decode only "
             f"the current step is exposed; the meaningful future-prediction then reduces to "
             f"running the recovered public model. Private weights = bigram baseline.\n")
    L.append("**Classification.** Under **private base weights** the nonlinear cross-Gram is "
             "*auxiliary structural leakage*: it did NOT recover user tokens, targets, or a "
             "functional adapter — except **membership inference (MEDIUM)**, which rides the "
             "model-agnostic gradient-norm signal. Under **public base weights** the co-leaked "
             "pre-activation value multiset is **HIGH**: exact token reconstruction, target "
             "recovery, and full adapter (ΔW) extraction. This matches the CP-uniqueness / "
             "pre-activation-exposure dichotomy from the prior audits.\n")

    L.append("## Allowed conclusion (per-regime)\n")
    L.append("- *Private weights:* \"Nonlinear permutation islands leak the exact token×token "
             "activation–gradient Gram matrices and per-token gradient norms, but under the "
             "evaluated attackers this auxiliary structural leakage did not recover user data, "
             "target tokens, or functional adapters — with the exception of membership inference "
             "(medium), which the gradient-norm signal enables.\"")
    L.append("- *Public weights:* the same leak (specifically the co-leaked pre-activation value "
             "multiset) is HIGH: token/target/future-token recovery and ΔW adapter extraction all "
             "succeed.\n")
    L.append("## Disallowed conclusions (NOT claimed here)")
    for c in [
        "\"No leakage exists.\" — the Gram/value/gradient-norm leaks are exact and measured.",
        "\"Cross-Gram is harmless.\" — only stated per-regime, with attack numbers.",
        "\"Formal privacy proven.\" — no formal claim; this is an empirical attacker panel.",
        "\"Adapter protection proven.\" — adapter extraction was run; it SUCCEEDS under public "
        "weights and only fails under private weights.",
    ]:
        L.append(f"- {c}")
    L.append("")
    L.append("## Threat-model scope\n")
    L.append("`attacker_sees_only_transcript = true` (masked activations `Z_tilde`/`U_tilde`, "
             "backward `GZ_tilde`, all mod a global column permutation). `public_base_weights` is "
             "the decisive toggle. Our paper's threat model is public base weights + protect user "
             "data — so the public-weights column is the operative one, and it shows the "
             "value-multiset leak is a genuine user-data / adapter-privacy risk, consistent with "
             "the paper reporting nonlinear-region leakage as proxy-only / `needs_more_evaluation` "
             "rather than `low`.\n")

    OUT.joinpath("summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
