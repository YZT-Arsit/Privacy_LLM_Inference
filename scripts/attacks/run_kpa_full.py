"""KPA (known-plaintext) full sample-complexity sweep on real Qwen — the decisive
default-vs-fresh discriminator.

For each defense we apply its REAL cloud-visible transform to the same captured
Qwen residual states, then, in a subspace of dimension ``subdim`` (so the sweep
is tractable at hidden=3584), fit a single global deobfuscation map from
``num_pairs`` known (X, X*) pairs and measure how well it generalises to HELD-OUT
tokens:

    heldout_deobfuscation_relative_l2_error = ‖X_te − X*_te M̂‖ / ‖X_te‖

A STATIC mask (ObfuscaTune fixed R, STIP fixed π, ours signed-perm) is a single
global linear map: once ``num_pairs ≥ subdim`` the solve recovers it and the
held-out error → 0 (attack succeeds). A FRESH per-token mask has no global map,
so the held-out error stays ≈ 1 (attack fails) — the intended protection.

Emits AttackResults (attack_id=kpa_known_plaintext) for the security table plus a
full sample-complexity grid. Subspace + subdim are written into every record so
nothing is silently truncated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

METHODS = ["plaintext_gpu", "stip_qwen", "obfuscatune_qwen_orthogonal",
           "ours_amulet_style_signed_perm", "ours_amulet_style_fresh_pad",
           "ours_non_isometric_variant"]
_FRESH = {"ours_amulet_style_fresh_pad", "ours_non_isometric_variant"}
_TRANSFORM = {
    "plaintext_gpu": "identity", "stip_qwen": "fixed_permutation",
    "obfuscatune_qwen_orthogonal": "fixed_orthogonal",
    "ours_amulet_style_signed_perm": "fixed_signed_permutation",
    "ours_amulet_style_fresh_pad": "fresh_orthogonal_per_token",
    "ours_non_isometric_variant": "fresh_non_orthogonal_cond",
}


def _protect(method, Hs, *, seed, cond, orthogonal_matrix, matrix_with_condition_number):
    import torch
    n, d = Hs.shape
    g = torch.Generator().manual_seed(seed)
    if method == "plaintext_gpu":
        return Hs.clone()
    if method == "stip_qwen":
        return Hs[:, torch.randperm(d, generator=g)]
    if method == "obfuscatune_qwen_orthogonal":
        R, _ = orthogonal_matrix(d, seed=seed, dtype=torch.float32)
        return Hs @ R
    if method == "ours_amulet_style_signed_perm":
        perm = torch.randperm(d, generator=g)
        signs = torch.where(torch.rand(d, generator=g) < 0.5, torch.tensor(-1.0), torch.tensor(1.0))
        return Hs[:, perm] * signs
    if method == "ours_amulet_style_fresh_pad":
        return torch.stack([Hs[i] @ orthogonal_matrix(d, seed=seed + 1 + i, dtype=torch.float32)[0]
                            for i in range(n)])
    if method == "ours_non_isometric_variant":
        return torch.stack([Hs[i] @ matrix_with_condition_number(d, cond=cond, seed=seed + 7 + i,
                                                                 dtype=torch.float32)[0]
                            for i in range(n)])
    raise ValueError(method)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--num-tokens", type=int, default=4400)
    p.add_argument("--subdims", default="128,256,512,1024")
    p.add_argument("--holdout", type=int, default=256)
    p.add_argument("--cond", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--table-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    subdims = [int(x) for x in args.subdims.split(",")]
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "outputs" / "attacks" / "kpa_full"
    tdir = Path(args.table_dir) if args.table_dir else out
    if args.dry_run:
        print(f"[dry-run] KPA full model={args.model_name_or_path or 'tiny_random_qwen'} "
              f"subdims={subdims} num_tokens={args.num_tokens} -> {out}")
        return

    import torch
    from pllo.attacks.real_qwen_representations import capture_plaintext_hidden
    from pllo.attacks.metrics import nearest_neighbor_scores, relative_l2_error, token_recovery_topk
    from pllo.attacks.schema import (AttackResult, empty_metrics, make_attack_config,
                                     make_attacker_knowledge, make_input_info, make_runtime)
    from pllo.attacks.result_io import write_jsonl
    from pllo.attacks.table_builder import write_measured_table
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config
    from pllo.baselines.obfuscatune.random_matrices import (matrix_with_condition_number,
                                                            orthogonal_matrix)

    if args.model_name_or_path:
        model = load_qwen(model_name_or_path=args.model_name_or_path, seed=args.seed, dtype=torch.float32)
        family, path = "qwen", args.model_name_or_path
    else:
        model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=args.seed, dtype=torch.float32)
        family, path = "qwen_tiny_random", None
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, model.config.vocab_size, (1, args.num_tokens), generator=g)
    H, table = capture_plaintext_hidden(model, ids)          # (N, hidden), (V, hidden)
    H = H.to(torch.float32)
    hidden = H.shape[1]
    subdims = [s for s in subdims if s <= hidden]
    flat_ids = ids.reshape(-1)

    def _fit_and_eval(Hs, Ps, num_pairs, holdout):
        N = Hs.shape[0]
        tr = slice(0, num_pairs)
        te = slice(num_pairs, num_pairs + holdout)
        Xtr, Ytr = Hs[tr].to(torch.float64), Ps[tr].to(torch.float64)
        M = torch.linalg.lstsq(Ytr, Xtr).solution          # X ≈ X* M  (deobfuscate)
        Xte_hat = (Ps[te].to(torch.float64) @ M)
        err = relative_l2_error(Xte_hat.to(torch.float32), Hs[te])
        return M, float(err), te

    grid = []
    results = []
    for method in METHODS:
        rep_sub = min(1024, hidden)
        rep_np = min(2 * rep_sub, args.num_tokens - args.holdout)
        rep_err, rep_tok = None, {}
        for sub in subdims:
            Hs = H[:, :sub]
            Ps = _protect(method, Hs, seed=args.seed, cond=args.cond,
                          orthogonal_matrix=orthogonal_matrix,
                          matrix_with_condition_number=matrix_with_condition_number)
            sweep_np = [max(1, sub // 4), sub // 2, sub, 2 * sub, 4 * sub]
            sweep_np = [n for n in sweep_np if n + args.holdout <= Hs.shape[0]]
            for np_ in sweep_np:
                M, err, te = _fit_and_eval(Hs, Ps, np_, args.holdout)
                gen = err < 0.1
                grid.append({"method": method, "transform_type": _TRANSFORM[method],
                             "whether_fresh_per_sample": method in _FRESH,
                             "subdim": sub, "num_pairs": np_, "hidden_dim": hidden,
                             "heldout_deobfuscation_relative_l2_error": err,
                             "whether_attack_generalizes_to_heldout": gen})
                if sub == rep_sub and np_ == rep_np:
                    rep_err = err
                    # subspace token recovery: deobfuscate held-out, NN vs table[:, :sub]
                    Xhat = (Ps[te].to(torch.float64) @ M).to(torch.float32)
                    scores = nearest_neighbor_scores(Xhat, table[:, :sub], "cosine")
                    rep_tok = token_recovery_topk(scores, flat_ids[te])
        if rep_err is None and grid:  # rep point not hit (small hidden) -> use best
            best = min([x for x in grid if x["method"] == method],
                       key=lambda x: x["heldout_deobfuscation_relative_l2_error"])
            rep_err = best["heldout_deobfuscation_relative_l2_error"]
            rep_sub, rep_np = best["subdim"], best["num_pairs"]
        success = max(0.0, min(1.0, 1.0 - (rep_err if rep_err is not None else 1.0)))
        m = empty_metrics()
        m["heldout_deobfuscation_relative_l2_error"] = rep_err
        m["matrix_recovery_error"] = rep_err
        m["relative_l2_error"] = rep_err
        m["attack_success_rate"] = success
        m["num_known_pairs"] = rep_np
        for k in ("token_recovery_top1", "token_recovery_top10", "token_recovery_top100"):
            if k in rep_tok:
                m[k] = rep_tok[k]
        trivial = method == "plaintext_gpu"
        note = (f"subspace KPA subdim={rep_sub} num_pairs={rep_np} hidden={hidden}; "
                f"heldout_deobf_err={rep_err:.3e}; fresh={method in _FRESH}; "
                f"generalizes={'yes' if success > 0.9 else 'no'}. "
                + ("TRIVIAL: plaintext, no obfuscation. " if trivial else "")
                + ("INTENDED PROTECTION: fresh per-token mask -> no global map; KPA fails."
                   if method in _FRESH else "static global map -> KPA recovers it once "
                   "num_pairs>=subdim."))
        results.append(AttackResult(
            attack_id="kpa_known_plaintext", attack_name="KPA (linear, subspace sweep)",
            attack_family="cryptanalysis", target_method=method,
            threat_model="known_plaintext", paper_source="KPA on linear obfuscation",
            implementation_level="full", model_family=family, model_name_or_path=path,
            attacker_knowledge=make_attacker_knowledge(has_known_plaintext_pairs=True,
                                                       has_embedding_table=True),
            input_info=make_input_info(hidden_size=hidden, num_samples=args.num_tokens,
                                       dtype="float32"),
            attack_config=make_attack_config(seed=args.seed),
            metrics=m, runtime=make_runtime(), status="measured", notes=note))
        print(f"{method:34s} heldout_deobf_err={rep_err:.3e} success={success:.3f} "
              f"tok1={m.get('token_recovery_top1')}")

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, out / "all_attacks.jsonl")
    (out / "kpa_sample_complexity.json").write_text(json.dumps(grid, indent=2))
    cols = ["method", "transform_type", "whether_fresh_per_sample", "subdim", "num_pairs",
            "hidden_dim", "heldout_deobfuscation_relative_l2_error",
            "whether_attack_generalizes_to_heldout"]
    lines = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in grid]
    (out / "kpa_sample_complexity.csv").write_text("\n".join(lines) + "\n")
    md = ["# KPA sample-complexity sweep (heldout deobfuscation error; lower = attack wins)", ""]
    md += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in grid:
        md.append("| " + " | ".join(f"{r[c]:.3e}" if isinstance(r[c], float) else str(r[c])
                                    for c in cols) + " |")
    (out / "kpa_sample_complexity.md").write_text("\n".join(md) + "\n")
    tdir.mkdir(parents=True, exist_ok=True)
    write_measured_table(results, tdir)
    print(f"wrote {out}/all_attacks.jsonl + kpa_sample_complexity.{{json,csv,md}}")
    print(f"wrote {tdir}/security_attack_table_measured.*")


if __name__ == "__main__":
    main()
