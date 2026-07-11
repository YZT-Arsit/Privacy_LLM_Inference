"""Gate 3 — compare protected (real-TDX) vs plaintext multi-step trajectories.
Per-step loss alignment + error growth + final-state Delta_W/next-logits alignment.
Runs on the H800 (both tensor files local). Small JSON + CSV output."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def rel(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float(((a - b).norm() / (b.norm() + 1e-12)).item())


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prot-json", required=True)
    ap.add_argument("--plain-json", required=True)
    ap.add_argument("--prot-tensors", required=True)
    ap.add_argument("--plain-tensors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    pj = json.loads(Path(args.prot_json).read_text())
    qj = json.loads(Path(args.plain_json).read_text())
    pt = pj["trajectory"]; qt = qj["trajectory"]
    n = min(len(pt), len(qt))
    rows = []; max_loss_abs = 0.0
    for i in range(n):
        d = abs(pt[i]["loss"] - qt[i]["loss"]); max_loss_abs = max(max_loss_abs, d)
        rows.append({"step": i, "loss_protected": pt[i]["loss"], "loss_plaintext": qt[i]["loss"],
                     "loss_abs_diff": d, "protected_finite": pt[i]["finite"]})
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss_protected", "loss_plaintext",
                                          "loss_abs_diff", "protected_finite"])
        w.writeheader(); w.writerows(rows)

    prot = torch.load(args.prot_tensors, map_location="cpu")
    plain = torch.load(args.plain_tensors, map_location="cpu")
    max_dw_rel = 0.0; min_dw_cos = 1.0
    for name in plain["deltaw_after"]:
        max_dw_rel = max(max_dw_rel, rel(prot["deltaw_after"][name], plain["deltaw_after"][name]))
        min_dw_cos = min(min_dw_cos, cos(prot["deltaw_after"][name], plain["deltaw_after"][name]))
    nl_rel = rel(prot["next_logits"], plain["next_logits"])
    top1 = float((prot["next_logits"].float().argmax(-1) == plain["next_logits"].float().argmax(-1)).float().mean())
    all_finite = all(r["protected_finite"] for r in rows)
    summary = {"steps": n, "max_step_loss_abs_diff": max_loss_abs,
               "final_max_deltaw_rel_error": max_dw_rel, "final_min_deltaw_cos": min_dw_cos,
               "final_next_logits_rel_error": nl_rel, "final_top1_agreement": top1,
               "all_protected_steps_finite": all_finite,
               "loss_trajectory_protected": [r["loss_protected"] for r in rows],
               "loss_trajectory_plaintext": [r["loss_plaintext"] for r in rows],
               "invocations_per_step": pj["invocations_per_step"]}
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("steps", "max_step_loss_abs_diff",
        "final_min_deltaw_cos", "final_max_deltaw_rel_error", "final_next_logits_rel_error",
        "final_top1_agreement", "all_protected_steps_finite")}, indent=2))


if __name__ == "__main__":
    main()
