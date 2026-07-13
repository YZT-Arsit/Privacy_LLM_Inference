#!/usr/bin/env python3
"""V3 known-plaintext signed-permutation recovery positive control.

This deliberately lies outside the primary threat model.  It receives paired
plaintext/transformed weights and uses the evaluator secret only after recovery
to score the estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from view_contracts import View, ViewRecord


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from pllo.ops.masked_training_kernels import orthogonal_signed_perm  # noqa: E402


def load_tilde(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return next(iter(obj.values())).double()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, choices=["V3"])
    ap.add_argument("--condition", required=True, choices=["KNOWN_PLAINTEXT_STRESS_TEST"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layers", default="0,12,23")
    args = ap.parse_args()
    ViewRecord.build(View(args.view), {
        "sample_id": "known-plaintext-control", "run_id": "v3_positive_control",
        "task": "alignment", "split": "diagnostic",
        "plaintext_weights": {"checkpoint": str(args.checkpoint)},
    })
    sd = load_file(str(args.checkpoint / "model.safetensors"), device="cpu")
    layers = [int(x) for x in args.layers.split(",")]
    plain_grams = []
    tilde_grams = []
    pair_meta = []
    for layer in layers:
        gamma = sd[f"model.layers.{layer}.input_layernorm.weight"].double()
        for proj in ("q_proj", "k_proj", "v_proj"):
            w = sd[f"model.layers.{layer}.self_attn.{proj}.weight"].double()
            wt_path = args.package / f"model.layers.{layer}.self_attn.{proj}_tilde.pt"
            wt = load_tilde(wt_path)
            wg = w * gamma.unsqueeze(0)
            plain_grams.append(wg.T @ wg)
            tilde_grams.append(wt.T @ wt)
            pair_meta.append({"layer": layer, "projection": proj,
                              "tilde_sha256": hashlib.sha256(wt_path.read_bytes()).hexdigest()})

    # Each transformed coordinate has a multi-Gram diagonal fingerprint equal to
    # one original coordinate's fingerprint; signs cancel on the diagonal.
    fp = torch.stack([g.diag() for g in plain_grams], dim=1)
    fpt = torch.stack([g.diag() for g in tilde_grams], dim=1)
    scale = fp.std(dim=0).clamp_min(1e-30)
    distance = torch.cdist(fpt / scale, fp / scale)
    perm = distance.argmin(dim=1)  # transformed column j -> plaintext coordinate perm[j]
    unique = int(perm.unique().numel())
    nearest_gap = distance.topk(2, largest=False).values

    # Relative signs use all Gram families against a stable anchor coordinate.
    anchor_scores = fp.abs().mean(dim=1)
    anchor_plain = int(anchor_scores.argmax())
    anchor_j_candidates = (perm == anchor_plain).nonzero().flatten()
    anchor_j = int(anchor_j_candidates[0]) if anchor_j_candidates.numel() else 0
    signs = torch.ones(perm.numel(), dtype=torch.float64)
    for j in range(perm.numel()):
        score = sum(
            tilde_grams[k][anchor_j, j] * plain_grams[k][perm[anchor_j], perm[j]]
            for k in range(len(plain_grams))
        )
        signs[j] = 1.0 if score >= 0 else -1.0
    nhat = torch.zeros(perm.numel(), perm.numel(), dtype=torch.float64)
    nhat[perm, torch.arange(perm.numel())] = signs

    # Evaluator-only ground truth scoring. Global sign is functionally irrelevant.
    ntrue = orthogonal_signed_perm(perm.numel(), seed=9000, dtype=torch.float64)
    direct = (nhat - ntrue).norm() / ntrue.norm()
    global_flip = (-nhat - ntrue).norm() / ntrue.norm()
    basis_rel_error = float(torch.minimum(direct, global_flip))
    true_perm = ntrue.abs().argmax(dim=0)
    permutation_accuracy = float((perm == true_perm).double().mean())
    est_sign = nhat[true_perm, torch.arange(perm.numel())]
    true_sign = ntrue[true_perm, torch.arange(perm.numel())]
    sign_accuracy = max(float((est_sign == true_sign).double().mean()),
                        float((-est_sign == true_sign).double().mean()))

    gram_errors = []
    for s, st in zip(plain_grams, tilde_grams):
        pred = nhat.T @ s @ nhat
        gram_errors.append(float((pred - st).norm() / st.norm().clamp_min(1e-30)))
    result = {
        "schema": "known_plaintext_alignment_stress", "version": "1.0",
        "condition": args.condition, "view": args.view,
        "outside_primary_threat_model": True,
        "plaintext_checkpoint_accessed": True,
        "paired_weight_families": len(pair_meta), "pairs": pair_meta,
        "dimension": perm.numel(), "unique_assignments": unique,
        "permutation_accuracy": permutation_accuracy,
        "sign_accuracy_up_to_global_flip": sign_accuracy,
        "basis_relative_error_up_to_global_flip": basis_rel_error,
        "gram_reconstruction_relative_error_mean": sum(gram_errors) / len(gram_errors),
        "gram_reconstruction_relative_error_max": max(gram_errors),
        "nearest_fingerprint_margin_mean": float((nearest_gap[:, 1] - nearest_gap[:, 0]).mean()),
        "positive_control_pass": permutation_accuracy > 0.99 and sign_accuracy > 0.99,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
