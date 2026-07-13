#!/usr/bin/env python3
"""Narrow V2-only alignment diagnostics on the frozen transformed package.

This primary condition never opens a plaintext checkpoint.  Its comparison
model is an independently initialized same-shape S0 control, not a trained
same-lineage model and not a known-plaintext pair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from view_contracts import View, ViewRecord


def load_transformed(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    tensor = next(iter(obj.values())) if isinstance(obj, dict) else obj
    return tensor.detach().float().contiguous()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xy = (x.T @ y).square().sum()
    denom = (x.T @ x).square().sum().sqrt() * (y.T @ y).square().sum().sqrt()
    return float(xy / denom.clamp_min(1e-30))


def profile(t: torch.Tensor, seed: int) -> dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    rows = min(128, t.shape[0])
    cols = min(128, t.shape[1])
    ridx = torch.randperm(t.shape[0], generator=g)[:rows]
    cidx = torch.randperm(t.shape[1], generator=g)[:cols]
    s = t[ridx][:, cidx]
    sv = torch.linalg.svdvals(s.double()).float()
    row_norm = t.norm(dim=1)
    flat = t.abs().flatten()
    return {
        "mean": float(t.mean()), "std": float(t.std()),
        "fro_norm": float(t.norm()),
        "row_norm_mean": float(row_norm.mean()),
        "row_norm_std": float(row_norm.std()),
        "abs_q50": float(torch.quantile(flat, 0.50)),
        "abs_q90": float(torch.quantile(flat, 0.90)),
        "abs_q99": float(torch.quantile(flat, 0.99)),
        "sketch_singular_max": float(sv.max()),
        "sketch_singular_mean": float(sv.mean()),
        "sketch_stable_rank": float(sv.square().sum() / sv.max().square().clamp_min(1e-30)),
    }


def compare(w: torch.Tensor, control: torch.Tensor, seed: int) -> dict[str, float]:
    g = torch.Generator().manual_seed(seed + 10000)
    rows = min(128, w.shape[0])
    cols = min(128, w.shape[1])
    ridx = torch.randperm(w.shape[0], generator=g)[:rows]
    cidx = torch.randperm(w.shape[1], generator=g)[:cols]
    x = w[ridx][:, cidx].double()
    y = control[ridx][:, cidx].double()
    sx = torch.linalg.svdvals(x)
    sy = torch.linalg.svdvals(y)
    spectral_cos = torch.nn.functional.cosine_similarity(sx, sy, dim=0)
    # Unpaired row-norm distribution matching (1-D OT/Wasserstein diagnostic).
    rx = w.norm(dim=1).sort().values
    ry = control.norm(dim=1).sort().values
    wasserstein = (rx - ry).abs().mean() / rx.mean().clamp_min(1e-30)
    # Procrustes in a bounded 128-D sketch; row ordering is deliberately unpaired.
    xs = x[x.norm(dim=1).argsort()]
    ys = y[y.norm(dim=1).argsort()]
    u, _, vh = torch.linalg.svd(ys.T @ xs, full_matrices=False)
    aligned = ys @ (u @ vh)
    proc = (aligned - xs).norm() / xs.norm().clamp_min(1e-30)
    return {
        "spectral_profile_cosine": float(spectral_cos),
        "linear_cka_unpaired_index_control": linear_cka(x.float(), y.float()),
        "row_norm_ot_relative_distance": float(wasserstein),
        "unpaired_norm_sorted_procrustes_relative_error": float(proc),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", required=True, choices=["V2"])
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layers", default="0,12,23")
    ap.add_argument("--projections", default="q_proj,k_proj,v_proj,o_proj")
    args = ap.parse_args()
    view = View(args.view)
    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projections.split(",")
    rows = []
    for layer in layers:
        for proj in projs:
            path = args.package / f"model.layers.{layer}.self_attn.{proj}_tilde.pt"
            w = load_transformed(path)
            # Validate that the attack receives only a transformed artifact record.
            ViewRecord.build(view, {
                "sample_id": f"layer-{layer}-{proj}", "run_id": "package_static",
                "task": "alignment", "split": "diagnostic",
                "transformed_base": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                "tensor_shapes": {"weight": list(w.shape)},
            })
            g = torch.Generator().manual_seed(args.seed + layer * 101 + len(proj))
            control = torch.randn(w.shape, generator=g, dtype=w.dtype) * 0.02
            row = {
                "condition": "PRIMARY_PRIVATE_BASE_PROXY",
                "view": "V2", "attacker_initialization": "S0_random_same_shape",
                "layer": layer, "projection": proj, "rows": w.shape[0], "cols": w.shape[1],
                **{f"transformed_{k}": v for k, v in profile(w, args.seed + layer).items()},
                **compare(w, control, args.seed + layer),
            }
            rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_layer_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "schema": "authorized_primary_alignment", "version": "1.0",
        "condition": "PRIMARY_PRIVATE_BASE_PROXY", "view": "V2",
        "plaintext_checkpoint_accessed": False,
        "control": "S0 independently randomized same-shape tensors",
        "trained_unrelated_model_available": False,
        "package": str(args.package.resolve()), "layers": layers, "projections": projs,
        "seed": args.seed, "cells": len(rows),
    }
    (args.output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
