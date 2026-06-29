#!/usr/bin/env python
"""Small-model Amulet right-mask + Linear-boundary pad MLP probe (real Qwen weights).

Loads a local Qwen2.5 model (e.g. Qwen2.5-0.5B-Instruct / Qwen2.5-3B-Instruct),
extracts one or more real decoder-layer SwiGLU MLP weight sets, and validates the
pipeline:

    X
     -> pad-enabled gate/up Linear  (G_tilde = G N_ff, U_tilde = U N_ff)
     -> Amulet right-mask SwiGLU    (A_tilde = [SiLU(G) * U] N_ff)
     -> pad-enabled down Linear     (Y_tilde = Y N_out)

The right-masked output `Y_tilde` is checked against the plaintext MLP output
`Y N_out` (and recovered `Y`). This is a SMALL-MODEL real-weight nonlinear-island
PROBE, not the full Qwen7B production path.

If transformers / the model files are unavailable, the script skips cleanly
(exit 0) and writes a report with `skipped: true`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.ops.amulet_right_mask_islands import (  # noqa: E402
    run_amulet_right_mask_qwen_mlp_with_linear_pad,
)


def _invertible(dim: int, dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:
    M = torch.randn(dim, dim, dtype=dtype, generator=g)
    while abs(float(torch.linalg.det(M).item())) < 1e-3:
        M = torch.randn(dim, dim, dtype=dtype, generator=g)
    return M


def _load_qwen_mlp_layers(model_path: str, num_layers: int, dtype: torch.dtype):
    """Return [(w_gate, w_up, w_down)] for the first ``num_layers`` decoder MLPs.

    Raises on any failure so the caller can skip cleanly."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    layers = model.model.layers
    out = []
    for i in range(min(num_layers, len(layers))):
        mlp = layers[i].mlp
        # row-vector convention y = x @ W ; nn.Linear stores [out, in] -> .T
        w_gate = mlp.gate_proj.weight.detach().to(dtype).T.contiguous()
        w_up = mlp.up_proj.weight.detach().to(dtype).T.contiguous()
        w_down = mlp.down_proj.weight.detach().to(dtype).T.contiguous()
        out.append((w_gate, w_up, w_down))
    cfg = model.config
    del model
    return out, int(cfg.hidden_size), int(cfg.intermediate_size)


def run(args: argparse.Namespace) -> dict:
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    device = torch.device(args.device)
    tol = 1e-6 if dtype == torch.float64 else 5e-2

    base = {
        "experiment": "qwen_small_amulet_pad_mlp_probe",
        "model_name": args.model_name,
        "model_path": args.model_path,
        "main_scheme":
            "linear_boundary_additive_pad_plus_amulet_right_mask_nonlinear",
        "dtype": args.dtype,
        "device": str(device),
        "kronecker_size": args.kronecker_size,
        "seq_len": args.seq_len,
        "paper_scope": "small_model_real_weight_nonlinear_island_probe",
        "production_qwen7b_integration": False,
        "formal_security_claim": False,
    }

    try:
        layers, hidden, inter = _load_qwen_mlp_layers(
            args.model_path, args.num_layers, dtype)
    except Exception as exc:  # noqa: BLE001 - intentional clean skip
        base.update({
            "skipped": True,
            "skip_reason": f"{type(exc).__name__}: {exc}",
            "uses_real_qwen_weights": False,
            "paper_ready_small_model_amulet_probe": False,
        })
        return base

    g = torch.Generator()
    g.manual_seed(args.seed)
    per_layer = []
    max_abs = 0.0
    max_rel = 0.0
    for li, (w_gate, w_up, w_down) in enumerate(layers):
        w_gate = w_gate.to(device)
        w_up = w_up.to(device)
        w_down = w_down.to(device)
        d = w_gate.shape[0]
        f = w_gate.shape[1]
        X = torch.randn(args.seq_len, d, dtype=dtype, generator=g, device=device)
        n_in = _invertible(d, dtype, g).to(device)
        n_in_inv = torch.linalg.inv(n_in)
        n_ff = _invertible(f, dtype, g).to(device)
        n_out = _invertible(d, dtype, g).to(device)
        r = run_amulet_right_mask_qwen_mlp_with_linear_pad(
            X, w_gate, None, w_up, None, w_down, None,
            n_in, n_in_inv, n_ff, n_out,
            k=args.kronecker_size, generator=g, pad_scale=args.pad_scale,
        )
        la = float(r["max_abs_error"])
        lr = float(r["relative_l2_error"])
        max_abs = max(max_abs, la)
        max_rel = max(max_rel, lr)
        per_layer.append({
            "layer": li, "hidden_size": d, "intermediate_size": f,
            "max_abs_error": la, "relative_l2_error": lr,
            "gate_clean_err": float(r["gate_clean_err"]),
            "up_clean_err": float(r["up_clean_err"]),
            "verified": la <= tol,
        })

    matched = all(p["verified"] for p in per_layer)
    base.update({
        "skipped": False,
        "uses_real_qwen_weights": True,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_layers_checked": len(per_layer),
        "linear_boundary_pad_enabled": True,
        "amulet_right_mask_swiglu_enabled": True,
        "gate_up_down_pad_enabled": True,
        "pad_enters_nonlinear_island": False,
        "tolerance": tol,
        "max_abs_error": max_abs,
        "relative_l2_error": max_rel,
        "tokens_or_mlp_output_match": bool(matched),
        "paper_ready_small_model_amulet_probe": bool(matched),
        "per_layer": per_layer,
    })
    if not matched:
        base["paper_ready"] = False
        base["paper_ready_blocker"] = (
            "small-model real-weight Amulet pad MLP probe exceeded tolerance "
            f"{tol}")
    return base


def _markdown(rep: dict) -> str:
    L = [f"# Small-model Amulet + Linear-pad MLP probe ({rep['model_name']})", ""]
    if rep.get("skipped"):
        L += [f"**SKIPPED**: {rep.get('skip_reason')}", "",
              "transformers / model files unavailable; nothing was run.", ""]
        return "\n".join(L)
    L += [
        f"- model_path: `{rep['model_path']}`",
        f"- uses_real_qwen_weights: {rep['uses_real_qwen_weights']}",
        f"- num_layers_checked: {rep['num_layers_checked']}  "
        f"(hidden={rep['hidden_size']}, intermediate={rep['intermediate_size']})",
        f"- dtype/device: {rep['dtype']}/{rep['device']}  k={rep['kronecker_size']}",
        "",
        "## Pipeline (per real decoder MLP)",
        "",
        "`X -> pad gate/up Linear -> Amulet SwiGLU -> pad down Linear -> Y N_out`",
        "",
        f"- linear_boundary_pad_enabled: {rep['linear_boundary_pad_enabled']}",
        f"- gate_up_down_pad_enabled: {rep['gate_up_down_pad_enabled']}",
        f"- amulet_right_mask_swiglu_enabled: {rep['amulet_right_mask_swiglu_enabled']}",
        f"- pad_enters_nonlinear_island: {rep['pad_enters_nonlinear_island']}",
        f"- max_abs_error: {rep['max_abs_error']:.3e}  "
        f"relative_l2_error: {rep['relative_l2_error']:.3e}  (tol {rep['tolerance']})",
        f"- tokens_or_mlp_output_match: {rep['tokens_or_mlp_output_match']}",
        f"- paper_ready_small_model_amulet_probe: "
        f"{rep['paper_ready_small_model_amulet_probe']}",
        "",
        "| layer | hidden | inter | max_abs_error | rel_l2 | verified |",
        "|---|---|---|---|---|---|",
    ]
    for p in rep["per_layer"]:
        L.append(f"| {p['layer']} | {p['hidden_size']} | {p['intermediate_size']} "
                 f"| {p['max_abs_error']:.3e} | {p['relative_l2_error']:.3e} "
                 f"| {'yes' if p['verified'] else 'NO'} |")
    L += ["",
          f"_paper_scope: {rep['paper_scope']}; "
          f"production_qwen7b_integration: {rep['production_qwen7b_integration']}_", ""]
    return "\n".join(L)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True,
                   help="local HF / ModelScope model directory (real Qwen weights)")
    p.add_argument("--model-name", default="Qwen2.5-0.5B-Instruct")
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--kronecker-size", type=int, default=3)
    p.add_argument("--pad-scale", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--linear-boundary-pad", dest="linear_boundary_pad",
                   action="store_true", default=True)
    p.add_argument("--output-json", type=Path,
                   default=PROJECT_ROOT / "outputs" / "amulet_small_qwen"
                   / "qwen_amulet_pad_mlp.json")
    p.add_argument("--output-md", type=Path,
                   default=PROJECT_ROOT / "outputs" / "amulet_small_qwen"
                   / "qwen_amulet_pad_mlp.md")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rep = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown(rep), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"\nWrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    if rep.get("skipped"):
        print("SKIPPED (model/transformers unavailable) — clean exit 0")
        return 0
    if not rep.get("tokens_or_mlp_output_match", False):
        raise SystemExit("FAILED: small-model Amulet pad MLP probe exceeded tol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
