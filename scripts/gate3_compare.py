"""Gate 3 — compare plaintext reference vs protected masked-domain-SGD (exactness).

Loads tensors_<mode>.pt from two runs and reports, per LoRA layer + overall:
  - grad Frobenius-norm match (orthogonal masks preserve ||.||_F exactly, so a correct
    rank-orthogonal mask gives protected gradA/gradB norm == plaintext, to fp tol);
  - Delta_W relative error AFTER one SGD step (B@A; mask cancels -> directly comparable);
  - next-step logits relative error + top-1 agreement;
  - finite rate.
This proves masked-domain SGD tracks plaintext SGD on real Qwen. bf16 tolerances apply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def rel(a, b):
    a = a.float(); b = b.float()
    d = (a - b).norm()
    n = b.norm()
    return float((d / (n + 1e-12)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="tensors_plaintext_ref.pt")
    ap.add_argument("--prot", required=True, help="tensors_protected_*.pt")
    ap.add_argument("--ref-json", required=True)
    ap.add_argument("--prot-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = torch.load(args.ref, map_location="cpu")
    prot = torch.load(args.prot, map_location="cpu")
    rj = json.loads(Path(args.ref_json).read_text())
    pj = json.loads(Path(args.prot_json).read_text())

    per_layer = {}
    max_deltaw_rel = 0.0
    for name in ref["deltaw_after"]:
        dw_ref = ref["deltaw_after"][name]
        dw_prot = prot["deltaw_after"][name]
        dw_rel = rel(dw_prot, dw_ref)
        max_deltaw_rel = max(max_deltaw_rel, dw_rel)
        gnA_ref = rj["grad_report"][name]["gradA_norm"]
        gnA_prot = pj["grad_report"][name]["gradA_norm"]
        gnB_ref = rj["grad_report"][name]["gradB_norm"]
        gnB_prot = pj["grad_report"][name]["gradB_norm"]
        per_layer[name] = {
            "deltaw_rel_error": dw_rel,
            "gradA_norm_ref": gnA_ref, "gradA_norm_prot": gnA_prot,
            "gradA_norm_rel": abs(gnA_prot - gnA_ref) / (abs(gnA_ref) + 1e-12),
            "gradB_norm_ref": gnB_ref, "gradB_norm_prot": gnB_prot,
            "gradB_norm_rel": abs(gnB_prot - gnB_ref) / (abs(gnB_ref) + 1e-12),
        }

    nl_ref = ref["next_logits"].float()
    nl_prot = prot["next_logits"].float()
    next_logits_rel = rel(nl_prot, nl_ref)
    top1_ref = nl_ref.argmax(-1)
    top1_prot = nl_prot.argmax(-1)
    top1_agreement = float((top1_ref == top1_prot).float().mean().item())

    summary = {
        "ref_mode": rj["mode"], "prot_mode": pj["mode"], "dtype": rj["dtype"],
        "loss_ref": rj["loss"], "loss_prot": pj["loss"],
        "loss_abs_diff": abs(rj["loss"] - pj["loss"]),
        "max_deltaw_rel_error": max_deltaw_rel,
        "next_logits_rel_error": next_logits_rel,
        "top1_agreement": top1_agreement,
        "max_gradA_norm_rel": max(v["gradA_norm_rel"] for v in per_layer.values()),
        "max_gradB_norm_rel": max(v["gradB_norm_rel"] for v in per_layer.values()),
        "num_layers": len(per_layer),
        "ref_invocations_total": rj["invocations_total"],
        "prot_invocations_total": pj["invocations_total"],
        "prot_packed_update_calls": pj["packed_update_calls"],
        "prot_trusted_optimizer_calls": pj["trusted_optimizer_calls"],
        "finite": bool(torch.isfinite(nl_prot).all() and torch.isfinite(nl_ref).all()),
        "per_layer": per_layer,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in (
        "loss_abs_diff", "max_deltaw_rel_error", "next_logits_rel_error",
        "top1_agreement", "max_gradA_norm_rel", "max_gradB_norm_rel",
        "prot_packed_update_calls", "prot_trusted_optimizer_calls", "finite")}, indent=2))


if __name__ == "__main__":
    main()
