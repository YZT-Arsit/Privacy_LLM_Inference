"""Gate 3 — compare protected_tdx (real TDX loss) vs plaintext reference (one step).

Reports loss error, dlogits rel error + cosine, per-layer recovered gradA/gradB rel
error + cosine (using U to un-mask the protected grads for ANALYSIS only), Delta_W rel
error + cosine, next-logits rel error, top1 agreement, finite. bf16 tolerances apply.
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--ref", required=True)          # tensors_plaintext_ref.pt (batch-matched)
    ap.add_argument("--prot", required=True)         # tensors_protected_tdx.pt
    ap.add_argument("--ref-json", required=True)
    ap.add_argument("--prot-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ref = torch.load(args.ref, map_location="cpu")
    prot = torch.load(args.prot, map_location="cpu")
    rj = json.loads(Path(args.ref_json).read_text())
    pj = json.loads(Path(args.prot_json).read_text())

    per_layer = {}
    max_dw_rel = 0.0; min_dw_cos = 1.0
    max_gA_rel = 0.0; min_gA_cos = 1.0; max_gB_rel = 0.0; min_gB_cos = 1.0
    for name in ref["deltaw_after"]:
        dw_rel = rel(prot["deltaw_after"][name], ref["deltaw_after"][name])
        dw_cos = cos(prot["deltaw_after"][name], ref["deltaw_after"][name])
        max_dw_rel = max(max_dw_rel, dw_rel); min_dw_cos = min(min_dw_cos, dw_cos)
        # recover protected grads to plaintext frame: gradA = U^T gradA_t, gradB = gradB_t U
        gA_t, gB_t = prot["grads"][name]; U = prot["U"][name]
        gA_rec = U.T.float() @ gA_t.float(); gB_rec = gB_t.float() @ U.float()
        gA_ref, gB_ref = ref["grads"][name]
        gA_rel = rel(gA_rec, gA_ref); gB_rel = rel(gB_rec, gB_ref)
        gA_cos = cos(gA_rec, gA_ref); gB_cos = cos(gB_rec, gB_ref)
        max_gA_rel = max(max_gA_rel, gA_rel); min_gA_cos = min(min_gA_cos, gA_cos)
        max_gB_rel = max(max_gB_rel, gB_rel); min_gB_cos = min(min_gB_cos, gB_cos)
        per_layer[name] = {"deltaw_rel": dw_rel, "deltaw_cos": dw_cos,
                           "gradA_rel": gA_rel, "gradA_cos": gA_cos,
                           "gradB_rel": gB_rel, "gradB_cos": gB_cos}

    nl_rel = rel(prot["next_logits"], ref["next_logits"])
    top1 = float((prot["next_logits"].float().argmax(-1) == ref["next_logits"].float().argmax(-1)).float().mean())
    dlog_rel = rel(prot["dlogits"], ref["dlogits"]) if "dlogits" in ref else None
    dlog_cos = cos(prot["dlogits"], ref["dlogits"]) if "dlogits" in ref else None

    loss_ref = rj["loss"]; loss_prot = pj["trajectory"][0]["loss"]
    summary = {
        "loss_ref": loss_ref, "loss_prot": loss_prot,
        "loss_abs_diff": abs(loss_ref - loss_prot),
        "dlogits_rel_error": dlog_rel, "dlogits_cos": dlog_cos,
        "max_deltaw_rel_error": max_dw_rel, "min_deltaw_cos": min_dw_cos,
        "max_gradA_rel_error": max_gA_rel, "min_gradA_cos": min_gA_cos,
        "max_gradB_rel_error": max_gB_rel, "min_gradB_cos": min_gB_cos,
        "next_logits_rel_error": nl_rel, "top1_agreement": top1,
        "finite": bool(torch.isfinite(prot["next_logits"]).all()),
        "prot_invocations": pj["invocations_per_step"],
        "num_layers": len(per_layer), "per_layer": per_layer,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in (
        "loss_abs_diff", "dlogits_rel_error", "dlogits_cos", "max_deltaw_rel_error",
        "min_deltaw_cos", "max_gradA_rel_error", "min_gradA_cos", "max_gradB_rel_error",
        "min_gradB_cos", "next_logits_rel_error", "top1_agreement", "finite")}, indent=2))


if __name__ == "__main__":
    main()
