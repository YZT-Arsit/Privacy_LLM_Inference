"""H800 package-native D4 worker (untrusted). Forward/backward split so the masked
logits can round-trip through the real TDX loss boundary without a long-lived process
(the LoRA state persists to disk; backward recomputes the deterministic forward).

Modes
  init      : create the masked LoRA state (rank-masked leaves) for all 7x24 targets.
  forward   : load state + package, run the 24-layer masked forward, SAVE masked logits.
  backward  : load state, RECOMPUTE forward (deterministic), apply the TDX-returned
              masked-domain dlogits, one GPU masked-SGD step, SAVE new state + effective
              masked dW per target + grad presence + counters.

The worker only ever touches the transformed package + masked LoRA + masked dlogits.
No plaintext base/embedding/hidden/logits/labels; no HF model; no vocab permutation.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from h800_unified_worker import (  # noqa: E402
    PackageNativeLoader, MaskedQwen, fail_closed_checks, compute_root_hash,
    new_counters, LORA_TARGETS, PKG, CKPT, EXPECTED_ROOT_HASH)

OUT = REPO / "results/aaai_private_base/gate0_d4"
MSG = OUT / "msg"
DTYPE = torch.float32     # deployment bf16 exercised separately; fp32 for clean equivalence


def load_model():
    device = torch.device("cuda")
    cfg = json.loads((CKPT / "config.json").read_text())
    loader = PackageNativeLoader(PKG, device, DTYPE)
    loader.load_all(cfg["num_hidden_layers"])
    model = MaskedQwen(loader.tensors, cfg, device, DTYPE)
    return model, loader, cfg, device


def rank_masked_init(l, proj, in_d, out_d, rank=8, seed_base=7000):
    """Masked-domain LoRA leaves A_t=U@A0, B_t=B0@U^T with an ORTHOGONAL rank mask U
    (paper_safe: naive masked SGD recovers plaintext SGD). Tiny nonzero B0 so both
    grads flow at step 0."""
    g = torch.Generator().manual_seed(seed_base + l * 10 + LORA_TARGETS.index(proj))
    A0 = torch.randn(rank, in_d, generator=g, dtype=torch.float64) * 0.02
    B0 = torch.randn(out_d, rank, generator=g, dtype=torch.float64) * 1e-4
    U, _ = torch.linalg.qr(torch.randn(rank, rank, generator=g, dtype=torch.float64))
    A_t = (U @ A0); B_t = (B0 @ U.T)
    return A_t.to(DTYPE), B_t.to(DTYPE)


def install_lora(model, state):
    for l in range(model.L):
        for proj in LORA_TARGETS:
            A, B = state[f"{l}.{proj}"]
            A = A.to(model.device, DTYPE).requires_grad_(True)
            B = B.to(model.device, DTYPE).requires_grad_(True)
            model.lora[(l, proj)] = (A, B)


def effective_dw(model):
    """Per-target effective masked delta-W = scale * B_t @ A_t (the masked-domain
    weight delta the LoRA adds). Returned to the trusted verifier for un-folding."""
    dw = {}
    for (l, proj), (A, B) in model.lora.items():
        dw[f"{l}.{proj}"] = (model.scale * (B.detach() @ A.detach())).cpu()
    return dw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["init", "forward", "backward"])
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--state", default=str(OUT / "masked_lora_state.pt"))
    ap.add_argument("--input-ids", default=str(
        REPO / "results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json"))
    ap.add_argument("--seq-len", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dlogits", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); MSG.mkdir(parents=True, exist_ok=True)

    if args.mode == "init":
        model, loader, cfg, device = load_model()
        state = {}
        for l in range(model.L):
            for proj in LORA_TARGETS:
                W = loader.tensors[f"L{l}.{proj}.w"]
                A, B = rank_masked_init(l, proj, W.shape[1], W.shape[0])
                state[f"{l}.{proj}"] = (A.cpu(), B.cpu())
        torch.save(state, args.state)
        # fail-closed + provenance snapshot
        root = compute_root_hash(PKG)
        fc = fail_closed_checks(cfg, "paper_safe", "package_native_A_rightmul", root)
        (OUT / "d4_fail_closed_tests.json").write_text(json.dumps(fc, indent=2))
        print(json.dumps({"mode": "init", "targets": len(state),
                          "root_hash_ok": root == EXPECTED_ROOT_HASH,
                          "fail_closed_all_pass": all(t["passed"] for t in fc)}))
        return

    meta = json.loads(Path(args.input_ids).read_text())
    ids = torch.tensor(meta["input_ids"][:args.seq_len])
    model, loader, cfg, device = load_model()
    state = torch.load(args.state, map_location="cpu")
    install_lora(model, state)
    input_ids = ids.to(device)
    counters = new_counters()

    if args.mode == "forward":
        t0 = time.time()
        logits_masked = model.forward(input_ids, counters)
        torch.cuda.synchronize(); fwd_t = time.time() - t0
        torch.save(logits_masked.detach().cpu(), MSG / f"logits_step{args.step}.pt")
        (MSG / f"forward_meta_step{args.step}.json").write_text(json.dumps({
            "step": args.step, "seq_len": int(input_ids.shape[0]),
            "vocab": int(logits_masked.shape[1]), "forward_sec": fwd_t,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "attention_score_exposures": model.attention_score_exposures,
            "finite": bool(torch.isfinite(logits_masked).all())}))
        print(json.dumps({"mode": "forward", "step": args.step, "fwd_sec": round(fwd_t, 3)}))
        return

    if args.mode == "backward":
        t0 = time.time()
        logits_masked = model.forward(input_ids, counters)      # deterministic recompute
        dlog = torch.load(args.dlogits, map_location=device).to(DTYPE)
        params, keys = [], []
        for (l, proj), (A, B) in model.lora.items():
            params += [A, B]; keys += [(l, proj, "A"), (l, proj, "B")]
        grads = torch.autograd.grad(logits_masked, params, grad_outputs=dlog,
                                    allow_unused=True)
        # one GPU masked-SGD step (masked/rank domain)
        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None:
                    p -= args.lr * g
        torch.cuda.synchronize(); bwd_t = time.time() - t0
        # per-target grad presence
        per_target = []
        for i in range(0, len(params), 2):
            l, proj, _ = keys[i]
            gA, gB = grads[i], grads[i + 1]
            per_target.append({"layer": l, "proj": proj,
                               "gradA_present": gA is not None and bool(torch.isfinite(gA).all()) and float(gA.abs().sum()) > 0,
                               "gradB_present": gB is not None and bool(torch.isfinite(gB).all()) and float(gB.abs().sum()) > 0})
        # save updated state + effective dW (before/after) for the verifier
        new_state = {f"{l}.{proj}": (A.detach().cpu(), B.detach().cpu())
                     for (l, proj), (A, B) in model.lora.items()}
        torch.save(new_state, args.state)
        # effective masked dW is derived by the trusted-eval verifier from the saved
        # LoRA state (17 MB); no multi-GB dW file is written or ferried.
        base_grad = any(t.requires_grad for t in loader.tensors.values())
        connected = sum(1 for r in per_target if r["gradA_present"] and r["gradB_present"])
        counters["attention_score_exposures"] = model.attention_score_exposures
        counters["total_logical_trusted_invocations_per_step"] = 2
        result = {"mode": "backward", "step": args.step, "backward_sec": bwd_t,
                  "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  "lora_targets_connected": connected, "lora_targets_total": len(per_target),
                  "all_targets_connected": connected == len(per_target),
                  "base_tensors_require_grad": base_grad,
                  "step_finite": all(bool(torch.isfinite(p).all()) for p in params),
                  "counters": counters}
        (OUT / f"backward_result_step{args.step}.json").write_text(json.dumps(result, indent=2))
        with open(OUT / "per_layer_gradients.csv", "w") as f:
            f.write("layer,proj,gradA_present,gradB_present\n")
            for r in per_target:
                f.write(f"{r['layer']},{r['proj']},{r['gradA_present']},{r['gradB_present']}\n")
        print(json.dumps({k: result[k] for k in
                          ["mode", "step", "all_targets_connected",
                           "base_tensors_require_grad", "step_finite"]}))
        return


if __name__ == "__main__":
    main()
