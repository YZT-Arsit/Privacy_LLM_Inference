"""Gate 3 — H800 single/multi step with REAL TDX private loss (direct H800<->TDX).

Two phases (so the Mac verifier stays in the control plane only):
  challenge : POST /train/challenge with the MAC-issued nonce, write bundle.json.
  run       : (after Mac authorizes) handshake + authorized init (register labels,
              receive pi) + N masked-SGD steps with the real TDX private CE, save
              tensors + per-step trajectory. Logits/dlogits go directly H800<->TDX.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate3_worker import (  # noqa: E402
    TARGET_ATTN, TARGET_MLP, build_batch, flat_logits_labels, inject_lora, load_model, log)
from gate3_tdx_client import TdxLossSession  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["challenge", "run"])
    ap.add_argument("--tdx-url", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config-digest", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--gradient-convention", default="nout_dual")
    ap.add_argument("--optimizer-mode", default="gpu_masked_sgd")
    ap.add_argument("--nonce", default="")
    ap.add_argument("--bundle-out", default="")
    ap.add_argument("--bundle-in", default="")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--targets", default="attn", choices=["attn", "attn_mlp"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--out", default="/root/gate3/out")
    args = ap.parse_args()

    sess = TdxLossSession(args.tdx_url)

    if args.phase == "challenge":
        bundle = sess.challenge(run_id=args.run_id, config_digest=args.config_digest,
                                model_id=args.model_id,
                                gradient_convention=args.gradient_convention,
                                optimizer_mode=args.optimizer_mode, verifier_nonce=args.nonce)
        Path(args.bundle_out).write_text(json.dumps(bundle))
        log(f"challenge done; quote_available={bundle.get('quote_available')} "
            f"appraisal_ok={bundle.get('appraisal_ok')} sig={bundle.get('signature_verified')} "
            f"-> {args.bundle_out}")
        return

    # ---- run phase: real single/multi step ----
    bundle = json.loads(Path(args.bundle_in).read_text())
    sess.handshake(bundle, run_id=args.run_id, config_digest=args.config_digest,
                   gradient_convention=args.gradient_convention,
                   optimizer_mode=args.optimizer_mode)
    log("attested AEAD session established (direct H800<->TDX)")

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    targets = TARGET_ATTN if args.targets == "attn" else TARGET_ATTN + TARGET_MLP
    model, tok = load_model(args.model_dir, dtype)
    vocab = model.config.vocab_size
    texts = json.loads(Path(args.data).read_text())[: args.batch]
    input_ids, attn, labels = build_batch(tok, texts, args.seqlen, "cuda")

    layers = inject_lora(model, targets, args.rank, args.alpha, args.seed, masked=True)
    trainable = [p for l in layers.values() for p in l.parameters() if p.requires_grad]
    opt = torch.optim.SGD(trainable, lr=args.lr, momentum=0.0, weight_decay=0.0)

    # authorized init: register the private labels (shifted, flat) in TDX + receive pi.
    # Labels are computed once and registered per-step id 0..steps-1 (fixed batch).
    with torch.no_grad():
        out0 = model(input_ids=input_ids, attention_mask=attn)
        _, lab0, _ = flat_logits_labels(out0.logits.float(), labels)
    lab_list = lab0.cpu().tolist()
    # gpu_masked_sgd: the GPU owns the LoRA + its masked optimizer. The TDX loss service
    # is loss-only (head permutation + private labels) and holds NO LoRA state, so the
    # init manifest is EMPTY -- the service registers labels + the vocab permutation only.
    sess.init(lora_manifest=[], vocab_size=vocab,
              labels_by_step={s: lab_list for s in range(args.steps)})
    log(f"authorized init: registered labels + received pi (len {sess.pi.numel()}); "
        f"TDX holds NO LoRA/optimizer state")

    traj = []
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps):
        t0 = time.time()
        out = model(input_ids=input_ids, attention_mask=attn)
        Z, lab, valid = flat_logits_labels(out.logits.float(), labels)
        fwd_t = time.time() - t0
        loss_val, G = sess.logits_loss(step, Z)             # REAL TDX private CE
        opt.zero_grad(set_to_none=True)
        t1 = time.time()
        (Z * G.to(Z.dtype)).sum().backward()                # inject TDX dlogits
        bwd_t = time.time() - t1
        finite = all(bool(torch.isfinite(p.grad).all()) for p in trainable if p.grad is not None)
        dw_before = {n: l.delta_w().detach().float().cpu() for n, l in layers.items()}
        opt.step()                                          # masked-domain SGD (no unmask)
        dw_after = {n: l.delta_w().detach().float().cpu() for n, l in layers.items()}
        traj.append({"step": step, "loss": loss_val, "finite": finite,
                     "fwd_s": fwd_t, "bwd_s": bwd_t})
        log(f"step {step}: loss={loss_val:.6f} finite={finite}")
        if step == args.steps - 1:              # save FINAL-step state for comparison
            with torch.no_grad():
                nfl, _, _ = flat_logits_labels(model(input_ids=input_ids, attention_mask=attn).logits.float(), labels)
            torch.save({"deltaw_before": dw_before, "deltaw_after": dw_after,
                        "next_logits": nfl.cpu(),
                        "grads": {n: (l.A_t.grad.detach().cpu(), l.B_t.grad.detach().cpu())
                                  for n, l in layers.items()},
                        "U": {n: l.U.detach().cpu() for n, l in layers.items()},
                        "dlogits": G.detach().cpu()}, outdir / "tensors_protected_tdx.pt")

    result = {"mode": "protected_tdx", "model_id": args.model_id, "vocab": vocab,
              "num_lora_layers": len(layers), "steps": args.steps, "lr": args.lr,
              "targets": list(targets), "rank": args.rank, "alpha": args.alpha,
              "trajectory": traj,
              "invocations_per_step": {"input_init_boundary": 1, "loss_boundary": 1,
                                       "packed_update": 0, "trusted_optimizer": 0,
                                       "layer_wise": 0, "total": 2,
                                       "note": "input/init boundary = authorized init "
                                       "(labels+pi), amortized once for the fixed batch; "
                                       "loss boundary per step"},
              "wire_bytes": sess.wire_bytes,
              "rpc_latency_s": sess.rpc_latency,
              "attestation_bundle_public": {k: v for k, v in bundle.items()
                                            if k not in ("quote_b64", "appraisal_jwt")}}
    (outdir / "result_protected_tdx.json").write_text(json.dumps(result, indent=2, default=str))
    log(f"protected_tdx done: {args.steps} step(s); wire_bytes={sess.wire_bytes}")


if __name__ == "__main__":
    main()
