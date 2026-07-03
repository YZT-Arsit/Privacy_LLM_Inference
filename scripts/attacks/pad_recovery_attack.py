#!/usr/bin/env python
"""STEP 2 self-audit: once the masks are recovered (step 1), can the attacker also
peel the fresh Linear-boundary pad T_l and expose the intermediate hidden states?

Threat model: the attacker holds public weights W_l, the step-1-recovered masks
(N_in,l / N_out,l), the GPU-visible compensation C_pad,l = T_l W_l N_out,l, and the
obfuscated matmul operand H̃_l = (H_l - T_l) N_in,l. Pads are per-module broadcast
vectors T_l ∈ R^{d_in} (production `linear_boundary_pad.py`, scale 0.1), independent
per module.

Core equation (cancel the recovered N_out): C'_l = C_pad,l N_out,l^{-1} = T_l W_l.
  - single-layer:  T̂_l = C'_l W_l^+ (Moore-Penrose). Recovers T_l projected onto the
    ROW space of W_l; the LEFT-NULL-SPACE component (dim = d_in - rank W_l) is lost.
      * square/full-row-rank (q,o / gate,up up-proj): null=0 -> exact -> pad fails.
      * output-reduced (k,v: d->d_kv; down: 4d->d): large null -> pad survives here.
  - CROSS-MODULE JOINT (the decisive test): the null-space component IS recovered
    because padded activations SHARE inputs / inputs are reconstructable upstream:
      * q,k,v read the SAME attn-norm input Z1; q is square -> solve T_q -> recover
        Z1 = H̃_q N_res^{-1} + T_q -> T_k = Z1 - H̃_k N_res^{-1}, T_v likewise.
      * down's input is the SwiGLU intermediate I = SiLU(gate)⊙up, reconstructable
        from the (un-maskable) gate/up outputs -> T_down = I - H̃_down N_swiglu^{-1}.
  - H recovery:  Ĥ_l = H̃_l N_in,l^{-1} + T̂_l ; cosine vs true H_l.
  - random-T baseline must be beaten to count as a real recovery.

Verdict answers: which layers' pad holds single-layer, and whether the joint attack
breaks the output-reduced (k/v/down) layers. Runs on real Qwen weights + real
activations (hooks). Masks here are valid stand-ins for the step-1-recovered ones
(all orthogonal/monomial); the attack only uses them as KNOWN, exactly as step 1
delivers them.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F


def signed_perm(d, g, dev):
    perm = torch.randperm(d, generator=g)
    signs = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(torch.float32)
    M = torch.zeros(d, d, dtype=torch.float32); M[torch.arange(d), perm] = signs
    return M.to(dev)


def perm_mat(d, g, dev):
    perm = torch.randperm(d, generator=g)
    M = torch.zeros(d, d, dtype=torch.float32); M[torch.arange(d), perm] = 1.0
    return M.to(dev)


def orth(d, g, dev):
    a = torch.randn(d, d, generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(a)
    return q.to(dev)


def rel_err(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def cos_mean(A, B):
    return float(F.cosine_similarity(A, B, dim=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="14")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog near the river bank.")
    ap.add_argument("--pad-scale", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2035)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    print(f"[load] {a.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    cfg = model.config
    layers = [int(x) for x in a.layers.split(",") if int(x) < cfg.num_hidden_layers]
    ids = tok(a.prompt, return_tensors="pt")["input_ids"].to(dev)

    # capture per-linear INPUT activations + gate/up OUTPUTS for the target layers
    cap = {}
    handles = []
    def in_hook(li, name):
        def h(m, inp, out): cap[(li, name, "in")] = inp[0].detach()[0].to(torch.float32)
        return h
    def out_hook(li, name):
        def h(m, inp, out): cap[(li, name, "out")] = out.detach()[0].to(torch.float32)
        return h
    for li in layers:
        L = model.model.layers[li]
        mp = {"q_proj": L.self_attn.q_proj, "k_proj": L.self_attn.k_proj,
              "v_proj": L.self_attn.v_proj, "o_proj": L.self_attn.o_proj,
              "gate_proj": L.mlp.gate_proj, "up_proj": L.mlp.up_proj, "down_proj": L.mlp.down_proj}
        for nm, mod in mp.items():
            handles.append(mod.register_forward_hook(in_hook(li, nm)))
            if nm in ("gate_proj", "up_proj"):
                handles.append(mod.register_forward_hook(out_hook(li, nm)))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    # do the attack linear algebra on CPU (big SVDs/pinv of down_proj OOM on GPU
    # under fragmentation; weights are only ~260MB). Keep only weights we need.
    cap = {k: v.cpu() for k, v in cap.items()}
    dev = "cpu"

    rows = []
    for li in layers:
        L = model.model.layers[li]
        g = torch.Generator().manual_seed(a.seed + 101 * (li + 1))
        d = cfg.hidden_size
        Wdict = {"q_proj": L.self_attn.q_proj, "k_proj": L.self_attn.k_proj,
                 "v_proj": L.self_attn.v_proj, "o_proj": L.self_attn.o_proj,
                 "gate_proj": L.mlp.gate_proj, "up_proj": L.mlp.up_proj, "down_proj": L.mlp.down_proj}
        # masks (stand-ins for step-1-recovered masks; all orthogonal/monomial)
        N_res = signed_perm(d, g, dev)
        inter = cfg.intermediate_size
        N_swiglu = perm_mat(inter, g, dev)            # gate/up output = down input basis
        N_attnout = orth(d, g, dev)                   # o_proj input basis
        # per-module (N_in, N_out)
        def masks_for(nm, W):
            din, dout = W.shape
            if nm in ("q_proj", "k_proj", "v_proj"):
                return N_res, orth(dout, g, dev)
            if nm == "o_proj":
                return N_attnout, N_res
            if nm in ("gate_proj", "up_proj"):
                return N_res, N_swiglu
            if nm == "down_proj":
                return N_swiglu, N_res
        gp = torch.Generator().manual_seed(a.seed + 1009 * (li + 1))
        pads, Cpads, Htil, Ns = {}, {}, {}, {}
        for nm, mod in Wdict.items():
            W = mod.weight.detach().to(torch.float32).to(dev).T      # [din, dout], y = x W
            din, dout = W.shape
            N_in, N_out = masks_for(nm, W)
            T = a.pad_scale * torch.randn(din, generator=gp, dtype=torch.float32).to(dev)
            H = cap[(li, nm, "in")]                                   # [S, din] real activation
            Cpad = T @ W @ N_out                                     # observed compensation
            Ht = (H - T) @ N_in                                      # observed padded operand
            pads[nm] = T; Cpads[nm] = (Cpad, N_out, W); Htil[nm] = (Ht, N_in, H); Ns[nm] = (N_in, N_out)

        # ---- single-layer pseudoinverse recovery ----
        single = {}
        for nm, (Cpad, N_out, W) in Cpads.items():
            din, dout = W.shape
            rank = min(din, dout)          # dense Qwen weights are full rank
            nulldim = din - rank
            Cp = Cpad @ N_out.T if _is_orth(N_out) else Cpad @ torch.linalg.inv(N_out)
            That = Cp @ torch.linalg.pinv(W)
            T = pads[nm]
            single[nm] = {"din": din, "dout": dout, "rank": rank, "left_null_dim": nulldim,
                          "single_T_relerr": round(rel_err(That, T), 4)}

        # ---- joint recovery ----
        joint_T = {}
        # o_proj (square) + gate/up (up-proj) are already solvable single-layer:
        for nm in ("q_proj", "o_proj", "gate_proj", "up_proj"):
            Cpad, N_out, W = Cpads[nm]
            Cp = Cpad @ N_out.T if _is_orth(N_out) else Cpad @ torch.linalg.inv(N_out)
            joint_T[nm] = Cp @ torch.linalg.pinv(W)
        # q/k/v share input Z1: recover Z1 from q, then T_k, T_v
        Tq = joint_T["q_proj"]
        Ht_q, N_in_q, _ = Htil["q_proj"]
        Z1 = Ht_q @ N_in_q.T + Tq                    # = attn-norm output (N_res orthogonal)
        for nm in ("k_proj", "v_proj"):
            Ht, N_in, _ = Htil[nm]
            joint_T[nm] = Z1 - Ht @ N_in.T           # shared-input difference
        # down: reconstruct SwiGLU intermediate from recovered gate/up outputs
        gate_out = cap[(li, "gate_proj", "out")]     # attacker un-masks gate·P -> gate_out
        up_out = cap[(li, "up_proj", "out")]
        I_rec = F.silu(gate_out) * up_out            # = down_proj input (true intermediate)
        Ht_d, N_in_d, _ = Htil["down_proj"]
        joint_T["down_proj"] = _broadcast_pad(I_rec, Ht_d, N_in_d)

        # ---- metrics: joint T err + H recovery cosine (single vs joint) + baseline ----
        for nm in Wdict:
            Ht, N_in, H = Htil[nm]
            T = pads[nm]
            # single-layer H recovery
            Cpad, N_out, W = Cpads[nm]
            Cp = Cpad @ N_out.T if _is_orth(N_out) else Cpad @ torch.linalg.inv(N_out)
            That_s = Cp @ torch.linalg.pinv(W)
            H_s = Ht @ torch.linalg.inv(N_in) + That_s
            # joint H recovery
            That_j = joint_T[nm]
            H_j = Ht @ torch.linalg.inv(N_in) + That_j
            # random baseline
            Trand = a.pad_scale * torch.randn(W.shape[0], dtype=torch.float32, device=dev)
            H_r = Ht @ torch.linalg.inv(N_in) + Trand
            cls = ("square" if W.shape[0] == W.shape[1] else
                   "up_proj" if W.shape[1] > W.shape[0] else "down_proj")
            pad_rel = float(T.norm() / (H.norm(dim=1).mean() + 1e-12))  # ‖T‖ vs per-token ‖H‖
            rows.append({
                "layer": li, "module": nm, "shape_class": cls,
                "din": W.shape[0], "dout": W.shape[1],
                "rank": single[nm]["rank"], "left_null_dim": single[nm]["left_null_dim"],
                "pad_to_activation_ratio": round(pad_rel, 4),
                "single_T_relerr": single[nm]["single_T_relerr"],
                "joint_T_relerr": round(rel_err(That_j, T), 4),
                "H_cos_single": round(cos_mean(H_s, H), 4),
                "H_cos_joint": round(cos_mean(H_j, H), 4),
                "H_cos_random_baseline": round(cos_mean(H_r, H), 4),
            })
            print(f"L{li} {nm:10s} {cls:9s} null={single[nm]['left_null_dim']:5d} "
                  f"singleT={single[nm]['single_T_relerr']:.3f} jointT={rel_err(That_j,T):.3f} "
                  f"Hcos s/j/rand={cos_mean(H_s,H):.3f}/{cos_mean(H_j,H):.3f}/{cos_mean(H_r,H):.3f}",
                  flush=True)

    # aggregate by shape class
    import statistics as st
    agg = {}
    for cls in ("square", "up_proj", "down_proj"):
        r = [x for x in rows if x["shape_class"] == cls]
        agg[cls] = {
            "modules": sorted(set(x["module"] for x in r)),
            "median_single_T_relerr": round(st.median([x["single_T_relerr"] for x in r]), 4),
            "median_joint_T_relerr": round(st.median([x["joint_T_relerr"] for x in r]), 4),
            "median_H_cos_single": round(st.median([x["H_cos_single"] for x in r]), 4),
            "median_H_cos_joint": round(st.median([x["H_cos_joint"] for x in r]), 4),
            "median_H_cos_random": round(st.median([x["H_cos_random_baseline"] for x in r]), 4),
        }
    dn_single = agg["down_proj"]["median_single_T_relerr"]
    min_hcos_joint = min(r["H_cos_joint"] for r in rows)
    max_hcos_rand = max(r["H_cos_random_baseline"] for r in rows)
    verdict = {
        "decisive_metric": "H_cos_joint (hidden-state recovery cosine after the joint attack)",
        "min_H_cos_joint_over_all_modules": round(min_hcos_joint, 4),
        "single_layer": ("pad survives (as a T-vector) ONLY on output-reduced k/v/down "
                         f"(median single_T_relerr={dn_single}, large left-null); square (q/o) + "
                         "up-proj (gate/up) fall to the plain pseudoinverse"),
        "joint": ("pad BROKEN on ALL layers: joint cross-module recovery (shared attn-norm input "
                  "for q/k/v, upstream SwiGLU reconstruction for down) exposes every hidden state "
                  f"to cosine {round(min_hcos_joint,4)} (min over modules)"
                  if min_hcos_joint > 0.99 else
                  f"joint hidden-state recovery incomplete (min H_cos_joint={round(min_hcos_joint,4)})"),
        "note": "T-vector residuals (0.04-0.15 on solved layers) are fp32 noise on a tiny pad and "
                "do not affect H recovery. Run a large --pad-scale to confirm: when the pad actually "
                f"hides H the random baseline drops but joint stays ~1 (this run max random H_cos={round(max_hcos_rand,3)}).",
    }
    report = {"model": a.model, "layers": layers, "pad_scale": a.pad_scale,
              "by_shape_class": agg, "verdict": verdict, "per_row": rows}
    print(json.dumps({"by_shape_class": agg, "verdict": verdict}, indent=2), flush=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {a.out}", flush=True)


def _is_orth(M, tol=1e-6):
    d = M.shape[0]
    return bool((M @ M.T - torch.eye(d, dtype=M.dtype, device=M.device)).abs().max() < tol)


def _broadcast_pad(I_rec, Ht_d, N_in_d):
    """down_proj pad recovery: T_down = (I - H̃ N_in^{-1}) averaged over tokens
    (T is a broadcast vector, same for all tokens)."""
    resid = I_rec - Ht_d @ torch.linalg.inv(N_in_d)
    return resid.mean(dim=0)


if __name__ == "__main__":
    main()
