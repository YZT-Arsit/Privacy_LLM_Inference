#!/usr/bin/env python
"""ObfuscaTune (Frikha et al., arXiv:2407.02960) real-model evaluation.

Two hardware-independent measurements (run on the 5090):

  MODE=utility  (their Table 2 trend, on a REAL LM)
    Obfuscate every attention+MLP linear of a real model with ObfuscaTune's
        X* = X Ra ,  W* = Ra^{-1} W   =>   X* W* = X W
    at a sweep of condition numbers kappa, and measure the numerical-error ->
    utility trend against the plaintext model: next-token top-1 agreement,
    mean logit MSE, and perplexity. kappa=1 (orthogonal, their choice) is
    ~lossless; large kappa degrades utility. Confirms the scheme is lossless at
    kappa=1 on a real architecture (=> the paper contrast is "same utility,
    different COST / security model", not a utility trade).

  MODE=exposure (security contrast, at scale)
    Because the projections cancel exactly, the untrusted device reconstructs
    the TRUE Q/K/V. We capture layer-0 q_proj input/output on a REAL model,
    recompute Q under ObfuscaTune obfuscation (kappa=1) and show
        max | Q_obf - Q_plain |  ~  0
    i.e. the accelerator sees plaintext Q/K/V. In the open-weight threat model
    this is *direct* token exposure -- contrast our A_rightmul, whose GPU sees
    only masked q_hat/k_hat (Q_hat K_hat^T = Q K^T but Q_hat != Q).

Faithful to the paper: obfuscation matmuls run in fp32; kappa=1 uses orthogonal
matrices (App. B: R = QA S QB). Raw obfuscation matrices are never serialised.
"""
from __future__ import annotations

import argparse
import json
import types

import torch
import torch.nn as nn

# attention + MLP projection names across GPT-2 / Llama / Qwen. These are the
# high-parameter linears ObfuscaTune pushes OUTSIDE the TEE. Embeddings, final
# norm and lm_head stay in the TEE (~5% of params) and are NOT obfuscated.
ATTN_MLP_KEYS = (
    "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj",          # gpt2
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",     # llama/qwen
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)

PASSAGE = (
    "The history of privacy-preserving machine learning is a story of trade-offs. "
    "Homomorphic encryption promises confidentiality but multiplies cost by orders "
    "of magnitude. Secure multi-party computation splits trust across parties yet "
    "demands heavy communication. Differential privacy bounds leakage with calibrated "
    "noise, sacrificing accuracy. Trusted execution environments move the boundary of "
    "trust into hardware, but their memory is small and their nonlinear throughput is "
    "limited. A practical system must therefore decide which computations stay inside "
    "the trusted base and which can be safely delegated to an untrusted accelerator, "
    "and it must justify that split against a concrete adversary. Obfuscation by "
    "invertible matrices hides intermediate activations from the accelerator while "
    "preserving the exact arithmetic of the underlying model, so that no accuracy is "
    "lost when the matrices are orthogonal. The remaining question is whether the "
    "quantities that the accelerator still observes are enough to reconstruct the "
    "private input, and at what additional cost the leaking computations can be "
    "relocated into the trusted base without changing the model's output."
)


def make_cond_matrices(d, kappa, device, gen):
    """(R, R_inv) with cond(R)=kappa, fp32. kappa<=1 => orthogonal (R_inv=R^T)."""
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    qa, _ = torch.linalg.qr(a)
    if kappa <= 1.0:
        R = qa
        Rinv = qa.transpose(0, 1).contiguous()
    else:
        b = torch.randn(d, d, generator=gen, dtype=torch.float64)
        qb, _ = torch.linalg.qr(b)
        sv = torch.linspace(1.0 / float(kappa), 1.0, d, dtype=torch.float64)
        R = qa @ torch.diag(sv) @ qb
        Rinv = qb.transpose(0, 1) @ torch.diag(1.0 / sv) @ qa.transpose(0, 1)
    return R.to(device, torch.float32), Rinv.to(device, torch.float32)


def _obf_linear_forward(self, x):
    """ObfuscaTune obfuscated linear: y = (x Ra)(Ra^{-1} Weff) + b, in fp32.

    Weff is the matrix with y_nobias = x @ Weff (Linear: W^T, Conv1D: W)."""
    Ra, Rainv, Weff, bias = self._obf
    xf = x.to(torch.float32)
    x_star = xf @ Ra                     # obfuscated activation leaves TEE
    y = (x_star @ (Rainv @ Weff))        # untrusted matmul; Ra Ra^{-1} = I
    if bias is not None:
        y = y + bias
    return y.to(x.dtype)


def obfuscate_model(model, kappa, seed=0):
    gen = torch.Generator().manual_seed(seed)
    n = 0
    for name, mod in model.named_modules():
        is_lin = isinstance(mod, nn.Linear)
        is_conv1d = mod.__class__.__name__ == "Conv1D"
        if not (is_lin or is_conv1d):
            continue
        if not any(k in name for k in ATTN_MLP_KEYS):
            continue
        W = mod.weight.data
        # in-dim + Weff such that y_nobias = x @ Weff
        if is_lin:
            d = W.shape[1]
            Weff = W.t().contiguous().to(torch.float32)
        else:  # Conv1D: weight is (in, out), y = x @ W
            d = W.shape[0]
            Weff = W.contiguous().to(torch.float32)
        Ra, Rainv = make_cond_matrices(d, kappa, W.device, gen)
        bias = None if mod.bias is None else mod.bias.data.to(torch.float32)
        mod._obf = (Ra, Rainv, Weff, bias)
        mod.forward = types.MethodType(_obf_linear_forward, mod)
        n += 1
    return n


@torch.no_grad()
def _logits_and_ppl(model, ids):
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0].float()          # [S, V]
    lp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0, 1:]
    nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    ppl = torch.exp(nll.mean()).item()
    return logits, ppl


def run_utility(model_path, arch, kappas, seed, out):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok(PASSAGE, return_tensors="pt")["input_ids"]
    print(f"[utility] {arch} tokens={ids.shape[1]}", flush=True)

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32)
        return m.eval().cuda()

    base = fresh()
    ids = ids.cuda()
    base_logits, base_ppl = _logits_and_ppl(base, ids)
    base_top1 = base_logits[:-1].argmax(-1)
    del base
    torch.cuda.empty_cache()

    rows = []
    for kappa in kappas:
        m = fresh()
        n = obfuscate_model(m, kappa, seed=seed)
        lg, ppl = _logits_and_ppl(m, ids)
        top1 = lg[:-1].argmax(-1)
        agree = (top1 == base_top1).float().mean().item()
        mse = (lg - base_logits).pow(2).mean().item()
        maxerr = (lg - base_logits).abs().max().item()
        rows.append({
            "condition_number": kappa,
            "obfuscated_linears": n,
            "next_token_top1_agreement": round(agree, 6),
            "logit_mse": mse,
            "logit_max_abs_err": maxerr,
            "perplexity": ppl,
        })
        print(f"  kappa={kappa:>6}: agree={agree:.4f} mse={mse:.3e} "
              f"maxerr={maxerr:.3e} ppl={ppl:.4f}", flush=True)
        del m
        torch.cuda.empty_cache()

    res = {"mode": "utility", "arch": arch, "model_path": model_path,
           "base_perplexity": base_ppl, "n_tokens": int(ids.shape[1]),
           "rows": rows}
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[utility] wrote {out}", flush=True)


def run_exposure(model_path, arch, seed, out):
    """Show ObfuscaTune reconstructs the TRUE layer-0 Q/K/V (kappa=1)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    ids = tok(PASSAGE, return_tensors="pt")["input_ids"].cuda()
    dtype = torch.bfloat16 if arch != "gpt2" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype).eval().cuda()

    # locate layer-0 q/k/v projections + their inputs
    cap = {}
    if arch == "gpt2":
        blk = model.transformer.h[0].attn
        mods = {"qkv": blk.c_attn}
    else:
        sa = model.model.layers[0].self_attn
        mods = {"q": sa.q_proj, "k": sa.k_proj, "v": sa.v_proj}

    handles = []
    for tag, mod in mods.items():
        def hook(m, inp, o, tag=tag):
            cap[tag] = (inp[0].detach(), o.detach())
        handles.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    gen = torch.Generator().manual_seed(seed)
    report = {"mode": "exposure", "arch": arch, "model_path": model_path,
              "condition_number": 1.0, "tensors": {}}
    for tag, mod in mods.items():
        x, _ = cap[tag]
        x = x[0].to(torch.float32)                  # [S, d_in]
        W = mod.weight.data
        if mod.__class__.__name__ == "Conv1D":
            Weff = W.to(torch.float32)
            d = W.shape[0]
        else:
            Weff = W.t().to(torch.float32)
            d = W.shape[1]
        bias = mod.bias
        # reference = the EXACT fp32 linear on the same input; comparing to the
        # bf16 forward output would conflate bf16 quantization with the (zero)
        # mask residual. ObfuscaTune's claim is X* W* = X W to working precision.
        y_plain = x @ Weff
        if bias is not None:
            y_plain = y_plain + bias.to(torch.float32)
        Ra, Rainv = make_cond_matrices(d, 1.0, x.device, gen)
        y_obf = (x @ Ra) @ (Rainv @ Weff)
        if bias is not None:
            y_obf = y_obf + bias.to(torch.float32)
        maxerr = (y_obf - y_plain).abs().max().item()
        rel = maxerr / (y_plain.abs().max().item() + 1e-9)
        report["tensors"][tag] = {
            "shape": list(y_plain.shape),
            "max_abs_err_obf_vs_plain": maxerr,
            "relative_err": rel,
            "gpu_sees_true_value": bool(rel < 1e-4),
        }
        print(f"  {tag}: maxerr={maxerr:.3e} rel={rel:.3e} "
              f"true_value_exposed={rel < 1e-4}", flush=True)

    report["contrast_ours_a_rightmul"] = {
        "gpu_sees": "masked q_hat/k_hat (Q_hat K_hat^T = Q K^T, but Q_hat != Q)",
        "true_qkv_exposed": False,
        "note": "ObfuscaTune assumes SECRET weights; open-weight => direct token exposure",
    }
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[exposure] wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["utility", "exposure"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", choices=["gpt2", "llama", "qwen"], required=True)
    ap.add_argument("--kappas", default="1,4,16,64,256")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "utility":
        ks = [float(x) for x in a.kappas.split(",")]
        run_utility(a.model, a.arch, ks, a.seed, a.out)
    else:
        run_exposure(a.model, a.arch, a.seed, a.out)
