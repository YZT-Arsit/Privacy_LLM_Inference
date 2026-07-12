"""S6 — KV-cache inversion under the frozen private-base threat model.

Purpose: does the masked KV cache leak recoverable tokens?

Fold (design_spec §C/§D): the untrusted GPU's layer-l KV cache holds
    K_tilde = RoPE(K_plain) @ Bk     (Bk = block_diag(rope-commuting rotation, n_kv))
    V_tilde = V_plain @ Sv           (Sv = block_diag(signed-perm, n_kv))
both ORTHOGONAL per-kv-head. Attacker receives only KV_tilde (no weights/masks/TDX).

Attack: representation-inversion decoders from S1 (linear + MLP) mapping KV -> token,
as closed-set token classification over the tokens present (clean chance baseline), plus
embedding-cosine regression. We use layer-0 KV (least contextualized => strongest,
attacker-favourable case).

Baselines / settings:
  plaintext KV   : decoder trained & tested on plaintext KV      -> recoverable (control)
  identity KV    : no mask (== plaintext)                          -> recoverable
  masked KV (transfer): decoder trained on PLAINTEXT KV, tested on MASKED KV
                        (attacker does not know the mask)          -> REDUCED recovery
  masked KV (adapted) : decoder trained AND tested on masked KV    -> recovers (orthogonal
                        mask is invertible given masked pairs => protection = secret mask/TEE)

Metrics: token top-1 / top-5 accuracy (+ chance), embedding cosine. CPU only.
Writes security/S6_kv_cache/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_harness import PrivateBaseOracle, stamp, cosine  # noqa: E402
from pllo.ops.masked_training_kernels import rmsnorm_core, apply_rope, rope_cos_sin  # noqa: E402
from gate0_build_private_package import block_diag  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/aaai_private_base/security/S6_kv_cache"
DATA = REPO / "results/aaai_private_base/datasets/tokenized"
torch.manual_seed(0)
DT = torch.float64
LAYER = 0
N_SENT = 120


@torch.no_grad()
def collect_kv(o: PrivateBaseOracle):
    """Layer-0 plaintext + masked K_rope,V for real SST-2 token positions."""
    o._load()
    tr = torch.load(DATA / "sst2_train_a10.pt", weights_only=False)
    g = torch.Generator().manual_seed(3)
    sel = torch.randperm(len(tr), generator=g)[:N_SENT].tolist()
    sd = o._sd
    p = f"model.layers.{LAYER}."
    ga = sd[p + "input_layernorm.weight"].to(DT)
    Wk = sd[p + "self_attn.k_proj.weight"].to(DT); bk = sd[p + "self_attn.k_proj.bias"].to(DT)
    Wv = sd[p + "self_attn.v_proj.weight"].to(DT); bv = sd[p + "self_attn.v_proj.bias"].to(DT)
    E = sd["model.embed_tokens.weight"].to(DT)
    Bk = block_diag(o.B[LAYER], o.nkv)                  # (128,128) rope-commuting
    Sv = block_diag(o.S[LAYER], o.nkv)                  # (128,128) signed-perm
    Kp, Km, Vp, Vm, toks, Emb = [], [], [], [], [], []
    for i in sel:
        ids = torch.tensor(tr[i]["input_ids"])
        h0 = E[ids]                                     # (T,896) layer-0 input = embeddings
        rms = rmsnorm_core(h0, o.eps) * ga
        k = rms @ Wk.T + bk                             # (T,128) pre-rope
        v = rms @ Wv.T + bv                             # (T,128)
        T = ids.shape[0]; hd = o.hd
        cos, sin = rope_cos_sin(T, hd, o.cfg["rope_theta"], DT)
        # per kv-head rope
        kh = k.view(T, o.nkv, hd)
        kr = torch.stack([apply_rope(kh[:, j], cos, sin) for j in range(o.nkv)], 1).reshape(T, o.nkv * hd)
        # masked
        kr_m = kr @ Bk
        v_m = v @ Sv
        for t in range(T):
            Kp.append(kr[t]); Km.append(kr_m[t]); Vp.append(v[t]); Vm.append(v_m[t])
            toks.append(int(ids[t])); Emb.append(E[ids[t]])
    return (torch.stack(Kp), torch.stack(Km), torch.stack(Vp), torch.stack(Vm),
            torch.tensor(toks), torch.stack(Emb))


def kv_feat(K, V):
    return torch.cat([K, V], 1).to(torch.float32)      # (N,256)


def closed_set(toks):
    uniq = sorted(set(toks.tolist()))
    remap = {t: i for i, t in enumerate(uniq)}
    y = torch.tensor([remap[int(t)] for t in toks])
    return y, len(uniq)


def train_classifier(X, y, nc, mlp=False, steps=400):
    d = X.shape[1]
    if mlp:
        net = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(), torch.nn.Linear(512, nc))
    else:
        net = torch.nn.Linear(d, nc)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(X), y)
        loss.backward(); opt.step()
    return net


def topk_acc(logits, y, k=5):
    top1 = float((logits.argmax(1) == y).float().mean())
    tk = logits.topk(k, dim=1).indices
    topk = float((tk == y.unsqueeze(1)).any(1).float().mean())
    return top1, topk


def run(o):
    Kp, Km, Vp, Vm, toks, Emb = collect_kv(o)
    N = toks.shape[0]
    y, nc = closed_set(toks)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(N, generator=g)
    ntr = int(N * 0.7)
    tr, te = perm[:ntr], perm[ntr:]
    Xp = kv_feat(Kp, Vp); Xm = kv_feat(Km, Vm)
    chance = 1.0 / nc
    res = {"n_positions": N, "n_unique_tokens": nc, "chance_top1": chance, "settings": {}}

    for name, mlp in [("linear_decoder", False), ("mlp_decoder", True)]:
        # train on plaintext
        clf_p = train_classifier(Xp[tr], y[tr], nc, mlp)
        with torch.no_grad():
            lp = clf_p(Xp[te]); li = clf_p(Xp[te])           # identity == plaintext
            lm_transfer = clf_p(Xm[te])                       # plaintext-calibrated -> masked
        # train on masked (adapted attacker)
        clf_m = train_classifier(Xm[tr], y[tr], nc, mlp)
        with torch.no_grad():
            lm_adapt = clf_m(Xm[te])
        p1, p5 = topk_acc(lp, y[te]); i1, i5 = topk_acc(li, y[te])
        mt1, mt5 = topk_acc(lm_transfer, y[te]); ma1, ma5 = topk_acc(lm_adapt, y[te])
        res["settings"][name] = {
            "plaintext_KV": {"top1": p1, "top5": p5},
            "identity_KV": {"top1": i1, "top5": i5},
            "masked_KV_transfer_plaintext_decoder": {"top1": mt1, "top5": mt5},
            "masked_KV_adapted_decoder": {"top1": ma1, "top5": ma5},
        }
    # embedding cosine: regress plaintext hidden token-embedding from KV (known-pairs)
    def emb_cos(X):
        W = torch.linalg.solve(X[tr].T @ X[tr] + 1e-2 * torch.eye(X.shape[1]),
                               X[tr].T @ Emb[tr].to(torch.float32))
        pred = X[te] @ W
        return float(torch.stack([torch.nn.functional.cosine_similarity(pred[i], Emb[te][i].float(), dim=0)
                                  for i in range(len(te))]).mean())
    res["embedding_cosine_knownpairs"] = {"plaintext_KV": emb_cos(Xp), "masked_KV": emb_cos(Xm),
                                          "note": "orthogonal KV mask is linearly invertible given masked pairs -> similar cosine"}
    return res


def main():
    t0 = time.time()
    o = PrivateBaseOracle()
    r = run(o)
    results = {"experiment": "S6_kv_cache", "model": "Qwen2.5-0.5B",
               "threat_model": "from_scratch_private_base", "layer": LAYER,
               "method": "S1 representation-inversion decoders (linear + MLP) on KV",
               "budget": {"decoders": "linear + 1x512 MLP, Adam lr1e-2 x400", "device": "cpu",
                          "seeds": [0, 3, 7], "n_sentences": N_SENT},
               **r, "positive_control_pass": None, "limitations": []}
    lin = results["settings"]["linear_decoder"]
    results["positive_control_pass"] = lin["plaintext_KV"]["top1"] > 5 * results["chance_top1"]
    results["interpretation"] = {
        "plaintext_recoverable": f"plaintext KV top1={lin['plaintext_KV']['top1']:.3f} vs chance {results['chance_top1']:.4f}",
        "masked_transfer_reduced": ("a plaintext-calibrated decoder applied to masked KV drops to "
            f"top1={lin['masked_KV_transfer_plaintext_decoder']['top1']:.3f} (attacker does not know the mask) "
            "=> masking REDUCES recovery for an attacker without the secret mask."),
        "masked_adapted_caveat": ("with masked pairs the orthogonal mask is invertible: an adapted decoder "
            f"recovers top1={lin['masked_KV_adapted_decoder']['top1']:.3f} => protection rests on mask secrecy "
            "(the TEE), not on information destruction."),
    }
    results["limitations"] = [
        "Layer-0 KV (least contextualized) is the attacker-favourable case; deeper layers mix context and are "
        "harder to invert to a single token.",
        "Closed-set token classification over tokens present gives a clean chance baseline but is easier than "
        "open-vocabulary recovery; the plaintext-vs-masked-transfer CONTRAST is the design-relevant quantity.",
        "The masked-adapted decoder assumes masked (KV_tilde, token) pairs; consistent with S1, the orthogonal "
        "KV mask provides no information-theoretic protection — confidentiality is the secret mask / TEE.",
        "KV masks are orthogonal per-kv-head (Bk rope-commuting, Sv signed-perm); norms/Grams are preserved as "
        "in S1 (documented structural leak).",
    ]
    results["wall_sec"] = round(time.time() - t0, 1)
    stamp(OUT, "s6_results.json", results)
    print("[S6] done %.1fs  linear: plaintext top1=%.3f  masked-transfer top1=%.3f  masked-adapted top1=%.3f  "
          "chance=%.4f  control=%s" % (
        results["wall_sec"], lin["plaintext_KV"]["top1"],
        lin["masked_KV_transfer_plaintext_decoder"]["top1"], lin["masked_KV_adapted_decoder"]["top1"],
        results["chance_top1"], results["positive_control_pass"]))


if __name__ == "__main__":
    main()
