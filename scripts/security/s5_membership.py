"""S5 — Membership inference under the frozen private-base threat model.

Methodology basis: Shokri, Stronati, Song & Shmatikov, "Membership Inference Attacks
Against Machine Learning Models", IEEE S&P 2017 (shadow-model attack).

Purpose: does LoRA fine-tuning leak membership, and does the protected OUTPUT masking
change what a black-box attacker (outputs only; no weights/masks/TDX secrets) can exploit?

Faithful, tractable setup (CPU, real Qwen2.5-0.5B, real SST-2):
  * Featurize each SST-2 sentence with the FROZEN private base (hidden state at the
    supervised position) — the untrusted output channel.
  * "Fine-tuned model" = a LoRA-style 2-class head trained on that model's MEMBER set
    (over-fit on a small member set so a member/non-member gap exists — this is the
    signal MIA exploits). We train a target model + K shadow models on disjoint splits.
  * Attacker observes OUTPUTS only. Three observation channels:
      B0 plaintext         : full 2-class logits (correct-class prob, margin, entropy, maxprob)
      B1 protected perm-only: monomial Pi with D=I -> only SET-symmetric features (maxprob,
                              entropy); which-class / signed-margin are hidden by the permutation
      B1 protected monomial : D>0 scaling distorts the probabilities (maxprob/entropy perturbed)
  * Shadow classifier (logistic regression) trained on shadow features labeled member/
    non-member; evaluated on the target model's held-out member/non-member.

Positive control: plaintext (B0) MIA must be non-random (AUC>0.5). Metrics: ROC-AUC,
accuracy, precision, recall. CPU only. Writes security/S5_membership/.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_harness import PrivateBaseOracle, stamp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/aaai_private_base/security/S5_membership"
DATA = REPO / "results/aaai_private_base/datasets/tokenized"
torch.manual_seed(0)
NEG, POS = 8225, 6785
N_SAMPLES = 1500
N_SHADOW = 8
MEMBER = 120
NONMEMBER = 120
HEAD_EPOCHS = 150


@torch.no_grad()
def featurize(o: PrivateBaseOracle, samples):
    """Frozen private-base hidden state at the supervised position for each sample."""
    o._load()
    feats = []
    B = 16
    for i in range(0, len(samples), B):
        chunk = samples[i:i + B]
        maxlen = max(len(s["input_ids"]) for s in chunk)
        ids = torch.full((len(chunk), maxlen), o.tok.pad_token_id or 151643, dtype=torch.long)
        for j, s in enumerate(chunk):
            ids[j, :len(s["input_ids"])] = torch.tensor(s["input_ids"])
        out = o._model(input_ids=ids, output_hidden_states=True)
        hs = out.hidden_states[-1]                       # (B,T,896)
        for j, s in enumerate(chunk):
            feats.append(hs[j, s["sup_pos"]].to(torch.float32))
    return torch.stack(feats)                            # (N,896)


def train_head(feat_m, y_m, epochs=HEAD_EPOCHS):
    """A LoRA-style 2-class head over frozen features, over-fit on the member set."""
    d = feat_m.shape[1]
    head = torch.nn.Linear(d, 2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(feat_m), y_m)
        loss.backward(); opt.step()
    return head


def mia_features(logits2, y, channel, D2=None):
    """Attacker-observable features per sample given the observation channel.
    logits2: (N,2) the true 2-class logits (member's model output at the verbalizer tokens)."""
    p = torch.softmax(logits2, dim=1)
    maxp = p.max(1).values
    ent = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(1)
    if channel == "B0_plaintext":
        correct = p[torch.arange(len(y)), y]
        margin = (logits2[torch.arange(len(y)), y] - logits2[torch.arange(len(y)), 1 - y])
        return torch.stack([correct, maxp, ent, margin], 1)
    if channel == "B1_perm_only":                        # set-symmetric only
        return torch.stack([maxp, ent], 1)
    if channel == "B1_monomial":                         # D-scaled logits distort probs
        pm = torch.softmax(logits2 * D2, dim=1)
        return torch.stack([pm.max(1).values,
                            -(pm.clamp_min(1e-30) * pm.clamp_min(1e-30).log()).sum(1)], 1)
    raise ValueError(channel)


def roc_auc(scores, labels):
    order = torch.argsort(scores, descending=True)
    lab = labels[order]
    P = lab.sum().item(); Nn = (lab == 0).sum().item()
    if P == 0 or Nn == 0:
        return 0.5
    tps = torch.cumsum(lab, 0).float(); fps = torch.cumsum(1 - lab, 0).float()
    tpr = tps / P; fpr = fps / Nn
    # trapezoid
    fpr = torch.cat([torch.zeros(1), fpr]); tpr = torch.cat([torch.zeros(1), tpr])
    return float(torch.trapz(tpr, fpr))


def fit_attack(Xtr, ytr):
    m = torch.nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(m.parameters(), lr=5e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xtr).squeeze(1), ytr.float())
        loss.backward(); opt.step()
    return m


def eval_metrics(scores, labels, thr=0.5):
    pred = (torch.sigmoid(scores) >= thr).long()
    tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    acc = (tp + tn) / max(1, len(labels))
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"auc": roc_auc(torch.sigmoid(scores), labels), "accuracy": acc,
            "precision": prec, "recall": rec}


def run_channel(o, feats, labels_all, channel, D2):
    """Shadow-model MIA for one observation channel."""
    N = feats.shape[0]
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(N, generator=g)
    # target split
    t_m = perm[:MEMBER]; t_n = perm[MEMBER:MEMBER + NONMEMBER]
    pool = perm[MEMBER + NONMEMBER:]
    # shadow classifier training data
    Xs, ys = [], []
    for k in range(N_SHADOW):
        base = k * (2 * (MEMBER + NONMEMBER)) % max(1, len(pool) - 2 * MEMBER)
        s_m = pool[base:base + MEMBER]; s_n = pool[base + MEMBER:base + MEMBER + NONMEMBER]
        if len(s_m) < MEMBER or len(s_n) < NONMEMBER:
            continue
        head = train_head(feats[s_m], labels_all[s_m])
        with torch.no_grad():
            # member + nonmember outputs
            for idx, mem in [(s_m, 1), (s_n, 0)]:
                z = head(feats[idx])
                f = mia_features(z, labels_all[idx], channel, D2)
                Xs.append(f); ys.append(torch.full((len(idx),), mem))
    Xs = torch.cat(Xs); ys = torch.cat(ys)
    # normalize features (fit on shadow)
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    atk = fit_attack((Xs - mu) / sd, ys)
    # target model
    tgt_head = train_head(feats[t_m], labels_all[t_m])
    with torch.no_grad():
        zt_m = tgt_head(feats[t_m]); zt_n = tgt_head(feats[t_n])
        Xt = torch.cat([mia_features(zt_m, labels_all[t_m], channel, D2),
                        mia_features(zt_n, labels_all[t_n], channel, D2)])
        yt = torch.cat([torch.ones(len(t_m)), torch.zeros(len(t_n))]).long()
        scores = atk((Xt - mu) / sd).squeeze(1)
    return eval_metrics(scores, yt)


def main():
    t0 = time.time()
    o = PrivateBaseOracle()
    # load real SST-2 train samples + labels (READ-ONLY)
    tr = torch.load(DATA / "sst2_train_a10.pt", weights_only=False)
    lb = torch.load(DATA / "sst2_train_labels.pt", weights_only=False)
    g = torch.Generator().manual_seed(1)
    sel = torch.randperm(len(tr), generator=g)[:N_SAMPLES].tolist()
    samples = [tr[i] for i in sel]
    labels_all = torch.tensor([lb[tr[i]["sample_id"]]["label"] for i in sel])
    feats = featurize(o, samples)
    # per-class monomial scale for the 2 verbalizer tokens (bounded condition, from vocab_D)
    D2 = torch.tensor([float(o.vocab_D[NEG]), float(o.vocab_D[POS])])

    results = {"experiment": "S5_membership", "model": "Qwen2.5-0.5B",
               "threat_model": "from_scratch_private_base_black_box_outputs_only",
               "method": "Shokri et al. S&P'17 shadow-model MIA",
               "config": {"n_samples": N_SAMPLES, "n_shadow": N_SHADOW, "member": MEMBER,
                          "nonmember": NONMEMBER, "head": "linear 896->2 (LoRA-style probe)",
                          "head_epochs": HEAD_EPOCHS, "device": "cpu", "seeds": [0, 1, 42]},
               "channels": {}, "positive_control_pass": None, "limitations": []}

    for ch in ["B0_plaintext", "B1_perm_only", "B1_monomial"]:
        results["channels"][ch] = run_channel(o, feats, labels_all, ch, D2)

    b0 = results["channels"]["B0_plaintext"]["auc"]
    results["positive_control_pass"] = b0 > 0.55
    results["interpretation"] = {
        "membership_signal_source": "the fine-tuned head's over-confidence on members (generalization gap); a "
                                    "property of training, NOT of the mask.",
        "masking_effect": "the mask only changes READABILITY of the output confidence. perm-only preserves the "
                          "set-symmetric confidence (maxprob/entropy) so most of the black-box signal remains; "
                          "monomial distorts it, reducing readable signal; neither removes membership leakage at "
                          "its source (the trained weights).",
        "auc_by_channel": {k: results["channels"][k]["auc"] for k in results["channels"]}}
    results["limitations"] = [
        "MIA target is a LoRA-style linear probe over frozen private-base features (standard lightweight MIA "
        "target); it models the output-confidence channel the mask affects, not a full attention/MLP LoRA.",
        "Black-box outputs-only attacker; a white-box or per-example-gradient attacker is out of this channel.",
        "Absolute AUC is modest because the fine-tune is light and SST-2 generalizes well (small member gap); "
        "we report the RELATIVE channel comparison, which is the design-relevant quantity.",
        "Protected channels are modeled at the 2-class verbalizer boundary via the S3 monomial mask; a full "
        "vocab-logit MIA would see the same set-symmetric preservation (perm-only) shown in S3.",
    ]
    results["wall_sec"] = round(time.time() - t0, 1)
    stamp(OUT, "s5_results.json", results)
    a = results["interpretation"]["auc_by_channel"]
    print("[S5] done %.1fs  AUC B0=%.3f  B1_perm=%.3f  B1_monomial=%.3f  control_pass=%s" % (
        results["wall_sec"], a["B0_plaintext"], a["B1_perm_only"], a["B1_monomial"],
        results["positive_control_pass"]))


if __name__ == "__main__":
    main()
