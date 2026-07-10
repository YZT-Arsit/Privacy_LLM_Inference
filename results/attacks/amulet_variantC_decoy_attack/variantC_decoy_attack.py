"""(a) Statistical real-vs-decoy column distinguishing attack on variant C.

Variant C GPU-visible lift z [m, d*k]: each hidden coordinate j has k columns;
the secret 'valid' column (offset b, then globally permuted) equals x[:,j]; the
other k-1 columns are decoys. The attacker sees z only and tries to identify the
d real columns (recover the selector) and/or the pre-activation.

Metrics per config: valid-column top-1 recovery, top-2 recovery, real-vs-decoy
AUC (best simple column statistic), and a distributional-distance proxy.
Vary: k, decoy mode, input type, seq len m, hidden dim d.  numpy fp64.
"""
import numpy as np

rng = np.random.default_rng(0)


def inv(M): return np.linalg.inv(M)


def spd(d, g):
    A = g.standard_normal((d, d)); return A @ A.T + d * np.eye(d)


def gaussian_x(m, d, g):
    return g.standard_normal((m, d))


def structured_x(m, d, g):
    """Transformer-like: a few outlier channels with large scale + correlations."""
    base = g.standard_normal((m, d))
    scales = np.ones(d); outliers = g.choice(d, size=max(1, d // 8), replace=False)
    scales[outliers] = g.uniform(6.0, 20.0, size=outliers.size)   # outlier features
    corr = g.standard_normal((d, d)) * 0.1 + np.eye(d)
    return (base @ corr) * scales[None, :]


def build_z(x, N_in, k, decoy_mode, g):
    m, d = x.shape
    Ninv = inv(N_in); x_tilde = x @ N_in
    b = int(g.integers(0, k))
    perm = g.permutation(d * k); invp = np.argsort(perm)
    L = np.zeros((d, d * k))
    for j in range(d):
        for q in range(k):
            col = j * k + q
            if q == b:
                L[:, col] = Ninv[:, j]                      # real: z col = x[:,j]
            else:
                if decoy_mode == "naive":
                    dec = g.standard_normal(d)              # unnormalized combo
                elif decoy_mode == "unit_norm":
                    dec = g.standard_normal(d); dec /= np.linalg.norm(dec)
                elif decoy_mode == "coord":
                    jj = int(g.integers(0, d)); dec = np.zeros(d); dec[jj] = 1.0  # another coord
                elif decoy_mode == "colnorm_matched":
                    dec = g.standard_normal(d)
                    dec *= np.linalg.norm(x[:, j]) / max(np.linalg.norm(x @ dec), 1e-9)
                else:
                    raise ValueError(decoy_mode)
                L[:, col] = Ninv @ dec
    L = L[:, perm]
    z = x_tilde @ L
    real_cols = np.array([invp[j * k + b] for j in range(d)])
    groups = np.array([[invp[j * k + q] for q in range(k)] for j in range(d)])  # [d,k]
    return z, real_cols, groups


def col_stats(z):
    """Per-column features an attacker can compute from z alone."""
    zc = z - z.mean(0, keepdims=True)
    var = zc.var(0)
    norm = np.linalg.norm(z, axis=0)
    kurt = (zc ** 4).mean(0) / (var ** 2 + 1e-12)
    maxabs = np.abs(z).max(0)
    return {"norm": norm, "var": var, "kurtosis": kurt, "maxabs_over_norm": maxabs / (norm + 1e-12)}


def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0: return 0.5
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def attack(z, real_cols, groups):
    d, k = groups.shape
    dk = z.shape[1]
    labels = np.zeros(dk); labels[real_cols] = 1
    stats = col_stats(z)
    best = {"stat": None, "auc": 0.5}
    for name, s in stats.items():
        a = max(auc(s, labels), auc(-s, labels))
        if a > best["auc"]:
            best = {"stat": name, "auc": a}
    # oracle-direction top-1 selector recovery (calibrated attacker upper bound):
    # for the best stat, pick the group column that is most extreme in the
    # direction (high or low) that best separates real from decoy.
    s = stats[best["stat"]] if best["stat"] else stats["norm"]
    real_mean = s[real_cols].mean(); decoy_mean = s[labels == 0].mean()
    sign = 1.0 if real_mean >= decoy_mean else -1.0
    top1 = 0
    for j in range(d):
        gcols = groups[j]
        pick = gcols[np.argmax(sign * s[gcols])]
        if pick == real_cols[j]: top1 += 1
    return {"best_stat": best["stat"], "auc": round(best["auc"], 3),
            "top1_recovery": round(top1 / d, 3), "chance_top1": round(1.0 / k, 3)}


def run():
    rows = []
    for input_name, xf in (("gaussian", gaussian_x), ("structured", structured_x)):
        for decoy in ("naive", "unit_norm", "colnorm_matched", "coord"):
            for k in (2, 4, 8):
                m, d = 64, 32
                g = np.random.default_rng([hash(input_name) % 999, hash(decoy) % 999, k])
                x = xf(m, d, g); N_in = spd(d, g)
                z, real_cols, groups = build_z(x, N_in, k, decoy, g)
                r = attack(z, real_cols, groups)
                rows.append({"input": input_name, "decoy": decoy, "k": k, **r})
    return rows


if __name__ == "__main__":
    rows = run()
    hdr = ["input", "decoy", "k", "best_stat", "auc", "top1_recovery", "chance_top1", "top2_recovery"]
    print(" | ".join(hdr))
    print("-" * 92)
    for r in rows:
        print(" | ".join(str(r[h]) for h in hdr))
