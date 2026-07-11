"""TDX trusted GRADIENT-CORRECTION boundary for the O1-C hybrid optimizer (L10/L11/L12).

Runs INSIDE the Intel TDX guest. Holds the TRUSTED correction bundle (per-layer
gram_inv = Nr^T diag(gamma^2) Nr, encoding the private RMSNorm-gain spectrum). Receives
ONLY the masked A-gradients of the gamma-fed targets (q/k/v/gate/up) from the untrusted
worker, right-multiplies each by the matching gram_inv IN-ENCLAVE, and returns ONLY the
corrected masked A-gradients.

This is the endpoint that makes O1-C exact while keeping O1-B's leaky correction matrix
off the GPU. The bundle / gamma / Nr are NEVER returned or logged.

Fail-closed:
  - HMAC-SHA256 auth + monotonic sequence (replay resistance);
  - reject any key that is not one of the 5 allowed gamma-fed targets (o_proj/down_proj
    and any B-gradient, mask secret, or token id are FORBIDDEN here);
  - reject if any expected (target,layer) is missing (correction_missing_targets must be 0).
"""
from __future__ import annotations
import argparse, hashlib, hmac, json, time
from pathlib import Path

import torch

ATTN_TARGETS = ("q_proj", "k_proj", "v_proj")
MLP_TARGETS = ("gate_proj", "up_proj")
ALLOWED = set(ATTN_TARGETS) | set(MLP_TARGETS)
FORBIDDEN_SUBSTR = ("o_proj", "down_proj", "_B", ".B", "N_inv", "Nr", "gamma", "pad",
                    "token", "input_ids", "label")


def orthogonal_signed_perm(n, seed, dtype=torch.float64):
    """Inlined (self-contained) copy of the builder's Nr generator -- bit-identical."""
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    N = torch.zeros(n, n, dtype=dtype)
    N[torch.arange(n), perm] = signs
    return N


def build_gram_inv(gamma_bundle):
    """Reconstruct the dense gram_inv matrices IN-ENCLAVE from the private gamma vectors
    + Nr seed. gram_inv = Nr^T diag(gamma^2) Nr. Never transferred over the wire."""
    H = gamma_bundle["hidden"]; L = gamma_bundle["num_layers"]
    Nr = orthogonal_signed_perm(H, gamma_bundle["nr_seed"], torch.float64)
    attn, mlp = {}, {}
    for l in range(L):
        ga = gamma_bundle["ga"][l].to(torch.float64)
        gm = gamma_bundle["gm"][l].to(torch.float64)
        attn[l] = Nr.T @ torch.diag(ga ** 2) @ Nr
        mlp[l] = Nr.T @ torch.diag(gm ** 2) @ Nr
    return {"attn": attn, "mlp": mlp}


def verify_message(payload_bytes, tag_hex, session_key, seq, last_seq):
    expect = hmac.new(session_key, payload_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, tag_hex):
        raise ValueError("message_authentication_failed")
    if seq <= last_seq:
        raise ValueError(f"replay_or_out_of_order seq={seq} last={last_seq}")
    return True


def correct(grads, bundle, num_layers, counters):
    """grads: {f'{l}.{proj}': gA_tilde}. Returns corrected dict; fail-closed on bad keys."""
    out = {}
    seen = set()
    for key, gA in grads.items():
        l_str, proj = key.split(".", 1)
        l = int(l_str)
        if proj not in ALLOWED:
            counters["forbidden_key_rejected"] += 1
            raise ValueError(f"forbidden_target_key {key}")
        if any(s in key for s in FORBIDDEN_SUBSTR):
            counters["forbidden_key_rejected"] += 1
            raise ValueError(f"forbidden_substring_in_key {key}")
        gram_inv = bundle["attn"][l] if proj in ATTN_TARGETS else bundle["mlp"][l]
        gA64 = gA.to(torch.float64)
        corrected = gA64 @ gram_inv                       # exact fp64 correction, in-enclave
        out[key] = corrected.to(gA.dtype)
        counters["correction_targets_processed"] += 1
        seen.add((l, proj))
    # completeness: 5 target types x num_layers expected
    expected = {(l, p) for l in range(num_layers) for p in ALLOWED}
    counters["correction_missing_targets"] = len(expected - seen)
    if seen != expected:
        raise ValueError(f"incomplete_correction_set missing={len(expected - seen)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--grads", required=True, help="masked A-grads .pt (worker -> TDX)")
    ap.add_argument("--bundle", required=True, help="trusted gamma bundle .pt (TDX-only); gram_inv rebuilt in-enclave")
    ap.add_argument("--session-key-hex", required=True)
    ap.add_argument("--last-seq", type=int, default=-1)
    ap.add_argument("--num-layers", type=int, default=24)
    ap.add_argument("--out-grads", required=True)
    ap.add_argument("--out-response", required=True)
    args = ap.parse_args()

    counters = {"correction_targets_processed": 0, "correction_missing_targets": -1,
                "forbidden_key_rejected": 0, "auth_failures": 0, "replay_rejected": 0,
                "bundle_returned_to_worker": 0, "gamma_returned_to_worker": 0}
    session_key = bytes.fromhex(args.session_key_hex)
    req = json.loads(Path(args.request).read_text())
    g_bytes = Path(args.grads).read_bytes()
    payload = hashlib.sha256(g_bytes).digest() + str(req["seq"]).encode() + req["run_id"].encode()
    try:
        verify_message(payload, req["hmac"], session_key, req["seq"], args.last_seq)
    except ValueError as e:
        kind = "replay_rejected" if "replay" in str(e) else "auth_failures"
        counters[kind] += 1
        Path(args.out_response).write_text(json.dumps(
            {"status": "REJECTED_FAIL_CLOSED", "reason": str(e), "counters": counters}))
        raise SystemExit(2)

    gamma_bundle = torch.load(args.bundle, map_location="cpu")
    bundle = build_gram_inv(gamma_bundle)          # reconstruct gram_inv in-enclave
    grads = torch.load(args.grads, map_location="cpu")
    t0 = time.time()
    try:
        corrected = correct(grads, bundle, args.num_layers, counters)
    except ValueError as e:
        Path(args.out_response).write_text(json.dumps(
            {"status": "REJECTED_FAIL_CLOSED", "reason": str(e), "counters": counters}))
        raise SystemExit(3)
    t_compute = time.time() - t0
    torch.save(corrected, args.out_grads)
    # in-enclave numerics summary per group (aggregate only; no raw grad logged)
    def rng(d, keys):
        vals = [d[k].abs() for k in d if k.split(".", 1)[1] in keys]
        if not vals:
            return {}
        cat = torch.cat([v.flatten() for v in vals]).to(torch.float64)
        return {"min_abs": float(cat[cat > 0].min()) if (cat > 0).any() else 0.0,
                "max_abs": float(cat.max()), "zeroed_frac": float((cat == 0).float().mean())}
    dl_bytes = Path(args.out_grads).read_bytes()
    resp_seq = req["seq"] + 1
    resp_payload = hashlib.sha256(dl_bytes).digest() + str(resp_seq).encode() + req["run_id"].encode()
    resp_tag = hmac.new(session_key, resp_payload, hashlib.sha256).hexdigest()
    Path(args.out_response).write_text(json.dumps({
        "status": "OK", "seq": resp_seq, "hmac": resp_tag, "run_id": req["run_id"],
        "counters": counters,
        "corrected_numerics": {"attn_in": rng(corrected, ATTN_TARGETS),
                               "mlp_in": rng(corrected, MLP_TARGETS)},
        "timing": {"tdx_correction_sec": t_compute,
                   "grads_bytes_received": len(g_bytes), "grads_bytes_returned": len(dl_bytes)},
        "bundle_or_gamma_returned": False}))
    print(json.dumps({"status": "OK", "corrected": counters["correction_targets_processed"],
                      "missing": counters["correction_missing_targets"],
                      "corr_sec": round(t_compute, 3)}))


if __name__ == "__main__":
    main()
