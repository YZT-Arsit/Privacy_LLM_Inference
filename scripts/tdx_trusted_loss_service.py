"""TDX trusted loss/dlogits boundary for the private-base D4 protocol.

Runs INSIDE the Intel TDX guest (tdx310). Holds the trusted-domain secrets: the
vocabulary monomial permutation (frozen builder seed 8000) and the real training
labels. Receives ONLY the frozen masked-logit representation from the untrusted H800
worker, and returns ONLY the correctly re-masked masked-domain gradient.

Per request (one training step):
  1. verify session + message integrity (HMAC over the payload with the session key;
     monotonically increasing sequence number; fail-closed on a bad tag / replay /
     malformed message);
  2. apply the exact inverse monomial-logit transform  plaintext = masked[:, perm];
  3. compute the real cross-entropy loss against the private labels (causal shift);
  4. compute plaintext-domain dlogits internally (softmax - onehot)/N;
  5. return the re-masked masked-domain gradient  dlogits_masked = dlogits_plain[:, perm_inv];
  6. never log plaintext labels / logits / dlogits -- only approved aggregate metrics.

CPU torch. No GPU, no HF model, no base weights here.
"""
from __future__ import annotations
import argparse
import hashlib
import hmac
import json
import time
from pathlib import Path

import torch


def vocab_perm(V: int, seed: int = 8000):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(V, generator=g)
    perm_inv = torch.empty_like(perm); perm_inv[perm] = torch.arange(V)
    return perm, perm_inv


def verify_message(payload_bytes: bytes, tag_hex: str, session_key: bytes,
                   seq: int, last_seq: int):
    """HMAC-SHA256 auth + monotonic sequence (replay resistance). Fail-closed."""
    expect = hmac.new(session_key, payload_bytes, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expect, tag_hex):
        raise ValueError("message_authentication_failed")
    if seq <= last_seq:
        raise ValueError(f"replay_or_out_of_order seq={seq} last={last_seq}")
    return True


def trusted_step(masked_logits, labels, perm, perm_inv, counters):
    """The trusted computation. Returns (loss_float, dlogits_masked, aggregate)."""
    T, V = masked_logits.shape
    # 2. inverse monomial-logit transform (un-permute columns) -> plaintext domain
    plain = masked_logits[:, perm]                       # (T,V)
    # 3. real CE with causal shift (positions 0..T-2 predict tokens 1..T-1)
    lg = plain[:-1].float(); tg = labels[1:]
    loss = torch.nn.functional.cross_entropy(lg, tg)
    counters["tdx_ce_computed"] += 1
    # 4. plaintext-domain dlogits = d CE / d plain
    probs = torch.softmax(lg, dim=-1)
    dplain = probs.clone()
    dplain[torch.arange(tg.shape[0]), tg] -= 1.0
    dplain = dplain / tg.shape[0]                         # mean reduction
    dlogits_plain = torch.zeros_like(plain)
    dlogits_plain[:-1] = dplain
    # 5. re-mask to masked domain: dmasked[:,k] = dplain[:, perm_inv[k]]
    dlogits_masked = dlogits_plain[:, perm_inv]
    counters["dlogits_remasked"] += 1
    # 6. approved aggregate metrics only (NO plaintext logged)
    aggregate = {"ce_loss": float(loss),
                 "dlogits_masked_l2": float(dlogits_masked.norm()),
                 "num_positions": int(tg.shape[0])}
    return float(loss), dlogits_masked, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, help="request JSON (msg envelope)")
    ap.add_argument("--masked-logits", required=True, help="masked logits .pt")
    ap.add_argument("--labels", required=True, help="labels json (private, trusted-side)")
    ap.add_argument("--session-key-hex", required=True)
    ap.add_argument("--last-seq", type=int, default=-1)
    ap.add_argument("--out-dlogits", required=True)
    ap.add_argument("--out-response", required=True)
    ap.add_argument("--vocab-seed", type=int, default=8000)
    args = ap.parse_args()

    counters = {"tdx_ce_computed": 0, "dlogits_remasked": 0,
                "malformed_rejected": 0, "replay_rejected": 0,
                "auth_failures": 0}
    session_key = bytes.fromhex(args.session_key_hex)
    req = json.loads(Path(args.request).read_text())
    labels = torch.tensor(json.loads(Path(args.labels).read_text())["input_ids"])

    # 1. verify session + message integrity (fail-closed)
    ml_bytes = Path(args.masked_logits).read_bytes()
    payload = hashlib.sha256(ml_bytes).digest() + str(req["seq"]).encode() \
        + req["run_id"].encode()
    t_net0 = req.get("worker_send_time")
    try:
        verify_message(payload, req["hmac"], session_key, req["seq"], args.last_seq)
    except ValueError as e:
        kind = "replay_rejected" if "replay" in str(e) else "auth_failures"
        counters[kind] += 1
        Path(args.out_response).write_text(json.dumps(
            {"status": "REJECTED_FAIL_CLOSED", "reason": str(e), "counters": counters}))
        raise SystemExit(2)

    t0 = time.time()
    masked_logits = torch.load(args.masked_logits, map_location="cpu").float()
    t_load = time.time() - t0
    V = masked_logits.shape[1]
    perm, perm_inv = vocab_perm(V, args.vocab_seed)
    t1 = time.time()
    loss, dlogits_masked, aggregate = trusted_step(
        masked_logits, labels, perm, perm_inv, counters)
    t_compute = time.time() - t1
    t2 = time.time()
    torch.save(dlogits_masked, args.out_dlogits)
    t_save = time.time() - t2
    # response envelope: authenticate the returned gradient too
    dl_bytes = Path(args.out_dlogits).read_bytes()
    resp_seq = req["seq"] + 1
    resp_payload = hashlib.sha256(dl_bytes).digest() + str(resp_seq).encode() \
        + req["run_id"].encode()
    resp_tag = hmac.new(session_key, resp_payload, hashlib.sha256).hexdigest()
    Path(args.out_response).write_text(json.dumps({
        "status": "OK", "seq": resp_seq, "hmac": resp_tag, "run_id": req["run_id"],
        "aggregate": aggregate, "counters": counters,
        "timing": {"tdx_load_sec": t_load, "tdx_transform_and_ce_sec": t_compute,
                   "tdx_save_sec": t_save,
                   "logits_bytes_received": len(ml_bytes),
                   "gradient_bytes_returned": len(dl_bytes)},
        "plaintext_logged": False}))
    print(json.dumps({"status": "OK", "ce_loss": loss,
                      "logits_bytes": len(ml_bytes), "grad_bytes": len(dl_bytes)}))


if __name__ == "__main__":
    main()
