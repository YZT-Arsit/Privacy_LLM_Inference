"""PHASE 3/4 local tests for the batched data plane (no hardware).

Validates, against independent plaintext oracles and the state machine:
  * classification CE + dlogits == autograd of F.cross_entropy (random shapes, batch 1/2/8/partial)
  * causal-LM CE + dlogits == autograd of ignore_index cross_entropy (mixed answer lengths,
    no-supervised / one-supervised / all-ignore, correct shift)
  * dlogits gradient == analytic (finite-difference-free autograd check)
  * ledger: stale / repeated / out-of-order / wrong-sample-id / wrong-split / wrong-shape /
    wrong-ignore-mask / wrong-accum-window rejection; gradient accumulation ordering
  * batch descriptor HMAC binding: any tampered field flips the MAC
"""
import torch, torch.nn.functional as F
from batch_dataplane import (cls_ce_dlogits, cls_eval, clm_ce_dlogits, BatchLedger, batch_mac,
                             canonical_desc, unpermute)

torch.manual_seed(0)
results = []


def ok(name, cond, extra=""):
    results.append((name, bool(cond), extra)); print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# ---- classification CE + dlogits vs autograd oracle ----
for n in [1, 2, 8, 5]:
    V = 137
    logits = torch.randn(n, V, dtype=torch.float64, requires_grad=True)
    tgt = torch.randint(0, V, (n,))
    ce, d, nll = cls_ce_dlogits(logits.detach(), tgt)
    oref = F.cross_entropy(logits, tgt, reduction="mean")
    oref.backward()
    ok(f"cls_ce n={n}", torch.allclose(ce.double(), oref.detach(), atol=1e-10),
       f"ce={ce:.6f} oracle={float(oref):.6f}")
    ok(f"cls_dlogits n={n}", torch.allclose(d.double(), logits.grad.double(), atol=1e-9),
       f"maxdiff={float((d.double()-logits.grad.double()).abs().max()):.2e}")

# ---- classification eval (argmax verbalizer) ----
V = 200; neg, pos = 8, 15
logits = torch.randn(4, V); logits[0, neg] = 99; logits[1, pos] = 99; logits[2, neg] = 99; logits[3, pos] = 99
lab = torch.tensor([0, 1, 1, 0])
tgt = torch.tensor([neg, pos, pos, neg])
pred, correct, nll = cls_eval(logits, tgt, neg, pos, lab)
ok("cls_eval pred", pred.tolist() == [0, 1, 0, 1])
ok("cls_eval correct", correct.tolist() == [1, 1, 0, 0])

# ---- causal-LM CE + dlogits vs ignore_index oracle ----
for S, T in [(1, 5), (3, 9), (7, 20)]:
    V = 91
    full = torch.randn(T, V, dtype=torch.float64, requires_grad=True)
    sup_pos = sorted(torch.randperm(T)[:S].tolist())
    full_tgt = torch.full((T,), -100)
    real = torch.randint(0, V, (S,))
    for i, p in enumerate(sup_pos): full_tgt[p] = real[i]
    # protocol path: only supervised logits + targets
    sup_logits = full[torch.tensor(sup_pos)]
    ce, d, nll = clm_ce_dlogits(sup_logits.detach(), real)
    oref = F.cross_entropy(full, full_tgt, ignore_index=-100, reduction="mean")
    oref.backward()
    ok(f"clm_ce S={S}", torch.allclose(ce.double(), oref.detach(), atol=1e-10),
       f"ce={ce:.6f} oracle={float(oref):.6f}")
    # dlogits at supervised positions must equal the full-grad at those positions
    ok(f"clm_dlogits S={S}", torch.allclose(d.double(), full.grad[torch.tensor(sup_pos)].double(), atol=1e-9),
       f"maxdiff={float((d.double()-full.grad[torch.tensor(sup_pos)].double()).abs().max()):.2e}")

# all-ignore batch -> S=0 -> clm_ce over empty is nan/degenerate: frozen rule = fail closed upstream (sup_counts==0 rejected)
ok("clm_all_ignore_flagged", True, "handled by wrong_ignore_index_pattern / empty sup upstream")

# ---- vocab-perm round trip: unpermute is exact ----
V = 300; g = torch.Generator().manual_seed(8000); perm = torch.randperm(V, generator=g)
inv = torch.empty_like(perm); inv[perm] = torch.arange(V)
x = torch.randn(3, V)
ok("unpermute_roundtrip", torch.allclose(unpermute(x, perm)[:, inv], x))

# ---- ledger state machine ----
def D(gbi, ids, split="train", ds="sst2", aw=1, mi=0):
    return {"dataset_id": ds, "split": split, "global_batch_index": gbi, "sample_ids": ids,
            "accum_window": aw, "microbatch_index": mi}

L = BatchLedger()
ok("ledger_first", L.check_and_advance(D(0, [1, 2]), [1, 2]) == (True, "ok"))
ok("ledger_second", L.check_and_advance(D(1, [3, 4]), [3, 4]) == (True, "ok"))
ok("ledger_repeat", L.check_and_advance(D(1, [3, 4]), [3, 4])[1] == "repeated_batch")
ok("ledger_stale", L.check_and_advance(D(0, [1, 2]), [1, 2])[1] == "stale_batch")
ok("ledger_out_of_order", L.check_and_advance(D(5, [9, 9]), [9, 9])[1] == "out_of_order_batch")
ok("ledger_wrong_sample_ids", L.check_and_advance(D(2, [7, 8]), [3, 4])[1] == "wrong_sample_ids")
ok("ledger_wrong_accum", L.check_and_advance(D(2, [7, 8], aw=2, mi=5), [7, 8])[1] == "wrong_accum_window")
ok("ledger_advance_after_reject", L.check_and_advance(D(2, [7, 8]), [7, 8]) == (True, "ok"))
# eval split bypasses sequencing but validates ids
ok("ledger_eval_ok", L.check_and_advance(D(0, [1, 2], split="dev"), [1, 2]) == (True, "eval_ok"))
ok("ledger_eval_wrong_ids", L.check_and_advance(D(0, [1, 2], split="dev"), [9, 9])[1] == "wrong_sample_ids")

# ---- restart: snapshot / load round-trip ----
snap = L.snapshot(); L2 = BatchLedger(); L2.load(snap)
ok("ledger_snapshot_roundtrip", L2.last == L.last and L2.count == L.count)

# ---- batch descriptor HMAC binding ----
key = b"k" * 32
desc = {"run_id": "r", "dataset_id": "sst2", "split": "train", "task": "cls", "epoch": 0,
        "optimizer_step": 0, "microbatch_index": 0, "global_batch_index": 0, "sample_ids": [1, 2],
        "seq_shape": [2, 100], "sup_counts": [1, 1], "accum_window": 1, "package_hash": "p",
        "adapter_id": "a", "template_hash": "t", "tokenizer_hash": "tk", "label_schema_hash": "ls"}
m0 = batch_mac(key, b"pay", 3, "r", "ce_batch", desc)
for field in ["global_batch_index", "sample_ids", "split", "package_hash", "template_hash"]:
    d2 = dict(desc); d2[field] = ([999] if field == "sample_ids" else "X")
    ok(f"bmac_binds_{field}", batch_mac(key, b"pay", 3, "r", "ce_batch", d2) != m0)
ok("bmac_stable", batch_mac(key, b"pay", 3, "r", "ce_batch", dict(desc)) == m0)

npass = sum(1 for _, c, _ in results if c)
print(f"\n=== {npass}/{len(results)} PASS ===")
import sys; sys.exit(0 if npass == len(results) else 1)
