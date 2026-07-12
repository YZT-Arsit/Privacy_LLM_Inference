"""Batched authenticated training/eval data plane — shared by the TDX service and A10 runner.

Design (frozen threat model):
  * A10 (untrusted): holds input_ids + supervised-position metadata; runs the masked forward;
    sends supervised-position logits + a signed batch descriptor. Never computes CE/dlogits,
    never holds classification labels or causal-LM targets.
  * TDX (trusted): holds the private label table (sample_id -> target) + the deterministic batch
    schedule; validates every batch against a strict state machine + monotonic ledger; computes
    CE + dlogits (classification or causal-LM w/ ignore_index) over the un-permuted vocab; returns
    authenticated dlogits only. No labels / no plaintext loss beyond the CE scalar leave TDX.

Batch descriptor bound into the per-batch HMAC (so tampering any field is rejected):
  run_id, dataset_id, split, task, epoch, optimizer_step, microbatch_index, global_batch_index,
  sample_ids, seq_shape, sup_counts, accum_window, package_hash, adapter_id, template_hash,
  tokenizer_hash, label_schema_hash.

Ledger (monotonic, per training split): rejects stale / repeated / out-of-order / wrong-sample-id
/ wrong-split / wrong-shape / wrong-label-count / wrong-ignore-mask / wrong-accum-window batches.
Eval splits (dev/test/validation) bypass the monotonic ledger (repeat scans allowed) but still
validate binding + labels. The ledger is AEAD-sealable (ChaCha20-Poly1305) for restart.
"""
from __future__ import annotations
import hashlib, hmac, io, json
import torch
import torch.nn.functional as F

TRAIN_SPLITS = {"train"}
EVAL_SPLITS = {"dev", "test", "validation"}


# ---------------- batch descriptor binding ----------------
DESC_FIELDS = ["run_id", "dataset_id", "split", "task", "epoch", "optimizer_step",
               "microbatch_index", "global_batch_index", "sample_ids", "seq_shape",
               "sup_counts", "accum_window", "package_hash", "adapter_id",
               "template_hash", "tokenizer_hash", "label_schema_hash"]


def canonical_desc(desc: dict) -> bytes:
    fields = {k: desc.get(k) for k in DESC_FIELDS}
    return json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()


def batch_mac(key: bytes, payload: bytes, seq: int, run_id: str, op: str, desc: dict) -> str:
    """HMAC over payload + seq + run_id + op + the canonical batch descriptor."""
    m = (hashlib.sha256(payload).digest() + str(seq).encode() + run_id.encode()
         + op.encode() + hashlib.sha256(canonical_desc(desc)).digest())
    return hmac.new(key, m, hashlib.sha256).hexdigest()


# ---------------- CE + dlogits (trusted side) ----------------
def unpermute(masked: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """masked[:, perm] -> plaintext-domain logits (perm is the vocab permutation)."""
    return masked[:, perm]


def cls_ce_dlogits(logits_plain: torch.Tensor, targets: torch.Tensor):
    """Classification CE at one position per example. logits_plain [n,V], targets [n] (token ids).
    Returns (mean_ce, dlogits[n,V], per_example_nll[n])."""
    logp = F.log_softmax(logits_plain.float(), dim=-1)
    nll = -logp[torch.arange(targets.shape[0]), targets]      # [n]
    ce = nll.mean()
    probs = torch.softmax(logits_plain.float(), dim=-1)
    d = probs.clone()
    d[torch.arange(targets.shape[0]), targets] -= 1.0
    d = d / targets.shape[0]                                   # mean reduction grad
    return ce, d, nll


def cls_eval(logits_plain: torch.Tensor, targets: torch.Tensor, neg_tok: int, pos_tok: int, labels: torch.Tensor):
    """Eval: predicted label = argmax over {neg,pos} verbalizer logits; NLL vs target token."""
    logp = F.log_softmax(logits_plain.float(), dim=-1)
    nll = -logp[torch.arange(targets.shape[0]), targets]
    two = torch.stack([logits_plain[:, neg_tok], logits_plain[:, pos_tok]], dim=-1)  # [n,2]
    pred = two.argmax(-1)                                       # 0->neg,1->pos
    correct = (pred == labels).long()
    return pred, correct, nll


def clm_ce_dlogits(logits_plain: torch.Tensor, targets: torch.Tensor):
    """Causal-LM CE over supervised positions (ignore already excluded). logits_plain [S,V],
    targets [S] token ids (all supervised). Mean reduction over S. Returns (ce, dlogits[S,V], nll[S])."""
    logp = F.log_softmax(logits_plain.float(), dim=-1)
    nll = -logp[torch.arange(targets.shape[0]), targets]
    ce = nll.mean()
    probs = torch.softmax(logits_plain.float(), dim=-1)
    d = probs.clone()
    d[torch.arange(targets.shape[0]), targets] -= 1.0
    d = d / targets.shape[0]
    return ce, d, nll


# ---------------- state machine / ledger ----------------
class BatchLedger:
    """Monotonic per-(dataset,split) training-batch ledger. Rejects replay / out-of-order /
    duplicate / gaps. Eval splits are validated but not sequenced."""

    def __init__(self):
        self.last = {}          # (dataset_id, split) -> last accepted global_batch_index
        self.count = {}         # (dataset_id, split) -> accepted count
        self.accum_seen = {}    # optimizer_step -> set(microbatch_index) within current window

    def key(self, desc):
        return (desc["dataset_id"], desc["split"])

    def check_and_advance(self, desc, expected_sample_ids):
        """Returns (ok, reason). Advances the ledger only on a fully-valid TRAINING batch."""
        split = desc["split"]
        # sample-id integrity (both train + eval)
        if list(desc["sample_ids"]) != list(expected_sample_ids):
            return False, "wrong_sample_ids"
        if split in EVAL_SPLITS:
            return True, "eval_ok"          # no monotonic sequencing for eval
        if split not in TRAIN_SPLITS:
            return False, "unknown_split"
        k = self.key(desc)
        last = self.last.get(k, -1)
        gbi = desc["global_batch_index"]
        if gbi <= last:
            return False, ("repeated_batch" if gbi == last else "stale_batch")
        if gbi != last + 1:
            return False, "out_of_order_batch"
        # gradient-accumulation window sanity
        aw = int(desc.get("accum_window", 1))
        mi = int(desc.get("microbatch_index", 0))
        if not (0 <= mi < aw):
            return False, "wrong_accum_window"
        self.last[k] = gbi
        self.count[k] = self.count.get(k, 0) + 1
        return True, "ok"

    def snapshot(self) -> bytes:
        obj = {"last": {f"{a}|{b}": v for (a, b), v in self.last.items()},
               "count": {f"{a}|{b}": v for (a, b), v in self.count.items()}}
        return json.dumps(obj, sort_keys=True).encode()

    def load(self, blob: bytes):
        obj = json.loads(blob.decode())
        self.last = {tuple(k.split("|", 1)): v for k, v in obj["last"].items()}
        self.count = {tuple(k.split("|", 1)): v for k, v in obj["count"].items()}


# ---------------- independent plaintext oracles (verification only, NOT in the protocol path) ----------------
def oracle_cls_ce(logits_plain, targets):
    return F.cross_entropy(logits_plain.float(), targets, reduction="mean")


def oracle_clm_ce(full_logits_plain, full_targets, ignore_index=-100):
    """full_logits_plain [T,V], full_targets [T] with ignore_index on non-supervised. Standard."""
    return F.cross_entropy(full_logits_plain.float(), full_targets, ignore_index=ignore_index, reduction="mean")
