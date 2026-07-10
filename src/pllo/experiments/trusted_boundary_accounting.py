"""CPU-only trusted-boundary call counting + communication/compute accounting (C, D).

This is PROTOCOL-LEVEL instrumentation ONLY. It counts logical invocations,
bytes, and operations. It measures NO real TEE latency, NO real GPU time, NO
enclave crossing cost. "invocation count" != "measured TEE latency". A CPU
`ProtocolCounter` simulates the schedule so the counts are observed, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DTYPE_BYTES = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2}


# ---------------------------------------------------------------------------
# C — trusted invocation counting via a simulated schedule
# ---------------------------------------------------------------------------
@dataclass
class ProtocolCounter:
    """Simulated per-step protocol. Counts logical trusted invocations and
    worker->trusted returns for a training step. No timing."""
    input_calls: int = 0
    loss_calls: int = 0
    update_calls: int = 0
    nonlinear_calls: int = 0
    worker_to_trusted_returns: int = 0
    log: list[str] = field(default_factory=list)

    def mask_input(self):
        self.input_calls += 1
        self.log.append("trusted:mask_input")

    def forward_layer_nonlinear(self):
        # nonlinear island runs ON THE WORKER — zero trusted calls
        self.log.append("worker:nonlinear_island")

    def logits_return_and_loss(self):
        self.worker_to_trusted_returns += 1          # worker returns masked logits
        self.loss_calls += 1                          # trusted computes loss + logit grad
        self.log.append("trusted:loss")

    def packed_update(self, num_lora_layers):
        self.worker_to_trusted_returns += 1          # worker returns ONE packed grad msg
        self.update_calls += 1                        # single trusted unmask+AdamW+remask
        self.log.append("trusted:packed_update")

    def per_layer_update(self, num_lora_layers):
        for _ in range(num_lora_layers):
            self.worker_to_trusted_returns += 1
            self.update_calls += 1
            self.log.append("trusted:per_layer_update")

    @property
    def total_invocations(self):
        return self.input_calls + self.loss_calls + self.update_calls + self.nonlinear_calls


def simulate_training_step(num_layers, num_lora_layers, schedule="packed") -> dict:
    c = ProtocolCounter()
    c.mask_input()                                    # 1 input masking invocation
    for _ in range(num_layers):
        c.forward_layer_nonlinear()                   # islands on worker (0 trusted)
    c.logits_return_and_loss()                        # 1 loss/logits invocation
    if schedule == "packed":
        c.packed_update(num_lora_layers)              # 1 packed update invocation
    else:
        c.per_layer_update(num_lora_layers)           # |S| per-layer updates
    total = c.total_invocations
    expected = 3 if schedule == "packed" else 2 + num_lora_layers
    return {
        "num_layers": num_layers, "num_lora_layers": num_lora_layers, "schedule": schedule,
        "input_calls": c.input_calls, "loss_calls": c.loss_calls,
        "update_calls": c.update_calls, "nonlinear_calls": c.nonlinear_calls,
        "total_invocations": total,
        "worker_to_trusted_returns": c.worker_to_trusted_returns,
        "logical_round_trips": c.input_calls + c.worker_to_trusted_returns,
        "expected_complexity": ("O(1) = 3/step" if schedule == "packed"
                                else "O(2 + |S|)/step"),
        "observed_match": bool(total == expected),
    }


def c_boundary_calls() -> list[dict]:
    rows = []
    for L in (1, 2, 4, 8, 16, 32):
        for schedule in ("packed", "per_layer"):
            rows.append(simulate_training_step(num_layers=L, num_lora_layers=L, schedule=schedule))
    return rows


# ---------------------------------------------------------------------------
# D1 — communication bytes (exact analytic accounting)
# ---------------------------------------------------------------------------
@dataclass
class WorkloadDims:
    batch: int = 1
    seq: int = 128            # m
    vocab: int = 32000        # V
    hidden: int = 2048        # d
    ffn: int = 5632           # d_ff
    layers: int = 22          # L
    rank: int = 8             # r
    lora_layers: int = 22     # |S| target modules with LoRA
    modules_per_layer: int = 1  # target modules (e.g. q_proj) per layer counted
    dtype: str = "float32"
    opt_state_dtype: str = "float32"


def d1_communication(dims: WorkloadDims, schedule="packed") -> dict:
    b, m, V, d, r = dims.batch, dims.seq, dims.vocab, dims.hidden, dims.rank
    S = dims.lora_layers
    wb = DTYPE_BYTES[dims.dtype]
    # per-adapter parameter/gradient element count (d_in=d_out=d assumed square proj)
    adapter_elems = S * r * (d + d)                    # sum_j r_j (d_in + d_out)
    input_hidden = b * m * d * wb                      # masked input dispatched once
    logits_to_trusted = b * m * V * wb                 # masked logits returned
    loss_grad_back = b * m * d * wb                    # grad wrt final hidden injected to worker
    packed_grads = adapter_elems * wb                  # packed gradA/gradB
    updated_adapters = adapter_elems * wb              # updated masked A/B returned
    metadata = S * 3 * 4                               # small mask-id ints per layer
    per_step_total = (input_hidden + logits_to_trusted + loss_grad_back +
                      packed_grads + updated_adapters + metadata)
    # comparison schedules
    full_activation_return = b * m * d * dims.layers * wb   # return every layer's activation
    full_param_update = S * d * d * wb                       # full weight update (not adapter)
    return {
        "schedule": schedule, "dtype": dims.dtype,
        "input_hidden_bytes": input_hidden,
        "logits_to_trusted_bytes": logits_to_trusted,
        "loss_grad_back_bytes": loss_grad_back,
        "packed_grad_bytes": packed_grads,
        "updated_adapter_bytes": updated_adapters,
        "metadata_bytes": metadata,
        "per_step_total_bytes": per_step_total,
        "per_token_bytes": round(per_step_total / (b * m), 2),
        "cmp_full_activation_return_bytes": full_activation_return,
        "cmp_full_param_update_bytes": full_param_update,
        "adapter_scale_formula": "O(sum_j r_j (d_in_j + d_out_j))",
        "logits_formula": "O(m V)",
        "dominant_channel": ("logits" if logits_to_trusted >= packed_grads + updated_adapters
                             else "adapter_update"),
    }


def d1_suite() -> list[dict]:
    rows = []
    small = WorkloadDims(seq=64, vocab=8000, hidden=512, ffn=1376, layers=8, rank=8, lora_layers=8)
    mid = WorkloadDims()  # ~1.1B-ish
    for name, dims in (("small", small), ("mid", mid)):
        for wb in ("float32", "bfloat16"):
            d = WorkloadDims(**{**dims.__dict__, "dtype": wb})
            row = d1_communication(d)
            row["config"] = name
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# D2 — trusted compute operation counts per mask family (NOT timing)
# ---------------------------------------------------------------------------
def d2_trusted_compute(dims: WorkloadDims, mask_family="dense_gl") -> dict:
    b, m, V, d, r = dims.batch, dims.seq, dims.vocab, dims.hidden, dims.rank
    S = dims.lora_layers
    # logit unmask + CE + logit grad: O(mV) elementwise/softmax
    logit_ops = b * m * V
    ce_ops = b * m * V
    # adapter-grad unmask depends on mask family
    if mask_family in ("permutation",):
        unmask_mults = 0                     # index gather only
        unmask_note = "O(adapter_elems) index permute, no matmul"
        unmask_cost = S * r * (d + d)
    elif mask_family in ("positive_diagonal",):
        unmask_mults = S * r * (d + d)       # elementwise scale
        unmask_note = "O(adapter_elems) elementwise scale"
        unmask_cost = unmask_mults
    else:                                     # orthogonal / dense_gl -> matmul with d x d mask
        unmask_mults = S * (d * d * r + r * d * d)   # N^T gradA (d x d)(d x r) etc.
        unmask_note = "O(d^2 r) matmul per adapter -- NOT O(adapter_elems)"
        unmask_cost = unmask_mults
    adamw_ops = S * r * (d + d) * 5          # elementwise m,v,update (~5 ops/elem)
    remask_ops = unmask_mults                 # same cost class as unmask
    mask_gen_ops = (0 if mask_family == "permutation"
                    else S * (d * d if mask_family in ("orthogonal", "dense_gl") else d))
    return {
        "mask_family": mask_family,
        "logit_unmask_ops": logit_ops, "ce_ops": ce_ops,
        "adapter_grad_unmask_mults": unmask_mults, "adapter_grad_unmask_note": unmask_note,
        "adamw_elementwise_ops": adamw_ops, "adapter_remask_ops": remask_ops,
        "mask_generation_ops": mask_gen_ops,
        "unmask_complexity": ("O(adapter_elems)" if mask_family in ("permutation", "positive_diagonal")
                              else "O(d^2 r)"),
        "honest_note": "adapter-SCALE communication does NOT imply adapter-scale compute: "
                       "dense/orthogonal unmask is O(d^2 r).",
    }


def d2_suite(dims: WorkloadDims | None = None) -> list[dict]:
    dims = dims or WorkloadDims(seq=64, vocab=8000, hidden=512, ffn=1376, layers=8, rank=8, lora_layers=8)
    return [d2_trusted_compute(dims, f) for f in
            ("permutation", "positive_diagonal", "orthogonal", "dense_gl")]


# ---------------------------------------------------------------------------
# D3 — trusted-side peak memory estimate (elements/bytes, NOT enclave memory)
# ---------------------------------------------------------------------------
def d3_memory(dims: WorkloadDims) -> dict:
    b, m, V, d, r = dims.batch, dims.seq, dims.vocab, dims.hidden, dims.rank
    S = dims.lora_layers
    wb = DTYPE_BYTES[dims.dtype]
    ob = DTYPE_BYTES[dims.opt_state_dtype]
    adapter = S * r * (d + d)
    params = adapter * wb
    grads = adapter * wb
    adam_mv = 2 * adapter * ob
    masks = S * (d * d + r * r) * wb                # N_out + rank masks (dense)
    temp_unmask = 2 * d * d * wb + 2 * adapter * wb  # transient unmask buffers
    logits_ws = b * m * V * wb
    peak = params + grads + adam_mv + masks + temp_unmask + logits_ws
    return {
        "params_bytes": params, "grads_bytes": grads, "adam_mv_bytes": adam_mv,
        "masks_bytes": masks, "temp_unmask_bytes": temp_unmask,
        "logits_workspace_bytes": logits_ws, "trusted_peak_bytes": peak,
        "trusted_peak_mib": round(peak / 2 ** 20, 3),
        "note": "operation/byte estimate on CPU; NOT real enclave memory or paging.",
    }


def run_accounting() -> dict:
    return {
        "C_boundary_calls": c_boundary_calls(),
        "D1_communication": d1_suite(),
        "D2_trusted_compute": d2_suite(),
        "D3_memory": [d3_memory(WorkloadDims(seq=64, vocab=8000, hidden=512, layers=8, lora_layers=8)),
                      d3_memory(WorkloadDims())],
    }
