"""Training-stage security audit for protected (masked) LoRA training.

During protected LoRA training the trusted boundary owns the training data, token
ids, LoRA adapter parameters (A, B), gradients (dA, dB), optimizer state, and the
mask secrets. It offloads only the **frozen base** matmuls of the adapted layers
to the untrusted GPU, sending masked activations and receiving masked outputs.
The LoRA terms, the loss, the gradients, and the optimizer updates never leave the
boundary.

This module defines the GPU-channel message schemas for that training protocol
and the audit that verifies, against the *exact* recorded GPU trace, that the
untrusted GPU never received any of:

    raw training samples, labels, raw prompt, input_ids, tokenized examples,
    plaintext hidden states, raw LoRA A / B, raw delta_W = B@A, LoRA gradients
    dA / dB, optimizer states, adapter updates dA/dB, recovered logits, or mask
    secrets.

numpy + standard library only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np

__all__ = [
    "LoRAMaskedInitRequest",
    "LoRAMaskedMatmulRequest",
    "LoRAMaskedMatmulResponse",
    "LoRATrainingTrace",
    "LoRATrainingAuditReport",
    "LORA_FORBIDDEN_FIELD_NAMES",
    "audit_lora_training_trace",
    "gather_arrays",
]


# ---------------------------------------------------------------------------
# GPU-channel messages (trusted boundary -> untrusted GPU worker)
# ---------------------------------------------------------------------------


@dataclass
class LoRAMaskedInitRequest:
    """Trusted -> GPU once. Folded (masked) frozen base weights + public meta.

    ``folded_base_weights[layer] = N^{-1} W M`` for each adapted layer's frozen
    base weight ``W``. These are masked artifacts; the raw masks never cross."""
    session_id: str
    folded_base_weights: dict[str, np.ndarray]
    public_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoRAMaskedMatmulRequest:
    """Trusted -> GPU per adapted layer per step. A masked activation only."""
    session_id: str
    layer: str
    step: int
    phase: str                              # "forward" | "backward"
    masked_input: np.ndarray                # X_tilde = c * (X @ N)
    batch_size: int
    in_features: int
    out_features: int


@dataclass
class LoRAMaskedMatmulResponse:
    """GPU -> trusted. The masked base output only."""
    session_id: str
    layer: str
    step: int
    masked_output: np.ndarray               # c * (X @ W) @ M


@dataclass
class LoRATrainingTrace:
    """Everything that crossed to the untrusted GPU during protected training."""
    inbound: list[Any] = field(default_factory=list)     # to GPU
    outbound: list[Any] = field(default_factory=list)    # from GPU
    gpu_calls: dict[str, int] = field(default_factory=dict)
    trusted_bytes: int = 0
    gpu_bytes: int = 0
    tee_used_on_gpu: bool = False

    def _bump(self, name: str) -> None:
        self.gpu_calls[name] = self.gpu_calls.get(name, 0) + 1

    def record_inbound(self, msg: Any) -> None:
        self.inbound.append(msg)
        self.gpu_bytes += _array_nbytes(msg)
        self._bump(type(msg).__name__)

    def record_outbound(self, msg: Any) -> None:
        self.outbound.append(msg)
        self.gpu_bytes += _array_nbytes(msg)

    @property
    def messages(self) -> list[Any]:
        return self.inbound + self.outbound


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------


@dataclass
class LoRATrainingAuditReport:
    gpu_visible_train_examples: bool
    gpu_visible_labels: bool
    gpu_visible_input_ids: bool
    gpu_visible_tokenized_examples: bool
    gpu_visible_lora_a: bool
    gpu_visible_lora_b: bool
    gpu_visible_delta_w: bool
    gpu_visible_lora_grad_a: bool
    gpu_visible_lora_grad_b: bool
    gpu_visible_optimizer_state: bool
    gpu_visible_adapter_update: bool
    gpu_visible_plain_hidden: bool
    gpu_visible_recovered_logits: bool
    leaked_secret_fields: list[str]
    forbidden_field_names: list[str]
    tee_used_on_gpu: bool
    audit_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Field names that must never appear in a GPU-channel message.
LORA_FORBIDDEN_FIELD_NAMES = frozenset({
    "raw_prompt", "prompt", "train_examples", "training_examples", "labels",
    "label", "input_ids", "token_ids", "tokenized", "tokenizer_output",
    "plain_hidden", "hidden_states", "lora_a", "lora_b", "delta_w", "deltaw",
    "grad_a", "grad_b", "lora_grad_a", "lora_grad_b", "dA", "dB",
    "optimizer_state", "adam_m", "adam_v", "adapter_update", "delta_a",
    "delta_b", "recovered_logits", "mask_secret", "mask_secrets", "in_perm",
    "in_signs", "out_perm", "out_scale", "seed",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gather_arrays(obj: Any) -> list[np.ndarray]:
    """All ndarray leaves inside a message/dataclass/dict/list."""
    out: list[np.ndarray] = []

    def walk(o: Any) -> None:
        if isinstance(o, np.ndarray):
            out.append(o)
        elif is_dataclass(o) and not isinstance(o, type):
            for f in fields(o):
                walk(getattr(o, f.name))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return out


def _array_nbytes(obj: Any) -> int:
    return sum(int(a.nbytes) for a in gather_arrays(obj))


def _names(obj: Any) -> list[str]:
    """All field/key names appearing in a message tree."""
    out: list[str] = []

    def walk(o: Any) -> None:
        if is_dataclass(o) and not isinstance(o, type):
            for f in fields(o):
                out.append(f.name)
                walk(getattr(o, f.name))
        elif isinstance(o, dict):
            for k, v in o.items():
                out.append(str(k))
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return out


def _arr_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if (np.issubdtype(a.dtype, np.floating)
            or np.issubdtype(b.dtype, np.floating)):
        return bool(np.allclose(a, b, atol=1e-8, rtol=1e-6))
    return bool(np.array_equal(a, b))


def _arr_contains(haystack: np.ndarray, needle: np.ndarray) -> bool:
    if _arr_equal(haystack, needle):
        return True
    h, n = np.asarray(haystack), np.asarray(needle)
    if (np.issubdtype(h.dtype, np.integer) and np.issubdtype(n.dtype, np.integer)
            and h.size == n.size and h.size > 0):
        return bool(np.array_equal(np.sort(h.ravel()), np.sort(n.ravel())))
    return False


def _appears(trace_arrays: list[np.ndarray], targets: Any) -> bool:
    """True if any provided target array appears among the trace arrays."""
    if targets is None:
        return False
    if isinstance(targets, np.ndarray):
        targets = [targets]
    for t in targets:
        if t is None:
            continue
        t = np.asarray(t)
        if t.size == 0:
            continue
        for h in trace_arrays:
            if _arr_contains(h, t):
                return True
    return False


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_lora_training_trace(
    trace: LoRATrainingTrace,
    plaintext: dict[str, Any] | None = None,
    secrets: dict[str, Any] | None = None,
    *,
    raise_on_fail: bool = False,
) -> LoRATrainingAuditReport:
    """Audit the recorded GPU trace of a protected LoRA training run.

    ``plaintext`` maps artifact names to the trusted-side array(s) (single array
    or list of per-step snapshots) to value-check against the trace:
    ``train_examples``, ``labels``, ``input_ids``, ``lora_a``, ``lora_b``,
    ``delta_w``, ``lora_grad_a``, ``lora_grad_b``, ``optimizer_state``,
    ``adapter_update``, ``plain_hidden``, ``recovered_logits``. ``secrets`` maps
    mask-secret names to arrays (e.g. ``in_perm``/``in_signs``/``out_perm``/
    ``out_scale``). Every ``gpu_visible_*`` should be False and
    ``leaked_secret_fields`` empty for a correct protected run."""
    plaintext = plaintext or {}
    secrets = secrets or {}
    arrays = [a for m in trace.messages for a in gather_arrays(m)]

    def vis(key: str) -> bool:
        return _appears(arrays, plaintext.get(key))

    # structural: forbidden field NAMES anywhere in the messages
    forbidden_names = sorted({n for m in trace.messages for n in _names(m)
                              if n in LORA_FORBIDDEN_FIELD_NAMES})

    # value-based: secret arrays literally present in the trace
    leaked = []
    for sname, sval in secrets.items():
        if sval is None:
            continue
        if _appears(arrays, np.asarray(sval)):
            leaked.append(sname)
    leaked.sort()

    report = LoRATrainingAuditReport(
        gpu_visible_train_examples=vis("train_examples"),
        gpu_visible_labels=vis("labels"),
        gpu_visible_input_ids=vis("input_ids"),
        gpu_visible_tokenized_examples=vis("tokenized_examples"),
        gpu_visible_lora_a=vis("lora_a"),
        gpu_visible_lora_b=vis("lora_b"),
        gpu_visible_delta_w=vis("delta_w"),
        gpu_visible_lora_grad_a=vis("lora_grad_a"),
        gpu_visible_lora_grad_b=vis("lora_grad_b"),
        gpu_visible_optimizer_state=vis("optimizer_state"),
        gpu_visible_adapter_update=vis("adapter_update"),
        gpu_visible_plain_hidden=vis("plain_hidden"),
        gpu_visible_recovered_logits=vis("recovered_logits"),
        leaked_secret_fields=leaked,
        forbidden_field_names=forbidden_names,
        tee_used_on_gpu=bool(trace.tee_used_on_gpu),
        audit_passed=False,
    )
    visible_any = any([
        report.gpu_visible_train_examples, report.gpu_visible_labels,
        report.gpu_visible_input_ids, report.gpu_visible_tokenized_examples,
        report.gpu_visible_lora_a,
        report.gpu_visible_lora_b, report.gpu_visible_delta_w,
        report.gpu_visible_lora_grad_a, report.gpu_visible_lora_grad_b,
        report.gpu_visible_optimizer_state, report.gpu_visible_adapter_update,
        report.gpu_visible_plain_hidden, report.gpu_visible_recovered_logits,
    ])
    report.audit_passed = bool(
        not visible_any and not leaked and not forbidden_names
        and not report.tee_used_on_gpu)

    if raise_on_fail and not report.audit_passed:
        raise AssertionError(f"LoRA training audit failed: {report.to_dict()}")
    return report
