"""Security audit for the TEE-boundary <-> GPU-worker protocol.

These functions inspect the *exact* messages that crossed to the untrusted GPU
worker (recorded in a :class:`~pllo.protocol.tee_gpu_messages.ProtocolTrace`) and
verify the boundary's confidentiality claims:

* :func:`assert_no_gpu_visible_plaintext` -- no raw prompt, ``input_ids``,
  generated token ids, recovered logits, or tokenizer output crossed.
* :func:`assert_no_mask_secret_leak` -- no mask perm / inverse / signs / scale /
  seed / :class:`MaskHandles` crossed.
* :func:`assert_wrong_mask_recovery_fails` -- recovering masked logits with the
  wrong mask does not reproduce the plaintext logits/token.

Each ``assert_*`` returns a list of human-readable findings (empty == clean) and,
by default (``raise_on_fail=True``), raises :class:`AssertionError` if non-empty.
The demo calls them with ``raise_on_fail=False`` to populate its report.

numpy + standard library only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pllo.protocol.tee_gpu_messages import (
    GPU_INBOUND_TYPES,
    GPU_OUTBOUND_TYPES,
    ProtocolTrace,
    iter_named_values,
)
from pllo.tee.runtime_api import MaskHandles, recover_vocab_logits

__all__ = [
    "PLAINTEXT_FIELD_NAMES",
    "MASK_SECRET_FIELD_NAMES",
    "assert_no_gpu_visible_plaintext",
    "assert_no_mask_secret_leak",
    "assert_wrong_mask_recovery_fails",
    "leaf_name",
]

# Field-name blocklists (structural check, independent of values).
PLAINTEXT_FIELD_NAMES = frozenset({
    "raw_prompt", "prompt", "prompts", "text", "input_ids", "input_id",
    "token_ids", "generated_token_ids", "generated_tokens", "next_token_id",
    "next_token_ids", "tokens", "tokenizer_output", "recovered_logits",
    "plain_logits", "plaintext_logits",
})
MASK_SECRET_FIELD_NAMES = frozenset({
    "residual_perm", "residual_inv_perm", "residual_signs", "vocab_perm",
    "vocab_inv_perm", "vocab_scale", "vocab_inv_scale", "input_pad", "seed",
    "mask_handles", "handles", "perm", "inv_perm", "signs", "scale",
})


def leaf_name(path: str) -> str:
    """Final attribute/key name of a dotted/indexed path from ``iter_named_values``."""
    return path.split(".")[-1].split("[")[0]


def _finish(findings: list[str], raise_on_fail: bool, what: str) -> list[str]:
    if findings and raise_on_fail:
        raise AssertionError(f"{what}: " + "; ".join(findings))
    return findings


# ---------------------------------------------------------------------------
# 1. No GPU-visible plaintext
# ---------------------------------------------------------------------------


def assert_no_gpu_visible_plaintext(
    trace: ProtocolTrace,
    *,
    raw_prompt: str | list[str] | None = None,
    input_ids: np.ndarray | None = None,
    generated_token_ids: np.ndarray | None = None,
    recovered_logits: np.ndarray | None = None,
    raise_on_fail: bool = True,
) -> list[str]:
    """Verify no plaintext crossed to the GPU worker.

    Two complementary checks: (a) **structural** -- no GPU message field is named
    like a plaintext field, and every GPU message is an allowed type; (b)
    **value** -- the actual known plaintext artifacts (raw prompt string,
    ``input_ids``, generated tokens, recovered logits) do not appear verbatim in
    any GPU message. Returns the list of leaked field paths (empty == clean)."""
    findings: list[str] = []
    allowed = GPU_INBOUND_TYPES + GPU_OUTBOUND_TYPES
    messages = trace.gpu_inbound_messages + trace.gpu_outbound_messages

    raw_strs: list[str] = []
    if isinstance(raw_prompt, str):
        raw_strs = [raw_prompt]
    elif isinstance(raw_prompt, list):
        raw_strs = [str(p) for p in raw_prompt]
    id_arr = None if input_ids is None else np.asarray(input_ids)
    gen_arr = None if generated_token_ids is None else np.asarray(
        generated_token_ids)
    rec_arr = None if recovered_logits is None else np.asarray(recovered_logits)

    for mi, msg in enumerate(messages):
        if not isinstance(msg, allowed):
            findings.append(f"msg[{mi}]: forbidden type {type(msg).__name__}")
            continue
        for path, val in iter_named_values(msg):
            name = leaf_name(path)
            if name in PLAINTEXT_FIELD_NAMES:
                findings.append(f"msg[{mi}].{path}: plaintext field name")
            if isinstance(val, str) and any(r and r in val for r in raw_strs):
                findings.append(f"msg[{mi}].{path}: raw prompt text present")
            if isinstance(val, np.ndarray):
                if id_arr is not None and _array_contains(val, id_arr):
                    findings.append(f"msg[{mi}].{path}: input_ids present")
                if gen_arr is not None and _array_contains(val, gen_arr):
                    findings.append(
                        f"msg[{mi}].{path}: generated token ids present")
                if rec_arr is not None and _array_equal(val, rec_arr):
                    findings.append(
                        f"msg[{mi}].{path}: recovered logits present")
    return _finish(findings, raise_on_fail, "gpu-visible plaintext")


# ---------------------------------------------------------------------------
# 2. No mask-secret leak
# ---------------------------------------------------------------------------


def assert_no_mask_secret_leak(
    trace: ProtocolTrace,
    handles: MaskHandles | None = None,
    *,
    raise_on_fail: bool = True,
) -> list[str]:
    """Verify no mask secret crossed to the GPU worker.

    Structural check: no GPU message field is named like a mask secret and no
    :class:`MaskHandles` instance crossed. Value check (if ``handles`` given):
    none of the secret arrays (perm/inv/signs/scale/pad) or the integer ``seed``
    appear verbatim in any GPU message. The *folded* LM head is allowed -- it is
    a transformed artifact, not equal to any raw secret array."""
    findings: list[str] = []
    messages = trace.gpu_inbound_messages + trace.gpu_outbound_messages

    secret_arrays: list[tuple[str, np.ndarray]] = []
    secret_seed: int | None = None
    if handles is not None:
        secret_seed = int(handles.seed)
        for nm in ("residual_perm", "residual_inv_perm", "residual_signs",
                   "vocab_perm", "vocab_inv_perm", "vocab_scale",
                   "vocab_inv_scale"):
            arr = getattr(handles, nm)
            if arr is not None:
                secret_arrays.append((nm, np.asarray(arr)))
        if handles.input_pad is not None:
            secret_arrays.append(("input_pad", np.asarray(handles.input_pad)))

    for mi, msg in enumerate(messages):
        if isinstance(msg, MaskHandles):
            findings.append(f"msg[{mi}]: MaskHandles object crossed")
            continue
        for path, val in iter_named_values(msg):
            name = leaf_name(path)
            if name in MASK_SECRET_FIELD_NAMES:
                findings.append(f"msg[{mi}].{path}: mask-secret field name")
            if isinstance(val, MaskHandles):
                findings.append(f"msg[{mi}].{path}: MaskHandles value")
            if isinstance(val, np.ndarray):
                for sn, sa in secret_arrays:
                    if _array_equal(val, sa):
                        findings.append(f"msg[{mi}].{path}: equals {sn}")
            if (secret_seed is not None and isinstance(val, (int, np.integer))
                    and not isinstance(val, bool) and int(val) == secret_seed):
                findings.append(f"msg[{mi}].{path}: equals mask seed")
    return _finish(findings, raise_on_fail, "mask-secret leak")


# ---------------------------------------------------------------------------
# 3. Wrong-mask recovery must fail
# ---------------------------------------------------------------------------


def assert_wrong_mask_recovery_fails(
    masked_logits: np.ndarray,
    correct_handles: MaskHandles,
    wrong_handles: MaskHandles,
    plaintext_logits: np.ndarray,
    *,
    tol: float = 1e-3,
    min_wrong_rel_err: float = 0.1,
    raise_on_fail: bool = True,
) -> dict[str, Any]:
    """Recovering with the wrong mask must not reproduce the plaintext.

    Confirms (a) the correct mask recovers the plaintext logits within ``tol``
    and recovers the plaintext argmax; (b) the wrong mask flips at least one
    argmax token AND its max abs error is a non-trivial fraction
    (``min_wrong_rel_err``) of the plaintext logit scale -- a magnitude-relative
    test so it holds whether logits are O(0.1) or O(10). Returns a metrics dict;
    raises if either property fails."""
    masked = np.asarray(masked_logits)
    plain = np.asarray(plaintext_logits)
    correct = recover_vocab_logits(masked, correct_handles)
    wrong = recover_vocab_logits(masked, wrong_handles)

    plain_scale = float(max(np.abs(plain).max(), 1e-9))
    correct_err = float(np.abs(correct - plain).max())
    wrong_err = float(np.abs(wrong - plain).max())
    plain_tok = plain.argmax(axis=-1)
    correct_tok = correct.argmax(axis=-1)
    wrong_tok = wrong.argmax(axis=-1)
    correct_recovers = bool(correct_err < tol
                            and np.array_equal(correct_tok, plain_tok))
    wrong_diverges = bool(wrong_err >= min_wrong_rel_err * plain_scale
                          and (wrong_tok != plain_tok).any())

    metrics = {
        "correct_max_abs_err": correct_err,
        "wrong_max_abs_err": wrong_err,
        "plaintext_logit_scale": plain_scale,
        "wrong_rel_err": float(wrong_err / plain_scale),
        "correct_recovers_plaintext": correct_recovers,
        "wrong_mask_diverges": wrong_diverges,
        "correct_token_match": bool(np.array_equal(correct_tok, plain_tok)),
        "wrong_token_match": bool(np.array_equal(wrong_tok, plain_tok)),
    }
    findings: list[str] = []
    if not correct_recovers:
        findings.append(
            f"correct mask did not recover plaintext (err={correct_err:.3e})")
    if not wrong_diverges:
        findings.append(
            f"wrong mask did NOT diverge (err={wrong_err:.3e}, "
            f"token_match={metrics['wrong_token_match']})")
    if findings and raise_on_fail:
        raise AssertionError("wrong-mask recovery: " + "; ".join(findings))
    metrics["findings"] = findings
    return metrics


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype,
                                                            np.floating):
        return bool(np.allclose(a, b))
    return bool(np.array_equal(a, b))


def _array_contains(haystack: np.ndarray, needle: np.ndarray) -> bool:
    """True if ``needle`` appears as the full content of ``haystack`` (any shape)
    or matches its flattened integer sequence -- catches token ids smuggled in
    a reshaped/!=-dtype array."""
    h = np.asarray(haystack)
    n = np.asarray(needle)
    if _array_equal(h, n):
        return True
    if (np.issubdtype(h.dtype, np.integer) and np.issubdtype(n.dtype,
                                                             np.integer)
            and h.size == n.size and h.size > 0):
        return bool(np.array_equal(np.sort(h.ravel()), np.sort(n.ravel())))
    return False
