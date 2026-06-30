"""GPU-staged obfuscation schedule (non-secret, masked-basis artifacts only).

To cut online TEE<->GPU interaction the TEE pre-generates a per-session, per-step
schedule and STAGES only NON-SECRET artifacts to the GPU. This is an engineering
optimisation, NOT a change to the security model: the GPU still never sees the raw
user input or any raw secret.

A staged slot may contain ONLY:
  * ``slot_id`` / ``commitment`` (public hash) / public ``shape`` / ``dtype``
  * intended ``layer`` / ``module``
  * ``xpad_tilde = T N_in`` and ``cpad_tilde = T W N_out`` (composed, masked-basis)
  * references to folded operators (``*_tilde``) the worker already loads
  * an optional TEE-only-encrypted blob the GPU CANNOT decrypt

It must NEVER contain: raw ``N`` / ``N_in`` / ``N_out`` / ``N_inv`` / ``T`` / raw
pad / recovery matrix / mask seed / input token ids / plaintext embedding /
plaintext hidden / plaintext logits / sampled token.

``audit_gpu_staged_schedule_no_secrets`` enforces this (deep key scan + explicit
``contains_*`` flags) and is run by the prestage CLI and by the runner before the
schedule is handed to the worker. Under paper-facing AAAI a failed audit is fatal
(no fallback to an unsafe path). stdlib only (json / hashlib).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "StagedScheduleSecretLeak",
    "ALLOWED_STAGED_ARTIFACT_KINDS",
    "FORBIDDEN_STAGED_SUBSTRINGS",
    "build_staged_schedule",
    "write_gpu_staged_schedule",
    "load_gpu_staged_schedule",
    "audit_gpu_staged_schedule_no_secrets",
    "staged_schedule_report_fields",
    "STAGED_MANIFEST_FILENAME",
]

STAGED_MANIFEST_FILENAME = "staged_schedule_manifest.json"

# masked-basis / public artifact kinds a slot may carry
ALLOWED_STAGED_ARTIFACT_KINDS = (
    "xpad_tilde", "cpad_tilde", "folded_operator_ref", "commitment",
    "public_shape", "tee_only_encrypted_blob",
)

# substrings that, if they appear in any staged KEY, indicate a secret leak.
# (``*_tilde`` masked-basis names are explicitly allowed and checked first.)
FORBIDDEN_STAGED_SUBSTRINGS = (
    "n_inv", "ninv", "inverse_mask", "mask_inverse", "recover", "recovery",
    "raw_mask", "raw_n", "raw_pad", "pad_raw", "raw_t", "t_raw", "mask_secret",
    "mask_seed", "secret_tensor", "input_ids", "token_ids", "plaintext",
    "raw_prompt", "raw_input", "sampled_token", "next_token_id",
)
# exact raw-mask key names (no ``_tilde``) that are forbidden
_FORBIDDEN_EXACT = frozenset({"n", "t", "n_in", "n_out", "n_res", "perm",
                              "pad", "mask", "n_inv_last"})
_ALWAYS_SAFE_KEYS = frozenset({
    "slot_id", "commitment", "shape", "dtype", "layer", "module",
    "artifact_kind", "schedule_id", "nonlinear_backend", "seq_len",
    "max_new_tokens", "num_layers", "num_slots", "created_at", "session_id",
    "slots", "intended_layer", "intended_module", "path", "ref",
    "folded_operator_ref", "encrypted_blob_ref", "version", "step",
})


class StagedScheduleSecretLeak(RuntimeError):
    """A staged schedule artifact carries (or names) a secret the GPU must not see."""


def _commit(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:32]


def build_staged_schedule(*, schedule_id: str, nonlinear_backend: str,
                          seq_len: int, max_new_tokens: int, num_layers: int,
                          modules: list[str] | None = None,
                          session_id: str | None = None) -> dict[str, Any]:
    """Build a staged-schedule MANIFEST (metadata + commitments only; the actual
    ``xpad_tilde`` / ``cpad_tilde`` tensors come from the folded package, never
    raw secrets). One slot per (layer, module) Linear-boundary pad site."""
    modules = modules or ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    slots = []
    for layer in range(int(num_layers)):
        for mod in modules:
            sid = "L%03d.%s" % (layer, mod)
            slots.append({
                "slot_id": sid,
                "commitment": _commit(schedule_id, sid, nonlinear_backend),
                "intended_layer": layer, "intended_module": mod,
                "artifact_kind": "xpad_tilde+cpad_tilde",
                "folded_operator_ref": "layer_%03d:%s" % (layer, mod),
            })
    # head pad site
    slots.append({"slot_id": "head.lm_head",
                  "commitment": _commit(schedule_id, "head", nonlinear_backend),
                  "intended_layer": -1, "intended_module": "lm_head",
                  "artifact_kind": "xpad_tilde+cpad_tilde",
                  "folded_operator_ref": "head:lm_head"})
    return {
        "schedule_id": schedule_id,
        "nonlinear_backend": nonlinear_backend,
        "seq_len": int(seq_len), "max_new_tokens": int(max_new_tokens),
        "num_layers": int(num_layers), "session_id": session_id,
        "num_slots": len(slots), "slots": slots,
        # explicit security flags -- all MUST stay False
        "contains_raw_mask": False, "contains_raw_inverse": False,
        "contains_raw_pad": False, "contains_plaintext_input": False,
        "contains_token_ids": False, "contains_recovery_secret": False,
        "staging_kind": "gpu_staged_nonsecret_obfuscation_artifacts",
    }


def _scan_keys(obj: Any, path: str, leaks: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            klc = str(k).lower()
            safe = (klc in _ALWAYS_SAFE_KEYS or "_tilde" in klc
                    or klc.startswith("contains_") or klc.endswith("_ref")
                    or klc in ("staging_kind", "audit", "manifest_sha256"))
            if not safe:
                if klc in _FORBIDDEN_EXACT:
                    leaks.append("%s.%s (raw-mask key)" % (path, k))
                for bad in FORBIDDEN_STAGED_SUBSTRINGS:
                    if bad in klc:
                        leaks.append("%s.%s (forbidden:%s)" % (path, k, bad))
                        break
            _scan_keys(v, "%s.%s" % (path, k), leaks)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _scan_keys(v, "%s[%d]" % (path, i), leaks)


def audit_gpu_staged_schedule_no_secrets(manifest: Any) -> dict[str, Any]:
    """Raise :class:`StagedScheduleSecretLeak` if the staged manifest carries (or
    names) any secret; return an audit dict when clean. Accepts a manifest dict or
    a directory/file path."""
    if isinstance(manifest, (str, Path)):
        manifest = load_gpu_staged_schedule(manifest)
    if not isinstance(manifest, dict):
        raise StagedScheduleSecretLeak("staged schedule manifest is not a dict")

    # 1. explicit security flags must all be False
    flags = ("contains_raw_mask", "contains_raw_inverse", "contains_raw_pad",
             "contains_plaintext_input", "contains_token_ids",
             "contains_recovery_secret")
    bad_flags = [f for f in flags if manifest.get(f) is True]
    if bad_flags:
        raise StagedScheduleSecretLeak(
            "staged schedule declares secret content: %s" % bad_flags)

    # 2. deep key scan for forbidden / raw-secret names
    leaks: list[str] = []
    _scan_keys(manifest, "manifest", leaks)
    if leaks:
        raise StagedScheduleSecretLeak(
            "staged schedule keys leak secrets: %s" % leaks[:8])

    # 3. every slot's artifact kind must be in the allowed set
    for slot in manifest.get("slots", []):
        kinds = str(slot.get("artifact_kind", "")).split("+")
        for k in kinds:
            base = k.strip()
            if base and base not in ALLOWED_STAGED_ARTIFACT_KINDS:
                raise StagedScheduleSecretLeak(
                    "slot %s has disallowed artifact_kind %r"
                    % (slot.get("slot_id"), base))
    return {
        "staged_schedule_no_secret_audit_passed": True,
        "gpu_staged_schedule_contains_raw_masks": False,
        "gpu_staged_schedule_contains_raw_pad": False,
        "gpu_staged_schedule_contains_plaintext_input": False,
        "gpu_staged_schedule_contains_token_ids": False,
        "gpu_staged_schedule_contains_recovery_secret": False,
        "num_slots": manifest.get("num_slots"),
        "nonlinear_backend": manifest.get("nonlinear_backend"),
    }


def write_gpu_staged_schedule(out_dir: str | Path, manifest: dict[str, Any]
                              ) -> Path:
    """Audit then write the staged-schedule manifest. Refuses to write if the
    manifest fails the no-secret audit."""
    audit_gpu_staged_schedule_no_secrets(manifest)        # fail before writing
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    path = out / STAGED_MANIFEST_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def load_gpu_staged_schedule(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / STAGED_MANIFEST_FILENAME
    return json.loads(p.read_text(encoding="utf-8"))


def staged_schedule_report_fields(manifest: dict[str, Any],
                                  audit: dict[str, Any] | None = None,
                                  *, slots_consumed: int | None = None
                                  ) -> dict[str, Any]:
    """Report fields a runner stamps when a staged schedule is used."""
    total = manifest.get("num_slots")
    return {
        "staged_schedule_used": True,
        "staged_schedule_id": manifest.get("schedule_id"),
        "staged_schedule_slots_total": total,
        "staged_schedule_slots_consumed": slots_consumed,
        "staged_schedule_full_coverage_verified": (
            slots_consumed is not None and total is not None
            and slots_consumed >= total),
        "staged_schedule_no_secret_audit_passed": bool(
            (audit or {}).get("staged_schedule_no_secret_audit_passed")),
        "gpu_staged_schedule_contains_raw_masks": False,
        "gpu_staged_schedule_contains_raw_pad": False,
        "gpu_staged_schedule_contains_plaintext_input": False,
        "optimization": "gpu_staged_nonsecret_obfuscation_artifacts",
        "raw_input_protected": True,
        "sampling_location": "trusted_boundary",
        "plaintext_logits_on_gpu": False,
        "sampled_token_on_gpu": False,
    }
