"""Trusted-side precomputed per-step obfuscation schedule.

Autoregressive decode needs a FRESH obfuscation domain at every step (to stop
cross-token hidden/KV alignment attacks). That per-step trusted-side work is on
the online critical path today. This module lets the trusted side PRECOMPUTE, in a
setup / warm-up phase, the per-step schedule -- one slot per decode step -- so the
online decode can simply *consume* the step's slot instead of deriving it inline.

What this is NOT:
* It does NOT remove per-step freshness: every slot carries a DISTINCT fresh
  ``mask_id`` / ``domain_id`` (derived deterministically from the session seed +
  step), never a single fixed mask.
* It does NOT precompute per-step folded weights (that would explode the ~26GB
  package). The folded model package stays fixed; only the small trusted-only
  per-step obfuscation material is scheduled.

Security invariants (audited):
* secret tensors (fresh mask material, pad values, inverses, PRG seed,
  inter-step transition relation) live ONLY in the trusted runtime object and are
  NEVER serialized into a GPU worker package or a remote request payload;
* the public, serializable surface (slots' metadata, ``to_dict`` / ``to_disk``)
  carries only non-secret ids + shapes;
* ``audit_*`` helpers raise loudly if any schedule secret reaches the GPU side.

Honesty: this module gives the schedule abstraction + per-step consumption hook +
audit + reporting. Whether the online remask/pad/inverse math is ACTUALLY moved
off the online path depends on the runtime that consumes it; until a runtime does
the full migration it must report ``schedule_used_for_metadata_only=True`` and
``online_remask_still_performed=True`` -- do not claim a speed-up that did not
happen.

stdlib only at import time; torch is imported lazily ONLY when generating real
secret tensors (``with_secret_tensors=True``), so the metadata path + audit run
on any machine with no torch / GPU / model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "ObfuscationScheduleSlot",
    "PrecomputedMaskSchedule",
    "ObfuscationSchedule",
    "ScheduleSlotAlreadyConsumed",
    "ScheduleSecretLeak",
    "SCHEDULE_SECRET_FIELDS",
    "audit_schedule_trusted_only",
    "audit_gpu_payload_no_schedule_secrets",
    "audit_worker_package_no_schedule_secrets",
    "schedule_report_fields",
    "default_schedule_report_fields",
]

SCHEDULE_REGISTRY_VERSION = "1.0"

# Field/key names that are SECRET and must never cross to the untrusted GPU side
# (matched case-insensitively as a substring of a payload key name). Tokens are
# SPECIFIC multi-word names so benign metadata keys (e.g. ``has_trusted_secrets``,
# ``pad_location``, ``session_fingerprint``) are never false-flagged; bare tokens
# like "seed"/"inverse" are intentionally NOT here (the server's exact-match guard
# carries "seed" for the wire).
SCHEDULE_SECRET_FIELDS = frozenset({
    "mask_secret", "mask_secrets", "mask_inverse", "mask_matrix",
    "residual_mask", "n_res", "n_res_inv", "pad_value", "pad_values",
    "pad_plaintext", "pad_tensor", "prg_seed", "schedule_seed", "session_seed",
    "secret_tensor", "secret_tensors", "schedule_secret", "transition_secret",
    "inter_step_transition", "raw_lora", "lora_a", "lora_b",
})


class ScheduleSlotAlreadyConsumed(RuntimeError):
    """A schedule slot was consumed twice (online decode reused a step)."""


class ScheduleSecretLeak(RuntimeError):
    """A schedule secret was found on a GPU-facing surface (payload / package)."""


# ---------------------------------------------------------------------------
# Slot (public, serializable metadata ONLY -- secrets live on the schedule)
# ---------------------------------------------------------------------------


@dataclass
class ObfuscationScheduleSlot:
    """One decode step's PUBLIC obfuscation metadata.

    The actual fresh secret tensors for this step are held trusted-only on the
    owning :class:`PrecomputedMaskSchedule` (``_secrets[step_id]``) and are never
    placed on this slot, never serialized, never sent to the GPU."""

    step_id: int
    mask_id: str
    domain_id: str
    dtype: str
    device: str
    hidden_size: int
    pad_meta: Dict[str, Any] = field(default_factory=dict)
    remask_meta: Dict[str, Any] = field(default_factory=dict)
    consumed: bool = False
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "mask_id": self.mask_id,
            "domain_id": self.domain_id,
            "dtype": self.dtype,
            "device": self.device,
            "hidden_size": self.hidden_size,
            "pad_meta": dict(self.pad_meta),
            "remask_meta": dict(self.remask_meta),
            "consumed": self.consumed,
            "audit": dict(self.audit),
        }


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def _hex(*parts: Any) -> str:
    h = hashlib.sha256()
    h.update("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()


class PrecomputedMaskSchedule:
    """Trusted-side per-step obfuscation schedule (one slot per decode step).

    Build with :meth:`precompute`. The online decode calls :meth:`consume`
    (``step_id``) once per generated token. Public metadata is serializable; the
    secret tensors are runtime-only and excluded from every serialization."""

    def __init__(self, *, session_seed: int, hidden_size: int, dtype: str,
                 device: str, mask_family: str, nonlinear_backend: str,
                 max_steps: int) -> None:
        self.session_seed = int(session_seed)
        self.hidden_size = int(hidden_size)
        self.dtype = str(dtype)
        self.device = str(device)
        self.mask_family = str(mask_family)
        self.nonlinear_backend = str(nonlinear_backend)
        self.max_steps = int(max_steps)
        self.slots: List[ObfuscationScheduleSlot] = []
        # runtime-only secret store: NEVER serialized / sent to GPU.
        self._secrets: Dict[int, Dict[str, Any]] = {}
        self.secret_tensors_present = False
        self.precompute_latency_s: Optional[float] = None
        self.strict_audit = True

    # -- construction -----------------------------------------------------
    @classmethod
    def precompute(cls, *, max_steps: int, hidden_size: int, seed: int,
                   seq_len: int = 0, max_new_tokens: int = 0,
                   dtype: str = "float32", device: str = "cpu",
                   mask_family: str = "pairwise_complex_scaling",
                   nonlinear_backend: str = "current",
                   with_secret_tensors: bool = True,
                   strict_audit: bool = True) -> "PrecomputedMaskSchedule":
        """Precompute a fresh per-step schedule (deterministic from ``seed``).

        ``with_secret_tensors`` generates the trusted-only secret material via
        torch (CPU is fine); set False for a metadata-only schedule (no torch).
        ``max_steps`` should cover ``max_new_tokens`` (extra slots are harmless
        headroom)."""
        sched = cls(session_seed=seed, hidden_size=hidden_size, dtype=dtype,
                    device=device, mask_family=mask_family,
                    nonlinear_backend=nonlinear_backend, max_steps=max_steps)
        sched.strict_audit = bool(strict_audit)
        t0 = time.perf_counter()
        tgen = _maybe_torch() if with_secret_tensors else None
        sched.secret_tensors_present = tgen is not None
        for step in range(int(max_steps)):
            # FRESH per step: ids derive from (seed, step) -> distinct every step.
            mask_id = _hex(seed, step, mask_family, "mask")[:32]
            domain_id = _hex(seed, step, nonlinear_backend, "domain")[:32]
            slot = ObfuscationScheduleSlot(
                step_id=step, mask_id=mask_id, domain_id=domain_id,
                dtype=dtype, device=device, hidden_size=hidden_size,
                pad_meta={"pad_present": True, "pad_dim": hidden_size,
                          "pad_location": "trusted_runtime_only"},
                remask_meta={"kind": "fresh_per_step",
                             "mask_family": mask_family,
                             "restore": "trusted_only",
                             "secret_location": "trusted_runtime_only"},
                audit={"secret_held": "trusted_runtime_only",
                       "serializable": True, "fresh_per_step": True})
            sched.slots.append(slot)
            if tgen is not None:
                sched._secrets[step] = _make_step_secret(
                    tgen, seed, step, hidden_size, dtype, device)
        sched.precompute_latency_s = round(time.perf_counter() - t0, 6)
        if strict_audit:
            audit_schedule_trusted_only(sched)
        return sched

    # -- online consumption ----------------------------------------------
    def slot(self, step_id: int) -> ObfuscationScheduleSlot:
        if not (0 <= step_id < len(self.slots)):
            raise IndexError("step_id %d out of range [0, %d)"
                             % (step_id, len(self.slots)))
        return self.slots[step_id]

    def consume(self, step_id: int, *,
                allow_reconsume: bool = False) -> ObfuscationScheduleSlot:
        """Consume the slot for ``step_id`` (marks it used). Raises on a repeat
        consume unless ``allow_reconsume`` -- a reused step would reuse an
        obfuscation domain, which is exactly what must not happen."""
        s = self.slot(step_id)
        if s.consumed and not allow_reconsume:
            raise ScheduleSlotAlreadyConsumed(
                "schedule slot step_id=%d already consumed (reusing an "
                "obfuscation domain across decode steps is forbidden)" % step_id)
        s.consumed = True
        return s

    def step_secret(self, step_id: int) -> Optional[Dict[str, Any]]:
        """Trusted-only fresh secret material for a step (None in metadata-only
        mode). NEVER serialize / send this to the GPU."""
        return self._secrets.get(int(step_id))

    # -- views / persistence ---------------------------------------------
    def slots_consumed(self) -> int:
        return sum(1 for s in self.slots if s.consumed)

    def public_metadata(self) -> Dict[str, Any]:
        """Non-secret schedule metadata (safe to log / persist). Excludes the
        raw seed value (a PRG seed is a secret) -- only a seed *fingerprint* is
        exposed for reproducibility bookkeeping."""
        return {
            "schedule_registry_version": SCHEDULE_REGISTRY_VERSION,
            "max_steps": self.max_steps,
            "slots_precomputed": len(self.slots),
            "slots_consumed": self.slots_consumed(),
            "hidden_size": self.hidden_size,
            "dtype": self.dtype,
            "device": self.device,
            "mask_family": self.mask_family,
            "nonlinear_backend": self.nonlinear_backend,
            "has_trusted_secrets": self.secret_tensors_present,
            "session_fingerprint": _hex(self.session_seed, "fp")[:16],
            "precompute_latency_s": self.precompute_latency_s,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Fully PUBLIC, JSON-safe view (metadata + per-slot public dicts). No
        secret tensors are included by construction."""
        return {
            **self.public_metadata(),
            "slots": [s.to_public_dict() for s in self.slots],
        }

    def to_disk(self, path, *, save_secret_tensors: bool = False,
                allow_secret_persist: bool = False) -> str:
        """Persist the PUBLIC schedule metadata as JSON. Secret tensors are NOT
        written by default. ``save_secret_tensors=True`` is refused unless the
        caller ALSO passes ``allow_secret_persist=True`` (strongly discouraged;
        secrets should stay in trusted runtime memory)."""
        if save_secret_tensors and not allow_secret_persist:
            raise ScheduleSecretLeak(
                "refusing to persist schedule secret tensors to disk "
                "(save_secret_tensors=True requires allow_secret_persist=True; "
                "secrets should stay in trusted runtime memory only)")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["secret_tensors_persisted"] = bool(
            save_secret_tensors and allow_secret_persist)
        p.write_text(json.dumps(payload, indent=2, default=str),
                     encoding="utf-8")
        if save_secret_tensors and allow_secret_persist:
            self._persist_secret_tensors(p.with_suffix(p.suffix + ".secret"))
        return str(p)

    def _persist_secret_tensors(self, path) -> None:  # pragma: no cover
        # Explicit opt-in only path; kept tiny + clearly separated from public.
        import torch  # lazy
        torch.save({k: v for k, v in self._secrets.items()}, str(path))

    def stats(self) -> Dict[str, Any]:
        return {
            **self.public_metadata(),
            "slots_consumed": self.slots_consumed(),
            "all_slots_consumed": (bool(self.slots)
                                   and self.slots_consumed() == len(self.slots)),
        }


# back-compat / alias name
ObfuscationSchedule = PrecomputedMaskSchedule


# ---------------------------------------------------------------------------
# Secret material (lazy torch)
# ---------------------------------------------------------------------------


def _maybe_torch():
    try:
        import torch
        return torch
    except Exception:                                       # noqa: BLE001
        return None


_DTYPES = {"float32": "float32", "float64": "float64", "bfloat16": "bfloat16",
           "float16": "float16"}


def _make_step_secret(torch, seed: int, step: int, hidden_size: int,
                      dtype: str, device: str) -> Dict[str, Any]:
    """A FRESH per-step trusted-only secret: pad vector + domain scale, generated
    on CPU from a step-specific generator (different step -> different secret).
    Stays in the trusted runtime; never serialized / sent to GPU."""
    gen = torch.Generator().manual_seed(int(seed) + int(step) + 1)
    dt = getattr(torch, _DTYPES.get(str(dtype), "float32"))
    pad = (torch.randn(hidden_size, generator=gen).to(dt))
    domain_scale = (torch.rand(1, generator=gen) + 0.5).to(dt)
    return {"pad": pad, "domain_scale": domain_scale,
            "device": str(device), "note": "trusted_runtime_only_secret"}


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _secret_key_paths(payload: Any,
                      names: frozenset = SCHEDULE_SECRET_FIELDS) -> List[str]:
    """Dotted paths of any key whose lowercased name contains a secret token."""
    found: List[str] = []

    def hit(k: str) -> bool:
        lk = str(k).lower()
        return any(tok in lk for tok in names)

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                here = "%s.%s" % (path, k) if path else str(k)
                if hit(k):
                    found.append(here)
                walk(v, here)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, "%s[%d]" % (path, i))

    walk(payload, "")
    return found


def audit_gpu_payload_no_schedule_secrets(payload: Any) -> Dict[str, Any]:
    """Raise :class:`ScheduleSecretLeak` if a schedule secret key appears in a
    GPU-facing request payload; return an audit record when clean."""
    paths = _secret_key_paths(payload)
    if paths:
        raise ScheduleSecretLeak(
            "schedule secret field(s) present in GPU payload: %s" % paths)
    return {"audit": "gpu_payload_no_schedule_secrets", "ok": True,
            "schedule_secret_paths": []}


def audit_worker_package_no_schedule_secrets(path) -> Dict[str, Any]:
    """Raise if a worker package directory/manifest carries schedule secrets.

    Accepts a directory path (scans file names + a manifest.json if present) or a
    dict (treated as a manifest). Lightweight + defensive -- it never loads model
    weights."""
    if isinstance(path, dict):
        paths = _secret_key_paths(path)
        if paths:
            raise ScheduleSecretLeak(
                "schedule secret field(s) in worker manifest: %s" % paths)
        return {"audit": "worker_package_no_schedule_secrets", "ok": True}
    p = Path(path)
    bad_files = []
    if p.exists():
        for f in (p.rglob("*") if p.is_dir() else [p]):
            ln = f.name.lower()
            if any(tok in ln for tok in ("schedule_secret", "mask_secret",
                                         ".secret")):
                bad_files.append(str(f))
        manifest = p / "manifest.json" if p.is_dir() else None
        if manifest and manifest.is_file():
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
                if _secret_key_paths(m):
                    bad_files.append("manifest.json:%s" % _secret_key_paths(m))
            except Exception:                                # noqa: BLE001
                pass
    if bad_files:
        raise ScheduleSecretLeak(
            "schedule secret artifact(s) in worker package: %s" % bad_files)
    return {"audit": "worker_package_no_schedule_secrets", "ok": True,
            "scanned": str(p)}


def audit_schedule_trusted_only(schedule: "PrecomputedMaskSchedule"
                                ) -> Dict[str, Any]:
    """Assert the schedule's serializable surface holds no secret tensors and no
    secret-named fields; secrets must live only in the runtime ``_secrets`` store.
    Raises :class:`ScheduleSecretLeak` on violation."""
    pub = schedule.to_dict()
    # 1) no secret-named keys in the public surface
    paths = _secret_key_paths(pub)
    if paths:
        raise ScheduleSecretLeak(
            "schedule public surface contains secret-named field(s): %s" % paths)
    # 2) the public surface must be plain JSON (no tensors leaked into slots)
    try:
        json.dumps(pub)
    except TypeError as exc:
        raise ScheduleSecretLeak(
            "schedule public surface is not JSON-serializable (a tensor likely "
            "leaked into a slot): %s" % exc)
    # 3) secret tensors, if any, are ONLY in the runtime store
    return {"audit": "schedule_trusted_only", "ok": True,
            "secret_tensors_present": schedule.secret_tensors_present,
            "secret_steps": len(schedule._secrets)}


# ---------------------------------------------------------------------------
# Report fields (all defaulted so old report consumers never break)
# ---------------------------------------------------------------------------


def default_schedule_report_fields() -> Dict[str, Any]:
    """The schedule report fields for a run with the feature DISABLED (all safe
    defaults -- old reports/consumers parse unchanged)."""
    return {
        "precompute_obfuscation_schedule": False,
        "schedule_max_steps": None,
        "schedule_slots_precomputed": 0,
        "schedule_slots_consumed": 0,
        "schedule_precompute_latency_s": None,
        "online_generation_latency_s": None,
        "latency_s_total_including_precompute": None,
        "latency_s_online_only": None,
        "boundary_calls": None,
        "boundary_calls_per_generated_token": None,
        "trusted_bytes": None,
        "gpu_bytes": None,
        "schedule_secret_leaked_to_gpu": False,
        "gpu_request_contains_schedule_secret": False,
        "schedule_used_for_metadata_only": False,
        "online_remask_still_performed": True,
    }


def schedule_report_fields(schedule: Optional["PrecomputedMaskSchedule"] = None,
                           *, enabled: bool,
                           online_generation_latency_s: Optional[float] = None,
                           boundary_calls: Optional[int] = None,
                           generated_tokens: Optional[int] = None,
                           trusted_bytes: Optional[int] = None,
                           gpu_bytes: Optional[int] = None,
                           schedule_used_for_metadata_only: bool = True,
                           online_remask_still_performed: bool = True,
                           schedule_secret_leaked_to_gpu: bool = False,
                           gpu_request_contains_schedule_secret: bool = False
                           ) -> Dict[str, Any]:
    """Build the schedule report fields for a run. ``schedule_used_for_metadata_
    only`` / ``online_remask_still_performed`` MUST reflect reality -- if the
    runtime did not actually move the remask off the online path, leave them True
    (do not claim a speed-up that did not happen)."""
    out = default_schedule_report_fields()
    if not enabled or schedule is None:
        # still surface any explicitly-measured costs (e.g. baseline online run)
        if online_generation_latency_s is not None:
            out["online_generation_latency_s"] = round(
                online_generation_latency_s, 6)
            out["latency_s_online_only"] = round(online_generation_latency_s, 6)
            out["latency_s_total_including_precompute"] = round(
                online_generation_latency_s, 6)
        out["boundary_calls"] = boundary_calls
        out["trusted_bytes"] = trusted_bytes
        out["gpu_bytes"] = gpu_bytes
        if boundary_calls is not None and generated_tokens:
            out["boundary_calls_per_generated_token"] = round(
                boundary_calls / generated_tokens, 6)
        return out

    pre = schedule.precompute_latency_s
    out.update({
        "precompute_obfuscation_schedule": True,
        "schedule_max_steps": schedule.max_steps,
        "schedule_slots_precomputed": len(schedule.slots),
        "schedule_slots_consumed": schedule.slots_consumed(),
        "schedule_precompute_latency_s": pre,
        "online_generation_latency_s": (None if online_generation_latency_s is None
                                        else round(online_generation_latency_s, 6)),
        "latency_s_online_only": (None if online_generation_latency_s is None
                                  else round(online_generation_latency_s, 6)),
        "latency_s_total_including_precompute": (
            None if online_generation_latency_s is None
            else round(online_generation_latency_s + (pre or 0.0), 6)),
        "boundary_calls": boundary_calls,
        "trusted_bytes": trusted_bytes,
        "gpu_bytes": gpu_bytes,
        "schedule_secret_leaked_to_gpu": bool(schedule_secret_leaked_to_gpu),
        "gpu_request_contains_schedule_secret": bool(
            gpu_request_contains_schedule_secret),
        "schedule_used_for_metadata_only": bool(schedule_used_for_metadata_only),
        "online_remask_still_performed": bool(online_remask_still_performed),
    })
    if boundary_calls is not None and generated_tokens:
        out["boundary_calls_per_generated_token"] = round(
            boundary_calls / generated_tokens, 6)
    return out
