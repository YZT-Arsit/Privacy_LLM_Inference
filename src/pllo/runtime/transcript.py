"""Stage 7.5c - the runtime transcript object.

What the accelerator backend hands back to the trusted controller and (in
turn) what is allowed to be published as a paper artifact. By construction
this object NEVER carries raw tensors, masks, pads, adapters, gradients,
or plaintext input -- only summary statistics, op names, shapes, and
boundary counts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeTranscript:
    """JSON-safe summary of what the accelerator observed.

    Mandatory invariant: ``contains_raw_secret`` MUST be ``False`` for any
    transcript that is written to disk or returned across the trust
    boundary. Tests in ``tests/test_runtime_boundary.py`` enforce this.
    """

    visible_tensor_shapes: list[list[int]] = field(default_factory=list)
    operation_names: list[str] = field(default_factory=list)
    boundary_calls: int = 0
    runtime_ms: float = 0.0
    mask_ids_redacted: list[str] = field(default_factory=list)
    contains_raw_secret: bool = False
    notes: str = ""

    def record_op(self, op: str, *, shapes: list[tuple[int, ...]] | None = None) -> None:
        self.operation_names.append(op)
        self.boundary_calls += 1
        if shapes is not None:
            for s in shapes:
                self.visible_tensor_shapes.append(list(s))

    def record_mask_id(self, raw_seed: int | str) -> None:
        """Store a *redacted* hash of a mask seed instead of the seed itself.

        We never publish raw seeds. The hash is for fingerprinting only.
        """
        digest = hashlib.blake2b(
            str(raw_seed).encode("utf-8"), digest_size=6,
        ).hexdigest()
        self.mask_ids_redacted.append(digest)

    def to_summary(self) -> dict[str, Any]:
        """JSON-safe summary dict."""
        return {
            "visible_tensor_shapes": list(self.visible_tensor_shapes),
            "operation_names": list(self.operation_names),
            "boundary_calls": int(self.boundary_calls),
            "runtime_ms": float(self.runtime_ms),
            "mask_ids_redacted": list(self.mask_ids_redacted),
            "contains_raw_secret": bool(self.contains_raw_secret),
            "notes": str(self.notes),
        }


__all__ = ["RuntimeTranscript"]
