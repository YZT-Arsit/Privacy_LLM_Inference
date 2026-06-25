"""Public benchmark dataset support for masked-Qwen task-utility evaluation.

Local-only (no downloads): converters turn user-provided local files into a
single normalized example schema, deterministic prompt builders and pure-Python
metrics score predictions, and a backend-pluggable runner produces honest
reports (always carrying ``dry_run`` / ``paper_ready`` / ``backend`` labels).
"""

from __future__ import annotations

__all__ = ["task_schemas", "prompt_templates", "metrics",
           "public_dataset_converters", "runners"]
