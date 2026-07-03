#!/usr/bin/env python3
"""Verify a base folded Qwen package (integrity + no-secret + nonlinear backend).

Thin CLI over ``pllo.deployment.folded_package.verify_package`` plus an
``--expected-nonlinear-backend`` check against the manifest. Emits the fields
the generation-eval runbook expects: ``package_valid``, ``nonlinear_backend_ok``,
``contains_mask_secrets``, ``contains_plaintext_inputs``, ``hash_mismatches``,
``missing_shards``.

Example::

    python scripts/verify_qwen7b_folded_package.py \
      --folded-package-path /root/.../qwen7b_folded_full_trusted_shortcut_seq1024_pad \
      --expected-nonlinear-backend trusted_shortcut \
      --output-json /root/.../verify_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pllo.deployment.folded_package import verify_package          # noqa: E402
from pllo.deployment.folded_package_manifest import load_manifest  # noqa: E402
from pllo.experiments.nonlinear_designs import (                   # noqa: E402
    normalize_nonlinear_backend)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folded-package-path", required=True)
    ap.add_argument("--expected-nonlinear-backend")
    ap.add_argument("--output-json")
    args = ap.parse_args(argv)

    result = dict(verify_package(args.folded_package_path))

    nl_ok = None
    nl_problems = []
    try:
        man = load_manifest(args.folded_package_path)
        actual = getattr(man, "nonlinear_backend", None)
        result["nonlinear_backend"] = actual
        if args.expected_nonlinear_backend:
            exp = normalize_nonlinear_backend(args.expected_nonlinear_backend)
            act = normalize_nonlinear_backend(actual) if actual else None
            result["expected_nonlinear_backend"] = exp
            nl_ok = (act == exp)
            if not nl_ok:
                nl_problems.append(
                    f"manifest nonlinear_backend {act!r} != expected {exp!r}")
    except Exception as e:                                          # noqa: BLE001
        nl_problems.append(f"manifest load failed: {e}")
        nl_ok = False
    result["nonlinear_backend_ok"] = nl_ok
    result["nonlinear_backend_problems"] = nl_problems

    ok = bool(result.get("package_valid")) and (nl_ok is not False)
    result["overall_ok"] = ok

    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
