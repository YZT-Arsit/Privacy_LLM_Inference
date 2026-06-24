"""Validate that a private LoRA adapter actually changes model behaviour.

Given a no-LoRA decode report JSON and a LoRA decode report JSON (from the folded
local/remote probes, which carry ``package_token_ids`` / ``reference_token_ids``),
report whether the LoRA path produced different tokens than the no-LoRA path.

This is a correctness guard: a LoRA decode that EXACTLY matches the no-LoRA decode
usually means the adapter was not merged (wrong package, empty adapter, or the
worker silently ran the base path). The script surfaces that as a warning.

Reported fields: ``no_lora_token_ids``, ``lora_token_ids``, ``tokens_differ``,
``token_diff_positions``, ``top1_changed`` (first decoded token, a proxy for the
next-token top-1 when logits are not stored), ``lora_has_effect``, ``warning``.

Example::

    python scripts/validate_lora_effect.py \\
        --no-lora-json outputs/qwen7b_folded_full_decode_probe.json \\
        --lora-json    outputs/qwen7b_lora_folded_remote_decode_probe.json \\
        --output-json  outputs/validate_lora_effect.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# token-id fields a decode/probe report may carry, in extraction priority order.
_TOKEN_KEYS = ("lora_token_ids", "no_lora_token_ids", "package_token_ids",
               "lora_tokens", "no_lora_tokens", "reference_token_ids",
               "expected_token_ids", "recovered_tokens")


def extract_token_ids(data, prefer=None):
    """Pull a flat list of token ids from a report dict (or a raw list)."""
    if data is None:
        return None
    if isinstance(data, list):
        seq = data
    elif isinstance(data, dict):
        keys = ([prefer] if prefer else []) + list(_TOKEN_KEYS)
        seq = None
        for k in keys:
            if k and data.get(k):
                seq = data[k]
                break
        if seq is None:
            return None
    else:
        return None
    if seq and isinstance(seq[0], list):     # [[...]] -> [...]
        seq = seq[0]
    try:
        return [int(x) for x in seq]
    except (TypeError, ValueError):
        return None


def compare_decodes(no_lora, lora, *, no_lora_key=None, lora_key=None) -> dict:
    """Compare two decode reports; return the validation dict (pure, no I/O)."""
    nl = extract_token_ids(no_lora, prefer=no_lora_key)
    ll = extract_token_ids(lora, prefer=lora_key)
    rep: dict = {
        "stage": "validate_lora_effect",
        "no_lora_token_ids": nl, "lora_token_ids": ll,
        "tokens_differ": None, "token_diff_positions": None,
        "top1_changed": None, "lora_has_effect": None, "warning": None,
    }
    if nl is None or ll is None:
        rep["warning"] = ("could not extract token ids from %s report"
                          % ("both" if nl is None and ll is None
                             else "no-lora" if nl is None else "lora"))
        return rep
    n = min(len(nl), len(ll))
    diff_pos = [i for i in range(n) if nl[i] != ll[i]]
    differ = bool(diff_pos) or len(nl) != len(ll)
    rep["tokens_differ"] = differ
    rep["token_diff_positions"] = diff_pos
    rep["top1_changed"] = (bool(nl[0] != ll[0]) if nl and ll else None)
    rep["top1_basis"] = "first_decoded_token"
    rep["lora_has_effect"] = differ
    if not differ:
        rep["warning"] = ("LoRA decode is IDENTICAL to no-LoRA decode; the adapter "
                          "may not have been applied (wrong/empty folded-LoRA "
                          "package, or the worker ran the base path)")
    return rep


def _load(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-lora-json", required=True)
    ap.add_argument("--lora-json", required=True)
    ap.add_argument("--no-lora-key", default=None,
                    help="explicit token-id field in the no-LoRA report")
    ap.add_argument("--lora-key", default=None,
                    help="explicit token-id field in the LoRA report")
    ap.add_argument("--require-effect", default="false",
                    help="exit non-zero if LoRA has no observable effect")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    rep = compare_decodes(_load(args.no_lora_json), _load(args.lora_json),
                          no_lora_key=args.no_lora_key, lora_key=args.lora_key)
    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    print("=== validate LoRA effect ===")
    print("no_lora_token_ids=%s" % rep["no_lora_token_ids"])
    print("lora_token_ids   =%s" % rep["lora_token_ids"])
    print("tokens_differ=%s token_diff_positions=%s top1_changed=%s"
          % (rep["tokens_differ"], rep["token_diff_positions"],
             rep["top1_changed"]))
    if rep["warning"]:
        print("WARNING: %s" % rep["warning"])

    require = str(args.require_effect).strip().lower() in {"1", "true", "yes"}
    if rep["lora_has_effect"] is None:
        return 2 if require else 0
    if require and not rep["lora_has_effect"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
