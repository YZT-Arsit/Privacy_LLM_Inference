#!/usr/bin/env python
"""Inspect a HuggingFace causal LM and write a JSON structure report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.model_zoo import (
    ExternalModelConfig,
    get_gpt2_module_spec,
    get_model_loader,
    inspect_model_modules,
    is_gpt2_like,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--source", default="huggingface")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "float16"])
    parser.add_argument("--output", type=Path, default=Path("outputs/hf_model_inspection.json"))
    return parser.parse_args()


def main() -> None:
    """Load and inspect a model."""
    args = parse_args()
    config = ExternalModelConfig(
        source=args.source,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    loader = get_model_loader(args.source)
    tokenizer, model = loader.load(config)
    inspection = inspect_model_modules(model)
    result = {
        "config": {
            "source": args.source,
            "model_id": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        },
        "inspection": inspection,
        "gpt2_spec": get_gpt2_module_spec(model) if is_gpt2_like(model) else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "model_class": inspection["model_class"],
                "total_parameters": inspection["total_parameters"],
                "linear_like_count": len(inspection["linear_like_modules"]),
                "recognizes_hf_conv1d": inspection["recognizes_hf_conv1d"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
