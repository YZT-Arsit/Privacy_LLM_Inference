#!/usr/bin/env python
"""Run a plain HuggingFace GPT-2 forward and greedy generation smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.model_zoo import ExternalModelConfig, get_model_loader


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "float16"])
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/hf_gpt2_smoke.json"))
    return parser.parse_args()


def main() -> None:
    """Run plain forward and greedy generation."""
    args = parse_args()
    config = ExternalModelConfig(
        source="huggingface",
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    tokenizer, model = get_model_loader("huggingface").load(config)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: tensor.to(torch.device(args.device)) for name, tensor in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    result = {
        "config": {
            "model_id": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
        },
        "forward": {
            "input_ids_shape": list(encoded["input_ids"].shape),
            "logits_shape": list(outputs.logits.shape),
        },
        "generation": {
            "generated_token_ids": generated.detach().cpu().tolist(),
            "decoded_text": decoded_text,
        },
        "stage4_scope": {
            "plain_huggingface": True,
            "obfuscated_gpt2": False,
            "modelscope": False,
            "real_tee": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "logits_shape": result["forward"]["logits_shape"],
                "generated_length": len(result["generation"]["generated_token_ids"][0]),
                "decoded_text": decoded_text,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
