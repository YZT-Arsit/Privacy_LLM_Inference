"""Capture Qwen2 intermediate representations to a .pt for offline attacks.

Uses a forward hook on a tiny random Qwen2 (default; no download) or a local
checkpoint. Writes a torch dict consumable by the attack runners. Does NOT run
7B by default.

Example:
    python scripts/attacks/capture_qwen_representations.py --num-tokens 12
    python scripts/attacks/capture_qwen_representations.py --model-name-or-path /path/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--num-tokens", type=int, default=12)
    p.add_argument("--layers", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    if args.dry_run:
        print(f"[dry-run] capture model={args.model_name_or_path or 'tiny_random_qwen'} "
              f"tokens={args.num_tokens} layers={layers}")
        return

    import torch
    from pllo.attacks.qwen_hooks import QwenActivationCapture
    from pllo.baselines.obfuscatune.qwen_config import load_qwen, make_tiny_qwen_config

    if args.model_name_or_path:
        model = load_qwen(model_name_or_path=args.model_name_or_path, seed=args.seed, dtype=torch.float32)
    else:
        model = load_qwen(tiny_random_qwen=make_tiny_qwen_config(), seed=args.seed, dtype=torch.float32)
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(0, model.config.vocab_size, (1, args.num_tokens), generator=g)
    cap = QwenActivationCapture(model, layers=layers)
    out = cap.run(ids)
    cap.remove()

    save = {"token_ids": ids[0], "embedding_table": out["embedding_table"],
            "plaintext_embeddings": out["embed_tokens"][0],
            "observed_intermediate": out.get(f"layer{layers[0]}_input", out["embed_tokens"])[0],
            "logits": out["logits"][0]}
    path = Path(args.output) if args.output else REPO_ROOT / "outputs" / "attacks" / "captures" / "qwen_capture.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save, path)
    print(f"wrote {path} (keys: {list(save.keys())})")


if __name__ == "__main__":
    main()
