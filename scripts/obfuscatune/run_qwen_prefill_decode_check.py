"""ObfuscaTune-Qwen prefill + decode correctness with KV cache.

Prefills a prompt, decodes one more token from the cache, and compares both
against the plaintext model (full recompute). Emits cache metrics: prefill /
decode cache length, shape match vs HF, and the cache obfuscation label.

Example:
    python scripts/obfuscatune/run_qwen_prefill_decode_check.py --tiny-random-qwen
"""

from __future__ import annotations

import argparse

from _qwen_common import (
    add_qwen_model_args, build_qwen, out_dir, qwen_input_ids, qwen_planned_config,
    torch_dtype, write_json,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_qwen_model_args(p)
    p.add_argument("--mode", choices=["orthogonal", "random", "cond"], default="orthogonal")
    p.add_argument("--condition-number", type=float, default=32.0)
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(qwen_planned_config(args, {"mode": args.mode, "stages": ["prefill", "decode"]}))
        return

    import torch
    from pllo.baselines.obfuscatune.qwen_obfuscatune import run_prefill, run_decode_step
    from pllo.baselines.obfuscatune.qwen_cache import cache_metrics, hf_cache_shapes
    from pllo.baselines.obfuscatune.qwen_metrics import method_id

    model, arch, src = build_qwen(args)
    ids = qwen_input_ids(arch.vocab_size, args.seq_len, args.seed, args.batch_size)
    dt = torch_dtype(args.dtype)
    g = torch.Generator().manual_seed(args.seed + 99)
    next_id = torch.randint(0, arch.vocab_size, (args.batch_size, 1), generator=g)

    pre = run_prefill(model, ids, args.mode, dtype=dt, seed=args.seed,
                      condition_number=args.condition_number)
    dec = run_decode_step(model, ids, next_id, args.mode, dtype=dt, seed=args.seed,
                          condition_number=args.condition_number)

    ref_out = model(ids, use_cache=True)
    ref_shapes = hf_cache_shapes(ref_out.past_key_values)
    cm = cache_metrics(
        enabled=True, prefill_cache_len=dec["prefill_cache_len"],
        decode_cache_len=dec["decode_cache_len"],
        obf_shapes=pre["cache"].shapes(), ref_shapes=ref_shapes)

    rmt = {"orthogonal": "orthogonal", "random": "random", "cond": "cond"}[args.mode]
    method = method_id(rmt, args.condition_number)
    result = {
        "method": method, "model_name_or_path": src, "model_family": "qwen",
        "prefill": {"max_abs_error": pre["correctness"]["max_abs_error"],
                    "logits_argmax_match_rate": pre["argmax_match_rate"],
                    "nan_or_inf_count": pre["nan_or_inf_count"]},
        "decode": {"max_abs_error": dec["correctness"]["max_abs_error"],
                   "logits_argmax_match_rate": dec["argmax_match_rate"],
                   "nan_or_inf_count": dec["nan_or_inf_count"]},
        "cache": cm,
    }
    print(f"prefill max_abs={pre['correctness']['max_abs_error']:.3e} argmax={pre['argmax_match_rate']:.3f}")
    print(f"decode  max_abs={dec['correctness']['max_abs_error']:.3e} argmax={dec['argmax_match_rate']:.3f} "
          f"prefill_len={dec['prefill_cache_len']} decode_len={dec['decode_cache_len']} "
          f"shape_match={cm['cache_shape_match']} obf={cm['cache_obfuscation']}")
    out = out_dir(args, "qwen_prefill_decode") / f"qwen_prefill_decode_{args.mode}_{args.dtype}.json"
    write_json(out, result)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
