"""ObfuscaTune-Qwen greedy-generation smoke vs unprotected.

Greedy-decodes ``max_new_tokens`` (default 8) under the unprotected model and an
obfuscation mode, then reports token_match_rate / exact_token_match. Fixed seed,
no sampling. Tiny random Qwen by default -- never downloads.

Example:
    python scripts/obfuscatune/run_qwen_generation_smoke.py --tiny-random-qwen --max-new-tokens 8
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
    p.add_argument("--max-new-tokens", type=int, default=8)
    args = p.parse_args()

    if args.dry_run:
        print("[dry-run] planned config:")
        print(qwen_planned_config(args, {"mode": args.mode,
                                         "max_new_tokens": args.max_new_tokens,
                                         "decoding": "greedy"}))
        return

    from pllo.baselines.obfuscatune.qwen_obfuscatune import generate_greedy
    from pllo.baselines.obfuscatune.qwen_metrics import method_id

    model, arch, src = build_qwen(args)
    ids = qwen_input_ids(arch.vocab_size, args.seq_len, args.seed, args.batch_size)
    dt = torch_dtype(args.dtype)

    ref = generate_greedy(model, ids, args.max_new_tokens, "unprotected", dtype=dt)
    obf = generate_greedy(model, ids, args.max_new_tokens, args.mode, dtype=dt,
                          seed=args.seed, condition_number=args.condition_number, use_cache=True)
    ref_t = ref["tokens"][0].tolist()
    obf_t = obf["tokens"][0].tolist()
    match = sum(a == b for a, b in zip(ref_t, obf_t))
    token_match_rate = match / max(1, len(ref_t))
    exact = ref_t == obf_t

    rmt = {"orthogonal": "orthogonal", "random": "random", "cond": "cond"}[args.mode]
    result = {
        "method": method_id(rmt, args.condition_number), "model_name_or_path": src,
        "model_family": "qwen", "mode": "generation", "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy", "dtype": args.dtype,
        "unprotected_tokens": ref_t, "obfuscated_tokens": obf_t,
        "correctness": {"token_match_rate": token_match_rate, "exact_token_match": exact},
    }
    print(f"unprotected: {ref_t}")
    print(f"{args.mode:11s}: {obf_t}")
    print(f"token_match_rate={token_match_rate:.3f} exact_token_match={exact}")
    out = out_dir(args, "qwen_generation") / f"qwen_generation_{args.mode}_{args.dtype}.json"
    write_json(out, result)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
