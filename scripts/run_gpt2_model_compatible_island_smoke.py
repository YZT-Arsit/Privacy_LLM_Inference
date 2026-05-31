#!/usr/bin/env python
"""Stage 5.3b smoke — GPT-2 model-level compatible nonlinear-island integration.

Runs the full ``ObfuscatedGPT2ModelWrapper`` with
``nonlinear_mode="compatible_islands"`` (both ``use_pad=False`` and
``use_pad=True``) and writes a JSON / Markdown smoke report covering:

* Full-model forward correctness vs. a plain HF reference forward.
* Greedy generation correctness vs. a hand-written plain HF greedy loop
  (``model.generate()`` is **not** used).
* The aggregate ``island_summary`` exposed by the model wrapper.

This is a *GPT-2 model-level* smoke. BERT and T5 wrappers are not
modified. The LM head, KV cache, and generation control flow are not
modified. ``compatible_islands`` is not the default mode.
"""

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

from pllo.evaluation import compute_correctness_metrics
from pllo.evaluation.correctness import (
    sequence_exact_match,
    token_match_rate,
    top1_match_rate,
)
from pllo.hf_wrappers import ObfuscatedGPT2ModelWrapper
from pllo.model_zoo import ExternalModelConfig, get_model_loader, torch_dtype_from_string


REPORT_VERSION = "stage-5.3b-v1"


def parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    return parser.parse_args()


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = a_flat.norm() * b_flat.norm()
    if float(denom) <= 0.0:
        return 0.0
    return float((a_flat @ b_flat / denom).item())


def _plain_greedy(model, input_ids: torch.Tensor, max_new_tokens: int):
    """Hand-written plain HF greedy loop — does NOT use model.generate()."""
    step_logits: list[torch.Tensor] = []
    with torch.no_grad():
        prefill = model(input_ids, use_cache=True)
        last_logits = prefill.logits[:, -1:, :]
        step_logits.append(last_logits)
        next_token = last_logits[:, -1, :].argmax(dim=-1)
        tokens = [next_token]
        past = prefill.past_key_values
        for _ in range(max_new_tokens - 1):
            step = model(
                next_token.unsqueeze(-1), past_key_values=past, use_cache=True
            )
            past = step.past_key_values
            step_logits.append(step.logits)
            next_token = step.logits[:, -1, :].argmax(dim=-1)
            tokens.append(next_token)
    generated = torch.cat([input_ids, torch.stack(tokens, dim=1)], dim=1)
    return generated, step_logits


def _run_single(
    model,
    input_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_pad: bool,
    max_new_tokens: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    plain_logits = model(input_ids).logits
    plain_generated, plain_step_logits = _plain_greedy(
        model, input_ids, max_new_tokens
    )

    torch.manual_seed(seed)
    wrapper = ObfuscatedGPT2ModelWrapper(
        model,
        dtype=dtype,
        device=device,
        use_pad=use_pad,
        nonlinear_mode="compatible_islands",
    )

    with torch.no_grad():
        recovered_logits = wrapper.forward(input_ids)
    forward_metrics = compute_correctness_metrics(
        plain_logits, recovered_logits, atol=1e-4, rtol=1e-4
    )
    forward_top1 = top1_match_rate(plain_logits, recovered_logits)

    with torch.no_grad():
        obf_generated, trace = wrapper.generate_greedy(input_ids, max_new_tokens)
    prompt_len = int(input_ids.shape[1])
    new_slice = slice(prompt_len, prompt_len + max_new_tokens)
    tokens_match = token_match_rate(
        plain_generated[:, new_slice], obf_generated[:, new_slice]
    )
    seq_match = sequence_exact_match(
        plain_generated[:, new_slice], obf_generated[:, new_slice]
    )
    full_top1 = top1_match_rate(plain_generated, obf_generated)

    obf_step_logits = trace.get("step_logits", [])
    max_logits_error = None
    if len(plain_step_logits) == len(obf_step_logits) and obf_step_logits:
        per_step = []
        for plain_s, obf_s in zip(plain_step_logits, obf_step_logits):
            if plain_s.shape == obf_s.shape:
                per_step.append(float((plain_s - obf_s).abs().max().item()))
        if per_step:
            max_logits_error = max(per_step)

    summary = dict(wrapper.island_summary)
    pad_reports = [dict(r) for r in wrapper.pad_reports]

    return {
        "use_pad": bool(use_pad),
        "nonlinear_mode": "compatible_islands",
        "full_forward": {
            "max_abs_error": float(forward_metrics["max_abs_error"]),
            "relative_l2_error": float(forward_metrics["relative_l2_error"]),
            "cosine_similarity": _cosine_similarity(plain_logits, recovered_logits),
            "allclose": bool(forward_metrics["allclose"]),
            "top1_match_rate": float(forward_top1),
        },
        "generation": {
            "max_new_tokens": int(max_new_tokens),
            "sequence_exact_match": float(seq_match),
            "token_match_rate": float(tokens_match),
            "top1_match_rate": float(full_top1),
            "max_logits_error": max_logits_error,
            "generated_token_ids": obf_generated[:, new_slice].tolist(),
            "plain_token_ids": plain_generated[:, new_slice].tolist(),
        },
        "island_summary": summary,
        "pad_report_mlp_c_fc_pad_per_block": [
            bool(r.get("mlp_c_fc_pad", False)) for r in pad_reports
        ],
        "pad_report_mlp_c_proj_pad_per_block": [
            bool(r.get("mlp_c_proj_pad", False)) for r in pad_reports
        ],
    }


def _render_md(payload: dict) -> str:
    cfg = payload["config"]
    runs = payload["runs"]
    lines: list[str] = []
    lines.append("# GPT-2 Model-Level Compatible Nonlinear Island — Stage 5.3b Smoke")
    lines.append("")
    lines.append(f"- model_id: `{cfg['model_id']}`")
    lines.append(f"- batch_size: {cfg['batch_size']}")
    lines.append(f"- seq_len: {cfg['seq_len']}")
    lines.append(f"- max_new_tokens: {cfg['max_new_tokens']}")
    lines.append(f"- dtype: {cfg['dtype']}")
    lines.append(f"- seed: {cfg['seed']}")
    lines.append(f"- report_version: {payload['report_version']}")
    lines.append("")
    lines.append("## Full-model forward correctness")
    lines.append("")
    lines.append(
        "| use_pad | allclose | max_abs_error | relative_l2_error | "
        "cosine_similarity | top1_match_rate |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        f = r["full_forward"]
        lines.append(
            f"| {r['use_pad']} | {f['allclose']} | "
            f"{f['max_abs_error']:.3e} | {f['relative_l2_error']:.3e} | "
            f"{f['cosine_similarity']:.6f} | {f['top1_match_rate']:.4f} |"
        )
    lines.append("")
    lines.append("## Greedy generation correctness")
    lines.append("")
    lines.append(
        "| use_pad | max_new_tokens | sequence_exact_match | token_match_rate | "
        "top1_match_rate | max_logits_error |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        g = r["generation"]
        max_err = g["max_logits_error"]
        max_err_str = "n/a" if max_err is None else f"{max_err:.3e}"
        lines.append(
            f"| {r['use_pad']} | {g['max_new_tokens']} | "
            f"{g['sequence_exact_match']:.4f} | {g['token_match_rate']:.4f} | "
            f"{g['top1_match_rate']:.4f} | {max_err_str} |"
        )
    lines.append("")
    lines.append("## Island audit summary")
    lines.append("")
    lines.append(
        "| use_pad | num_blocks | blocks_with_compatible_islands | "
        "total_mlp_island_permutation_draws | online_extra_matmul_count | "
        "pad_placement | layernorm_remains_trusted |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in runs:
        s = r["island_summary"]
        lines.append(
            f"| {r['use_pad']} | {s['num_blocks']} | "
            f"{s['blocks_with_compatible_islands']} | "
            f"{s['total_mlp_island_permutation_draws']} | "
            f"{s['online_extra_matmul_count']} | "
            f"`{s['pad_placement']}` | {s['layernorm_remains_trusted']} |"
        )
    lines.append("")
    lines.append("## Wrapper integration scope")
    lines.append("")
    lines.append(
        "- This is GPT-2 model-level wrapper integration (Stage 5.3b)."
    )
    lines.append("- LayerNorm remains trusted.")
    lines.append("- LM head is not modified.")
    lines.append("- KV cache and greedy generation control flow are not modified.")
    lines.append("- BERT and T5 wrappers are not integrated.")
    lines.append(
        "- `compatible_islands` is not the default; default mode remains "
        "`trusted` for every wrapper."
    )
    lines.append(
        "- This is a measured GPT-2 model-level smoke, not a full "
        "cross-architecture measurement."
    )
    lines.append("")
    lines.append("## Security caveats (Stage 5.2b)")
    lines.append("")
    lines.append(
        "- Security relies on Stage 5.2b mitigations: fresh permutation per "
        "session, dense sandwich at Linear boundaries, and pad at Linear "
        "boundaries only."
    )
    lines.append(
        "- Compatible mask families are weaker than unrestricted dense masks "
        "inside nonlinear islands."
    )
    lines.append(
        "- Fresh permutation, dense sandwiching, and pad at Linear boundaries "
        "are required mitigations."
    )
    lines.append(
        "- Only GPT-2 model-level wrapper is integrated; BERT/T5 not integrated."
    )
    lines.append("- This is not a real TEE measurement.")
    lines.append(
        "- This stage does not claim formal security; `compatible_islands` "
        "remains `proxy-evaluated, not formal`."
    )
    lines.append("")
    lines.append("## Next stage")
    lines.append("")
    lines.append(
        "- Stage 5.3c — BERT and T5 wrapper / probe selective integration of "
        "the same `nonlinear_mode` feature flag."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_string(args.dtype, args.device)
    device = torch.device(args.device)
    loader_cfg = ExternalModelConfig(
        source="huggingface",
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
    )
    _, model = get_model_loader("huggingface").load(loader_cfg)
    model.eval()

    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, model.config.vocab_size, (args.batch_size, args.seq_len), device=device
    )

    runs: list[dict] = []
    for idx, use_pad in enumerate((False, True)):
        runs.append(
            _run_single(
                model,
                input_ids,
                dtype=dtype,
                device=device,
                use_pad=use_pad,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed + idx,
            )
        )

    payload = {
        "report_version": REPORT_VERSION,
        "config": {
            "model_id": args.model_id,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "runs": runs,
        "wrapper_integration_status": {
            "gpt2_single_block": "implemented",
            "gpt2_model_level": "implemented",
            "bert": "not_yet",
            "t5": "not_yet",
        },
        "measured_integration_scope": "gpt2_model_level",
        "caveats": [
            "LayerNorm remains trusted.",
            "Only GPT-2 model-level wrapper is integrated; BERT/T5 not integrated.",
            "Security relies on Stage 5.2b mitigations.",
            "Compatible mask families are weaker than unrestricted dense masks.",
            "Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.",
            "This is not a real TEE measurement.",
            "compatible_islands is not enabled by default.",
            "Measured GPT-2 model-level smoke, not full cross-architecture measurement.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gpt2_model_compatible_island_smoke.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "gpt2_model_compatible_island_smoke.md").write_text(
        _render_md(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
