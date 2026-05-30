#!/usr/bin/env python
"""Stage 5.3a smoke — GPT-2 single-block compatible-nonlinear-island integration.

Runs one GPT-2 transformer block through ``ObfuscatedGPT2BlockWrapper`` with
``nonlinear_mode="compatible_islands"`` (both ``use_pad=False`` and
``use_pad=True``) and writes a JSON / Markdown smoke report. The report
records max-abs / relative-L2 / cosine error vs. the plain block, plus the
wrapper's audit metadata (``mlp_island_permutation_dim``,
``mlp_island_pad_placement``, ``online_extra_matmul_count``) and the
Stage 5.2b security caveats.

This is a *single-block* integration smoke. The LM head, GPT-2 model-level
wrapper, KV cache, generation path, BERT, and T5 wrappers are NOT modified
in this stage.
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
from pllo.hf_wrappers import ObfuscatedGPT2BlockWrapper
from pllo.model_zoo import ExternalModelConfig, get_model_loader, torch_dtype_from_string


REPORT_VERSION = "stage-5.3a-v1"


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
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64"]
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    return parser.parse_args()


def _first_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = a_flat.norm() * b_flat.norm()
    if float(denom) <= 0.0:
        return 0.0
    return float((a_flat @ b_flat / denom).item())


def _run_single(
    block,
    model_config,
    hidden_states: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_pad: bool,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    with torch.no_grad():
        plain = _first_hidden(block(hidden_states, attention_mask=None, use_cache=False))
    wrapper = ObfuscatedGPT2BlockWrapper(
        block,
        model_config,
        dtype=dtype,
        device=device,
        use_pad=use_pad,
        nonlinear_mode="compatible_islands",
    )
    with torch.no_grad():
        recovered = wrapper.forward(hidden_states)

    metrics = compute_correctness_metrics(plain, recovered, atol=1e-4, rtol=1e-4)
    cos = _cosine_similarity(plain, recovered)
    rep = dict(wrapper.island_report)
    pad = dict(wrapper.pad_report)

    return {
        "use_pad": bool(use_pad),
        "nonlinear_mode": "compatible_islands",
        "max_abs_error": float(metrics["max_abs_error"]),
        "relative_l2_error": float(metrics["relative_l2_error"]),
        "cosine_similarity": cos,
        "allclose": bool(metrics["allclose"]),
        "mlp_gelu_island_active": bool(rep["mlp_gelu_island_active"]),
        "permutation_dim": rep["mlp_island_permutation_dim"],
        "intermediate_size": rep["mlp_island_intermediate_size"],
        "pad_placement": rep["mlp_island_pad_placement"],
        "uses_fresh_permutation": bool(rep["mlp_island_uses_fresh_permutation"]),
        "online_extra_matmul_count": int(rep["online_extra_matmul_count"]),
        "layernorm_remains_trusted": bool(rep["layernorm_remains_trusted"]),
        "lm_head_not_modified": bool(rep["lm_head_not_modified"]),
        "generation_path_not_modified": bool(rep["generation_path_not_modified"]),
        "security_profile": rep["security_profile"],
        "security_caveats": list(rep["security_caveats"]),
        "mlp_c_fc_pad": bool(pad.get("mlp_c_fc_pad", False)),
        "mlp_c_proj_pad": bool(pad.get("mlp_c_proj_pad", False)),
    }


def _render_md(payload: dict) -> str:
    cfg = payload["config"]
    runs = payload["runs"]
    lines: list[str] = []
    lines.append("# GPT-2 Compatible Nonlinear Island — Stage 5.3a Smoke")
    lines.append("")
    lines.append(f"- model_id: `{cfg['model_id']}`")
    lines.append(f"- block_index: {cfg['block_index']}")
    lines.append(f"- batch_size: {cfg['batch_size']}")
    lines.append(f"- seq_len: {cfg['seq_len']}")
    lines.append(f"- dtype: {cfg['dtype']}")
    lines.append(f"- seed: {cfg['seed']}")
    lines.append(f"- report_version: {payload['report_version']}")
    lines.append("")
    lines.append("## Per-configuration results")
    lines.append("")
    lines.append(
        "| use_pad | allclose | max_abs_error | relative_l2_error | "
        "cosine_similarity | permutation_dim | pad_placement | "
        "online_extra_matmul_count |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in runs:
        lines.append(
            f"| {r['use_pad']} | {r['allclose']} | "
            f"{r['max_abs_error']:.3e} | {r['relative_l2_error']:.3e} | "
            f"{r['cosine_similarity']:.6f} | {r['permutation_dim']} | "
            f"`{r['pad_placement']}` | {r['online_extra_matmul_count']} |"
        )
    lines.append("")
    lines.append("## Wrapper integration scope")
    lines.append("")
    lines.append("- This is single-block wrapper integration.")
    lines.append("- LayerNorm remains trusted.")
    lines.append("- LM head is not modified.")
    lines.append("- Generation path is not modified.")
    lines.append("- GPT-2 model-level wrapper is not modified.")
    lines.append("- BERT / T5 wrappers are not modified.")
    lines.append(
        "- `compatible_islands` is not enabled by default; default mode "
        "remains `trusted`."
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
    lines.append("- This is not a real TEE measurement.")
    lines.append(
        "- This stage does not claim formal security; the `compatible_islands` "
        "mode is `proxy-evaluated, not formal`."
    )
    lines.append("")
    lines.append("## Next stage")
    lines.append("")
    lines.append(
        "- Stage 5.3b — GPT-2 model-level wrapper integration of the same "
        "feature flag, followed by BERT / T5 wrappers."
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
    block = model.transformer.h[args.block_index]
    hidden_size = int(model.config.n_embd)

    torch.manual_seed(args.seed)
    hidden_states = torch.randn(
        args.batch_size, args.seq_len, hidden_size, dtype=dtype, device=device
    )

    runs: list[dict] = []
    for idx, use_pad in enumerate((False, True)):
        runs.append(
            _run_single(
                block,
                model.config,
                hidden_states,
                dtype=dtype,
                device=device,
                use_pad=use_pad,
                seed=args.seed + idx,
            )
        )

    payload = {
        "report_version": REPORT_VERSION,
        "config": {
            "model_id": args.model_id,
            "block_index": args.block_index,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "runs": runs,
        "wrapper_integration_status": {
            "gpt2_single_block": "implemented",
            "gpt2_model_level": "not_yet",
            "bert": "not_yet",
            "t5": "not_yet",
        },
        "caveats": [
            "Compatible mask families are weaker than unrestricted dense masks.",
            "Fresh permutation, dense sandwiching, and pad at Linear boundaries are required mitigations.",
            "This is not a real TEE measurement.",
            "Compatible nonlinear island is not yet enabled by default.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gpt2_compatible_island_smoke.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (args.output_dir / "gpt2_compatible_island_smoke.md").write_text(
        _render_md(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
