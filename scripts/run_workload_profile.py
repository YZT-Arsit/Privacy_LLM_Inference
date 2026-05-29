#!/usr/bin/env python
"""Stage 5.0.1 workload profile — calibrated TEE/GPU cost-model comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments import (
    DEFAULT_COST_MODEL,
    INTERACTION_CATEGORIES,
    MODULE_CATEGORIES,
    WORKLOAD_METHODS,
    WorkloadProfileConfig,
    run_workload_profile,
)
from pllo.experiments.report_utils import (
    fmt,
    markdown_table,
    write_csv,
    write_json,
    write_text,
)


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
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64", "float16"]
    )
    parser.add_argument("--use-pad", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


CSV_FIELDS = (
    "method",
    "title",
    "implemented",
    "wall_time_source",
    "measured_wall_time_ms",
    "projected_wall_time_ms",
    "online_boundary_calls",
    "online_trusted_compute_ops",
    "online_trusted_transfer_bytes",
    "online_gpu_ops",
    "preprocessing_trusted_ops",
    "preprocessing_transfer_bytes",
    "boundary_calls_formula",
)


def _method_rows(profile: dict) -> list[dict]:
    rows = []
    for name, payload in profile["methods"].items():
        rows.append(
            {
                "method": name,
                "title": payload["title"],
                "implemented": payload["implemented"],
                "wall_time_source": payload["wall_time_source"],
                "measured_wall_time_ms": payload["measured_wall_time_ms"],
                "projected_wall_time_ms": payload["projected_wall_time_ms"],
                "online_boundary_calls": payload["online_boundary_calls"],
                "online_trusted_compute_ops": payload["online_trusted_compute_ops"],
                "online_trusted_transfer_bytes": payload["online_trusted_transfer_bytes"],
                "online_gpu_ops": payload["online_gpu_ops"],
                "preprocessing_trusted_ops": payload["preprocessing_trusted_ops"],
                "preprocessing_transfer_bytes": payload["preprocessing_transfer_bytes"],
                "boundary_calls_formula": payload["boundary_calls_formula"],
            }
        )
    return rows


def _build_markdown(profile: dict) -> str:
    out: list[str] = []
    cfg = profile["config"]
    out.append("# Privacy LLM Obfuscation — Calibrated Workload Profile (Stage 5.0.1)")
    out.append("")
    out.append(
        "Cost model splits every method into four explicit slices: **preprocessing"
        " trusted cost** (amortised), **online boundary crossings** (true"
        " trusted↔untrusted round trips), **online trusted compute**"
        " (LayerNorm / GELU / sampling / recovery FLOPs in the TEE), and"
        " **online GPU obfuscated compute** (linear matmuls, attention,"
        " LM head). Internal Python bookkeeping such as mask-state creation"
        " is **not** counted as a boundary call."
    )
    out.append("")
    out.append(
        f"`model_id={cfg['model_id']}`, `batch_size={cfg['batch_size']}`,"
        f" `prompt_len={cfg['prompt_len']}`, `max_new_tokens={cfg['max_new_tokens']}`,"
        f" `device={cfg['device']}`, `dtype={cfg['dtype']}`, `use_pad={cfg['use_pad']}`,"
        f" `warmup={cfg['warmup']}`, `repeat={cfg['repeat']}`."
    )
    out.append("")
    cal = profile["calibration"]
    out.append(
        f"GPU-FLOPs/ms calibration constant: `{cal['gpu_flops_per_ms']:.3e}`"
        f" (derived from measured `{cal['calibrated_from']}` wall time)."
    )
    out.append("")
    out.append(f"> **Warning:** {profile['interpretation']['cost_model_warning']}.")
    out.append("")

    out.append("## Method comparison")
    headers = [
        "method",
        "impl?",
        "wall_time_ms (measured/proj.)",
        "boundary calls",
        "boundary formula",
        "trusted compute (ops)",
        "trusted transfer (bytes)",
        "gpu (ops)",
    ]
    rows = []
    for method in WORKLOAD_METHODS:
        m = profile["methods"][method.name]
        wt = (
            f"{m['measured_wall_time_ms']:.3f}"
            if m["wall_time_source"] == "measured"
            else f"{m['projected_wall_time_ms']:.3f} (proj.)"
        )
        rows.append(
            [
                method.name,
                m["implemented"],
                wt,
                m["online_boundary_calls"],
                m["boundary_calls_formula"],
                m["online_trusted_compute_ops"],
                m["online_trusted_transfer_bytes"],
                m["online_gpu_ops"],
            ]
        )
    out.append(markdown_table(headers, rows))
    out.append("")

    out.append("## Preprocessing (amortised; excluded from online latency)")
    rows = [
        [
            method.name,
            profile["methods"][method.name]["preprocessing_trusted_ops"],
            profile["methods"][method.name]["preprocessing_transfer_bytes"],
        ]
        for method in WORKLOAD_METHODS
    ]
    out.append(
        markdown_table(["method", "preprocessing_trusted_ops", "preprocessing_transfer_bytes"], rows)
    )
    out.append("")

    out.append("## Interaction breakdown (online slice by interaction type)")
    headers = ["interaction"] + [m.name for m in WORKLOAD_METHODS]
    for column_name, key in (
        ("boundary calls", "online_boundary_calls"),
        ("transfer bytes", "online_trusted_transfer_bytes"),
        ("trusted compute (ops)", "online_trusted_compute_ops"),
    ):
        out.append(f"### {column_name} per interaction")
        rows = []
        for category in INTERACTION_CATEGORIES:
            row = [category]
            for method in WORKLOAD_METHODS:
                row.append(profile["interaction_breakdown"][category][method.name][key])
            rows.append(row)
        out.append(markdown_table(headers, rows))
        out.append("")

    out.append("## Module breakdown (online slice by module category)")
    out.append("")
    headers = ["module"] + [m.name for m in WORKLOAD_METHODS]
    for column_name, key in (
        ("boundary calls", "online_boundary_calls"),
        ("trusted compute (ops)", "online_trusted_compute_ops"),
        ("trusted transfer (bytes)", "online_trusted_transfer_bytes"),
        ("gpu ops", "online_gpu_ops"),
    ):
        out.append(f"### {column_name} per module")
        rows = []
        for category in MODULE_CATEGORIES:
            row = [category]
            for method in WORKLOAD_METHODS:
                row.append(profile["module_breakdown"][category][method.name][key])
            rows.append(row)
        out.append(markdown_table(headers, rows))
        out.append("")

    pm = profile["paper_metrics"]
    out.append("## Paper metrics")
    out.append("")
    out.append(
        f"- `boundary_call_reduction_vs_tslp` = {pm['boundary_call_reduction_vs_tslp']:.2%}"
        f" (ours_current vs tslp)"
    )
    out.append(
        f"- `trusted_transfer_reduction_vs_tslp` = {pm['trusted_transfer_reduction_vs_tslp']:.2%}"
    )
    out.append(
        f"- `online_trusted_compute_reduction_vs_tslp` = {pm['online_trusted_compute_reduction_vs_tslp']:.2%}"
    )
    out.append(f"- `gpu_offload_ratio` (ours_current) = {pm['gpu_offload_ratio']:.2%}")
    out.append(f"- `preprocessing_amortized` = `{pm['preprocessing_amortized']}`")
    out.append("- `boundary_calls_per_forward` =")
    for name, value in pm["boundary_calls_per_forward"].items():
        out.append(f"  - `{name}`: {value}")
    out.append("")

    interp = profile["interpretation"]
    out.append("## Interpretation")
    out.append("")
    out.append(f"- **Main online bottleneck (ours_current):** `{interp['main_online_bottleneck']}`")
    out.append(
        f"- **Next primitive to obfuscate on GPU:** `{interp['next_primitive_to_replace']}`"
    )
    out.append("")
    out.append(
        "*Note on ours_current vs TSLP boundary calls:* ours_current crosses"
        " the boundary once per obfuscated linear (4 per layer) while TSLP"
        " crosses once per non-linear (3 per layer plus ln_f). This is an"
        " **architectural** difference, not a bookkeeping artefact. Each"
        " ours_current crossing moves a smaller activation than a TSLP"
        " LayerNorm crossing, and the measured wall time is consistent with"
        " that tradeoff."
    )
    out.append("")

    out.append("## Method semantics & citation caveats")
    for method in WORKLOAD_METHODS:
        out.append(f"### `{method.name}` — {method.title}")
        out.append("")
        out.append(method.summary)
        out.append("")
        out.append(f"- Implemented: **{method.implemented}**")
        out.append(f"- Implementation note: {method.implementation_note}")
        out.append(f"- Caveat: {method.citation_caveat}")
        out.append("")

    out.append("## Limitations")
    for line in profile["limitations"]:
        out.append(f"- {line}")
    out.append("")

    out.append("## Reproducibility")
    out.append("")
    out.append("```bash")
    out.append(
        f"python scripts/run_workload_profile.py --batch-size {cfg['batch_size']}"
        f" --prompt-len {cfg['prompt_len']} --max-new-tokens {cfg['max_new_tokens']}"
        f" --warmup {cfg['warmup']} --repeat {cfg['repeat']} --use-pad {cfg['use_pad']}"
    )
    out.append("```")
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = WorkloadProfileConfig(
        model_id=args.model_id,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        use_pad=args.use_pad,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        repeat=args.repeat,
        cost_model=DEFAULT_COST_MODEL,
    )
    profile = run_workload_profile(cfg)

    out_dir: Path = args.output_dir
    write_json(out_dir / "workload_profile.json", profile)
    write_csv(out_dir / "workload_profile.csv", _method_rows(profile), CSV_FIELDS)
    write_text(out_dir / "workload_profile.md", _build_markdown(profile))

    headline = {
        name: {
            "boundary": m["online_boundary_calls"],
            "trusted_compute_ops": m["online_trusted_compute_ops"],
            "trusted_transfer_bytes": m["online_trusted_transfer_bytes"],
            "gpu_ops": m["online_gpu_ops"],
            "wall_time_ms": m["measured_wall_time_ms"]
            if m["wall_time_source"] == "measured"
            else m["projected_wall_time_ms"],
            "wall_time_source": m["wall_time_source"],
        }
        for name, m in profile["methods"].items()
    }
    print(
        f"main_online_bottleneck={profile['interpretation']['main_online_bottleneck']}, "
        f"next_primitive={profile['interpretation']['next_primitive_to_replace']}, "
        f"gpu_offload_ratio={profile['paper_metrics']['gpu_offload_ratio']:.3f}, "
        f"output_dir={out_dir}\n"
        f"headline={headline}"
    )


if __name__ == "__main__":
    main()
