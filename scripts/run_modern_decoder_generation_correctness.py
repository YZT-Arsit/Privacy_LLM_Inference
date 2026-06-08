"""Runner for the Stage 7.6e padded modern-decoder generation correctness
experiment.

Writes ``outputs/modern_decoder_generation_correctness.json`` and the
matching ``.md`` summary. CPU-only, ``float64``, ``use_pad=True`` by
default. ``use_pad=False`` is exercised as an ablation row.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.modern_decoder_generation_correctness import (  # noqa: E402
    GenerationCorrectnessConfig,
    run_modern_decoder_generation_correctness,
    write_reports,
)


def main() -> None:
    cfg = GenerationCorrectnessConfig()
    report = run_modern_decoder_generation_correctness(
        cfg=cfg, include_no_pad_ablation=True
    )
    json_path, md_path = write_reports(report, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print()
    print(f"status={report['status']}")
    print(f"main_mode={report['main_mode']}")
    print(
        "use_pad={use_pad} fresh_pad={fresh_pad} fresh_mask={fresh_mask}".format(
            **{k: report[k] for k in ("use_pad", "fresh_pad", "fresh_mask")}
        )
    )
    corr = report["correctness"]
    print(
        "greedy_token_match_rate={r} sequence_exact_match={m} "
        "prefill_max={p} decode_max={d} kv_inv_max={k}".format(
            r=corr["greedy_token_match_rate"],
            m=corr["sequence_exact_match"],
            p=corr["prefill_logits_max_abs_error"],
            d=corr["decode_step_logits_max_abs_error_max"],
            k=corr["kv_cache_invariant_max_abs_error"],
        )
    )
    sec = report["security_relevant_checks"]
    print(
        "same_input_two_runs_same_output={a} "
        "different_masked_fingerprints={b}".format(
            a=sec["same_input_two_runs_same_output"],
            b=sec["same_input_two_runs_different_masked_fingerprints"],
        )
    )


if __name__ == "__main__":
    main()
