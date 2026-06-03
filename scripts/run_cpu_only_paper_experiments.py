"""Stage 7.5b - master runner for all CPU-only paper experiments.

Runs in order: paper_toy_tasks -> paper_baseline_comparison ->
paper_ablation_study -> paper_stability_study -> cpu_runtime_completion.
Each step writes its own ``outputs/<name>.json``/``.csv``/``.md`` files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pllo.experiments.cpu_runtime_completion import (  # noqa: E402
    CPURuntimeCompletionConfig,
    run_cpu_runtime_completion,
)
from pllo.experiments.paper_ablation_study import (  # noqa: E402
    PaperAblationStudyConfig,
    run_paper_ablation_study,
)
from pllo.experiments.paper_baseline_comparison import (  # noqa: E402
    PaperBaselineComparisonConfig,
    run_paper_baseline_comparison,
)
from pllo.experiments.paper_stability_study import (  # noqa: E402
    PaperStabilityStudyConfig,
    run_paper_stability_study,
)
from pllo.experiments.paper_toy_tasks import (  # noqa: E402
    PaperToyTaskConfig,
    run_paper_toy_tasks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--quick", action="store_true",
                   help="Run a smaller / faster sweep (handy for smoke testing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = str(args.output_dir)
    seed = int(args.seed)
    if args.quick:
        cfgs = (
            ("paper_toy_tasks", PaperToyTaskConfig(
                output_dir=out, seed=seed, num_samples=32,
                num_train_steps=5, batch_size=4,
            ), run_paper_toy_tasks),
            ("paper_baseline_comparison", PaperBaselineComparisonConfig(
                output_dir=out, seed=seed, num_repeats=2,
            ), run_paper_baseline_comparison),
            ("paper_ablation_study", PaperAblationStudyConfig(
                output_dir=out, seed=seed, num_trials=4,
            ), run_paper_ablation_study),
            ("paper_stability_study", PaperStabilityStudyConfig(
                output_dir=out,
                seeds=(2021, 2022),
                batch_sizes=(1, 2),
                seq_lens=(4,),
                hidden_sizes=(16,),
                true_ranks=(2,),
                padded_ranks=(8,),
            ), run_paper_stability_study),
            ("cpu_runtime_completion", CPURuntimeCompletionConfig(
                output_dir=out, seed=seed,
                num_warmup=1, num_repeats=3,
                batch_sizes=(1,), seq_lens=(4,), hidden_sizes=(16,),
            ), run_cpu_runtime_completion),
        )
    else:
        cfgs = (
            ("paper_toy_tasks", PaperToyTaskConfig(output_dir=out, seed=seed),
             run_paper_toy_tasks),
            ("paper_baseline_comparison",
             PaperBaselineComparisonConfig(output_dir=out, seed=seed),
             run_paper_baseline_comparison),
            ("paper_ablation_study",
             PaperAblationStudyConfig(output_dir=out, seed=seed),
             run_paper_ablation_study),
            ("paper_stability_study",
             PaperStabilityStudyConfig(output_dir=out),
             run_paper_stability_study),
            ("cpu_runtime_completion",
             CPURuntimeCompletionConfig(output_dir=out, seed=seed),
             run_cpu_runtime_completion),
        )

    for name, cfg, runner in cfgs:
        t0 = time.perf_counter()
        runner(cfg)
        elapsed = time.perf_counter() - t0
        print(f"[{name}] wrote {args.output_dir}/{name}.json ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
