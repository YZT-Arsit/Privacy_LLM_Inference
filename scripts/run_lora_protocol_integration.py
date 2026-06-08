"""Runner for Stage 7.7b LoRA protocol integration."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.lora_protocol_integration import (  # noqa: E402
    LoRAProtocolConfig,
    run_lora_protocol_integration,
    write_reports,
)


def main() -> None:
    cfg = LoRAProtocolConfig()
    rep = run_lora_protocol_integration(cfg=cfg)
    j, m = write_reports(rep, outputs_dir=REPO_ROOT / "outputs")
    print(f"Wrote: {j}")
    print(f"Wrote: {m}")
    print(f"status={rep['status']} sites={rep['supported_lora_sites']}")
    for c in rep["merged_weights_generation_combos"]:
        print(
            f"  [{c['norm_mask_granularity']:8s} ck={c['norm_chunk_size']} "
            f"{c['attention_privacy_mode']:30s}] greedy={c['greedy_token_match_rate']} "
            f"lm_max={c['lm_head_recovery_max_abs_error']:.2e}"
        )


if __name__ == "__main__":
    main()
