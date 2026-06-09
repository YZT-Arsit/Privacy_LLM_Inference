"""Runner for Stage 7.6 -- Masked-Gradient LoRA Training."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.experiments.lora_training_inference_lifecycle import (  # noqa: E402
    build_lifecycle_report,
    write_reports as write_lifecycle_reports,
)
from pllo.experiments.masked_gradient_lora_security_proxy import (  # noqa: E402
    MaskedGradientLoRASecurityProxyConfig,
    run_masked_gradient_lora_security_proxy,
    write_reports as write_proxy_reports,
)
from pllo.experiments.masked_gradient_lora_training import (  # noqa: E402
    run_masked_gradient_lora_training,
    write_reports as write_training_reports,
)
from pllo.experiments.stage_7_6_claims_consistency import (  # noqa: E402
    build_claims_consistency_report,
    write_reports as write_claims_reports,
)
from pllo.ops.masked_gradient_lora import MaskedGradientLoRAConfig  # noqa: E402


def main() -> None:
    outputs_dir = REPO_ROOT / "outputs"
    cfg = MaskedGradientLoRAConfig(
        d_in=16, d_out=8, true_rank=2, padded_rank=6,
        batch_size=8, lr=0.01, momentum=0.9, use_momentum=False,
        use_rank_padding=True, dummy_strategy="paired_cancellation",
        seed=0, dtype="float64",
    )
    report = run_masked_gradient_lora_training(cfg, num_steps=6)
    j, c, m = write_training_reports(report, outputs_dir=str(outputs_dir))
    print(f"Wrote: {j}")
    print(f"Wrote: {c}")
    print(f"Wrote: {m}")

    proxy_cfg = MaskedGradientLoRASecurityProxyConfig(
        base=cfg, num_trials=3, num_steps=4,
        fresh_masks_per_step=True, fixed_masks_baseline=True,
    )
    proxy_report = run_masked_gradient_lora_security_proxy(proxy_cfg)
    pj, pc, pm = write_proxy_reports(proxy_report, outputs_dir=str(outputs_dir))
    print(f"Wrote: {pj}")
    print(f"Wrote: {pc}")
    print(f"Wrote: {pm}")

    lifecycle_report = build_lifecycle_report()
    lj, lc, lm = write_lifecycle_reports(
        lifecycle_report, outputs_dir=str(outputs_dir),
    )
    print(f"Wrote: {lj}")
    print(f"Wrote: {lc}")
    print(f"Wrote: {lm}")

    claims_report = build_claims_consistency_report(repo_root=REPO_ROOT)
    cj, cc, cm = write_claims_reports(
        claims_report, outputs_dir=str(outputs_dir),
    )
    print(f"Wrote: {cj}")
    print(f"Wrote: {cc}")
    print(f"Wrote: {cm}")

    print(
        f"status={report['status']} "
        f"formal_security_claim={report['formal_security_claim']} "
        f"adamw_status={report['adamw_dense_mask_unsupported']['status']} "
        f"claims_passes={claims_report['passes_consistency_check']}"
    )


if __name__ == "__main__":
    main()
