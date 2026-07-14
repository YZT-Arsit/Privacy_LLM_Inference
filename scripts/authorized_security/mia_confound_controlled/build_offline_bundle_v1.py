#!/usr/bin/env python3
"""Build isolated CPU-only cost, fidelity, takeover, and paper-preparation artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


SEEDS = (7, 1234, 2025)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def cost_rows(repo: Path) -> list[dict]:
    root = repo / "results/aaai_private_base/generative_lora"
    rows = []
    for seed in SEEDS:
        g1_meta = json.loads((root / f"adapters/g1_final_s{seed}.meta.json").read_text())
        g1_gen = json.loads((root / f"generations/g1_final_s{seed}_test.jsonl.profile.json").read_text())
        suffix = "s1234_full" if seed == 1234 else f"s{seed}_extension"
        g2_run = json.loads((root / f"protected_runs/e2e_L12_{suffix}.json").read_text())
        g2_gen = json.loads((root / f"generations/g2_protected_e2e_L12_{suffix}.jsonl.profile.json").read_text())
        counter_path = root / f"protected_runs/e2e_L12_{suffix}.training_tdx_counters.json"
        g2_count = json.loads(counter_path.read_text())["counters"] if counter_path.exists() else None
        g2_adapter = root / f"protected_runs/e2e_L12_{suffix}.adapter.pt"
        g1_adapter = root / f"adapters/g1_final_s{seed}/adapter_model.safetensors"
        rows.extend([
            {"method": "G1_plaintext", "seed": seed, "steps": g1_meta["steps"],
             "train_wall_sec": g1_meta["wall_sec"], "generation_wall_sec": g1_gen["wall_sec"],
             "total_measured_wall_sec": g1_meta["wall_sec"] + g1_gen["wall_sec"],
             "generation_samples": g1_gen["n"], "total_new_tokens": g1_gen["total_new_tokens"],
             "tokens_per_sec": g1_gen["tokens_per_sec"], "tdx_train_calls": 0,
             "tdx_decode_calls": 0, "adapter_size_bytes": g1_adapter.stat().st_size,
             "hardware": "NVIDIA A10", "scope": "frozen measured artifacts"},
            {"method": "G2_protected_L12", "seed": seed, "steps": g2_run["steps_run"],
             "train_wall_sec": sum(float(item["wall"]) for item in g2_run["trajectory"]),
             "generation_wall_sec": g2_gen["wall_sec"],
             "total_measured_wall_sec": sum(float(item["wall"]) for item in g2_run["trajectory"]) + g2_gen["wall_sec"],
             "generation_samples": g2_gen["n"], "total_new_tokens": g2_gen["total_new_tokens"],
             "tokens_per_sec": g2_gen["tokens_per_sec"],
             "tdx_train_calls": (int(g2_count.get("ce_batch_calls", 0)) + int(g2_count.get("adamw_step_calls", 0))
                                 if g2_count is not None else "unavailable_counter_not_local"),
             "tdx_decode_calls": g2_gen["tdx_decode_calls"], "adapter_size_bytes": g2_adapter.stat().st_size,
             "hardware": "NVIDIA A10 + Intel TDX", "scope": "frozen measured artifacts"},
        ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--mia-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--paired-json", type=Path, required=True)
    parser.add_argument("--active-log-json", type=Path, required=True)
    parser.add_argument("--negative-controls", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--adapter-controls", type=Path, required=True)
    args = parser.parse_args()
    args.mia_root.mkdir(parents=True, exist_ok=True)
    args.offline_root.mkdir(parents=True, exist_ok=True)
    for directory in ("shadow_runs", "v0", "v2"):
        (args.mia_root / directory).mkdir(exist_ok=True)

    now = datetime.now().astimezone().isoformat()
    active = json.loads(args.active_log_json.read_text())
    status = git(args.repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    takeover = {
        "schema": "confound_controlled_mia_takeover", "version": "1.0", "timestamp": now,
        "role": "OFFLINE_MIA_ANALYSIS_ONLY", "branch": git(args.repo, "branch", "--show-current"),
        "head": git(args.repo, "rev-parse", "HEAD"), "dirty_paths": status,
        "active_jobs": active["jobs"], "gpu_or_tdx_idle": False,
        "remote_mutations": False, "shared_registry_updated": False, "commits_created": False,
    }
    write_new(args.mia_root / "takeover_state.json", json.dumps(takeover, indent=2) + "\n")
    snapshot_source = args.active_log_json.parent / "active_job_snapshot.md"
    write_new(args.mia_root / "active_job_snapshot.md", snapshot_source.read_text())
    write_new(args.mia_root / "write_conflict_audit.md", """# Write-Conflict Audit

- Active T1 and T2 remote logs, drivers, sessions, adapters, and result paths are immutable.
- `generative_lora/extension/live_job_registry.json` and `experiment_queue.json` are immutable.
- Existing G0/G1/G2, target-ablation, and V2 MIA artifacts are immutable.
- Existing untracked paired-statistics files are owned by another workflow and are untouched.
- New writes are restricted to `mia/confound_controlled_v2/`, the isolated offline-analysis root,
  and new versioned scripts under `scripts/authorized_security/mia_confound_controlled/`.
- No active-job script is modified by this analysis.
""")

    sources = [
        args.repo / "results/aaai_private_base/authorized_security_eval/corpora/e2e_train_member_v2_s1234.jsonl",
        args.repo / "results/aaai_private_base/authorized_security_eval/corpora/e2e_test_v2_s1234.jsonl",
        args.repo / "results/aaai_private_base/authorized_security_eval/mia/v2_pilot_s7_s1234_s2025/feature_audit.csv",
        args.repo / "results/aaai_private_base/generative_lora/data/train.jsonl",
        args.repo / "scripts/authorized_security/capture_v2_prefill.py",
        args.repo / "scripts/authorized_security/run_v2_membership_pilot.py",
    ]
    write_new(args.mia_root / "source_hashes_before.sha256",
              "\n".join(f"{sha256(path)}  {path.relative_to(args.repo)}" for path in sources) + "\n")
    write_new(args.mia_root / "invalid_cohort_record.md", """# Invalid Initial V2 Cohort Record

Status: **AUDIT_EVIDENCE_ONLY**  
Formal status: `INITIAL_COHORT_INVALID_FOR_FINAL_MIA`; `SOURCE_COHORT_CONFOUND_CONFIRMED`.

- Members: 500 seed-20260714 samples from the frozen official E2E train-derived population,
  captured as `e2e_train_member_v2_s1234.jsonl`.
- Nonmembers: 500 unique-MR official E2E test prompts, captured as `e2e_test_v2_s1234.jsonl`.
- Both used `capture_v2_prefill.py`, run ID `e2e_L12_s1234_full`, package root
  `bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1`, and adapter
  SHA-256 `1e29821f0b4cf4eb90bd748f171a14c2d2607b6e086b7d23bd2ad18b16e90cd2`.
- Exact-length matching retained 215 records per class and controlled sequence length completely.
- 446/803 features nevertheless had disjoint member/nonmember ranges; the classifier reached AUC 1.0.

Length matching could not remove official train/test semantic and construction differences. Because
membership and source split were identical, AUC 1.0 is source-cohort classification and is invalid as
final membership evidence. The corrective design uses 1,000 unique MR groups from one train-derived
pool, constrained-random 500/500 membership per shadow, identical C0 collection, and leave-shadow-run-out evaluation.
""")

    local_registry = {"schema": "confound_controlled_mia_local_job_registry", "version": "1.0",
                      "scope": "this directory only; never synchronized to shared registry", "jobs": []}
    write_new(args.mia_root / "job_registry.json", json.dumps(local_registry, indent=2) + "\n")
    for name in ("negative_controls.csv", "negative_control_report.md", "pipeline_leakage_audit.json"):
        write_new(args.mia_root / name, (args.negative_controls / name).read_text())
    for name in ("confound_feature_attribution.csv", "confound_family_summary.csv", "confound_attribution_report.md"):
        write_new(args.mia_root / name, (args.attribution / name).read_text())
    for name in ("adapter_manifest_negative_controls.json", "adapter_manifest_negative_controls.md"):
        write_new(args.mia_root / name, (args.adapter_controls / name).read_text())

    costs = cost_rows(args.repo)
    cost_dir = args.offline_root / "cost"
    write_csv(cost_dir / "unified_cost_table.csv", costs)
    comparisons = []
    for seed in SEEDS:
        g1 = next(row for row in costs if row["seed"] == seed and row["method"] == "G1_plaintext")
        g2 = next(row for row in costs if row["seed"] == seed and row["method"] == "G2_protected_L12")
        comparisons.append({
            "seed": seed,
            "train_wall_delta_sec_g2_minus_g1": g2["train_wall_sec"] - g1["train_wall_sec"],
            "train_wall_slowdown_x": g2["train_wall_sec"] / g1["train_wall_sec"],
            "generation_wall_delta_sec_g2_minus_g1": g2["generation_wall_sec"] - g1["generation_wall_sec"],
            "generation_wall_ratio_x": g2["generation_wall_sec"] / g1["generation_wall_sec"],
            "total_wall_slowdown_x": g2["total_measured_wall_sec"] / g1["total_measured_wall_sec"],
            "adapter_size_delta_bytes": g2["adapter_size_bytes"] - g1["adapter_size_bytes"],
        })
    write_csv(cost_dir / "cost_comparison_vs_g1.csv", comparisons)
    lines = ["# Unified Frozen-Artifact Cost Table", "",
             "No latency measurement was run. Training wall is the recorded G1 wall or the sum of frozen "
             "G2 per-step walls; total combines training and the matching 500-sample generation.", "",
             "| Method | Seed | Train h | Generation min | Total h | tok/s | TDX train calls | TDX decode calls | Adapter MiB |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in costs:
        lines.append(f"| {row['method']} | {row['seed']} | {row['train_wall_sec']/3600:.3f} | "
                     f"{row['generation_wall_sec']/60:.2f} | {row['total_measured_wall_sec']/3600:.3f} | "
                     f"{row['tokens_per_sec']:.2f} | {row['tdx_train_calls']} | {row['tdx_decode_calls']} | "
                     f"{row['adapter_size_bytes']/2**20:.2f} |")
    write_new(cost_dir / "unified_cost_table.md", "\n".join(lines) + "\n")
    comparison_lines = ["# G2 Cost Delta versus Seed-Matched G1", "",
                        "| Seed | Train slowdown | Generation wall ratio | Total slowdown | Adapter Δ KiB |",
                        "|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        comparison_lines.append(f"| {row['seed']} | {row['train_wall_slowdown_x']:.2f}× | "
                                f"{row['generation_wall_ratio_x']:.3f}× | "
                                f"{row['total_wall_slowdown_x']:.2f}× | "
                                f"{row['adapter_size_delta_bytes']/1024:.1f} |")
    comparison_lines += ["", "Generation wall ratios are descriptive, not a latency benchmark: frozen outputs "
                         "have seed/method-dependent token counts. No timing run was launched."]
    write_new(cost_dir / "cost_comparison_vs_g1.md", "\n".join(comparison_lines) + "\n")

    obf_dir = args.offline_root / "obfuscatune_preparation"
    obf_sources = sorted((args.repo / "src/pllo/baselines/obfuscatune").glob("*.py"))
    checklist = [
        {"gate": "paper_method_mapping", "required": True, "current": "documented", "launch_blocking": True},
        {"gate": "qwen_prefill_equivalence", "required": True, "current": "implemented_not_revalidated_here", "launch_blocking": True},
        {"gate": "qwen_decode_cache_equivalence", "required": True, "current": "implemented_not_revalidated_here", "launch_blocking": True},
        {"gate": "lora_training_path", "required": True, "current": "not_implemented", "launch_blocking": True},
        {"gate": "nonlinear_tee_boundary", "required": True, "current": "simulator_only", "launch_blocking": True},
        {"gate": "qkv_attention_exposure_accounted", "required": True, "current": "implemented", "launch_blocking": True},
        {"gate": "same_e2e_pool_and_schedule", "required": True, "current": "prepared", "launch_blocking": True},
        {"gate": "matched_cost_instrumentation", "required": True, "current": "prepared", "launch_blocking": True},
    ]
    write_csv(obf_dir / "fidelity_checklist.csv", checklist)
    obf_config = {
        "schema": "obfuscatune_fidelity_preparation", "version": "1.0", "status": "PREPARATION_ONLY_DO_NOT_LAUNCH",
        "paper_identifier": "arXiv:2407.02960", "model": "Qwen2.5-0.5B", "dataset": "E2E NLG",
        "seeds": list(SEEDS), "matrix": "orthogonal", "condition_number": 1.0,
        "lora": {"rank": 8, "alpha": 16, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 "steps": 750, "lr": 0.0002},
        "boundary": {"rmsnorm": "TEE", "rope": "TEE", "softmax": "TEE", "silu_swiglu": "TEE",
                     "qkv_plaintext_visible": True, "attention_scores_visible": True, "kv_cache_protected": False},
        "launch_allowed": False, "blocking_gates": [row["gate"] for row in checklist if row["current"] not in ("implemented", "documented")],
        "source_hashes": {str(path.relative_to(args.repo)): sha256(path) for path in obf_sources},
    }
    write_new(obf_dir / "pilot_config.json", json.dumps(obf_config, indent=2) + "\n")
    write_new(obf_dir / "protocol.md", """# ObfuscaTune Protocol/Fidelity Preparation

Status: **PREPARATION ONLY — GPU launch prohibited**.

The repository already contains a Qwen inference simulator with orthogonal/random/conditioned
matrices, explicit nonlinear TEE boundaries, GQA handling, decode cache support, and honest exposure
labels. It does not yet contain a validated ObfuscaTune LoRA training path on the frozen E2E protocol.

Before any run, implement the training path in a new versioned module and require: orthogonal
condition number 1; plaintext-reference equivalence; prefill and cached-decode agreement; exact
Qwen2.5-0.5B, E2E pool, target set, rank, alpha, optimizer, schedule, dtype, and decoding match;
explicit Q/K/V, attention-score, and plaintext-KV exposure accounting; no claim that simulator timing
is real TEE timing; and fail-closed rejection of silent plaintext or trusted-shortcut fallback.

Required negative controls include wrong inverse, non-orthogonal ill-conditioned matrix, stale matrix,
wrong cache position, missing nonlinear boundary, mismatched adapter target set, and unreported QKV/KV
exposure. The fidelity checklist remains launch-blocking until every required implementation gate passes.
""")

    paired = json.loads(args.paired_json.read_text())
    paper_dir = args.offline_root / "paper_templates"
    agg = {row["metric"]: row for row in paired["aggregate"]}
    table = ["# Paper Table Template — Frozen Three-Seed Evidence", "",
             "| Comparison | Seeds | BLEU Δ | chrF Δ | ROUGE-L Δ | Evidence status |",
             "|---|---:|---:|---:|---:|---|"]
    table.append(f"| G2 L12 − matched G1 | 3 | {agg['BLEU']['mean_delta_across_seeds']:.3f} ± {agg['BLEU']['seed_std']:.3f} | "
                 f"{agg['chrF']['mean_delta_across_seeds']:.3f} ± {agg['chrF']['seed_std']:.3f} | "
                 f"{agg['ROUGE_L']['mean_delta_across_seeds']:.3f} ± {agg['ROUGE_L']['seed_std']:.3f} | frozen measured |")
    table += ["| Corrected paired-shadow V2 − V0 MIA | — | — | — | — | pending GPU/TDX shadows |", "",
              "Do not insert the invalid old-cohort AUC=1.0 into a paper-facing security table."]
    write_new(paper_dir / "paper_table_template.md", "\n".join(table) + "\n")
    write_new(args.mia_root / "final_mia_table.md", """# Confound-Controlled MIA Table

Status: `PENDING_CORRECTED_SHADOW_RESULTS`.

| Mode | Feature set | Leave-sample-out AUC | Leave-shadow-out AUC | 95% CI | V2−V0 | Status |
|---|---|---:|---:|---:|---:|---|
| C0 V0 | output only | — | — | — | reference | not collected |
| C0 V2 | F0/F2/F3/F4/F5 | — | — | — | — | not collected |

The old AUC=1.0 is excluded because source split and membership were identical.
""")
    write_new(args.mia_root / "claim_evidence_matrix.md", """# Claim–Evidence Matrix

| Claim | Status | Current evidence | Missing evidence |
|---|---|---|---|
| Initial V2 cohort is invalid due to source confounding | SUPPORTED | 446/803 disjoint invariant features; N3 source control | none |
| Per-prompt transforms do not remove invariant cohort differences | SUPPORTED | all disjoint features are invariant | none |
| Corrected V2 has membership advantage over V0 | NOT YET TESTED | none | 3 shadows, matched V0/V2, leave-shadow-out CI |
| Attention adds incremental membership signal | NOT YET TESTED | old cohort invalid | corrected F3 vs F5 ablation |
| Universal MIA resistance | FORBIDDEN | no supporting design | not a permitted claim |
""")
    write_new(args.mia_root / "limitations.md", """# Limitations

- Corrected shadow adapters, V0 outputs, and V2 C0 features do not yet exist.
- Three shadows will provide only coarse trained-model uncertainty.
- The same-pool pilot uses a 1,000-example subset of one E2E source population.
- Current negative controls diagnose the invalid frozen cohort; they do not substitute for corrected controls.
- ObfuscaTune preparation is not a completed faithful training baseline.
""")
    write_new(args.mia_root / "reproducibility_audit.md", """# Reproducibility Audit

- Candidate pool and membership assignments are deterministic and hash-bound.
- Membership is evaluator metadata and is excluded from feature files.
- Every sample changes membership state across the three planned shadows.
- Attack preprocessing is fit on attack-training partitions only.
- Planned primary split is leave-shadow-run-out with semantic/duplicate grouping.
- Existing source hashes are recorded in `source_hashes_before.sha256`.
- The local seed-1234 G2 JSONL was found truncated to exactly 524,288 bytes. Paired analysis uses
  a read-only isolated copy retrieved from the completed remote artifact and verified against the
  frozen completion-manifest SHA-256 `d3b5ad9820c13e473ee7a995480a9ea6b5b3f5a786a4ba3276774cf116e46e31`.
- No GPU/TDX work was launched during offline preparation.
""")
    write_new(args.mia_root / "final_status.md", """# Final Status

`CONFOUND_CONTROLLED_MIA_PARTIAL`

Completed offline: takeover audit, invalid-cohort preservation, frozen-feature attribution,
negative-control code/results, same-pool construction, adapter-manifest guards, paired G1/G2
statistics with bootstrap intervals, active-log parsing, cost table, ObfuscaTune fidelity preparation,
and paper/claim templates.

Active unrelated jobs: see `active_job_snapshot.md`.

Prepared but not launched: three shadow LoRA jobs, matched C0 V0/V2 collection, leave-shadow-run-out attack.

Missing corrected cells: 3 shadow adapters; 3×1,000 V0 outputs; 3×1,000 V2 C0 records; corrected controls and CIs.

Estimated resource: approximately 17–18 GPU/TDX hours. No device was idle at takeover.

Next safe command remains CPU-only:

```bash
python3 scripts/authorized_security/mia_confound_controlled/validate_prepared_shadows_v1.py \
  --root results/aaai_private_base/authorized_security_eval/mia/confound_controlled_v2/shadow_runs
```

No launch script is implemented during offline-only preparation; GPU/TDX execution remains fail-closed.
""")


if __name__ == "__main__":
    main()
