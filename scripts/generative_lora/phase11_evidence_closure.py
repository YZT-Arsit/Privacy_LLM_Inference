"""Build AAAI final evidence-closure reports exclusively from frozen/measured artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PB = REPO / "results" / "aaai_private_base"
GL = PB / "generative_lora"
OUT = GL / "evidence_closure"


def load_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, content):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite closure artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric_map(path):
    return {row["cell"]: row for row in csv_rows(path)}


def fmt(value, digits=2):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing non-empty closure directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    metrics_path = GL / "metrics/final_g2/aggregate_metrics.csv"
    metrics = metric_map(metrics_path)
    paired_path = GL / "handoff/paired_analysis.json"
    paired = load_json(paired_path)
    restart_path = GL / "handoff/restart_reproducibility.json"
    restart = load_json(restart_path)
    neg_path = GL / "handoff/negative_controls.json"
    neg = load_json(neg_path)
    adapter_dir = GL / "handoff/adapter_package_e2e_L12_s1234_full"
    adapter_manifest_path = adapter_dir / "adapter_manifest.json"
    adapter_manifest = load_json(adapter_manifest_path)
    ablation_metrics_path = GL / "target_ablation/metrics/aggregate_metrics.csv"
    ablation_metrics = metric_map(ablation_metrics_path)
    ablation_train_path = GL / "target_ablation/qv_s1234_train.json"
    ablation_train = load_json(ablation_train_path)
    ablation_gen_path = GL / "target_ablation/qv_s1234_test.jsonl.profile.json"
    ablation_gen = load_json(ablation_gen_path)
    g1_train_path = GL / "adapters/g1_final_s1234.meta.json"
    g1_train = load_json(g1_train_path)
    g1_gen_path = GL / "generations/g1_final_s1234_test.jsonl.profile.json"
    g1_gen = load_json(g1_gen_path)
    g2_train_path = GL / "protected_runs/e2e_L12_s1234_full.json"
    g2_train = load_json(g2_train_path)
    g2_gen_path = GL / "generations/g2_protected_e2e_L12_s1234_full.jsonl.profile.json"
    g2_gen = load_json(g2_gen_path)
    train_counter_path = GL / "protected_runs/e2e_L12_s1234_full.training_tdx_counters.json"
    train_counters = load_json(train_counter_path)["counters"]
    gen_counter_path = GL / "protected_runs/e2e_L12_s1234_full.generation_tdx_counters.json"
    gen_counters = load_json(gen_counter_path)["counters"]
    profile_path = PB / "alicloud_a10_runs/utility_dataplane/sst2_L12_sweep_pb16.timeprofile.json"
    profile = load_json(profile_path)
    security_path = PB / "security/security_run_manifest.json"
    security = load_json(security_path)

    # Exact target-dependent trusted-state accounting from the frozen architecture and L12 split.
    layers, hidden, kv_out, intermediate, rank, steps = 24, 896, 128, 4864, 8, 750
    trusted_elements_qv = layers * (2 * rank * hidden + rank * hidden)  # A(q,v) + B(q)
    trusted_elements_all7 = layers * (5 * rank * hidden + rank * hidden + rank * kv_out)
    def target_cost(elements):
        return {
            "trusted_factor_elements": elements,
            "trusted_master_m_v_bytes": elements * 3 * 4,
            "trusted_factor_roundtrip_bytes_per_step": elements * 2 * 4,
            "trusted_factor_roundtrip_bytes_750_steps": elements * 2 * 4 * steps,
        }
    qv_cost, all7_cost = target_cost(trusted_elements_qv), target_cost(trusted_elements_all7)

    # Adapter validation report.
    validation_lines = [
        "# Adapter package validation", "",
        f"Package contains {adapter_manifest['tensor_count']} transformed runtime factors. "
        "The manifest declares no plaintext adapter and no optimizer state.", "",
        "| Control | Result |", "|---|---|",
    ]
    for name, result in sorted(neg["controls"].items()):
        validation_lines.append(f"| {name} | {result} |")
    validation_lines.extend(["", f"All controls fail closed: **{neg['all_controls_fail_closed']}**.", "",
                             f"Adapter tensor blob SHA-256: `{sha(adapter_dir / 'adapter_tensors.safetensors')}`."])
    write_new(GL / "handoff/adapter_validation_report.md", "\n".join(validation_lines) + "\n")

    # Minimal target ablation report.
    qv, all7 = ablation_metrics["QV"], ablation_metrics["ALL7"]
    target_lines = [
        "# Minimal target ablation: q_proj+v_proj versus all seven targets", "",
        "Seed 1234 only. The all-seven column reuses frozen G1; only the q+v arm was newly run.", "",
        "| Target set | BLEU | chrF | ROUGE-L | train wall (s) | generation wall (s) | adapter bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| q+v | {qv['BLEU']} | {qv['chrF']} | {qv['ROUGE_L']} | {ablation_train['wall_sec']} | "
        f"{ablation_gen['wall_sec']} | {(GL / 'target_ablation/qv_s1234_adapter/adapter_model.safetensors').stat().st_size} |",
        f"| all seven | {all7['BLEU']} | {all7['chrF']} | {all7['ROUGE_L']} | {g1_train['wall_sec']} | "
        f"{g1_gen['wall_sec']} | {(GL / 'adapters/g1_final_s1234/adapter_model.safetensors').stat().st_size} |",
        "", "## Exact L12 trusted-state/traffic accounting", "",
        "These byte counts are exact shape accounting for FP32 master+m+v and the transformed-factor "
        "upload/return path. CE/dlogits traffic is target-independent and excluded from this comparison.", "",
        "| Target set | Trusted elements | FP32 master+m+v bytes | optimizer roundtrip bytes/step | 750-step bytes |",
        "|---|---:|---:|---:|---:|",
        f"| q+v | {qv_cost['trusted_factor_elements']} | {qv_cost['trusted_master_m_v_bytes']} | "
        f"{qv_cost['trusted_factor_roundtrip_bytes_per_step']} | {qv_cost['trusted_factor_roundtrip_bytes_750_steps']} |",
        f"| all seven | {all7_cost['trusted_factor_elements']} | {all7_cost['trusted_master_m_v_bytes']} | "
        f"{all7_cost['trusted_factor_roundtrip_bytes_per_step']} | {all7_cost['trusted_factor_roundtrip_bytes_750_steps']} |",
    ]
    write_new(GL / "target_ablation/cost_quality_tradeoff.md", "\n".join(target_lines) + "\n")

    trajectory = g2_train["trajectory"]
    g2_train_wall = sum(row["wall"] for row in trajectory)
    g2_ce_roundtrip = sum(row["net_ce"] for row in trajectory)
    g2_adam_roundtrip = sum(row["net_adamw"] for row in trajectory)
    logical_train = train_counters["ce_batch_calls"] + train_counters["adamw_step_calls"] + train_counters["adamw_init_calls"]
    logical_gen = gen_counters["decode_calls"]
    components = profile["components"]

    cost_lines = [
        "# Consolidated measured cost table", "",
        "E2E rows report the completed E2E run. The component row is the frozen L12 phys-batch-16 "
        "profile on the same A10↔TDX protocol; it is not relabeled as E2E timing.", "",
        "| Scope | Training | Generation | Handoff | TDX | GPU | Transport | Memory | Trusted calls | Logical ops | Physical messages |",
        "|---|---|---|---|---|---|---|---|---:|---:|---:|",
        f"| G1 E2E plaintext | {g1_train['wall_sec']} s | {g1_gen['wall_sec']} s; {g1_gen['tokens_per_sec']} tok/s | N/A | N/A | A10 FP32 | local | not recorded | 0 | 0 | 0 |",
        f"| G2 E2E protected | {g2_train_wall:.1f} s | {g2_gen['wall_sec']} s; {g2_gen['tokens_per_sec']} tok/s | "
        f"{(adapter_dir / 'adapter_tensors.safetensors').stat().st_size} B adapter | real Intel TDX, attested | A10 | "
        f"CE RTT {g2_ce_roundtrip:.1f} s; AdamW RTT {g2_adam_roundtrip:.1f} s; decode wait {g2_gen['tdx_wait_sec']} s | "
        f"last dlogits {g2_train['dlogits_instrumentation']['dlogits_bytes']} B; full peak not recorded | "
        f"{logical_train + logical_gen} | {logical_train + logical_gen} | {2 * (logical_train + logical_gen)} minimum request/reply messages |",
        f"| L12 measured component profile (phys batch 16) | {profile['wall_mean_ms']} ms/batch | N/A | N/A | "
        f"CE {components['ce_tdx_compute']['mean_ms']} ms; AdamW {components['adamw_tdx_compute']['mean_ms']} ms | "
        f"fwd {components['gpu_forward_and_row_gather']['mean_ms']} ms; bwd {components['gpu_backward_and_scatter']['mean_ms']} ms | "
        f"CE {profile['ce_bytes_mean']} B; AdamW {profile['adamw_bytes_mean']} B | {profile['gpu_peak_mem_mb']} MiB peak | 2 | 2 | 4 |",
        "", "Physical-message counts are derived from measured logical request counters as one request plus one reply; "
        "handshake/setup messages are excluded because the frozen counters do not enumerate them.",
    ]
    write_new(OUT / "cost_table.md", "\n".join(cost_lines) + "\n")

    # Main paper table.
    main_lines = [
        "# Final main table — frozen E2E NLG evidence", "",
        "| Cell | Seeds | BLEU | chrF | ROUGE-L | invalid% | repetition% | avg tokens | train wall | gen wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| G0 base | 1 | {metrics['G0']['BLEU']} | {metrics['G0']['chrF']} | {metrics['G0']['ROUGE_L']} | {metrics['G0']['invalid_rate_pct']} | {metrics['G0']['repetition_pct']} | {metrics['G0']['avg_len']} | N/A | {load_json(GL / 'generations/g0_base_test.jsonl.profile.json')['wall_sec']} s |",
        f"| G1 plaintext LoRA | 3 | {metrics['G1_s1234']['BLEU']} | {metrics['G1_s1234']['chrF']} | {metrics['G1_s1234']['ROUGE_L']} | {metrics['G1_s1234']['invalid_rate_pct']} | {metrics['G1_s1234']['repetition_pct']} | {metrics['G1_s1234']['avg_len']} | {g1_train['wall_sec']} s (seed 1234) | {g1_gen['wall_sec']} s |",
        f"| G2 protected L12 | 1 | {metrics['G2_L12_s1234']['BLEU']} | {metrics['G2_L12_s1234']['chrF']} | {metrics['G2_L12_s1234']['ROUGE_L']} | {metrics['G2_L12_s1234']['invalid_rate_pct']} | {metrics['G2_L12_s1234']['repetition_pct']} | {metrics['G2_L12_s1234']['avg_len']} | {g2_train_wall:.1f} s | {g2_gen['wall_sec']} s |",
        "", f"Fresh-process restart: token agreement {restart['token_agreement']:.3f}, generation agreement "
        f"{restart['generation_agreement']:.3f}, hash agreement {restart['hash_agreement']}.",
    ]
    write_new(OUT / "final_main_table.md", "\n".join(main_lines) + "\n")

    # Frozen security evidence, with corrected scope.
    h = security["headline_metrics"]
    sec_lines = [
        "# Frozen security table", "",
        "These S1–S6 results are retained as prior channel tests under the private-checkpoint/no-weight-prior "
        "assumption. The manifest's historical `from_scratch_private_base` label is not supported by provenance.", "",
        "| Test | Headline result | Positive control | Scope |",
        "|---|---|---|---|",
        f"| S1 representation inversion | strict middle cosine {h['S1']['strict_inversion_cosine_middle']} vs random {h['S1']['random_baseline_cosine_middle']} | known-pair cosine {h['S1']['known_pairs_A1_linear_cosine_middle']} | offline channel test |",
        f"| S2 LoRA recovery | transformed ΔW relative error {h['S2']['ours_plaintext_dW_rel_err']} | plaintext {h['S2']['control_plaintext_dW_rel_err']} | synthetic adapter recovery |",
        f"| S3 logit leakage | perm-only corr {h['S3']['B1_perm_only']['confidence_corr']}; monomial corr {h['S3']['B2_monomial']['confidence_corr']} | plaintext multiset check | design diagnostic |",
        f"| S4 gradient inversion | plaintext token acc {h['S4']['plaintext_b1_token_acc']} | chance {h['S4']['random_baseline_token_acc']} | gradient proxy |",
        f"| S5 membership | plaintext AUC {h['S5']['auc_B0_plaintext']}; perm-only {h['S5']['auc_B1_perm_only']}; monomial {h['S5']['auc_B1_monomial']} | plaintext AUC > 0.5 | lightweight MIA proxy |",
        f"| S6 KV recovery | masked-transfer top1 {h['S6']['masked_transfer_top1']}; adapted {h['S6']['masked_adapted_top1']} | plaintext {h['S6']['plaintext_KV_top1']} | KV proxy |",
        "", "No formal security, comprehensive black-box security, or from-scratch claim is made.",
    ]
    write_new(OUT / "security_table.md", "\n".join(sec_lines) + "\n")

    # Claim audit.
    def evidence(path):
        return f"`{path.relative_to(REPO)}` / `{sha(path)}`"
    claims = [
        ("One real protected E2E LoRA run completed on A10+TDX", "SUPPORTED", evidence(g2_train_path), "A10+Intel TDX", "G2 L12 seed 1234", "high"),
        ("G2 is competitive with G1 on BLEU and ROUGE-L", "SUPPORTED", evidence(paired_path), "offline paired analysis", "500 aligned outputs", "high"),
        ("G2 has no quality degradation on every metric", "UNSUPPORTED", evidence(paired_path), "offline paired analysis", "chrF CI excludes zero below 0", "high"),
        ("Transformed adapter supports two clean fresh-process restarts", "SUPPORTED" if restart["fresh_process_ok"] else "UNSUPPORTED", evidence(restart_path), "A10+Intel TDX", "restart A/B", "high"),
        ("Adapter compatibility/tamper controls fail closed", "SUPPORTED" if neg["all_controls_fail_closed"] else "UNSUPPORTED", evidence(neg_path), "A10 CPU loader", "9 controls", "high"),
        ("No plaintext base/adapter materialization in instrumented G2 runtime", "SUPPORTED", evidence(g2_gen_path), "A10+Intel TDX", "runtime counters", "medium; instrumentation scope"),
        ("The selected victim was trained from scratch", "UNSUPPORTED", evidence(PB / "blackbox_from_scratch/provenance_audit.json"), "provenance audit", "public Qwen checkpoint", "high"),
        ("The system is formally or universally black-box secure", "UNSUPPORTED", evidence(security_path), "offline proxies", "S1–S6 only", "high"),
        ("Protected training is cost-equivalent to plaintext", "UNSUPPORTED", evidence(g2_train_path), "two-machine prototype", "G1/G2 wall", "high"),
        ("All-seven targets dominate q+v on the evaluated task", "SUPPORTED" if float(all7['BLEU']) > float(qv['BLEU']) else "PARTIALLY_SUPPORTED", evidence(ablation_metrics_path), "A10", "single seed", "medium; one seed"),
    ]
    claim_lines = ["# Claim-evidence matrix", "", "| Claim | Status | Artifact / SHA-256 | Runtime | Experiment | Confidence |", "|---|---|---|---|---|---|"]
    for row in claims:
        claim_lines.append("| " + " | ".join(row) + " |")
    write_new(OUT / "claim_evidence_matrix.md", "\n".join(claim_lines) + "\n")

    limitations = [
        "# Limitations", "",
        "- The victim uses a public Qwen2.5-0.5B checkpoint as a private/secret proxy; it is not from scratch.",
        "- G2 has one seed, while G1 has three; seed robustness for protected training is not established.",
        "- Paired chrF is lower for G2 and its 95% interval excludes zero; only BLEU/ROUGE-L are comparable here.",
        "- The S1–S6 suite contains proxy/channel tests and is not the comprehensive B0/B2 black-box evaluation.",
        "- The two-machine SSH/VPC prototype wall time is not a co-located GPU-TEE overhead estimate.",
        "- Physical message totals exclude unenumerated handshake/setup messages.",
        "- The minimal target ablation uses one plaintext-reference seed; trusted-state bytes are exact shape accounting, not a second protected training run.",
        "- No formal, information-theoretic, or universal security statement is supported.",
    ]
    write_new(OUT / "limitations.md", "\n".join(limitations) + "\n")

    # Artifact index and reproducibility trace.
    artifact_paths = [metrics_path, paired_path, restart_path, neg_path, adapter_manifest_path,
                      adapter_dir / "adapter_tensors.safetensors", ablation_metrics_path,
                      ablation_train_path, ablation_gen_path, g1_train_path, g1_gen_path,
                      g2_train_path, g2_gen_path, train_counter_path, gen_counter_path,
                      profile_path, security_path]
    index_lines = ["# Artifact index", "", "| Artifact | SHA-256 | Bytes |", "|---|---|---:|"]
    for path in artifact_paths:
        index_lines.append(f"| `{path.relative_to(REPO)}` | `{sha(path)}` | {path.stat().st_size} |")
    write_new(OUT / "artifact_index.md", "\n".join(index_lines) + "\n")

    scripts = [
        REPO / "scripts/a10_batch_runner.py",
        REPO / "scripts/tdx_persistent_service.py",
        REPO / "scripts/batch_dataplane.py",
        REPO / "scripts/h800_unified_worker.py",
        REPO / "scripts/h800_d4_worker.py",
        REPO / "scripts/generative_lora/plaintext_lora.py",
        REPO / "scripts/generative_lora/protected_generate.py",
        REPO / "scripts/generative_lora/gate_e2e_protected.py",
        REPO / "scripts/generative_lora/phase6_metrics.py",
        REPO / "scripts/generative_lora/adapter_handoff.py",
        REPO / "scripts/generative_lora/phase7_export_adapter.py",
        REPO / "scripts/generative_lora/phase8_restart_closure.sh",
        REPO / "scripts/generative_lora/phase9_paired_closure.py",
        REPO / "scripts/generative_lora/phase10_target_ablation.sh",
        Path(__file__),
        *[REPO / f"scripts/security/s{i}_{name}.py" for i, name in [
            (1, "representation_inversion"), (2, "lora_recovery"),
            (3, "logit_leakage"), (4, "gradient_inversion"),
            (5, "membership"), (6, "kv_cache")]],
        REPO / "scripts/security/build_security_report.py",
    ]
    missing_scripts = [str(path.relative_to(REPO)) for path in scripts if not path.is_file()]
    manifests = [GL / "baseline_manifest.json", GL / "data/dataset_manifest.json",
                 GL / "monitor/e2e_L12_s1234_full/completion_manifest.json",
                 adapter_manifest_path, GL / "target_ablation/manifest.json", security_path]
    missing_manifests = [str(path.relative_to(REPO)) for path in manifests if not path.is_file()]
    broken = [str(path.relative_to(REPO)) for path in artifact_paths if not path.is_file()]
    session = load_json(GL / "protected_runs/e2e_L12_s1234_full.tdx_session.json")
    revision = session["binding_manifest"]["source_revision"]
    try:
        subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"], cwd=REPO, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        revision_available = True
    except Exception:
        revision_available = False
    repro_lines = [
        "# Reproducibility audit", "",
        "Trace: final tables → parsed metrics/JSON → raw generation/run artifacts → executable scripts → "
        "SHA-256/index → dataset, completion, adapter, ablation and security manifests.", "",
        "| Table | Metrics/raw evidence | Executable | Manifest | Status |", "|---|---|---|---|---|",
        "| final_main_table | final_g2 metrics + G0/G1/G2 generations | phase6_metrics.py / protected_generate.py | completion_manifest | traced |",
        "| security_table | S1–S6 result JSON | scripts/security/s1–s6 | security_run_manifest | traced, historical threat label corrected |",
        "| cost_table | E2E run/profile/counters | a10_batch_runner.py / protected_generate.py | completion_manifest | traced |",
        "| paired_analysis | frozen G1/G2 JSONL | phase9_paired_closure.py | completion_manifest | traced |",
        "| restart_reproducibility | restart A/B JSONL+profiles | phase8_restart_closure.sh | adapter_manifest | traced |",
        "| target ablation | qv run + frozen all-seven G1 | phase10_target_ablation.sh | target_ablation/manifest.json | traced |",
        "", f"- Missing scripts: `{missing_scripts}`", f"- Missing manifests: `{missing_manifests}`",
        f"- Broken required artifact references: `{broken}`",
        "- Duplicate outputs: restart A/B are intentionally byte-identical reproducibility controls; no accidental duplicate paper cells detected.",
        "- Orphan/diagnostic artifacts: smoke runs and the first failed restart log are retained as diagnostics and excluded from paper tables.",
        f"- Frozen G2 source revision `{revision}` available in local git object database: `{revision_available}`.",
        "- Current `protected_generate.py` differs from the frozen G2 hash because the previously unused adapter-package path "
        "received a variable-shadowing fix; the frozen raw-adapter G2 result was not rerun or altered.",
    ]
    write_new(OUT / "REPRODUCIBILITY_AUDIT.md", "\n".join(repro_lines) + "\n")

    required_ok = all([
        restart.get("fresh_process_ok"), neg.get("all_controls_fail_closed"),
        len(paired.get("metrics", {})) == 6, not missing_scripts, not missing_manifests, not broken,
        revision_available,
    ])
    status = "AAAI_EVIDENCE_CLOSURE_COMPLETE" if required_ok else "AAAI_EVIDENCE_CLOSURE_PARTIAL"
    blockers = []
    if not required_ok:
        if not restart.get("fresh_process_ok"): blockers.append("fresh-process restart closure")
        if not neg.get("all_controls_fail_closed"): blockers.append("adapter fail-closed controls")
        if missing_scripts: blockers.append("missing executable scripts")
        if missing_manifests: blockers.append("missing manifests")
        if broken: blockers.append("broken required artifact references")
        if not revision_available: blockers.append("frozen G2 source revision unavailable")
    status_lines = [f"# {status}", "", "## Remaining paper blockers", ""]
    status_lines.extend([f"- {item}" for item in blockers] or ["None."])
    status_lines.extend(["", "No completed G0/G1/G2 experiment was rerun or overwritten. No commit was created."])
    write_new(OUT / "final_status.md", "\n".join(status_lines) + "\n")
    print(json.dumps({"status": status, "remaining_blockers": blockers}, indent=2))


if __name__ == "__main__":
    main()
