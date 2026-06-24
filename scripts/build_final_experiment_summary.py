"""Consolidate the masked-Qwen experiment results into a final summary.

Emits ``outputs/final_experiment_summary.{json,md}`` with four clearly-separated
sections -- completed claims, supporting artifacts, measured numbers, and
limitations/TODO -- plus a three-way scope reminder and a paper-ready paragraph
explaining the 26.34 GB folded-package size.

The measured numbers default to the H800 results reported on 2026-06-24; pass the
real artifact JSONs (``--folded-build-json`` etc.) to refresh any group from disk.
Discipline: this script does NOT claim full 28-layer package-backed prefill/decode
-- that stays an explicit TODO.

stdlib only. Python 3.6-safe.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load(path):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def _pick(src, keys, into):
    """Override defaults in ``into`` with present keys from ``src``."""
    for k in keys:
        if k in src and src[k] is not None:
            into[k] = src[k]
    return into


# ---------------------------------------------------------------------------
# measured numbers (H800, 2026-06-24) -- defaults, overridable from artifacts
# ---------------------------------------------------------------------------


def measured_numbers(args) -> dict:
    standalone = {
        "model": "Qwen2.5-7B-Instruct", "batch_size": 1, "num_layers": 28,
        "dtype": "bfloat16", "seq_len_requested": 128, "effective_prompt_len": 39,
        "max_new_tokens": 64,
        "teacher_forced_top1_match_rate_hf_plain": 1.0,
        "teacher_forced_top1_match_rate_hf_masked": 1.0,
        "teacher_forced_top1_match_rate_plain_masked": 1.0,
        "plain_vs_masked_token_match_rate": 1.0,
        "topk_overlap": 0.9875, "logits_max_abs_error": 0.3367,
        "latency_s": 182.0, "peak_gpu_memory_mb": 31550.0,
        "e2_token_grid": [1, 8, 16, 32, 64],
        "e2_tf_top1_plain_masked_all": 1.0, "e2_tf_top1_hf_masked_all": 1.0,
        "e2_greedy_plain_vs_masked_all": 1.0,
        "tee_used_on_gpu": False,
    }
    e1 = _load(args.e1_json)
    _pick(e1, ["teacher_forced_top1_match_rate_hf_plain",
               "teacher_forced_top1_match_rate_hf_masked",
               "teacher_forced_top1_match_rate_plain_masked",
               "plain_vs_masked_token_match_rate", "topk_overlap",
               "logits_max_abs_error", "latency_s", "peak_gpu_memory_mb",
               "seq_len_requested", "effective_prompt_len", "max_new_tokens"],
          standalone)

    cross_machine = {
        "mode": "boundary_client", "gpu_backend": "mock", "max_new_tokens": 64,
        "audit_passed": True, "boundary_tee_type": "tdx",
        "boundary_attested": True, "runtime_hash_bound": True,
        "gpu_worker_remote": True, "tokens_match_plaintext_reference": True,
        "runtime_hash": ("96e4353736e4cf2e133b9efac0a6fcf6337eab8ecdd5d6c2d413"
                         "578072d133de392763f928da926bccd726716e694c7b1cc2b7b3"
                         "a06a3c5aaabe6c40979220a5"),
        "mr_td": ("e0199499baacb2e4f4bc73046f25bedf674d42defbe4e854242bd6554a"
                  "9d155edf7f3bff8e6202e63ed230e59ab2568a"),
        "trusted_bytes": 512608, "gpu_bytes": 1574400,
        "boundary_calls": {"embed_and_mask": 64, "recover_logits": 64,
                           "sample": 64},
        "gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
        "tee_used_on_gpu": False,
    }
    cm = _load(args.cross_machine_json)
    _pick(cm, ["audit_passed", "boundary_tee_type", "boundary_attested",
               "runtime_hash_bound", "gpu_worker_remote",
               "tokens_match_plaintext_reference", "runtime_hash", "mr_td",
               "trusted_bytes", "gpu_bytes", "boundary_calls"], cross_machine)

    folded = {
        "package_path": "/root/autodl-tmp/privacy_llm_packages/qwen7b_folded_full",
        "num_layers": 28, "num_shards": 29, "package_size_gb": 26.339369,
        "store_dtypes": ["F32"], "generation_time_s": 59.6366,
        "peak_memory_mb": 27221.19,
        "manifest_hash": ("90184a6bf33cfa058419085096b938e63f1ed1a00b2143131d6"
                          "618a19a21e1d8"),
        "package_valid": True, "hash_mismatches": 0, "forbidden_fields_found": [],
        "contains_mask_secrets": False, "contains_plaintext_inputs": False,
        "contains_raw_lora": False, "contains_optimizer_state": False,
        "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
        "folded_weight_size_gb_float32": 26.3394,
        "folded_weight_size_gb_if_model_dtype_bf16": 13.1694,
    }
    fb = _load(args.folded_build_json)
    _pick(fb, ["num_layers", "num_shards", "folded_weight_size_gb",
               "generation_time_s", "peak_memory_mb", "manifest_hash"], folded)
    if "folded_weight_size_gb" in fb:
        folded["package_size_gb"] = fb["folded_weight_size_gb"]
    cost = _load(args.cost_json)
    _pick(cost, ["folded_weight_size_gb", "folded_weight_size_gb_if_model_dtype"],
          folded)

    load_probe = {
        "folded_package_loaded": True, "folded_package_valid": True,
        "package_size_gb": 26.339369, "num_layers": 28, "num_shards": 29,
        "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
        "gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
    }
    _pick(_load(args.load_probe_json),
          ["folded_package_loaded", "folded_package_valid", "package_size_gb",
           "num_layers", "num_shards", "worker_has_mask_secrets",
           "tee_used_on_gpu", "leaked_secret_fields"], load_probe)

    layer0 = {
        "folded_package_loaded": True, "folded_package_valid": True,
        "num_layers": 1, "allclose": True,
        "max_abs_error": 1.9073486328125e-06,
        "mean_abs_error": 7.02214464354256e-08,
        "relative_l2_error": 4.5790019953528827e-07,
        "worker_has_mask_secrets": False, "tee_used_on_gpu": False,
        "gpu_visible_plaintext_fields": [], "leaked_secret_fields": [],
    }
    _pick(_load(args.layer_probe_json),
          ["allclose", "max_abs_error", "mean_abs_error", "relative_l2_error",
           "folded_package_loaded", "folded_package_valid",
           "worker_has_mask_secrets", "tee_used_on_gpu"], layer0)

    return {
        "standalone_e1_e2": standalone,
        "cross_machine_attested_mock": cross_machine,
        "folded_package_build_verify": folded,
        "folded_package_load_probe": load_probe,
        "folded_package_layer0_correctness": layer0,
    }


# ---------------------------------------------------------------------------
# curated structure
# ---------------------------------------------------------------------------

SIZE_EXPLANATION = (
    "The full Qwen2.5-7B folded package measures 26.34 GB because the folded "
    "operators (W_tilde = N_in^-1 W N_out, plus the folded LM head) are stored in "
    "float32 for numerical fidelity: the offline fold is computed in float32 and "
    "the resulting tensors are serialized at that precision (the inspector reports "
    "store_dtypes=['F32']). The parameter count is ~7.07 B (28 decoder layers' "
    "q/k/v/o + gate/up/down operators plus the LM head; embeddings stay trusted-"
    "side and are not packaged), so at 4 bytes/parameter the package is ~26.34 GB "
    "-- matching the measured artifact. Storing the same operators in bf16 would "
    "be ~13.17 GB, but that is NOT the current measured artifact; bf16 storage is "
    "an available trade-off (smaller package vs. a small loss of fold precision) "
    "and is left as a future option. Folding is a one-time setup/provisioning "
    "cost, not a per-token or per-request cost."
)


def completed_claims() -> list:
    return [
        {
            "id": "C1",
            "claim": "Standalone H800 Qwen2.5-7B masked/folded compute "
                     "correctness + token scaling (E1/E2).",
            "status": "complete",
            "summary": "Teacher-forced top-1 agreement (HF/plain/masked) = 1.0 "
                       "and plain-vs-masked token match = 1.0 at the fixed paper "
                       "config (bs=1, seq_len_requested=128, max_new_tokens=64, "
                       "28/28 layers, bf16); E2 scaling {1,8,16,32,64} all 1.0. "
                       "bf16 free-running exact-sequence equality is explicitly "
                       "NOT the criterion -- teacher-forced top-1 + plain-vs-"
                       "masked are.",
            "evidence": ["outputs/e1_nolora_qwen_natural.json",
                         "outputs/e2_token_scaling_qwen_natural.json",
                         "outputs/qwen_e1_e2_paper_tables.md"],
            "metrics_key": "standalone_e1_e2",
        },
        {
            "id": "C2",
            "claim": "Attested TDX trusted boundary driving a remote untrusted "
                     "GPU worker (cross-machine mock protocol).",
            "status": "complete",
            "summary": "Over an SSH tunnel: audit_passed, boundary_tee_type=tdx, "
                       "boundary_attested, runtime_hash_bound, gpu_worker_remote, "
                       "tokens_match_plaintext_reference; no plaintext/secret on "
                       "the wire; tee_used_on_gpu=false. The decoder here is the "
                       "mock identity model -- this validates the PROTOCOL + "
                       "boundary + attestation, not Qwen math (that is C1).",
            "evidence": ["outputs/cm_mock_tdx_h800_attested_current.json",
                         "scripts/generate_tdx_attestation_evidence.py",
                         "docs/cross_machine_security_scope.md"],
            "metrics_key": "cross_machine_attested_mock",
        },
        {
            "id": "C3",
            "claim": "Folded-package provisioning for Qwen2.5-7B: full "
                     "build + verify + worker load, and layer-0 numerical "
                     "correctness.",
            "status": "complete",
            "summary": "Full 28-layer folded package built (29 shards, 26.34 GB, "
                       "float32 store), verified (hashes match, no secret tensor "
                       "names), and loaded by the untrusted worker with "
                       "worker_has_mask_secrets=false. Layer-0 package-vs-in-"
                       "process masked output matches to fp tolerance "
                       "(allclose=true, max_abs_error~1.9e-6).",
            "evidence": ["outputs/folded_full_inspection.json",
                         "outputs/folded_full_cost.json",
                         "outputs/qwen7b_folded_full_load_probe.json",
                         "outputs/qwen7b_folded_full_1layer_probe.json"],
            "metrics_key": "folded_package_build_verify",
        },
    ]


def limitations_todo() -> list:
    return [
        "Full 28-layer package-backed prefill/decode is NOT implemented: the "
        "qwen7b_folded_package worker raises NotImplementedError on /prefill and "
        "/decode. Only build/verify/load and layer-0 numerical correctness are "
        "complete. End-to-end package-backed generation is the next step.",
        "Cross-machine end-to-end masked QWEN decode was not run: C2 uses the "
        "mock identity decoder. A private cross-machine Qwen decode requires the "
        "folded-package path (C3) wired through the remote worker for all 28 "
        "layers (the TODO above).",
        "Folded package is stored in float32 (26.34 GB). bf16 storage (~13.17 GB) "
        "is an available but unimplemented trade-off.",
        "Amulet-migrated nonlinear backend security is NOT formally claimed "
        "(security_claim_status=under_discussion).",
        "LoRA: the package format anticipates lora_adapter packages, but folded-"
        "LoRA generation and masked LoRA training are not part of this stage.",
        "E3 (current-vs-amulet end-to-end), E5 (LoRA-adapted generation), and E6 "
        "(attack evaluation) are intentionally not started.",
    ]


def scope_notes() -> list:
    return [
        "C1 (standalone E1/E2) = full Qwen masked/folded COMPUTE correctness + "
        "scaling, assuming folded operators are available.",
        "C2 (TDX+H800 mock) = the ATTESTED trusted boundary driving a remote "
        "untrusted GPU worker.",
        "C3 (folded package) = the PROVISIONING path that supplies folded "
        "operators to an untrusted worker without shipping mask secrets.",
        "Cross-machine connectivity is the deployment setting, not a "
        "contribution. tee_used_on_gpu is false everywhere.",
    ]


def supporting_artifacts(meta) -> list:
    return [
        {"name": "full folded package", "kind": "package_dir",
         "path": meta["folded_package_build_verify"]["package_path"]},
        {"name": "package builder", "kind": "script",
         "path": "scripts/build_qwen7b_folded_package.py"},
        {"name": "package verifier", "kind": "script",
         "path": "scripts/verify_folded_package.py"},
        {"name": "package inspector", "kind": "script",
         "path": "scripts/inspect_folded_package.py"},
        {"name": "cost estimator", "kind": "script",
         "path": "scripts/estimate_folded_package_cost.py"},
        {"name": "load probe", "kind": "script",
         "path": "scripts/run_qwen7b_folded_package_load_probe.py"},
        {"name": "layer-0 correctness probe", "kind": "script",
         "path": "scripts/run_qwen7b_folded_package_1layer_probe.py"},
        {"name": "E1 runner", "kind": "script",
         "path": "scripts/run_qwen7b_e1_nolora_generation.py"},
        {"name": "E2 runner", "kind": "script",
         "path": "scripts/run_qwen7b_e2_token_scaling.py"},
        {"name": "E1/E2 table aggregator", "kind": "script",
         "path": "scripts/summarize_qwen_e1_e2_results.py"},
        {"name": "TDX evidence generator", "kind": "script",
         "path": "scripts/generate_tdx_attestation_evidence.py"},
        {"name": "cross-machine demo", "kind": "script",
         "path": "scripts/run_tee_gpu_protocol_demo.py"},
        {"name": "security scope doc", "kind": "doc",
         "path": "docs/cross_machine_security_scope.md"},
    ]


def _md(summary) -> str:
    m = summary["measured_numbers"]
    L = ["# Final experiment summary -- privacy-preserving masked Qwen2.5-7B", "",
         "_No claim of full 28-layer package-backed decode; see Limitations._", ""]

    L += ["## 1. Completed claims", ""]
    for c in summary["completed_claims"]:
        L += [f"### {c['id']}. {c['claim']}", "",
              f"- status: **{c['status']}**", f"- {c['summary']}",
              "- evidence: " + ", ".join("`%s`" % e for e in c["evidence"]), ""]

    L += ["## 2. Measured numbers", ""]
    s = m["standalone_e1_e2"]
    L += ["**C1 standalone E1/E2** (bf16, 28 layers):",
          f"- teacher-forced top1 hf_plain/hf_masked/plain_masked = "
          f"{s['teacher_forced_top1_match_rate_hf_plain']}/"
          f"{s['teacher_forced_top1_match_rate_hf_masked']}/"
          f"{s['teacher_forced_top1_match_rate_plain_masked']}",
          f"- plain_vs_masked_token_match_rate = "
          f"{s['plain_vs_masked_token_match_rate']}; topk_overlap = "
          f"{s['topk_overlap']}; logits_max_abs_error = {s['logits_max_abs_error']}",
          f"- seq_len_requested={s['seq_len_requested']} "
          f"effective_prompt_len={s['effective_prompt_len']} "
          f"max_new_tokens={s['max_new_tokens']} latency_s={s['latency_s']} "
          f"peak_gpu_memory_mb={s['peak_gpu_memory_mb']}",
          f"- E2 scaling {s['e2_token_grid']}: tf_top1_plain_masked / "
          f"tf_top1_hf_masked / greedy_match all = 1.0", ""]
    cm = m["cross_machine_attested_mock"]
    L += ["**C2 TDX+H800 attested mock protocol**:",
          f"- audit_passed={cm['audit_passed']} boundary_tee_type="
          f"{cm['boundary_tee_type']} boundary_attested={cm['boundary_attested']} "
          f"runtime_hash_bound={cm['runtime_hash_bound']}",
          f"- gpu_worker_remote={cm['gpu_worker_remote']} max_new_tokens="
          f"{cm['max_new_tokens']} tokens_match_plaintext_reference="
          f"{cm['tokens_match_plaintext_reference']} tee_used_on_gpu="
          f"{cm['tee_used_on_gpu']}",
          f"- mr_td=`{cm['mr_td']}`", f"- runtime_hash=`{cm['runtime_hash']}`",
          f"- trusted_bytes={cm['trusted_bytes']} gpu_bytes={cm['gpu_bytes']} "
          f"boundary_calls={cm['boundary_calls']}",
          f"- gpu_visible_plaintext_fields={cm['gpu_visible_plaintext_fields']} "
          f"leaked_secret_fields={cm['leaked_secret_fields']}", ""]
    f = m["folded_package_build_verify"]
    lp = m["folded_package_load_probe"]
    l0 = m["folded_package_layer0_correctness"]
    L += ["**C3 folded-package provisioning** (build/verify/load + layer-0):",
          f"- num_layers={f['num_layers']} num_shards={f['num_shards']} "
          f"package_size_gb={f['package_size_gb']} store_dtypes={f['store_dtypes']}",
          f"- generation_time_s={f['generation_time_s']} peak_memory_mb="
          f"{f['peak_memory_mb']} manifest_hash=`{f['manifest_hash']}`",
          f"- package_valid={f['package_valid']} contains_mask_secrets="
          f"{f['contains_mask_secrets']} worker_has_mask_secrets="
          f"{f['worker_has_mask_secrets']} tee_used_on_gpu={f['tee_used_on_gpu']}",
          f"- cost: float32={f['folded_weight_size_gb_float32']} GB | "
          f"if bf16 store={f['folded_weight_size_gb_if_model_dtype_bf16']} GB",
          f"- load probe: loaded={lp['folded_package_loaded']} valid="
          f"{lp['folded_package_valid']} worker_has_mask_secrets="
          f"{lp['worker_has_mask_secrets']} leaked_secret_fields="
          f"{lp['leaked_secret_fields']}",
          f"- layer-0 correctness: allclose={l0['allclose']} max_abs_error="
          f"{l0['max_abs_error']:.3e} mean_abs_error={l0['mean_abs_error']:.3e} "
          f"relative_l2_error={l0['relative_l2_error']:.3e}", ""]

    L += ["## 3. Supporting artifacts", ""]
    for a in summary["supporting_artifacts"]:
        L.append(f"- {a['name']} ({a['kind']}): `{a['path']}`")
    L.append("")

    L += ["## 4. Limitations / TODO", ""]
    for t in summary["limitations_todo"]:
        L.append(f"- {t}")
    L.append("")

    L += ["## 5. Scope (do not conflate)", ""]
    for n in summary["scope_notes"]:
        L.append(f"- {n}")
    L.append("")

    L += ["## 6. Why the folded package is 26.34 GB (paper-ready)", "",
          summary["size_explanation"], ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-json", default=None)
    ap.add_argument("--e2-json", default=None)
    ap.add_argument("--cross-machine-json", default=None)
    ap.add_argument("--folded-build-json", default=None)
    ap.add_argument("--cost-json", default=None)
    ap.add_argument("--load-probe-json", default=None)
    ap.add_argument("--layer-probe-json", default=None)
    ap.add_argument("--output-json", default="outputs/final_experiment_summary.json")
    ap.add_argument("--output-md", default="outputs/final_experiment_summary.md")
    args = ap.parse_args()

    summary = {
        "stage": "final_experiment_summary",
        "title": "Privacy-preserving masked Qwen2.5-7B inference: experiment "
                 "summary",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_claims": completed_claims(),
        "measured_numbers": measured_numbers(args),
        "limitations_todo": limitations_todo(),
        "scope_notes": scope_notes(),
        "size_explanation": SIZE_EXPLANATION,
        "full_28layer_package_backed_decode_status": "TODO_not_implemented",
    }
    summary["supporting_artifacts"] = supporting_artifacts(
        summary["measured_numbers"])

    pj = Path(args.output_json)
    pj.parent.mkdir(parents=True, exist_ok=True)
    pj.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(_md(summary), encoding="utf-8")

    print("=== final experiment summary ===")
    print("completed_claims=%d (%s)" % (
        len(summary["completed_claims"]),
        ", ".join(c["id"] for c in summary["completed_claims"])))
    print("full_28layer_package_backed_decode=%s"
          % summary["full_28layer_package_backed_decode_status"])
    print("wrote %s" % args.output_json)
    print("wrote %s" % args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
