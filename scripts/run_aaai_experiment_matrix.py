"""AAAI experiment matrix: plan / run / validate the generation experiments.

The AAAI comparison is ONLY: plaintext-GPU vs ours (A_rightmul, folded remote +
TDX/H800). No pure-TEE, no LoRA, no amulet_secure_R (a baseline slot is reserved
in ``BASELINES`` for a future Slalom-style / pure-TEE backend, but it is NOT run).

Models: Qwen2.5-7B-Instruct (main), Llama-7B (structure generalization), GPT-2
(correctness sanity). Datasets: IFEval, GSM8K, MT-Bench. Config: seq_len=1024,
max_new_tokens=512, EOS on, greedy, batch 1.

Run modes:
* ``--run-mode plan``         -- print the full command plan (no execution);
* ``--run-mode run``          -- execute generation for every (model,dataset,backend);
* ``--run-mode validate-only``-- run the validator for every (model,dataset).

All hosts/paths are placeholders/args; no passwords, no SSH keys in the plan.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("ifeval", "gsm8k", "mt_bench", "humaneval", "sensitive_prompt_1024",
            "longbench_1024_lite")
MODELS = {
    "qwen2.5-7b-instruct": {"role": "main", "model_arg": "Qwen2.5-7B-Instruct"},
    "llama-7b": {"role": "generalization", "model_arg": "Llama-7B"},
    "gpt2": {"role": "sanity", "model_arg": "gpt2"},
}
# AAAI backends actually run: plaintext baseline + ours (unstaged) + ours (staged).
# The pure-TEE / Slalom slot is reserved, NOT planned/run here.
BACKENDS = ("plaintext_local", "folded_remote_unstaged", "folded_remote_staged")
BASELINES_RESERVED = ("pure_tee_slalom_style",)   # future; never planned/run here


def _gen_cmd(*, dataset, backend, model_key, model_path, dataset_jsonl,
             gpu_worker_url, embedding_path, evidence, expected_mr_td,
             out_dir, paper_facing, sanity, staged_dir):
    out = Path(out_dir) / model_key / backend / dataset
    run_id = "%s_%s_%s" % (model_key, dataset, backend)
    is_folded = backend.startswith("folded_remote")
    is_staged = backend == "folded_remote_staged"
    cmd = [sys.executable, str(REPO_ROOT / "scripts"
                               / "run_aaai_generation_benchmark.py"),
           "--dataset", dataset, "--dataset-jsonl", dataset_jsonl,
           "--backend", ("folded_remote" if is_folded else "plaintext_local"),
           "--model-name", MODELS[model_key]["model_arg"],
           "--model-path", model_path or "<MODEL_PATH>",
           "--seq-len", "1024", "--max-new-tokens", "512",
           "--use-chat-template", "--run-id", run_id, "--resume",
           "--output-response-jsonl", str(out / "responses.jsonl"),
           "--output-report-json", str(out / "report.json"),
           "--status-json", str(out / "status.json"),
           "--heartbeat-json", str(out / "heartbeat.json")]
    if is_folded:
        cmd += ["--nonlinear-backend", "A_rightmul", "--require-real",
                "--tdx-boundary-client",
                "--gpu-worker-url", gpu_worker_url or "<H800_WORKER_URL>",
                "--embedding-path", embedding_path or "<EMBEDDING_ARTIFACT>",
                "--attestation-evidence-json", evidence or "<EVIDENCE_JSON>"]
        if expected_mr_td:
            cmd += ["--expected-mr-td", expected_mr_td]
        if is_staged:
            cmd += ["--use-gpu-staged-schedule", "--require-staged-schedule",
                    "--gpu-staged-schedule-dir", staged_dir or "<STAGED_DIR>"]
    else:
        cmd += ["--require-real"]
    # paper-facing only for the AAAI headline models (main + generalization),
    # not the GPT-2 sanity check.
    if paper_facing and not sanity:
        cmd += ["--paper-facing-aaai"]
    return run_id, str(out), cmd


def _val_cmd(*, dataset, model_key, out_dir, card, evidence, expected_mr_td):
    base = Path(out_dir) / model_key
    out = base / "validation"
    cmd = [sys.executable, str(REPO_ROOT / "scripts"
                               / "validate_aaai_generation_results.py"),
           "--dataset", dataset,
           "--plaintext-report",
           str(base / "plaintext_local" / dataset / "report.json"),
           "--plaintext-responses",
           str(base / "plaintext_local" / dataset / "responses.jsonl"),
           "--ours-report",
           str(base / "folded_remote_unstaged" / dataset / "report.json"),
           "--ours-responses",
           str(base / "folded_remote_unstaged" / dataset / "responses.jsonl"),
           "--run-id", "%s_%s" % (model_key, dataset),
           "--output-json", str(out / ("aaai_validation_%s.json" % dataset)),
           "--output-md", str(out / ("aaai_validation_%s.md" % dataset)),
           "--output-csv", str(out / ("aaai_tables_%s.csv" % dataset))]
    if card:
        cmd += ["--dataset-card", card]
    if evidence:
        cmd += ["--attestation-evidence-json", evidence]
    if expected_mr_td:
        cmd += ["--expected-mr-td", expected_mr_td]
    return str(out), cmd


def build_plan(args):
    models = args.models or list(MODELS)
    datasets = args.datasets or list(DATASETS)
    plan = {"stage": "aaai_experiment_matrix_plan",
            "comparison": ["plaintext_local", "folded_remote(A_rightmul,TDX,H800)"],
            "excluded": ["pure_tee", "lora", "amulet_secure_R"],
            "reserved_baseline_slot": list(BASELINES_RESERVED),
            "config": {"seq_len": 1024, "max_new_tokens": 512, "eos": True,
                       "decoding": "greedy", "batch_size": 1},
            "generation": [], "validation": []}
    for mk in models:
        sanity = MODELS[mk]["role"] == "sanity"
        mp = {"qwen2.5-7b-instruct": args.qwen_path, "llama-7b": args.llama_path,
              "gpt2": args.gpt2_path}.get(mk)
        for ds in datasets:
            ds_jsonl = str(Path(args.dataset_dir) / ("%s.jsonl" % ds))
            card = str(Path(args.dataset_dir) / "cards" / ("%s_card.json" % ds))
            for be in BACKENDS:
                is_folded = be.startswith("folded_remote")
                run_id, out, cmd = _gen_cmd(
                    dataset=ds, backend=be, model_key=mk, model_path=mp,
                    dataset_jsonl=ds_jsonl, gpu_worker_url=args.gpu_worker_url,
                    embedding_path=args.embedding_path,
                    evidence=args.attestation_evidence_json,
                    expected_mr_td=args.expected_mr_td, out_dir=args.output_dir,
                    paper_facing=args.paper_facing_aaai, sanity=sanity,
                    staged_dir=args.gpu_staged_schedule_dir)
                plan["generation"].append({
                    "run_id": run_id, "model": mk, "role": MODELS[mk]["role"],
                    "dataset": ds, "backend": be, "output_dir": out,
                    "needs_h800": is_folded, "needs_tdx": is_folded,
                    "needs_quote": is_folded,
                    "needs_staged_schedule": be == "folded_remote_staged",
                    "estimated_runtime": None, "command": cmd})
            vout, vcmd = _val_cmd(
                dataset=ds, model_key=mk, out_dir=args.output_dir, card=card,
                evidence=args.attestation_evidence_json,
                expected_mr_td=args.expected_mr_td)
            plan["validation"].append({
                "model": mk, "dataset": ds, "output_dir": vout, "command": vcmd})
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-mode", default="plan",
                    choices=["plan", "run", "validate-only"])
    ap.add_argument("--dataset-dir", required=True,
                    help="dir with <dataset>.jsonl + cards/<dataset>_card.json")
    ap.add_argument("--output-dir", default="outputs/aaai")
    ap.add_argument("--models", nargs="*", default=None,
                    choices=list(MODELS))
    ap.add_argument("--datasets", nargs="*", default=None, choices=list(DATASETS))
    ap.add_argument("--qwen-path", default=None)
    ap.add_argument("--llama-path", default=None)
    ap.add_argument("--gpt2-path", default=None)
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--embedding-path", default=None)
    ap.add_argument("--attestation-evidence-json", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--gpu-staged-schedule-dir", default=None,
                    help="staged-schedule dir for the folded_remote_staged backend")
    ap.add_argument("--paper-facing-aaai", action="store_true", default=False)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--continue-on-error", action="store_true", default=False)
    args = ap.parse_args()

    plan = build_plan(args)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(plan, indent=2),
                                          encoding="utf-8")

    if args.run_mode == "plan":
        print(json.dumps(plan, indent=2))
        return 0

    items = (plan["generation"] if args.run_mode == "run" else plan["validation"])
    rc_final = 0
    for it in items:
        print("\n=== %s ===" % it.get("run_id", it.get("dataset")), flush=True)
        print(" ".join(it["command"]), flush=True)
        rc = subprocess.call(it["command"])
        if rc != 0:
            rc_final = rc
            print("WARNING: command exited %d" % rc, file=sys.stderr)
            if not args.continue_on_error:
                return rc
    return rc_final


if __name__ == "__main__":
    raise SystemExit(main())
