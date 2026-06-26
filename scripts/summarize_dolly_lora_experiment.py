"""Final paper summary for the Dolly-15k private-LoRA experiment.

Joins the five stage JSONs (training, base plaintext gen, LoRA plaintext gen,
Dolly evaluator, folded E6 pipeline) into a single paper-facing summary:
JSON + markdown + CSV. Pure aggregation (import-safe, unit-tested) -- no model.

Example::

    python scripts/summarize_dolly_lora_experiment.py \\
        --training-json   outputs/lora_dolly/train_qwen7b_lora_dolly.json \\
        --base-gen-json   outputs/lora_dolly/base_plaintext_report.json \\
        --lora-gen-json   outputs/lora_dolly/lora_plaintext_report.json \\
        --eval-json       outputs/lora_dolly/dolly_eval.json \\
        --folded-json     outputs/lora_dolly/dolly_lora_e6.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:                                        # noqa: BLE001
        return {}


def build_summary(train, base_gen, lora_gen, ev, folded):
    """Assemble the final summary dict from the five stage reports (each a dict;
    missing fields default to None / safe values)."""
    b = ev.get("base", {}) if isinstance(ev, dict) else {}
    l = ev.get("lora", {}) if isinstance(ev, dict) else {}
    paper_ready = bool(
        train.get("paper_ready") and folded.get("lora_package_valid")
        and not folded.get("contains_raw_lora", True)
        and not folded.get("worker_has_raw_lora", True)
        and not folded.get("contains_mask_secrets", True)
        and folded.get("audit_passed") is not False)
    return {
        "stage": "dolly_lora_final_summary",
        "dataset": "databricks-dolly-15k",
        "model": "Qwen2.5-7B-Instruct",
        # sizes
        "train_size": train.get("num_train_examples"),
        "valid_size": train.get("num_valid_examples"),
        "test_size": ev.get("num_examples"),
        # lora config
        "lora_rank": train.get("lora_rank"),
        "lora_alpha": train.get("lora_alpha"),
        "lora_dropout": train.get("lora_dropout"),
        "target_modules": train.get("target_modules"),
        "trainable_param_ratio": train.get("trainable_param_ratio"),
        "adapter_param_count": train.get("adapter_param_count"),
        # training
        "train_loss_final": train.get("train_loss_final"),
        "valid_loss_final": train.get("valid_loss_final"),
        "train_runtime_s": train.get("train_runtime_s"),
        # generation stats (base vs lora)
        "base_avg_words": b.get("avg_words"),
        "lora_avg_words": l.get("avg_words"),
        "response_length_delta_lora_minus_base":
            ev.get("response_length_delta_lora_minus_base"),
        "lora_unigram_f1_avg": l.get("unigram_f1_avg"),
        "lora_rouge_l_like_lcs_avg": l.get("rouge_l_like_lcs_avg"),
        "base_unigram_f1_avg": b.get("unigram_f1_avg"),
        "lora_empty_response_count": l.get("empty_response_count"),
        "lora_contains_human_marker_count": l.get("contains_human_marker_count"),
        "base_gen_latency_s": base_gen.get("online_generation_latency_s"),
        "lora_gen_latency_s": lora_gen.get("online_generation_latency_s"),
        "base_prompt_format": base_gen.get("prompt_format"),
        "lora_prompt_format": lora_gen.get("prompt_format"),
        # folded package validity + SECURITY
        "folded_package_valid": folded.get("lora_package_valid"),
        "contains_raw_lora": folded.get("contains_raw_lora"),
        "contains_optimizer_state": folded.get("contains_optimizer_state"),
        "contains_training_data": folded.get("contains_training_data"),
        "contains_mask_secrets": folded.get("contains_mask_secrets"),
        "worker_has_raw_lora": folded.get("worker_has_raw_lora"),
        "worker_has_mask_secrets": folded.get("worker_has_mask_secrets"),
        "tee_used_on_gpu": folded.get("tee_used_on_gpu"),
        "gpu_visible_plaintext_fields": folded.get("gpu_visible_plaintext_fields"),
        "leaked_secret_fields": folded.get("leaked_secret_fields"),
        "audit_passed": folded.get("audit_passed"),
        # folded correctness + cost
        "local_allclose": folded.get("local_allclose"),
        "local_max_abs_error": folded.get("local_max_abs_error"),
        "tokens_exact_match": folded.get("tokens_exact_match"),
        "token_match_rate": folded.get("token_match_rate"),
        "latency_s": folded.get("latency_s"),
        "trusted_bytes": folded.get("trusted_bytes"),
        "gpu_bytes": folded.get("gpu_bytes"),
        "boundary_calls": folded.get("boundary_calls"),
        "peak_gpu_memory_mb": folded.get("peak_gpu_memory_mb"),
        "paper_ready": paper_ready,
    }


_SECTIONS = [
    ("dataset / model", ["dataset", "model", "train_size", "valid_size",
                         "test_size"]),
    ("lora", ["lora_rank", "lora_alpha", "lora_dropout", "target_modules",
              "trainable_param_ratio", "adapter_param_count", "train_loss_final",
              "valid_loss_final", "train_runtime_s"]),
    ("generation", ["base_avg_words", "lora_avg_words",
                    "response_length_delta_lora_minus_base", "base_unigram_f1_avg",
                    "lora_unigram_f1_avg", "lora_rouge_l_like_lcs_avg",
                    "lora_empty_response_count", "lora_contains_human_marker_count",
                    "base_gen_latency_s", "lora_gen_latency_s",
                    "base_prompt_format", "lora_prompt_format"]),
    ("folded package security", ["folded_package_valid", "contains_raw_lora",
                                 "contains_optimizer_state", "contains_training_data",
                                 "contains_mask_secrets", "worker_has_raw_lora",
                                 "worker_has_mask_secrets", "tee_used_on_gpu",
                                 "gpu_visible_plaintext_fields",
                                 "leaked_secret_fields", "audit_passed"]),
    ("folded correctness / cost", ["local_allclose", "local_max_abs_error",
                                   "tokens_exact_match", "token_match_rate",
                                   "latency_s", "trusted_bytes", "gpu_bytes",
                                   "boundary_calls", "peak_gpu_memory_mb",
                                   "paper_ready"]),
]


def _markdown(s):
    L = ["# Dolly-15k private-LoRA experiment summary", ""]
    for title, keys in _SECTIONS:
        L += ["## %s" % title, "", "| field | value |", "|---|---|"]
        L += ["| %s | %s |" % (k, s.get(k)) for k in keys]
        L += [""]
    return "\n".join(L) + "\n"


def _write_csv(path, s):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field", "value"])
        for _title, keys in _SECTIONS:
            for k in keys:
                w.writerow([k, s.get(k)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-json", default=None)
    ap.add_argument("--base-gen-json", default=None)
    ap.add_argument("--lora-gen-json", default=None)
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--folded-json", default=None)
    ap.add_argument("--out-dir", default="outputs/lora_dolly")
    args = ap.parse_args()

    s = build_summary(_load(args.training_json), _load(args.base_gen_json),
                      _load(args.lora_gen_json), _load(args.eval_json),
                      _load(args.folded_json))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dolly_lora_final_summary.json").write_text(
        json.dumps(s, indent=2, default=str), encoding="utf-8")
    (out / "dolly_lora_final_summary.md").write_text(_markdown(s),
                                                     encoding="utf-8")
    _write_csv(out / "dolly_lora_final_summary.csv", s)

    print("=== Dolly LoRA final summary ===")
    print("train=%s valid=%s test=%s rank=%s train_loss=%s valid_loss=%s"
          % (s["train_size"], s["valid_size"], s["test_size"], s["lora_rank"],
             s["train_loss_final"], s["valid_loss_final"]))
    print("folded_package_valid=%s contains_raw_lora=%s worker_has_raw_lora=%s "
          "audit_passed=%s paper_ready=%s"
          % (s["folded_package_valid"], s["contains_raw_lora"],
             s["worker_has_raw_lora"], s["audit_passed"], s["paper_ready"]))
    print("summary -> %s" % (out / "dolly_lora_final_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
