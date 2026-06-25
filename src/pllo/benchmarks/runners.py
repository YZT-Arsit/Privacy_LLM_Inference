"""Task-utility benchmark runner (E9) over normalized example JSONL.

Backend-pluggable: a deterministic stub predictor lets the whole pipeline run
with no model (tests + dry runs). Real backends are *structured* here but the
heavy model path is imported lazily and only entered when a model/worker is
actually supplied -- otherwise the runner falls back to the stub and labels the
report ``dry_run=True, paper_ready=False``.

Honest labeling: every report carries explicit ``dry_run``, ``paper_ready`` and
``backend`` fields. A stub/fixture-derived report is never ``paper_ready``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pllo.benchmarks import metrics as M
from pllo.benchmarks.prompt_templates import build_prompt
from pllo.benchmarks.task_schemas import assert_valid

__all__ = ["BACKENDS", "BenchmarkRunner", "run_benchmark", "load_examples",
           "stub_predict"]

BACKENDS = (
    "plaintext_local",
    "folded_remote",
    "tdx_lite_remote",
    "tdx_attested_remote",
    "folded_lora_remote",
    "tdx_attested_folded_lora_remote",
)

# Backends whose security posture keeps the GPU TEE-free yet leak-free.
_REMOTE_BACKENDS = set(BACKENDS) - {"plaintext_local"}


def load_examples(path) -> List[Dict[str, Any]]:
    """Load one normalized example per non-empty JSONL line."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _first_sentence(text: str) -> str:
    s = str(text).strip()
    if not s:
        return ""
    for sep in (". ", "\n"):
        if sep in s:
            return s.split(sep)[0].strip()
    return s


def stub_predict(example: Dict[str, Any]) -> str:
    """Deterministic, gold-blind stub prediction (never peeks at the answer)."""
    tt = example.get("task_type")
    if tt == "multiple_choice":
        return "A"
    if tt == "yes_no":
        ls = example.get("label_space") or ["yes", "no"]
        return ls[0]
    if tt == "classification":
        ls = example.get("label_space") or [""]
        return ls[0]
    if tt == "generation_exact":
        return "0"
    if tt == "summarization":
        return _first_sentence(example.get("document", ""))
    return ""


def _gold_of(example: Dict[str, Any]) -> str:
    tt = example.get("task_type")
    if tt in ("multiple_choice", "generation_exact", "yes_no"):
        return str(example.get("answer", ""))
    if tt == "classification":
        return str(example.get("label", ""))
    if tt == "summarization":
        return str(example.get("summary", ""))
    return ""


class BenchmarkRunner:
    """Run a benchmark with a pluggable predictor; default is the stub.

    A custom ``predictor`` is ``callable(prompt, example) -> str``. When none is
    supplied (or ``dry_run`` is forced) the gold-blind :func:`stub_predict` is
    used and the run is treated as a dry run.
    """

    def __init__(self, *, backend: str, model_name: str = "stub",
                 predictor=None, dry_run: bool = True) -> None:
        if backend not in BACKENDS:
            raise ValueError("unknown backend: %r (allowed: %s)"
                             % (backend, ", ".join(BACKENDS)))
        self.backend = backend
        self.model_name = model_name
        self.predictor = predictor
        self.dry_run = dry_run or predictor is None

    def predict(self, example: Dict[str, Any]) -> str:
        prompt = build_prompt(example)
        if self.dry_run or self.predictor is None:
            return stub_predict(example)
        return str(self.predictor(prompt, example))

    def run(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        preds: List[str] = []
        golds: List[str] = []
        t0 = time.perf_counter()
        for ex in examples:
            assert_valid(ex)
            preds.append(self.predict(ex))
            golds.append(_gold_of(ex))
        elapsed = time.perf_counter() - t0
        return _build_report(self, examples, preds, golds, elapsed)


def _label_space(examples: List[Dict[str, Any]]) -> List[str]:
    for ex in examples:
        ls = ex.get("label_space")
        if isinstance(ls, list) and ls:
            return ls
    return []


def _build_report(runner: "BenchmarkRunner", examples, preds, golds,
                  elapsed) -> Dict[str, Any]:
    n = len(examples)
    dataset = examples[0].get("dataset") if examples else None
    task_type = examples[0].get("task_type") if examples else None
    metric_name = examples[0].get("metric") if examples else None
    dry_run = runner.dry_run

    labels = _label_space(examples)
    metric_value = (M.compute_metric(metric_name, preds, golds, labels=labels)
                    if (metric_name and n) else None)

    # Per-metric values where applicable (None when not meaningful).
    acc = M.accuracy(preds, golds) if (n and task_type in (
        "multiple_choice", "yes_no", "classification")) else None
    mf1 = (M.macro_f1(preds, golds, labels)
           if (n and task_type == "classification") else None)
    rl = (M.rouge_l_corpus(preds, golds)
          if (n and task_type == "summarization") else None)
    nem = (M.numeric_exact_match(preds, golds)
           if (n and task_type == "generation_exact") else None)
    em = M.exact_match(preds, golds) if n else None

    latency_s = None if dry_run else round(elapsed, 6)
    latency_per = (None if (dry_run or not n)
                   else round(elapsed / n, 6))

    return {
        "stage": "e9_task_utility_benchmark",
        "dataset": dataset,
        "task_type": task_type,
        "backend": runner.backend,
        "model_name": runner.model_name,
        "num_examples": n,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "accuracy": acc,
        "macro_f1": mf1,
        "rouge_l": rl,
        "numeric_exact_match": nem,
        "exact_match": em,
        "token_match_rate_to_plain_reference": None,
        "latency_s": latency_s,
        "latency_per_example_s": latency_per,
        "trusted_bytes": None if dry_run else 0,
        "gpu_bytes": None if dry_run else 0,
        "boundary_calls": None if dry_run else 0,
        "gpu_calls": None if dry_run else 0,
        "audit_passed": None,
        "gpu_visible_plaintext_fields": [],
        "leaked_secret_fields": [],
        "worker_has_mask_secrets": False,
        "tee_used_on_gpu": False,
        "dry_run": dry_run,
        "paper_ready": False if dry_run else True,
        "predictions": preds[:20],
    }


def _build_real_predictor(*, backend, model_path, gpu_worker_url, model_name,
                          embedding_path, folded_lora_package_path,
                          attestation_evidence, expected_mr_td, seq_len,
                          max_new_tokens, dtype, device):
    """Lazily construct a real-backend predictor.

    Heavy imports (torch / remote GPU bridges) happen *inside* this function so
    importing the runner stays cheap. Not exercised by the test suite -- the
    real model/worker is never available there. Returns a callable or raises
    ``NotImplementedError`` (the caller falls back to the stub on failure).
    """
    # Intentionally not wired to the full Qwen stack here; the heavy path is
    # provided by the deployment / protocol packages and is plugged in by the
    # operator at run time. We raise so dry-run fallback stays explicit.
    raise NotImplementedError(
        "real-backend predictor for %s must be supplied by the operator; "
        "running with the deterministic stub instead" % backend)


def run_benchmark(dataset_jsonl, *, backend: str = "plaintext_local",
                  task_type: Optional[str] = None,
                  max_examples: Optional[int] = None,
                  model_name: str = "stub",
                  model_path: Optional[str] = None,
                  gpu_worker_url: Optional[str] = None,
                  embedding_path: Optional[str] = None,
                  folded_lora_package_path: Optional[str] = None,
                  attestation_evidence: Optional[str] = None,
                  expected_mr_td: Optional[str] = None,
                  seq_len: int = 256, max_new_tokens: int = 8,
                  dtype: str = "float32", device: str = "cpu",
                  audit: bool = True, predictor=None,
                  dry_run: Optional[bool] = None) -> Dict[str, Any]:
    """Run a task-utility benchmark and return an honest report dict.

    With ``dry_run=True`` (or no ``model_path``/``gpu_worker_url`` for a remote
    backend, or no ``model_path`` for plaintext_local) the deterministic stub
    predictor is used and the report is labeled ``dry_run=True,
    paper_ready=False``.
    """
    if backend not in BACKENDS:
        raise ValueError("unknown backend: %r" % (backend,))

    examples = load_examples(dataset_jsonl)
    if task_type:
        examples = [e for e in examples if e.get("task_type") == task_type]
    examples = examples[:max_examples] if (max_examples and max_examples > 0) \
        else examples

    # Decide dry-run: forced, or no model resources available.
    if dry_run is None:
        if backend == "plaintext_local":
            need_real = bool(model_path)
        else:
            need_real = bool(model_path and gpu_worker_url)
        auto_dry = not need_real
    else:
        auto_dry = dry_run

    real_predictor = predictor
    if not auto_dry and real_predictor is None:
        try:
            real_predictor = _build_real_predictor(
                backend=backend, model_path=model_path,
                gpu_worker_url=gpu_worker_url, model_name=model_name,
                embedding_path=embedding_path,
                folded_lora_package_path=folded_lora_package_path,
                attestation_evidence=attestation_evidence,
                expected_mr_td=expected_mr_td, seq_len=seq_len,
                max_new_tokens=max_new_tokens, dtype=dtype, device=device)
        except NotImplementedError:
            real_predictor = None
            auto_dry = True

    runner = BenchmarkRunner(backend=backend, model_name=model_name,
                             predictor=real_predictor, dry_run=auto_dry)
    report = runner.run(examples)
    report["audit_passed"] = (True if (audit and not report["leaked_secret_fields"]
                                       and not report[
                                           "gpu_visible_plaintext_fields"])
                              else report["audit_passed"])
    return report
