"""Stage 7.3 -- repository hygiene, output-size guard, and evidence manifest.

This stage adds **no** masking functionality and **no** security claim. It
makes the repository safer, smaller, and easier to reproduce / present:

* :func:`run_repo_hygiene_audit` -- read-only audit of tracked / untracked
  generated artifacts (never deletes anything);
* :func:`ensure_gitignore_entries` -- idempotently append ignore patterns
  (never removes tracked files, never duplicates entries);
* :func:`check_output_sizes` -- warn / fail on oversized generated reports
  (excludes local model checkpoints; never deletes);
* :func:`generate_evidence_manifest` -- a compact, paper-ready manifest of the
  Stage 6.4 -> 7.6 evidence (no tensor dumps, no occurrence lists).

CPU-only. No CUDA, transformers, network, or real checkpoints. ``git`` is used
when available and degrades gracefully when it is not.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "RECOMMENDED_GITIGNORE_ENTRIES",
    "RepoHygieneConfig",
    "check_output_sizes",
    "ensure_gitignore_entries",
    "evidence_manifest_markdown",
    "generate_evidence_manifest",
    "render_audit_markdown",
    "run_repo_hygiene_audit",
]


# ---------------------------------------------------------------------------
# Config + constants
# ---------------------------------------------------------------------------


@dataclass
class RepoHygieneConfig:
    repo_root: str = "."
    output_dir: str = "outputs"
    warn_mb: int = 10
    fail_mb: int = 100
    include_git_tracked_scan: bool = True
    # Cap listed paths so the audit report can never balloon.
    max_listed_paths: int = 500


RECOMMENDED_GITIGNORE_ENTRIES: tuple[str, ...] = (
    "__pycache__/",
    "*.py[cod]",
    "*.pyo",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".coverage",
    "htmlcov/",
    "build/",
    "dist/",
    "*.egg-info/",
    "outputs/paper_artifacts/",
    "outputs/paper_sections/",
    "outputs/*claims_consistency*.json",
    "outputs/*claims_consistency*.csv",
    "outputs/*claims_consistency*.md",
    "outputs/*probe*.json",
    "outputs/*probe*.md",
    "outputs/*cost*.json",
    "outputs/*cost*.md",
    "outputs/*cost*.csv",
)

GITIGNORE_HEADER = "# Generated artifacts and local caches"

# Local model checkpoint extensions -- excluded from the output-size guard.
_CHECKPOINT_SUFFIXES: tuple[str, ...] = (
    ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".gguf", ".onnx", ".h5",
    ".msgpack", ".tflite",
)
_CHECKPOINT_DIR_NAMES: frozenset[str] = frozenset(
    {"models", "checkpoints", "model", "checkpoint", "hf_cache",
     "modelscope", ".cache"})


# ---------------------------------------------------------------------------
# Generated-artifact classification (pure path heuristics)
# ---------------------------------------------------------------------------


def _is_generated_path(rel_path: str) -> bool:
    """True if ``rel_path`` looks like a regenerable build/cache/output."""
    p = rel_path.replace("\\", "/")
    parts = p.split("/")
    if "__pycache__" in parts:
        return True
    if any(seg.endswith(".egg-info") for seg in parts):
        return True
    if p.endswith((".pyc", ".pyo", ".pyd")):
        return True
    for cache_dir in (".pytest_cache", ".ruff_cache", ".mypy_cache",
                      "htmlcov", "build", "dist"):
        if cache_dir in parts:
            return True
    if p == ".coverage" or p.endswith("/.coverage"):
        return True
    if parts[0] == "outputs" and p.endswith((".json", ".md", ".csv")):
        return True
    return False


# ---------------------------------------------------------------------------
# git helpers (degrade gracefully)
# ---------------------------------------------------------------------------


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(repo_root), capture_output=True,
            text=True, check=True, timeout=30)
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _git_tracked_files(repo_root: Path) -> list[str] | None:
    out = _run_git(repo_root, ["ls-files"])
    if out is None:
        return None
    return [ln for ln in out.splitlines() if ln]


def _git_untracked_files(repo_root: Path) -> list[str] | None:
    # --others = untracked; --exclude-standard honours .gitignore.
    out = _run_git(repo_root,
                   ["ls-files", "--others", "--exclude-standard"])
    if out is None:
        return None
    return [ln for ln in out.splitlines() if ln]


def _fs_scan_relpaths(repo_root: Path) -> list[str]:
    """Filesystem fallback when git is unavailable: list files, skipping the
    ``.git`` directory."""
    rels: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in filenames:
            full = Path(dirpath) / fn
            rels.append(str(full.relative_to(repo_root)).replace("\\", "/"))
    return rels


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _cap(paths: list[str], limit: int) -> tuple[list[str], bool]:
    return paths[:limit], len(paths) > limit


def run_repo_hygiene_audit(config: RepoHygieneConfig) -> dict[str, Any]:
    """Read-only audit of generated artifacts. Never deletes anything."""
    repo_root = Path(config.repo_root).resolve()
    limit = config.max_listed_paths

    tracked = _git_tracked_files(repo_root) if config.include_git_tracked_scan \
        else []
    git_available = tracked is not None
    if tracked is None:
        tracked = []
    untracked = _git_untracked_files(repo_root)
    if untracked is None:
        # Filesystem fallback: treat generated files not obviously source as
        # untracked candidates (best-effort, no git).
        untracked = [p for p in _fs_scan_relpaths(repo_root)
                     if _is_generated_path(p)]

    tracked_generated = sorted(p for p in tracked if _is_generated_path(p))
    untracked_generated = sorted(p for p in untracked if _is_generated_path(p))

    # Large outputs (> warn_mb) regardless of tracking.
    size_report = check_output_sizes(
        os.path.join(str(repo_root), config.output_dir),
        warn_mb=config.warn_mb, fail_mb=config.fail_mb)

    # Safe cleanup: ONLY untracked generated artifacts.
    untracked_dirs: list[str] = []
    untracked_files: list[str] = []
    seen_dirs: set[str] = set()
    for p in untracked_generated:
        parts = p.split("/")
        cache_dir = next(
            (seg for seg in parts
             if seg in {"__pycache__", ".pytest_cache", ".ruff_cache",
                        ".mypy_cache", "htmlcov", "build", "dist"}
             or seg.endswith(".egg-info")), None)
        if cache_dir is not None:
            prefix = p[: p.index(cache_dir) + len(cache_dir)]
            if prefix not in seen_dirs:
                seen_dirs.add(prefix)
                untracked_dirs.append(prefix)
        else:
            untracked_files.append(p)
    safe_cleanup_commands = [f"rm -rf {d}" for d in untracked_dirs] + [
        f"rm -f {f}" for f in untracked_files]

    tracked_capped, tracked_trunc = _cap(tracked_generated, limit)
    untracked_capped, untracked_trunc = _cap(untracked_generated, limit)
    cleanup_capped, cleanup_trunc = _cap(safe_cleanup_commands, limit)

    return {
        "stage": "7.3_repo_hygiene",
        "status": "ok",
        "git_available": git_available,
        "repo_root": str(repo_root),
        "summary": {
            "tracked_generated_count": len(tracked_generated),
            "untracked_generated_count": len(untracked_generated),
            "large_output_warnings": len(size_report["warnings"]),
            "large_output_failures": len(size_report["failures"]),
            "outputs_total_size_mb": size_report["total_size_mb"],
        },
        "tracked_generated_candidates": tracked_capped,
        "tracked_generated_candidates_truncated": tracked_trunc,
        "untracked_generated_candidates": untracked_capped,
        "untracked_generated_candidates_truncated": untracked_trunc,
        "large_outputs": size_report["warnings"],
        "recommended_gitignore_entries": list(RECOMMENDED_GITIGNORE_ENTRIES),
        "safe_cleanup_commands": cleanup_capped,
        "safe_cleanup_commands_truncated": cleanup_trunc,
        "safe_to_delete_untracked_paths": (untracked_dirs + untracked_files)[
            :limit],
        "manual_decision_needed": {
            "note": (
                "Tracked generated artifacts are listed here for a MANUAL "
                "decision. This stage never deletes tracked files. To stop "
                "tracking without deleting working copies: "
                "`git rm -r --cached <path>` then commit."),
            "tracked_generated_candidates": tracked_capped,
            "count": len(tracked_generated),
        },
        "caveats": [
            "Read-only audit: nothing is deleted by this function.",
            "safe_cleanup_commands target only UNtracked generated artifacts.",
            "Tracked artifacts require a manual `git rm --cached` decision.",
        ],
    }


def render_audit_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    s = report["summary"]
    w("# Stage 7.3 — Repository Hygiene Audit")
    w()
    w(f"- git available: **{report['git_available']}**")
    w(f"- tracked generated candidates: **{s['tracked_generated_count']}**")
    w(f"- untracked generated candidates: "
      f"**{s['untracked_generated_count']}**")
    w(f"- outputs total size: **{s['outputs_total_size_mb']} MB** | "
      f"warnings: **{s['large_output_warnings']}** | "
      f"failures: **{s['large_output_failures']}**")
    w()
    w("> Read-only audit. Nothing is deleted. Tracked artifacts require a "
      "manual `git rm --cached` decision.")
    w()
    w("## Recommended .gitignore entries")
    w()
    for e in report["recommended_gitignore_entries"]:
        w(f"- `{e}`")
    w()
    w("## Safe cleanup (untracked generated only)")
    w()
    if not report["safe_cleanup_commands"]:
        w("(none)")
    else:
        for cmd in report["safe_cleanup_commands"]:
            w(f"- `{cmd}`")
        if report.get("safe_cleanup_commands_truncated"):
            w("- _(list truncated)_")
    w()
    w("## Manual decision needed (tracked generated artifacts)")
    w()
    md = report["manual_decision_needed"]
    w(f"{md['note']}")
    w()
    w(f"- tracked generated count: **{md['count']}**")
    if report.get("tracked_generated_candidates_truncated"):
        w("- _(sample truncated below)_")
    w()
    sample = report["tracked_generated_candidates"][:20]
    for p in sample:
        w(f"  - `{p}`")
    if md["count"] > len(sample):
        w(f"  - … and {md['count'] - len(sample)} more")
    w()
    w("## Large outputs (> warn threshold)")
    w()
    if not report["large_outputs"]:
        w("(none)")
    else:
        w("| file | size_mb |")
        w("|---|---|")
        for item in report["large_outputs"]:
            w(f"| `{item['path']}` | {item['size_mb']} |")
    w()
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# .gitignore helper
# ---------------------------------------------------------------------------


def ensure_gitignore_entries(
    repo_root: str | os.PathLike[str], entries: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Idempotently append missing ignore ``entries`` under a header.

    Creates ``.gitignore`` if missing. Never removes anything; never
    duplicates an entry that is already present (anywhere in the file).
    """
    path = Path(repo_root) / ".gitignore"
    if path.exists():
        existing_text = path.read_text(encoding="utf-8")
        existing_lines = [ln.strip() for ln in existing_text.splitlines()]
    else:
        existing_text = ""
        existing_lines = []
    present = {ln for ln in existing_lines if ln and not ln.startswith("#")}

    added: list[str] = []
    for e in entries:
        e = e.strip()
        if e and e not in present:
            added.append(e)
            present.add(e)  # guard against duplicates within `entries`

    already_present = [e.strip() for e in entries
                       if e.strip() and e.strip() not in added]

    if added:
        chunk_lines: list[str] = []
        if existing_text and not existing_text.endswith("\n"):
            chunk_lines.append("")
        if existing_text.strip():
            chunk_lines.append("")
        chunk_lines.append(GITIGNORE_HEADER)
        chunk_lines.extend(added)
        new_text = existing_text + "\n".join(chunk_lines) + "\n"
        path.write_text(new_text, encoding="utf-8")

    return {
        "gitignore_path": str(path),
        "added_entries": added,
        "already_present_entries": already_present,
        "created": not existing_text,
    }


# ---------------------------------------------------------------------------
# Output-size guard
# ---------------------------------------------------------------------------


def _is_checkpoint(path: Path) -> bool:
    if path.suffix.lower() in _CHECKPOINT_SUFFIXES:
        return True
    return any(part in _CHECKPOINT_DIR_NAMES for part in path.parts)


def check_output_sizes(
    output_dir: str | os.PathLike[str], warn_mb: int = 10, fail_mb: int = 100,
) -> dict[str, Any]:
    """Recursively size-check generated reports under ``output_dir``.

    Excludes local model checkpoints. Never deletes. Returns warnings (files
    > ``warn_mb``), failures (files > ``fail_mb``), the largest file, and the
    total scanned size in MB.
    """
    root = Path(output_dir)
    mb = 1024 * 1024
    warnings: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_bytes = 0
    max_file: dict[str, Any] | None = None

    if root.is_dir():
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                fpath = Path(dirpath) / fn
                if fpath.is_symlink() or not fpath.is_file():
                    continue
                if _is_checkpoint(fpath):
                    continue
                try:
                    size = fpath.stat().st_size
                except OSError:
                    continue
                total_bytes += size
                rel = str(fpath)
                size_mb = round(size / mb, 4)
                entry = {"path": rel, "size_mb": size_mb}
                if max_file is None or size > max_file["_bytes"]:
                    max_file = {**entry, "_bytes": size}
                if size > fail_mb * mb:
                    failures.append(entry)
                elif size > warn_mb * mb:
                    warnings.append(entry)

    if max_file is not None:
        max_file = {k: v for k, v in max_file.items() if k != "_bytes"}

    return {
        "output_dir": str(root),
        "warn_mb": warn_mb,
        "fail_mb": fail_mb,
        "warnings": sorted(warnings, key=lambda d: -d["size_mb"]),
        "failures": sorted(failures, key=lambda d: -d["size_mb"]),
        "max_file": max_file,
        "total_size_mb": round(total_bytes / mb, 4),
        "passed": len(failures) == 0,
    }


# ---------------------------------------------------------------------------
# Evidence manifest
# ---------------------------------------------------------------------------


_GLOBAL_CAVEATS: tuple[str, ...] = (
    "No semantic, cryptographic, or formal security is claimed.",
    "Attention scores / probabilities remain GPU-visible.",
    "Vocab permutation+scaling is weaker than dense vocab masking.",
    "Real production full-model inference is not validated "
    "(extracted-weight reference, tiny/random models, greedy decode).",
)


def _stage_entries() -> list[dict[str, Any]]:
    return [
        {
            "stage": "6.4",
            "purpose": "RoPE-compatible masked attention and KV cache.",
            "key_files": ["src/pllo/ops/rope.py",
                          "src/pllo/ops/gqa_attention.py"],
            "key_tests": ["tests/test_rope_gqa_attention.py"],
            "commands": ["pytest tests/test_rope_gqa_attention.py -q"],
            "latest_known_result":
                "masked attention/KV invariants hold at float64 precision.",
            "requires_transformers": False,
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": ["attention scores remain GPU-visible."],
        },
        {
            "stage": "6.4.1",
            "purpose":
                "Pairwise complex-scaling RoPE masks and leakage proxy.",
            "key_files": ["src/pllo/ops/rope.py",
                          "src/pllo/ops/gqa_attention.py",
                          "src/pllo/experiments/rope_gqa_probe.py"],
            "key_tests": ["tests/test_rope_gqa_attention.py"],
            "commands": ["python scripts/run_rope_gqa_probe.py"],
            "latest_known_result":
                "complex-scaling default; rotation baseline retained; "
                "leakage proxy reduces cross-session linkability.",
            "requires_transformers": False,
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": [
                "RoPE/nonlinear-island masks weaker than dense masks."],
        },
        {
            "stage": "6.5",
            "purpose": "Synthetic LLaMA/Qwen-like decoder block.",
            "key_files": ["src/pllo/ops/llama_synthetic_block.py"],
            "key_tests": ["tests/test_llama_synthetic_block.py"],
            "commands": ["python scripts/run_llama_synthetic_block_probe.py"],
            "latest_known_result":
                "masked block == plain @ N at float64; KV-cache verified.",
            "requires_transformers": False,
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": ["synthetic weights only."],
        },
        {
            "stage": "6.6",
            "purpose": "HF single-decoder-layer adapter (extracted weights).",
            "key_files": ["src/pllo/hf_wrappers/llama_qwen_single_block.py"],
            "key_tests": ["tests/test_hf_single_block_wrapper.py"],
            "commands": ["python scripts/run_hf_single_block_probe.py"],
            "latest_known_result":
                "masked-vs-plain invariant holds for extracted LLaMA/Qwen2 "
                "layer weights (bias-aware).",
            "requires_transformers": "optional; skip cleanly if unavailable",
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": ["single decoder layer; no full model."],
        },
        {
            "stage": "6.7",
            "purpose":
                "Trusted embedding boundary, masked logits boundary, "
                "trusted sampling.",
            "key_files": ["src/pllo/ops/causal_lm_boundaries.py"],
            "key_tests": ["tests/test_causal_lm_boundaries.py"],
            "commands": ["python scripts/run_causal_lm_boundary_probe.py"],
            "latest_known_result":
                "GPU sees masked embeddings + masked logits only; TEE recovers "
                "logits and samples at float64 precision.",
            "requires_transformers": False,
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": [
                "vocab permutation+scaling weaker than dense vocab masking."],
        },
        {
            "stage": "6.8",
            "purpose": "Multi-layer masked CausalLM skeleton.",
            "key_files": ["src/pllo/ops/masked_causal_lm_skeleton.py"],
            "key_tests": ["tests/test_masked_causal_lm_skeleton.py"],
            "commands":
                ["python scripts/run_masked_causal_lm_skeleton_probe.py"],
            "latest_known_result":
                "per-layer residual masks N_0..N_L with honest handoff; "
                "greedy token match 1.0.",
            "requires_transformers": False,
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": [
                "per-layer handoff skip term needs one [H,H] GEMM "
                "(not zero-cost)."],
        },
        {
            "stage": "6.9",
            "purpose":
                "HF-style full CausalLM skeleton with random tiny "
                "LLaMA/Qwen2 (extracted-weight reference).",
            "key_files": ["src/pllo/hf_wrappers/hf_causal_lm_skeleton.py",
                          "src/pllo/experiments/hf_causal_lm_skeleton_probe.py"],
            "key_tests": ["tests/test_hf_causal_lm_skeleton.py"],
            "commands": [
                "python scripts/run_hf_causal_lm_skeleton_probe.py "
                "--model-family llama --max-layers 1 --prefill-seq-len 3 "
                "--decode-steps 1 --max-vocab-size 128 "
                "--output outputs/hf_causal_lm_skeleton_probe_llama.json"],
            "latest_known_result":
                "LLaMA + Qwen2 tiny random models: prefill allclose, decode "
                "token-match rate 1.0 (errors ~1e-16).",
            "requires_transformers": "optional; skip cleanly if unavailable",
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": [
                "tiny/random model by default; no tokenizer; greedy only; "
                "no production inference."],
        },
        {
            "stage": "7.0",
            "purpose": "Full-pipeline cost / leakage / ablation evaluation.",
            "key_files":
                ["src/pllo/experiments/full_pipeline_cost_leakage.py"],
            "key_tests": ["tests/test_full_pipeline_cost_leakage.py"],
            "commands": ["python scripts/run_full_pipeline_cost_leakage.py"],
            "latest_known_result":
                "analytical cost + leakage surfaces for 7 variants; no new "
                "masking.",
            "requires_transformers": False,
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": [
                "analytical cost model, not a hardware benchmark."],
        },
        {
            "stage": "7.1",
            "purpose": "Paper artifact generation (theorems / tables / audit).",
            "key_files": ["src/pllo/experiments/paper_artifact_generator.py"],
            "key_tests": ["tests/test_paper_artifact_generator.py"],
            "commands": ["python scripts/generate_paper_artifacts.py"],
            "latest_known_result":
                "artifacts generated; forbidden phrasing confined to a marked "
                "unsafe-claims section.",
            "requires_transformers": False,
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": [
                "artifacts summarize verified correctness/cost/leakage only; "
                "not a security proof."],
        },
        {
            "stage": "7.6_scanner_fix",
            "purpose":
                "Claims-scanner consistency and bounded report writer "
                "(no multi-GB reports).",
            "key_files": [
                "src/pllo/experiments/stage_7_6_claims_consistency.py"],
            "key_tests": ["tests/test_stage_7_6_claims_report.py",
                          "tests/test_lora_training_inference_lifecycle.py"],
            "commands": [
                "python scripts/run_stage_7_6_claims_consistency.py "
                "--output-dir outputs"],
            "latest_known_result":
                "compact reports by default; full occurrence dumps disabled "
                "unless --write-full-occurrences; hard size guard.",
            "requires_transformers": False,
            "requires_gpu": False,
            "writes_large_outputs": False,
            "limitations": [
                "lexical scan only; full occurrence dumps disabled by "
                "default to keep reports bounded."],
        },
    ]


def generate_evidence_manifest(output_dir: str = "outputs") -> dict[str, Any]:
    """Build + write the compact paper-ready evidence manifest.

    Writes ``evidence_manifest.json`` and ``evidence_manifest.md`` under
    ``output_dir``. No tensor dumps, no occurrence lists.
    """
    stages = _stage_entries()
    manifest = {
        "stage": "7.3_evidence_manifest",
        "title": "Privacy LLM Inference — Evidence Manifest",
        "description":
            "Compact, paper-ready summary of verified correctness, cost, and "
            "leakage-accounting evidence across Stages 6.4–7.6. No tensor "
            "dumps or occurrence lists.",
        "stages": stages,
        "stage_count": len(stages),
        "global_caveats": list(_GLOBAL_CAVEATS),
        "reproducibility": {
            "lightweight_entrypoint": "scripts/run_lightweight_repro.py",
            "output_size_guard": "scripts/check_output_sizes.py",
            "hygiene_audit": "scripts/run_repo_hygiene_audit.py",
            "notes": [
                "CPU-only; no CUDA / network / real checkpoints required.",
                "transformers is optional; HF stages skip cleanly if absent.",
                "Do not run the full test suite or the full Stage 7.6 scanner "
                "for a lightweight reproduction.",
            ],
        },
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "evidence_manifest.json"
    md_path = out / "evidence_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md_path.write_text(evidence_manifest_markdown(manifest), encoding="utf-8")
    manifest["written_files"] = [str(json_path), str(md_path)]
    return manifest


def evidence_manifest_markdown(manifest: dict[str, Any]) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w(f"# {manifest['title']}")
    w()
    w(manifest["description"])
    w()
    w(f"Stages documented: **{manifest['stage_count']}**")
    w()
    w("## Global caveats")
    w()
    for c in manifest["global_caveats"]:
        w(f"- {c}")
    w()
    w("## Stages")
    w()
    for s in manifest["stages"]:
        w(f"### Stage {s['stage']}")
        w()
        w(f"- **Purpose:** {s['purpose']}")
        w(f"- **Key files:** {', '.join(f'`{f}`' for f in s['key_files'])}")
        w(f"- **Key tests:** {', '.join(f'`{t}`' for t in s['key_tests'])}")
        w(f"- **Commands:** {', '.join(f'`{c}`' for c in s['commands'])}")
        w(f"- **Latest known result:** {s['latest_known_result']}")
        w(f"- **Requires transformers:** {s['requires_transformers']}")
        w(f"- **Requires GPU:** {s['requires_gpu']}")
        w(f"- **Writes large outputs:** {s['writes_large_outputs']}")
        w(f"- **Limitations:** {'; '.join(s['limitations'])}")
        w()
    w("## Reproducibility")
    w()
    repro = manifest["reproducibility"]
    w(f"- Lightweight entrypoint: `{repro['lightweight_entrypoint']}`")
    w(f"- Output size guard: `{repro['output_size_guard']}`")
    w(f"- Hygiene audit: `{repro['hygiene_audit']}`")
    for n in repro["notes"]:
        w(f"- {n}")
    w()
    return "\n".join(lines) + "\n"
