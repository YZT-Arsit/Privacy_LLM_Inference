"""Guard: deployment-critical code must not use Python 3.9+/3.10+ only APIs.

The Alibaba Cloud TDX VM that runs the trusted boundary may be on an older
Python (e.g. 3.6). The boundary / cross-machine / attestation code that executes
there must avoid 3.9+ only APIs:

* ``str.removeprefix`` / ``str.removesuffix`` (3.9)
* ``functools.cache`` (3.9)
* ``math.lcm`` / ``math.prod`` (3.9 / 3.8)
* ``match`` / ``case`` structural pattern matching (3.10)

This scans the AST of the deployment-critical files (parsing alone also confirms
they are syntactically valid on this interpreter). Comments/strings are ignored
because we walk the AST, not the text.

Run: python -m pytest tests/test_no_python39_only_apis.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files that run inside / drive the TDX boundary or cross the wire.
DEPLOYMENT_FILES = [
    "src/pllo/protocol/attestation.py",
    "src/pllo/protocol/remote.py",
    "src/pllo/protocol/wire.py",
    "src/pllo/protocol/orchestrator.py",
    "src/pllo/protocol/gpu_worker.py",
    "src/pllo/protocol/tee_gpu_messages.py",
    "src/pllo/protocol/security_audit.py",
    "src/pllo/protocol/lora_training_audit.py",
    # TDX-lite boundary runtime surface now measured into the attestation hash
    # (must stay py3.6-safe -- it runs inside the TD).
    "src/pllo/experiments/folded_probe_common.py",
    "src/pllo/ops/causal_lm_boundaries.py",
    "src/pllo/ops/nonlinear_islands.py",
    "src/pllo/ops/mitigation_bundles.py",
    "src/pllo/nonlinear/backends.py",
    "src/pllo/nonlinear/current_backend.py",
    "src/pllo/nonlinear/amulet_backend.py",
    "src/pllo/nonlinear/registry.py",
    # folded-weight provisioning may run in the TDX trusted-setup env
    "src/pllo/deployment/folded_package.py",
    "src/pllo/deployment/folded_package_manifest.py",
    "src/pllo/deployment/embedding_artifact.py",
    "src/pllo/deployment/lora_folded_package.py",
    "src/pllo/training/lora_private_trainer.py",
    "scripts/build_qwen7b_folded_package.py",
    "scripts/build_qwen7b_embedding_artifact.py",
    "scripts/build_qwen7b_lora_folded_package.py",
    "scripts/verify_qwen7b_lora_folded_package.py",
    "scripts/run_qwen7b_lora_folded_local_probe.py",
    "scripts/run_qwen7b_lora_folded_remote_decode_probe.py",
    "scripts/run_lora_private_training_tiny_probe.py",
    # TDX-side LoRA helpers (may run on the older-Python TDX VM)
    "scripts/prepare_tdx_lora_lite_inputs.py",
    "scripts/check_tdx_measurement_coverage.py",
    "scripts/verify_folded_package.py",
    "scripts/estimate_folded_package_cost.py",
    "scripts/inspect_folded_package.py",
    "scripts/run_qwen7b_folded_package_load_probe.py",
    "scripts/run_tee_gpu_protocol_demo.py",
    "src/pllo/experiments/e3_remote_decode_scaling.py",
    "scripts/run_e3_remote_decode_scaling.py",
    "scripts/generate_tdx_attestation_evidence.py",   # runs on the TDX VM
    "scripts/write_tee_boundary_runtime_hash.py",
    "scripts/run_nonlinear_backend_microbench.py",
    "scripts/run_lora_training_protection_experiments.py",
]

BANNED_METHOD_ATTRS = {"removeprefix", "removesuffix"}     # str, 3.9
BANNED_DOTTED = {("functools", "cache"), ("math", "lcm")}  # 3.9


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        # str.removeprefix / .removesuffix
        if isinstance(node, ast.Attribute) and node.attr in BANNED_METHOD_ATTRS:
            out.append(f"{path.name}:{node.lineno} .{node.attr}() (py3.9+)")
        # functools.cache / math.lcm
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and (node.value.id, node.attr) in BANNED_DOTTED):
            out.append(f"{path.name}:{node.lineno} "
                       f"{node.value.id}.{node.attr} (py3.9+)")
        # match/case (py3.10) — ast.Match exists only on 3.10+ interpreters
        if hasattr(ast, "Match") and isinstance(node, ast.Match):
            out.append(f"{path.name}:{node.lineno} match-case (py3.10+)")
    return out


@pytest.mark.parametrize("rel", DEPLOYMENT_FILES)
def test_no_py39_only_apis_in_deployment_files(rel) -> None:
    path = REPO_ROOT / rel
    assert path.exists(), f"missing deployment file {rel}"
    violations = _violations(path)
    assert violations == [], "; ".join(violations)


def test_all_deployment_files_parse() -> None:
    for rel in DEPLOYMENT_FILES:
        ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
