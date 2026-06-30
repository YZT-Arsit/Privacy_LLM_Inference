#!/usr/bin/env bash

_fail() {
  echo "[bootstrap][ERROR] $*" >&2
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 1
  else
    exit 1
  fi
}

_info() {
  echo "[bootstrap] $*"
}

if [[ -z "${PLLO_ROLE:-}" ]]; then
  _fail "PLLO_ROLE is required. Use PLLO_ROLE=tdx or PLLO_ROLE=h800."
fi

if [[ "$PLLO_ROLE" != "tdx" && "$PLLO_ROLE" != "h800" ]]; then
  _fail "Invalid PLLO_ROLE=$PLLO_ROLE. Expected tdx or h800."
fi

export PLLO_REPO="${PLLO_REPO:-/root/Privacy_LLM_Inference}"

if [[ ! -d "$PLLO_REPO" ]]; then
  _fail "Repository directory does not exist: $PLLO_REPO"
fi

cd "$PLLO_REPO" || _fail "Cannot cd to $PLLO_REPO"

if [[ -n "${PLLO_COMMIT:-}" ]]; then
  if [[ -d .git ]]; then
    current_commit="$(git rev-parse HEAD)"
  elif [[ -f .pllo_commit ]]; then
    current_commit="$(cat .pllo_commit | tr -d '[:space:]')"
  else
    _fail "Cannot verify commit: neither .git nor .pllo_commit exists."
  fi

  if [[ "$current_commit" != "$PLLO_COMMIT" ]]; then
    _fail "Commit mismatch. current=$current_commit expected=$PLLO_COMMIT"
  fi
fi

if [[ "$PLLO_ROLE" == "tdx" ]]; then
  export PLLO_RUNTIME=tdx
  export PLLO_BACKEND=folded_remote
  export PLLO_PAPER_FACING=1

  export PLLO_CONDA_ENV="${PLLO_CONDA_ENV:-tdx310}"

  if [[ ! -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    _fail "conda.sh not found at /root/miniconda3/etc/profile.d/conda.sh"
  fi

  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$PLLO_CONDA_ENV" || _fail "Failed to activate conda env: $PLLO_CONDA_ENV"

  export PLLO_DATA_DIR="${PLLO_DATA_DIR:-$PLLO_REPO/data/aaai}"
  export PLLO_OUTPUT_DIR="${PLLO_OUTPUT_DIR:-$PLLO_REPO/outputs/aaai}"
  export PLLO_STATUS_DIR="${PLLO_STATUS_DIR:-$PLLO_REPO/outputs/status}"
  export PLLO_TDX_ARTIFACT_DIR="${PLLO_TDX_ARTIFACT_DIR:-/root/privacy_llm_tee_artifacts/alibaba_aaai}"
  export PLLO_EVIDENCE_JSON="${PLLO_EVIDENCE_JSON:-$PLLO_TDX_ARTIFACT_DIR/attestation_evidence.json}"
  export PLLO_GPU_WORKER_URL="${PLLO_GPU_WORKER_URL:-http://127.0.0.1:18082}"
fi

if [[ "$PLLO_ROLE" == "h800" ]]; then
  export PLLO_RUNTIME=gpu
  export PLLO_BACKEND="${PLLO_BACKEND:-h800_worker}"
  export PLLO_PAPER_FACING=1

  if [[ -f /root/miniconda3/etc/profile.d/conda.sh && -n "${PLLO_CONDA_ENV:-}" ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate "$PLLO_CONDA_ENV" || _fail "Failed to activate conda env: $PLLO_CONDA_ENV"
  fi

  export PLLO_DATA_DIR="${PLLO_DATA_DIR:-$PLLO_REPO/data/aaai}"
  export PLLO_OUTPUT_DIR="${PLLO_OUTPUT_DIR:-$PLLO_REPO/outputs/aaai}"
  export PLLO_STATUS_DIR="${PLLO_STATUS_DIR:-$PLLO_REPO/outputs/status}"
  export PLLO_MODEL_PATH="${PLLO_MODEL_PATH:-/root/models/Qwen2.5-7B-Instruct}"
  export PLLO_FOLDED_PACKAGE_DIR="${PLLO_FOLDED_PACKAGE_DIR:-$PLLO_REPO/artifacts/qwen25_7b_A_rightmul_pkg}"
  export PLLO_EMBEDDING_ARTIFACT="${PLLO_EMBEDDING_ARTIFACT:-$PLLO_FOLDED_PACKAGE_DIR/embedding_artifact}"
  export PLLO_GPU_WORKER_HOST="${PLLO_GPU_WORKER_HOST:-0.0.0.0}"
  export PLLO_GPU_WORKER_PORT="${PLLO_GPU_WORKER_PORT:-18082}"
fi

export PYTHONPATH="$PWD/src:$PWD/src/pllo/third_party"

mkdir -p "$PLLO_DATA_DIR" "$PLLO_OUTPUT_DIR" "$PLLO_STATUS_DIR" "$PWD/artifacts"

python - <<'PY'
import os
import sys
from pathlib import Path

repo = Path.cwd()
src = repo / "src"
third_party = repo / "src" / "pllo" / "third_party"

assert src.exists(), f"missing src: {src}"
assert third_party.exists(), f"missing third_party: {third_party}"

print("[bootstrap] python =", sys.executable)
print("[bootstrap] python_version =", sys.version.replace("\n", " "))
print("[bootstrap] role =", os.environ.get("PLLO_ROLE"))
print("[bootstrap] runtime =", os.environ.get("PLLO_RUNTIME"))
print("[bootstrap] backend =", os.environ.get("PLLO_BACKEND"))
print("[bootstrap] repo =", str(repo))
print("[bootstrap] data_dir =", os.environ.get("PLLO_DATA_DIR"))
print("[bootstrap] output_dir =", os.environ.get("PLLO_OUTPUT_DIR"))
print("[bootstrap] status_dir =", os.environ.get("PLLO_STATUS_DIR"))
PY

_info "bootstrap finished"
