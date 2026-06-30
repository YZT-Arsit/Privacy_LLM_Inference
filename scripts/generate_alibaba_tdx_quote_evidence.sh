#!/usr/bin/env bash
# Thin convenience wrapper for the Alibaba Cloud TDX quote/evidence generator.
#
# Re-generates REAL TDX attestation evidence bound to the CURRENT code hash +
# A_rightmul metadata, using the Alibaba quote-generation + quote-verification
# samples. A quote binds the current runtime hash, so it MUST be regenerated
# before every formal experiment -- an old quote is stale and will fail the
# tdx_reportdata == runtime_hash binding check.
#
# No passwords / SSH keys are read or written here. All real attestation requires
# a real Alibaba TDX guest; off-TDX use --simulate (unsigned, paper_facing=false).
#
# Usage:
#   scripts/generate_alibaba_tdx_quote_evidence.sh \
#       <OUTPUT_DIR> <OUTPUT_EVIDENCE_JSON> [EXPECTED_MR_TD] [extra args...]
#
# Example (on the Alibaba TDX guest, before a formal run):
#   scripts/generate_alibaba_tdx_quote_evidence.sh \
#       /root/privacy_llm_tee_artifacts/alibaba_aaai \
#       /root/privacy_llm_tee_artifacts/alibaba_aaai/attestation_evidence.json \
#       e0199499baacb2e4...9ab2568a
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:?output dir required}"
OUT_EV="${2:?output evidence json required}"
EXPECTED_MR_TD="${3:-}"
shift || true; shift || true; [ "$#" -gt 0 ] && shift || true

MRTD_ARG=()
if [ -n "${EXPECTED_MR_TD}" ]; then
  MRTD_ARG=(--expected-mr-td "${EXPECTED_MR_TD}")
fi

export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"
exec python "${HERE}/scripts/generate_alibaba_tdx_quote_evidence.py" \
  --nonlinear-backend A_rightmul \
  --qgen-app /opt/alibaba/tdx-quote-generation-sample/app \
  --qverify-dir /opt/alibaba/tdx-quote-verification-sample \
  --output-dir "${OUT_DIR}" \
  --output-evidence "${OUT_EV}" \
  "${MRTD_ARG[@]}" \
  "$@"
