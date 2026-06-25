"""Deployment truth checker -- pure parsing of experiment report JSON.

Reads a single remote/folded/attested decode report (as produced by
``scripts/run_tee_gpu_protocol_demo.py``) and infers what was *actually*
demonstrated: real vs mock GPU, real TDX vs simulated, attestation binding,
folded-package backing, LoRA reality, transport. From that truth it derives the
set of claims a paper is allowed to make and the ones that are forbidden.

stdlib only. Defensive: any key may be missing -- ``_g`` returns ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "load_json", "infer_deployment_truth", "allowed_and_forbidden_claims",
    "deployment_truth_report",
]


def load_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


def _g(d, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _contains_h800(report: dict) -> bool:
    def _scan(v) -> bool:
        if isinstance(v, str):
            return "h800" in v.lower()
        if isinstance(v, dict):
            return any(_scan(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return any(_scan(x) for x in v)
        return False
    if _scan(report.get("gpu_name")):
        return True
    if _scan(report.get("server_health")):
        return True
    return False


def infer_deployment_truth(report: dict) -> dict:
    report = report if isinstance(report, dict) else {}
    dry_run = _g(report, "dry_run") is True
    backend = _g(report, "gpu_backend")
    mock_backend = backend == "mock"

    gpu_worker_remote = bool(_g(report, "gpu_worker_remote"))
    cuda_evidence = False
    sh = _g(report, "server_health")
    if isinstance(sh, dict):
        for k in ("device", "cuda", "gpu"):
            v = sh.get(k)
            if isinstance(v, str) and "cuda" in v.lower():
                cuda_evidence = True
            elif v is True and k in ("cuda", "gpu"):
                cuda_evidence = True
    if isinstance(_g(report, "device"), str) and "cuda" in _g(report,
                                                              "device").lower():
        cuda_evidence = True

    gpu_real = (
        (not mock_backend)
        and backend in ("qwen7b", "qwen7b_folded_package")
        and (gpu_worker_remote or cuda_evidence)
        and not dry_run
    )

    gpu_type = "h800" if _contains_h800(report) else "unknown"

    # --- TEE / attestation ---
    att = _g(report, "attestation")
    att = att if isinstance(att, dict) else None
    boundary_attested = _g(report, "boundary_attested") is True

    if att is not None and att.get("tee_type"):
        tee_type = att.get("tee_type")
        tee_real = att.get("tee_type") == "tdx" and att.get("verified") is True
    else:
        btt = _g(report, "boundary_tee_type")
        tee_type = btt if btt else "unknown"
        tee_real = False

    if tee_type not in ("tdx", "process", "simulated", "unknown"):
        # normalise unexpected strings to unknown for the typed contract
        tee_type = tee_type if tee_type in ("tdx", "process",
                                            "simulated") else "unknown"

    attestation_evidence_present = (
        (att is not None and att.get("available") is True)
        or ("boundary_attested" in report)
    )
    attestation_verified = (
        boundary_attested or (att is not None and att.get("verified") is True)
    )
    runtime_hash_bound = _g(report, "runtime_hash_bound") is True
    mr_td_verified = (att is not None and att.get("mr_td_match") is True)

    # --- folded package ---
    folded_package_loaded = _g(report, "folded_package_loaded")
    folded_package_valid = _g(report, "folded_package_valid")
    package_backed_prefill = _g(report, "package_backed_prefill")
    package_backed_decode = _g(report, "package_backed_decode")
    folded_package_real = (folded_package_loaded is True) and not dry_run

    # --- boundary mode ---
    bm = _g(report, "boundary_mode")
    boundary_mode = bm if bm in ("full_reference", "lite") else "unknown"

    def _boundary_holds(mode):
        if mode == "full_reference":
            return True
        if mode == "lite":
            return False
        return "unknown"

    boundary_holds_full_model = _boundary_holds(boundary_mode)
    boundary_holds_full_folded_package = _boundary_holds(boundary_mode)

    # --- LoRA ---
    lora_enabled = bool(_g(report, "lora_enabled"))
    lora_mode = _g(report, "lora_mode") or _g(report, "adapter_source")
    if lora_mode == "synthetic":
        lora_synthetic: Any = True
        lora_real_hf_adapter: Any = False
    elif lora_mode in ("hf", "hf_peft"):
        lora_synthetic = False
        lora_real_hf_adapter = True
    else:
        lora_synthetic = "unknown"
        lora_real_hf_adapter = "unknown"
    folded_lora_loaded = _g(report, "folded_lora_loaded")
    folded_lora_valid = _g(report, "folded_lora_valid")

    production_transport = bool(_g(report, "production_transport", default=False))
    research_prototype_transport = not production_transport

    return {
        "dry_run": dry_run,
        "mock_backend": mock_backend,
        "gpu_real": gpu_real,
        "gpu_type": gpu_type,
        "gpu_worker_remote": gpu_worker_remote,
        "tee_real": tee_real,
        "tee_type": tee_type,
        "attestation_evidence_present": attestation_evidence_present,
        "attestation_verified": attestation_verified,
        "runtime_hash_bound": runtime_hash_bound,
        "mr_td_verified": mr_td_verified,
        "folded_package_real": folded_package_real,
        "folded_package_loaded": folded_package_loaded,
        "folded_package_valid": folded_package_valid,
        "package_backed_prefill": package_backed_prefill,
        "package_backed_decode": package_backed_decode,
        "boundary_mode": boundary_mode,
        "boundary_holds_full_model": boundary_holds_full_model,
        "boundary_holds_full_folded_package": boundary_holds_full_folded_package,
        "lora_enabled": lora_enabled,
        "lora_synthetic": lora_synthetic,
        "lora_real_hf_adapter": lora_real_hf_adapter,
        "folded_lora_loaded": folded_lora_loaded,
        "folded_lora_valid": folded_lora_valid,
        "production_transport": production_transport,
        "research_prototype_transport": research_prototype_transport,
    }


def allowed_and_forbidden_claims(truth: dict) -> dict:
    allowed: list = []
    forbidden: list = []
    warnings: list = []

    if (truth.get("tee_real") and truth.get("tee_type") == "tdx"
            and truth.get("attestation_verified")
            and truth.get("gpu_worker_remote")
            and truth.get("folded_package_loaded") is True
            and truth.get("package_backed_decode") is True
            and not truth.get("mock_backend")):
        allowed.append("real_tdx_remote_h800_package_backed_decode")
    else:
        forbidden.append("real_tdx_remote_h800_package_backed_decode")

    if (truth.get("folded_package_loaded") is True
            and truth.get("package_backed_decode") is True
            and not truth.get("lora_enabled")):
        allowed.append("no_lora_package_backed_decode")

    if (truth.get("lora_enabled")
            and truth.get("folded_lora_loaded") is True
            and truth.get("package_backed_decode") is True):
        allowed.append("folded_lora_package_backed_decode")

    # production_ready_secure_serving ALWAYS forbidden unless production_transport
    if truth.get("production_transport") is True:
        allowed.append("production_ready_secure_serving")
    else:
        forbidden.append("production_ready_secure_serving")

    # real_lora_tdx_attested
    if (truth.get("lora_enabled") and truth.get("lora_real_hf_adapter") is True
            and truth.get("attestation_verified") and truth.get("tee_real")):
        allowed.append("real_lora_tdx_attested")
    else:
        forbidden.append("real_lora_tdx_attested")

    # --- warnings ---
    if truth.get("dry_run") is True:
        warnings.append("report is dry_run; not paper evidence")
    if truth.get("lora_synthetic") is True:
        warnings.append("synthetic LoRA, not real adapter utility")
    if truth.get("mock_backend"):
        warnings.append("mock GPU backend; not a real-compute result")
    if truth.get("tee_type") in ("simulated", "process"):
        warnings.append(
            "TEE is %s, not a hardware-attested TDX boundary"
            % truth.get("tee_type"))
    if (truth.get("folded_package_loaded") is True
            and truth.get("folded_package_valid") is False):
        warnings.append("folded package loaded but reported invalid")
    if (truth.get("lora_enabled") and truth.get("folded_lora_loaded") is True
            and truth.get("folded_lora_valid") is False):
        warnings.append("folded LoRA loaded but reported invalid")
    if truth.get("research_prototype_transport"):
        warnings.append(
            "transport is research-prototype; not production-hardened")

    return {"allowed_claims": allowed, "forbidden_claims": forbidden,
            "warnings": warnings}


def deployment_truth_report(report: dict) -> dict:
    report = report if isinstance(report, dict) else {}
    truth = infer_deployment_truth(report)
    claims = allowed_and_forbidden_claims(truth)
    return {
        "stage": "deployment_truth",
        "source_stage": _g(report, "stage"),
        "dry_run": _g(report, "dry_run"),
        "truth": truth,
        "allowed_claims": claims["allowed_claims"],
        "forbidden_claims": claims["forbidden_claims"],
        "warnings": claims["warnings"],
    }
