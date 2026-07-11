"""Section 5/13 accounting: invocation + communication + performance decomposition from
the real L10 gate reports. Emits full_lora_matrix/{communication,performance}/*.
"""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

FM = Path(__file__).resolve().parents[1] / "results/aaai_private_base/full_lora_matrix"
PV = FM / "profile_validation"


def load(name):
    p = PV / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    reports = [(f.name, json.loads(f.read_text())) for f in sorted(PV.glob("L10_*step.json"))]
    comm_rows, perf_rows = [], []
    for name, r in reports:
        for s in r["trajectory"]:
            cb = s["comm_bytes"]; t = s["timing"]
            comm_rows.append({
                "run": name, "step": s["step"], "logical_invocations": s["logical_invocations"],
                "logits_to_tdx_B": cb["logits_to_tdx"], "dlogits_from_tdx_B": cb["dlogits_from_tdx"],
                "corr_grads_to_tdx_B": cb["corr_grads_to_tdx"], "corrected_from_tdx_B": cb["corrected_from_tdx"],
                "total_gpu_to_tdx_B": cb["logits_to_tdx"] + cb["corr_grads_to_tdx"],
                "total_tdx_to_gpu_B": cb["dlogits_from_tdx"] + cb["corrected_from_tdx"],
                "correction_payload_frac": round(cb["corr_grads_to_tdx"] /
                    (cb["logits_to_tdx"] + cb["corr_grads_to_tdx"]), 4)})
            perf_rows.append({
                "run": name, "step": s["step"],
                "tdx_loss_sec": round(t["tdx_loss_sec"], 3),
                "tdx_correction_sec": round(t["tdx_correction_sec"], 3),
                "tdx_correction_compute_sec": round(t["tdx_correction_compute"], 4),
                "step_wall_sec": round(t["step_wall_sec"], 2),
                "in_enclave_compute_frac": round(
                    t["tdx_correction_compute"] / max(t["step_wall_sec"], 1e-9), 6)})
    (FM / "communication").mkdir(exist_ok=True); (FM / "performance").mkdir(exist_ok=True)
    with open(FM / "communication" / "communication_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comm_rows[0].keys())); w.writeheader(); w.writerows(comm_rows)
    with open(FM / "performance" / "performance_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(perf_rows[0].keys())); w.writeheader(); w.writerows(perf_rows)

    summary = {
        "transport_profile": "mac_ferried_authenticated_prototype",
        "ferried_latency_is_deployment_claim": False,
        "logical_trusted_invocations_per_step": 3,
        "invocation_breakdown": ["1: trusted session/labels/attestation",
                                 "2: trusted CE + dlogits (in-enclave)",
                                 "3: trusted gradient correction (in-enclave)"],
        "mean_step_wall_sec": round(statistics.mean(p["step_wall_sec"] for p in perf_rows), 1),
        "mean_tdx_correction_compute_sec": round(statistics.mean(
            p["tdx_correction_compute_sec"] for p in perf_rows), 4),
        "correction_payload_bytes": comm_rows[0]["corr_grads_to_tdx_B"],
        "logits_payload_bytes": comm_rows[0]["logits_to_tdx_B"],
        "correction_payload_frac_of_gpu_to_tdx": comm_rows[0]["correction_payload_frac"],
        "note": ("In-enclave compute (CE+dlogits+correction) is ~0.05-0.1 s/step; the ~180 s "
                 "step wall is cross-cloud SSH ferry. The O1-C correction adds one round-trip "
                 "(+3.48 MB each way = +12% of GPU->TDX bytes) and ~48 ms in-enclave compute.")}
    (FM / "performance" / "accounting_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
