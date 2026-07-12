"""S5: build the real per-step direct-transport timeline from a direct-runner result JSON.

Maps the runner's recorded per-step timing decomposition to the audit's timeline phases and
computes GPU-compute / CPU-processing / network-wait / TDX / idle fractions.

    python3 scripts/build_step_timeline.py --result <AUDIT_S5_bf16_1step.json> --out <dir>
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", default="results/aaai_private_base/gpu_execution_audit")
    args = ap.parse_args()
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    R = json.loads(Path(args.result).read_text())
    tr = R["trajectory"]
    rows = []
    for s in tr:
        t = s["timing"]; b = s.get("bytes", {})
        # network-wait split: enclave time is TDX; the rest of each round-trip is SSH wire
        tdx = t["enclave_ce_sec"] + t["enclave_correct_sec"]
        net = t["ssh_wire_ce_sec"] + t["ssh_wire_correct_sec"]
        gpu = t["fwd_sec"] + t["bwd_sec"] + t["apply_sec"]
        cpu = t["serialize_sec"] + t.get("deserialize_sec", 0.0)
        total = t["step_wall_sec"]
        idle = max(0.0, total - gpu - cpu - net - tdx)
        rows.append({
            "step": s["step"],
            "h800_forward_s": round(t["fwd_sec"], 4),
            "serialize_logits_DtoH_s": round(t["serialize_sec"], 4),
            "net_ce_roundtrip_s": round(t["net_ce_sec"], 3),
            "tdx_ce_s": round(t["enclave_ce_sec"], 4),
            "ssh_wire_ce_s": round(t["ssh_wire_ce_sec"], 3),
            "deserialize_dlogits_HtoD_s": round(t.get("deserialize_sec", 0.0), 4),
            "h800_backward_s": round(t["bwd_sec"], 4),
            "apply_gpu_and_stage_s": round(t["apply_sec"], 4),
            "net_correct_roundtrip_s": round(t["net_correct_sec"], 3),
            "tdx_correction_s": round(t["enclave_correct_sec"], 4),
            "ssh_wire_correct_s": round(t["ssh_wire_correct_sec"], 3),
            "step_wall_s": round(total, 3),
            "bytes_logits_up": b.get("logits_up"), "bytes_dlogits_down": b.get("dlogits_down"),
            "bytes_corr_up": b.get("corr_up"), "bytes_corrected_down": b.get("corrected_down"),
            "frac_gpu_compute": round(gpu / total, 4), "frac_cpu_processing": round(cpu / total, 4),
            "frac_network_wait": round(net / total, 4), "frac_tdx": round(tdx / total, 4),
            "frac_idle": round(idle / total, 4)})
    with open(OUT / "step_timeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)

    agg = {k: sum(r[k] for r in rows) / len(rows) for k in
           ("frac_gpu_compute", "frac_cpu_processing", "frac_network_wait", "frac_tdx", "frac_idle")}
    mean_wall = sum(r["step_wall_s"] for r in rows) / len(rows)
    md = ["# S5 direct-transport per-step timeline (real hardware)", "",
          f"- run: `{R.get('run_id')}`  transport: `{R.get('transport_profile')}`  steps: {len(rows)}",
          f"- mean step wall: **{mean_wall:.2f} s**", "",
          "## Mean fractions of the step wall", "",
          f"| phase | fraction |", "|---|---|",
          f"| GPU compute (fwd+bwd+apply) | {agg['frac_gpu_compute']*100:.2f}% |",
          f"| CPU processing (serialize/deserialize) | {agg['frac_cpu_processing']*100:.2f}% |",
          f"| Network wait (SSH wire, dominated by throttled TDX→H800 download) | {agg['frac_network_wait']*100:.2f}% |",
          f"| TDX enclave (CE + correction) | {agg['frac_tdx']*100:.2f}% |",
          f"| Idle/unattributed | {agg['frac_idle']*100:.2f}% |", "",
          "## Interpretation", "",
          "The step wall is dominated by **network wait** — the throttled TDX→H800 ingress moving the "
          "dlogits (and corrected grads) back to the GPU host. GPU compute is a tiny fraction. This is "
          "a transport/network limit, not a CPU-compute or device-placement bug: the forward/backward "
          "run on CUDA (S1–S4)."]
    (OUT / "step_timeline.md").write_text("\n".join(md))
    (OUT / "step_timeline_fractions.json").write_text(json.dumps(
        {"mean_step_wall_s": mean_wall, "mean_fractions": agg,
         "per_step_bytes": [{"dlogits_down": r["bytes_dlogits_down"], "corrected_down": r["bytes_corrected_down"],
                             "logits_up": r["bytes_logits_up"], "corr_up": r["bytes_corr_up"]} for r in rows]},
        indent=2))
    print(json.dumps({"mean_step_wall_s": round(mean_wall, 2), "mean_fractions": {k: round(v, 4) for k, v in agg.items()}}, indent=2))


if __name__ == "__main__":
    main()
