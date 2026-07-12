#!/usr/bin/env bash
# S4 GPU/CPU utilization sampling during a back-to-back protected-compute loop (runs ON H800).
# Samples nvidia-smi at 200ms + process CPU/RSS, then computes duty cycle.
set -u
OUT=/root/pllo_pb/results/aaai_private_base/gpu_execution_audit
mkdir -p "$OUT"
IDS=/root/pllo_pb/results/aaai_private_base/h800_unified_worker/dry_run_input_ids.json
export PYTHONPATH=/root/pllo_pb/src
export PB_PKG_DIR=/root/pllo_pb/results/aaai_private_base/private_package/gpu_package
export PB_CKPT_DIR=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen2___5-0___5B
export HF_HUB_OFFLINE=1
PY=/root/miniconda3/bin/python
STEPS="${1:-300}"

# GPU sampler @200ms (utilization, memory, power, clocks, pcie)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.reserved,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pstate \
  --format=csv,noheader,nounits -lms 200 > "$OUT/gpu_timeseries.csv" 2>/dev/null &
SMIPID=$!

# CPU/RSS sampler @200ms (of the python compute proc, once it starts)
( echo "unix,proc_cpu_pct,proc_rss_mb,loadavg1"
  while true; do
    P=$(pgrep -f "gpu_execution_audit.py --dtype" | head -1)
    if [ -n "$P" ]; then
      CPU=$(ps -p "$P" -o %cpu= 2>/dev/null | tr -d ' ')
      RSS=$(ps -p "$P" -o rss= 2>/dev/null | tr -d ' ')
      LA=$(awk '{print $1}' /proc/loadavg)
      echo "$(date +%s.%N),${CPU:-NA},$(awk "BEGIN{print ${RSS:-0}/1024}"),$LA"
    fi
    sleep 0.2
  done ) > "$OUT/cpu_timeseries.csv" 2>/dev/null &
CPUPID=$!

# run the back-to-back compute loop
$PY /root/pllo_pb/scripts/gpu_execution_audit.py --dtype bf16 --seq-len 42 \
    --input-ids "$IDS" --out "$OUT" --loop-steps "$STEPS" 2>/dev/null

kill $SMIPID $CPUPID 2>/dev/null
wait 2>/dev/null

# duty-cycle summary from the GPU timeseries (bounded to the loop window)
$PY - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path
OUT=Path(sys.argv[1])
mk=json.loads((OUT/"loop_marker.json").read_text())
rows=[]
with open(OUT/"gpu_timeseries.csv") as f:
    for r in csv.reader(f):
        if len(r)<7: continue
        try: util=float(r[1]); mem=float(r[2]); memused=float(r[3]); pw=float(r[5])
        except: continue
        rows.append((util,mem,memused,pw))
utils=[u for u,_,_,_ in rows]
n=len(utils) or 1
active=sum(1 for u in utils if u>0)
summ={
  "device": mk.get("device"), "loop_steps": mk.get("steps"),
  "loop_total_wall_s": round(mk.get("total_wall_s",0),3),
  "mean_step_wall_s": round(mk.get("mean_step_wall_s",0),5),
  "gpu_samples": n,
  "gpu_util_avg_pct": round(sum(utils)/n,2),
  "gpu_util_max_pct": max(utils) if utils else 0,
  "active_duty_cycle_pct_samples_util_gt0": round(100*active/n,2),
  "mem_used_max_mb": max((mu for _,_,mu,_ in rows), default=0),
  "power_avg_w": round(sum(p for *_,p in rows)/n,1),
  "power_max_w": max((p for *_,p in rows), default=0),
  "note": ("nvidia-smi GPU-Util is a coarse duty proxy; the intrinsic per-step compute duty "
           "(kernel_time/step_wall) is reported in audit_summary.json S3. This loop is pure "
           "back-to-back compute (NO network/TDX) — a REAL direct step adds seconds of transport, "
           "driving end-to-end GPU duty far lower."),
}
(OUT/"utilization_summary.json").write_text(json.dumps(summ,indent=2))
print(json.dumps(summ,indent=2))
PY
