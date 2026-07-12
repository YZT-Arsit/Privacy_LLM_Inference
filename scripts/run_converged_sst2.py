"""Converged SST-2 utility run driver (frozen config; selected: physical batch 16, batched forward).

Runs, sequentially on real A10 + TDX: L0 base floor (once), then L5 (fp32) and L12 (bf16) for each of
the 3 frozen seeds, full 12630 optimizer steps (3 epochs × ceil(67349/16)) with eval_every 200 and
best-dev checkpoint selection + early stopping (patience 3) — all per PHASE4_utility_budget_freeze.json.

Each profile run goes through gate0_a10_batch_orchestrator.py (pushes code + label tables + frozen
seed schedule, launches the batched runner). Results land in .../utility_dataplane/<tag>.json.
Idempotent-ish: skips a cell whose result json already exists (so it can resume across restarts).
Do not commit.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

REPO = Path("/Users/Hoshino/Desktop/privacy_llm_obfuscation")
ORCH = REPO / "scripts/gate0_a10_batch_orchestrator.py"
OUT = REPO / "results/aaai_private_base/alicloud_a10_runs/utility_dataplane"
SEEDS = [1234, 2025, 7]
STEPS = 12630          # 3 epochs * ceil(67349/16)
EVAL_EVERY = 200
PER_RUN_TIMEOUT = 9000  # ~2.5h ceiling per profile run


def run(tag, profile, seed, max_steps, eval_every, eval_max):
    res = OUT / f"{tag}.json"
    if res.exists():
        try:
            d = json.loads(res.read_text())
            if d.get("steps_run", 0) >= max_steps - 5 or profile == "L0":
                print(f"[skip] {tag} already complete (steps_run={d.get('steps_run')})"); return d.get("eval")
        except Exception:
            pass
    cmd = [sys.executable, str(ORCH), "--dataset", "sst2", "--task", "cls", "--profile", profile,
           "--seed", str(seed), "--max-steps", str(max_steps), "--eval-every", str(eval_every),
           "--eval-max", str(eval_max), "--batched-forward", "--run-tag", tag,
           "--timeout", str(PER_RUN_TIMEOUT)]
    print(f"[run] {tag} profile={profile} seed={seed} steps={max_steps} @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=PER_RUN_TIMEOUT + 600)
    tail = "\n".join(p.stdout.splitlines()[-6:])
    print(f"[done] {tag} rc={p.returncode} wall={round((time.time()-t0)/60,1)}min\n{tail}", flush=True)
    if p.returncode != 0:
        print("[stderr]", p.stderr[-800:], flush=True)
    try:
        return json.loads((OUT / f"{tag}.json").read_text()).get("eval")
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"config": {"physical_batch": 16, "batched_forward": True, "steps": STEPS,
                          "eval_every": EVAL_EVERY, "seeds": SEEDS}, "cells": {}}
    # L0 base floor — ONCE (seed-independent), full dev
    summary["cells"]["L0"] = run("sst2_conv_L0", "L0", SEEDS[0], 0, 0, 0)
    # L5 + L12 for each seed, full dev (eval_max 0 = all 872)
    for seed in SEEDS:
        for profile in ["L5", "L12"]:
            tag = f"sst2_conv_{profile}_s{seed}"
            summary["cells"][f"{profile}_s{seed}"] = run(tag, profile, seed, STEPS, EVAL_EVERY, 0)
            (OUT / "CONVERGED_SST2_progress.json").write_text(json.dumps(summary, indent=2))
    (OUT / "CONVERGED_SST2_summary.json").write_text(json.dumps(summary, indent=2))
    print("[ALL DONE]", json.dumps(summary["cells"], indent=2))


if __name__ == "__main__":
    main()
