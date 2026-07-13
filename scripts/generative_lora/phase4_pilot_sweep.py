"""PHASE 4 — plaintext-LoRA pilot sweep on the A10 GPU. Selects the training config by VALIDATION
BLEU (never test). Trains each config on the frozen train subset + schedule, greedy-generates the val
generation subset, computes multi-reference BLEU, and writes pilot_sweep.json. Freeze the winner for
G1/G2. Runs entirely on the real A10 GPU (plaintext reference cells).
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

GL = "/root/pllo_pb/results/aaai_private_base/generative_lora"
PY = "/usr/bin/python3"
SCRIPT = "/root/pllo_pb/scripts/generative_lora/plaintext_lora.py"
TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
SEED = 1234


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def bleu(genf):
    import sacrebleu
    recs = [json.loads(l) for l in open(genf)]
    hyps = [r["generated_text"] for r in recs]
    maxr = max(len(r["references"]) for r in recs)
    refs = [[(r["references"][i] if i < len(r["references"]) else r["references"][0]) for r in recs]
            for i in range(maxr)]
    b = sacrebleu.corpus_bleu(hyps, refs).score
    al = sum(r["generated_token_count"] for r in recs) / len(recs)
    rp = sum(r["repetition_bigram_frac"] for r in recs) / len(recs)
    iv = sum(r["invalid_output"] for r in recs) / len(recs)
    return round(b, 2), round(al, 1), round(rp, 3), round(iv, 3)


def main():
    # (lr, epochs->max_steps at batch16 over 4000: 1ep=250,2ep=500,3ep=750)
    grid = [(1e-4, 250), (2e-4, 250), (1e-4, 500), (2e-4, 500), (2e-4, 750)]
    results = []
    for lr, steps in grid:
        tag = f"g1pilot_lr{lr:.0e}_st{steps}".replace("-", "")
        adir = f"{GL}/adapters/{tag}"; gout = f"{GL}/generations/{tag}_val.jsonl"
        t0 = time.time()
        r = sh(f"cd /root/pllo_pb && HF_HUB_OFFLINE=1 {PY} {SCRIPT} train --targets {TARGETS} --rank 8 "
               f"--alpha 16 --dtype fp32 --seed {SEED} --train-data {GL}/data/e2e_train_a10.pt "
               f"--schedule {GL}/data/e2e_schedule_s{SEED}.json --max-steps {steps} --lr {lr} "
               f"--out-adapter {adir} --out-meta {adir}.meta.json")
        if r.returncode != 0:
            print("TRAIN FAIL", tag, r.stderr[-400:]); continue
        meta = json.loads(Path(f"{adir}.meta.json").read_text())
        r = sh(f"cd /root/pllo_pb && HF_HUB_OFFLINE=1 {PY} {SCRIPT} generate --targets {TARGETS} --rank 8 "
               f"--alpha 16 --dtype fp32 --adapter {adir} --gen-in {GL}/data/e2e_val_gen.json "
               f"--gen-out {gout} --gen-max 100 --max-new 96 --cell {tag}")
        if r.returncode != 0:
            print("GEN FAIL", tag, r.stderr[-400:]); continue
        b, al, rp, iv = bleu(gout)
        row = {"tag": tag, "lr": lr, "steps": steps, "final_loss": round(meta["final_loss"], 4),
               "val_bleu": b, "avg_len": al, "rep": rp, "invalid": iv, "wall_sec": round(time.time()-t0, 1)}
        results.append(row); print("PILOT", json.dumps(row), flush=True)
    results.sort(key=lambda x: -x["val_bleu"])
    best = results[0] if results else None
    out = {"phase": "4_pilot_sweep", "selection_metric": "validation_bleu_100", "grid": grid,
           "targets": TARGETS, "rank": 8, "alpha": 16, "seed": SEED, "results": results,
           "selected": best, "note": "config selected on VALIDATION only; frozen for G1/G2/G3."}
    Path(f"{GL}/data/pilot_sweep.json").write_text(json.dumps(out, indent=2))
    print("SELECTED", json.dumps(best))


if __name__ == "__main__":
    main()
