# Server run plan

Suggested commands for the GPU / server experiments. **Not executed here** — the
user opens the server. All local model loading is offline (`local_files_only`);
fetch checkpoints manually (e.g. `modelscope download --model Qwen/Qwen2.5-7B-Instruct
--local_dir /models/qwen2.5-7b`).

## A. Local CPU (works now, no server)
- Full toy attack matrix, tiny-Qwen smoke + capture, baseline audits, security
  table. See `docs/attack_evaluation_plan.md` §3.

## B. Recommended single GPU
```bash
# baseline correctness on a real Qwen2.5 checkpoint
python scripts/attacks/run_baseline_audit.py --model-name-or-path /models/qwen2.5-0.5b

# capture Qwen-7B intermediates (forward hook), then attack from the .pt
python scripts/attacks/capture_qwen_representations.py \
    --model-name-or-path /models/qwen2.5-7b --num-tokens 64 --layers 0,8,16 \
    --output outputs/attacks/captures/qwen7b.pt

# EIA / BiSR-forward / PIA with more steps + restarts on captured reps
python scripts/attacks/run_eia_optimization.py --source tiny_qwen --num-steps 2000 --num-restarts 4
python scripts/attacks/run_pia_prompt_inversion.py --source tiny_qwen --num-steps 2000
```
Resources: ~1 GPU (24-40 GB for 7B fp16 forward + hooks); minutes-to-hours per
optimization attack depending on steps × restarts × sequence length.

## C. Must be server (multi-GPU / long runs)
```bash
# Qwen-7B / Llama-8B full attack matrix across all target methods
python scripts/attacks/run_all_attacks.py \
    --source stip --model-name-or-path /models/qwen2.5-7b \
    --target-methods plaintext_gpu,stip_qwen,obfuscatune_qwen_orthogonal,ours_amulet_style \
    --num-steps 2000 --max-samples 128 --device cuda

# permutation Alg.3 autoregressive decode on real Qwen (O(V*N) forwards -> big)
#   (drive via pllo.attacks.permutation_multiset_attack.run_autoregressive_decode
#    with a proposal model to cut the per-step candidate set)

# BiSR BACKWARD (currently blocked): requires a split-learning FINE-TUNING loop
#   that records grad(x̃_trk). Implement an SL trainer, capture activation
#   gradients per step, then run gradient matching (Eq.9). Server-only.

python scripts/attacks/run_security_table.py       # measured Table B from real runs
```
Resources: multi-GPU for 7B/8B at scale; multi-seed optimization sweeps and long
prompts are the main cost drivers; the Alg.3 decode is `O(vocab × seq)` forward
passes (use a proposal model + KV-cache-across-candidates to make it tractable).

## What still needs data / code before a server run
- **Real Qwen-7B / Llama-8B checkpoints** (local, offline) — user provides paths.
- **BiSR-backward**: a split-FT training loop that emits `grad(x̃_trk)` (blocked
  until implemented).
- **PIA oracle-LLM `S_s`**: an auxiliary fluent LM for semantic speculation
  (omitted in the best_effort version).
- **Alg.3 proposal model + cross-candidate KV cache** for tractable full decode.
- **External ArrowMatch results** (if comparing to the authors' numbers) via
  `load_external_arrowmatch`.
