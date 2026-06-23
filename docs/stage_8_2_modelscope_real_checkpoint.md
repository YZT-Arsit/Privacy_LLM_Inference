# Stage 8.2 — Real ModelScope Checkpoint GPU Experiment

Moves beyond random tiny models to bounded, **real** Qwen2.5 checkpoints
(0.5B → 1.5B → 3B → optional 7B) loaded **only** through ModelScope (never
Hugging Face remote download). For each checkpoint it evaluates three paths
over a short prefill + bounded greedy decode horizon:

1. **HF baseline** — `model.generate` greedy (full model), diagnostic.
2. **Extracted-weight plaintext reference** — Stage 6.9 extraction path
   (possibly a *partial-layer* stack when `--max-layers` < total).
3. **Masked runtime** — trusted embedding boundary → masked decoder → masked
   logits → trusted recovery + greedy, with a **simulated TEE only**.

> The reference is our extracted-weight forward (adjacent-pair RoPE), not HF
> `forward`/`generate`. HF parity is diagnostic only; the primary correctness
> metric is **masked vs extracted-plain token match** on the same stack.

## Large-model engineering choices

| Concern | Choice | Why |
|---|---|---|
| dtype | bf16 (if supported) → fp16; never float64 | float64 is infeasible for real hidden sizes |
| residual mask | `signed_permutation` (default) | orthogonal + RMSNorm-compatible, cheap to build; honestly weaker than dense |
| mask alternatives | `block_orthogonal`, `dense_orthogonal` | block mixes within blocks; dense refused for hidden > 1024 without `--allow-dense-large-mask` |
| residual strategy | `shared` (default for real) | shared mask ⇒ every handoff is identity ⇒ **no online [H,H] GEMM**; per-layer only for small/diagnostic |
| layers | `--max-layers 1/2/4/all` | start partial; grow only if memory is stable |
| horizon | prefill 16 / decode 8 (then 64/16) | bounded smoke; no long context |
| reports | compact JSON/MD; no tensor dumps | JSON kept < 10 MB (hard guard) |

Mask modes are exact-orthogonal and validated: with `signed_permutation` +
`shared`, the tiny-model masked vs extracted-plain token match is 1.0 and the
handoff GEMM is skipped.

## Files

- [src/pllo/experiments/modelscope_real_checkpoint_probe.py](../src/pllo/experiments/modelscope_real_checkpoint_probe.py)
- [scripts/run_modelscope_real_checkpoint_probe.py](../scripts/run_modelscope_real_checkpoint_probe.py)
- [tests/test_modelscope_real_checkpoint_probe.py](../tests/test_modelscope_real_checkpoint_probe.py) — 10 tests (no network/CUDA/checkpoint)
- Additive mask support in [src/pllo/hf_wrappers/hf_causal_lm_skeleton.py](../src/pllo/hf_wrappers/hf_causal_lm_skeleton.py): `make_residual_mask`, mask-mode/strategy config fields, shared-mask handoff short-circuit (Stage 6.9 defaults unchanged).

## CLI (run smallest first)

```bash
python scripts/run_modelscope_real_checkpoint_probe.py \
    --model-id Qwen/Qwen2.5-0.5B-Instruct --cache-dir /root/modelscope_cache \
    --device cuda --dtype bfloat16 --max-layers 1 \
    --prefill-seq-len 16 --decode-steps 8 \
    --mask-mode signed_permutation --residual-mask-strategy shared \
    --output outputs/modelscope_qwen2_5_0_5b_stage8_2.json
```

Progressive order: 0.5B → 1.5B → 3B → (optional) 7B short smoke. After each
run: `python scripts/check_output_sizes.py --output-dir outputs --warn-mb 10
--fail-mb 100` and `nvidia-smi`.

## Metrics (compact)

model_id, local path, dtype, device, max_layers/total, prefill/decode, mask
mode/strategy; HF baseline latency + tokens; extracted-plain latency; masked
token_match_rate + recovered/masked logits error; peak CUDA memory (per phase);
boundary calls, TEE→GPU / GPU→TEE bytes, handoff GEMM count/FLOPs, recovery
FLOPs; report sizes.

## Stop conditions

ModelScope/transformers unavailable; download fails; CUDA unavailable & CPU too
slow; OOM; report > 10 MB / file approaching 100 MB; any attempted HF remote
download. On OOM, reduce in order: max_layers → decode_steps → prefill_seq_len
→ model size → dtype (fp32 → bf16/fp16). Do not retry large models after OOM.

## Caveats (always reported)

- Simulated TEE only; no real TEE hardware.
- `signed_permutation` weaker than dense orthogonal masking.
- `shared` residual mask weaker than per-layer masks.
- Attention scores remain GPU-visible.
- No semantic / cryptographic / formal security is claimed.
- `max_layers` < total ⇒ partial model; HF tokens need not match the
  partial-stack tokens.

## Next

**Stage 8.3** — full-layer 0.5B/1.5B experiment; **Stage 8.4** — final paper
table consolidation.
