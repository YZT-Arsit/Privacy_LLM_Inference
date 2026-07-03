# Generation-task evaluation — `current` backend

## 0. Real TDX attestation

- worker backend: `tdx_attested_remote` | attestation requested: yes
- **boundary_attested: yes** | tee_type: tdx | runtime_hash_bound: yes
- mr_td: `8a56b29ae530d23ee602628eff3b7fdf5e159d4a55d62b2454e8e4fbde424d67dfbe256f0473bcf4e4b2acba999419e0`

## 1. Backend verification (no silent fallback)

- expected nonlinear_backend: `current` → op_backend `current`
- worker actual: nonlinear_backend `current`, op_backend `current`
- **verification passed: yes** | silent fallback detected: no

## 2. Nonlinear execution evidence

- nonlinear_backend / op_backend (measured): `current` / `current`
- nonlinear_execution_evidence_missing: no
- lifted_nonlinear_ops_count: 0 | lift_k: 0 | lifted_gpu_bytes: 0 | trusted_nonlinear_ops_count: 56726
- nonlinear ops by type: rmsnorm=28614, softmax=14056, silu=14056, gelu=None

## 3. Dimension audit completeness

- sections present: 8/8 (model, inputs, attention, kv_cache, mlp, nonlinear, lm_head, masks_pad_package)
- model: 28 layers, hidden=3584, inter=18944, heads=28, kv_heads=4, head_dim=128, vocab=152064, dtype=bfloat16, seq_len=1024
- full per-layer/KV/MLP/nonlinear/LM-head/mask shapes in `dimension_audit.json` + `per_layer_dimensions.csv`.
- seq_len compatibility: ok

## 4. Generation quality

- prompts: 10 | clean: 10 | empty: 0 | repeated: 0 | early-EOS: 0 | length-truncated: 5 | malformed: 0

## 5. Performance

- total prompts: 10 | total tokens: 502 | total latency: 24.6579s
- tokens/sec: 20.359 | mean per-prompt latency: 2.4658s | mean first-token latency: Nones
- peak GPU memory: 27343.98828125 MB | resident cache active: yes
- boundary calls: 995 | gpu calls: 503 | trusted bytes: 305344512 | gpu bytes: 322633728
- decode bottleneck stage: gpu_worker_roundtrip

## 7. Generation examples

**smoke-01** (current, finish=eos, 29 tokens)
> prompt: 请用一句话解释什么是大语言模型。
> output: 大语言模型是一种能够理解和生成自然语言的深度学习模型，通常基于大规模的文本数据训练，具有强大的文本处理能力。

**smoke-02** (current, finish=eos, 39 tokens)
> prompt: In one sentence, explain what a transformer neural network is.
> output: A transformer neural network is a type of deep learning model architecture that uses self-attention mechanisms to process and generate sequences of data, excelling in tasks like machine translation and text summarization.

**smoke-03** (current, finish=length, 64 tokens)
> prompt: A shop sells apples at $3 each. If I buy 7 apples and pay with a 50 dollar bill, how much change do I get? Show your reasoning and give the final number.
> output: To determine the change you would receive, let's follow these steps:

1. **Calculate the total cost of the apples:**
   - The price of one apple is $3.
   - You are buying 7 apples.
   - Therefore, the total cost is \( 7 \times 3 = 2

## 8. Errors / failures

- none

## 9. Next steps

- scale IFEval subset (20 → 100 → 541) once smoke passes
- run the paired backend (current ↔ trusted_shortcut) for the comparison file if not already present
