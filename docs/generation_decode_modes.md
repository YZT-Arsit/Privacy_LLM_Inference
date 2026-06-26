# Folded-remote generation: decode modes

The trusted boundary owns tokenization, logits recovery, generation-config logit
processors, sampling/argmax, and the stop decision. The untrusted GPU worker only
ever executes folded layers over **masked** tensors. Two decode modes trade off
output quality/performance against output-length leakage. **Pick one per run and
never mix their numbers in the paper.**

## 1. Default EOS-stop mode (paper performance + IFEval quality)

- Trusted-side **EOS stopping is ON by default**, aligned with HuggingFace
  `model.generate` (stops at `generation_config.eos_token_id`, e.g. Qwen2.5
  `[151645, 151643]`). Optionally `--align-generation-config` applies the
  baseline's `repetition_penalty` trusted-side (after recovery, before argmax).
- Best quality and latency; this is the configuration used for IFEval / paper
  results.
- **Leakage:** the GPU observes the **number of decode rounds**, i.e. a coarse
  upper bound on output length (true if the boundary stops early at EOS). This is
  inherent to any online remote decode — the worker sees how many requests arrive.
- Report: `length_hiding_enabled=false`, `stop_on_eos=true`,
  `finish_reason_per_example`, `generated_tokens_per_example`,
  `gpu_decode_rounds_per_example` (== generated count + prefill).

CLI:

```
--align-generation-config --repetition-penalty 1.05      # EOS stop is default-on
# --disable-eos-stop   # ONLY to reproduce the old fixed-length behaviour
```

## 2. Strict length-hiding mode (hides the true stop step)

- After the trusted side detects EOS, it **keeps issuing dummy masked decode
  rounds** to a fixed `max_new_tokens` budget. The GPU therefore sees a **constant
  decode-round count** for every example and cannot infer the true output length
  or the EOS step.
- The dummy token id is **trusted-only**; the GPU sees only its masked embedding.
  Dummy recovered logits/tokens are **discarded** — never appended to the returned
  output, never recovered, never reported, never sent to the GPU.
- The **returned response is identical** to default mode (same real tokens); only
  extra GPU rounds + latency are added.
- **Cost:** up to `max_new_tokens − true_length` extra GPU rounds per example.
  Report this separately: `length_hiding_overhead_tokens`,
  `length_hiding_overhead_ratio`, `dummy_decode_rounds_per_example`,
  `true_output_latency_s` vs `dummy_decode_latency_s`,
  `latency_per_returned_token_s` vs `latency_per_gpu_decode_round_s`.

CLI:

```
--length-hide-generation        # (alias: --dummy-decode-after-eos)
```

Report: `length_hiding_enabled=true`, `dummy_decode_after_eos=true`,
`true_generated_tokens_per_example` vs `gpu_decode_rounds_per_example`
(== `max_new_tokens`), `length_hiding_security_note`.

## Security invariants (BOTH modes)

Never crosses the trusted boundary to the GPU: user prompt / `input_ids`,
plaintext hidden states, recovered plaintext logits, generated token ids / token
history, the sampling/argmax decision, the EOS decision / finish step, the dummy
token id, and mask / inverse / pad / PRG seed / schedule secret / raw LoRA / KV
plaintext. GPU requests carry only masked embeddings + public metadata. Audited
by `forbidden_fields_in_payload` (canonical, attestation-measured) and the
non-measured extended audit `pllo.security.length_hiding_audit`
(`scan_length_hiding_transcript` / `audit_gpu_request_payloads`), which also
forbids `token_ids` / `plaintext_logits` / `dummy_token_id` / `eos_decision` /
`finish_reason` / `generated_token_history`. `plaintext_logits_or_sampling_on_gpu`
stays `false` in both modes.

> The extended audit lives in a **non-measured** module on purpose, so adding it
> does not change the attestation runtime hash. The canonical, hash-bound
> `transcript_scanner` is unchanged; harden it (and re-bind the TDX quote) only
> when you next regenerate the quote.
