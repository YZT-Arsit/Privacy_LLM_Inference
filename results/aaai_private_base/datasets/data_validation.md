# Dataset validation (real, official)

Loaded with HuggingFace `datasets` 5.0.0 from the canonical sources. No paraphrase,
no synthetic augmentation, no regenerated labels, no hidden-test-label use.

## GSM8K (openai/gsm8k, config `main`) — MIT
| split | rows | official | match |
|---|---|---|---|
| train | 7473 | 7473 | ✓ |
| test | 1319 | 1319 | ✓ |
- Schema: `question`, `answer`. Content SHA-256 pinned (`checksums.sha256`,
  `gsm8k_manifest.json`).
- Train/test question overlap: **0** (disjoint). Internal duplicate questions:
  train 0, test 0.
- Answer extraction: official `#### <number>` final-answer regex (frozen).
- NOT TinyGSM; NOT synthetic.

## SST-2 (glue/sst2) — CC-BY-4.0
| split | rows | official | match |
|---|---|---|---|
| train | 67349 | 67349 | ✓ |
| validation (dev) | 872 | 872 | ✓ |
- Schema: `sentence`, `label`, `idx`. Labels: `['negative','positive']`.
- Eval on the official **dev** split; the hidden test labels are NOT used.
- Train/dev sentence overlap: **0** (disjoint). Internal duplicate sentences:
  train **371** (a genuine, known property of SST-2 — recorded, not removed), dev 0.
- Prompt template `Review: {sentence}\nSentiment:`; verbalizer `0→negative`,
  `1→positive` (frozen).

## Sample identity
`sample_id_lists/{gsm8k_train_ids,gsm8k_test_ids,sst2_train_ids,sst2_dev_ids}.json`
preserve official ordering / `idx`. Preregistered subsets for the deployment smoke and
the 1000-sample equivalence set will be drawn as fixed prefixes of these lists and
pinned before any protected run.

## Provenance notes
- `contamination_report.json`: exact normalized-text overlap method; no external
  instruction/fine-tuning corpus is available locally to cross-check against, so
  cross-corpus contamination is UNVERIFIED (recorded honestly, not claimed clean).
- Synthetic data is used ONLY for algebraic unit tests + attack positive controls,
  never for a main task.
