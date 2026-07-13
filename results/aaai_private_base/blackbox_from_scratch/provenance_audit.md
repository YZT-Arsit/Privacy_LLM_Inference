# From-scratch provenance audit

## Verdict

`FROM_SCRATCH_CLAIM_UNSUPPORTED_PUBLIC_CHECKPOINT`

The selected paper-facing victim is not a backbone trained from random initialization in this repository. The frozen registry identifies it as `Qwen2.5-0.5B` with source `Qwen/Qwen2.5-0.5B (HF)`. The recorded checkpoint was copied from an H800 ModelScope cache, and the trusted package builder directly loads its `model.safetensors`.

Under the stage's Phase 0 rule, this is a hard stop. No attack corpus was collected, no attack or surrogate was run, and no GPU job was launched.

## Decisive evidence

| Requirement | Result | Evidence |
|---|---|---|
| Initialization provenance | **FAIL** — public pretrained checkpoint | `experiment_registry.yaml` records `checkpoint_source: Qwen/Qwen2.5-0.5B (HF)` |
| Backbone training code hash | **Missing** | No selected-victim pretraining entrypoint; the authoritative script is a package builder |
| Initialization seed commitment | **Missing** | Existing seeds bind masks/LoRA experiments, not backbone initialization |
| Private pretraining-data manifest | **Missing** | Existing data manifests are downstream/LoRA data |
| Absence of public checkpoint loading | **FAIL** | `gate0_build_private_package.py` loads `model.safetensors`; the evaluator calls `from_pretrained` |
| Absence of weight import | **FAIL** | All plaintext tensors are imported with `safetensors.torch.load_file` in trusted setup |
| Tokenizer provenance | Public Qwen tokenizer | SHA-256 `c0382117…7539` |
| Architecture provenance | Public Qwen2 architecture | 24 layers, hidden size 896, 14 query heads, 2 KV heads |
| Final checkpoint hash | Present but not from-scratch | `88c14255…342` is the imported Qwen checkpoint |
| Transformed package root | Present | `bfd578b8…cde1` |

Key source locations:

- `results/aaai_private_base/experiment_registry.yaml:15-20`
- `results/aaai_private_base/checkpoint/source_inventory.json:2-4`
- `scripts/a10_env_setup.py:1-6`
- `scripts/gate0_build_private_package.py:94-98`
- `scripts/gate0_d3_operator_contract.py:92-96`

## Important distinction

The repository's existing experiments can treat the plaintext Qwen checkpoint as unavailable to an attacker and evaluate transformed-artifact confidentiality under that access assumption. That does not establish the stronger claim that the private backbone was trained completely from scratch, with a private initialization seed and private pretraining lineage.

The current package may remain useful for a **private imported-checkpoint** threat model. It cannot be used as evidence for the frozen **from-scratch private-base** stage requested here.

## Search audit

The text search covered Python/shell sources plus Markdown/YAML/JSON/text/log evidence under `scripts`, `configs`, and `results/aaai_private_base`. It found 64 `from_pretrained` matches and 38 `model.safetensors` matches. No paper-facing private-backbone training entrypoint, initialization commitment, or pretraining-data manifest was found. Tiny randomly initialized Qwen models in diagnostic scripts are controls only and are not the selected victim.

## Required remediation before this stage can run

Provide or train a genuine private backbone from random initialization and freeze:

1. architecture/tokenizer provenance;
2. initialization seed commitment;
3. private pretraining-data manifest hash;
4. training code and environment hashes;
5. logs proving no public checkpoint or weight import;
6. final private checkpoint hash;
7. a newly generated transformed package root bound to that checkpoint.

Only then should Q0 continue to immutable attack-view/corpus manifests and preregistered margins, followed by Q1 collection.
