# External literature-baseline registry (PLACEHOLDER — not implemented)

Per the frozen plan: **do not implement external private-training baselines yet, and
do not invent unsupported capabilities.** This registry records candidates; each row's
capabilities stay blank/`TBD` until confirmed from the *original paper AND code*. A
candidate is NOT a valid training baseline until matching LoRA/PEFT training support is
verified.

## Inference-only methods (training_support = N/A)
These are retained ONLY in the existing generation/inference comparison; they are NOT
reshaped into training baselines and NOT attributed a training variant they don't have.
- **STIP** — inference obfuscation. `training_support = N/A`.
- **ObfuscaTune** — inference obfuscation. `training_support = N/A`.
- **Amulet-style nonlinear islands** — inference nonlinear handling. `training_support = N/A`.

## Explicitly NOT used as main privacy-system baselines at this stage
(these are PEFT/efficiency or DP methods, not from-scratch private-base secure training,
and were named as out-of-scope for the main comparison): QLoRA, full fine-tuning, DP-SGD,
LoRA-FA, AdaLoRA, DoRA. They may appear later only as clearly-labeled references if a
fair comparison dimension is established.

## Candidate secure-LoRA / private-fine-tuning papers (to research)
To be filled during the dedicated literature pass. Each needs: title, venue/year,
official link, code repo, threat model, base-weight privacy, input/label/adapter/gradient
protection, LoRA/PEFT + backward support, optimizer location, TEE/MPC/HE/FHE/split-learning
requirement, decoder-only + generation support, real model scale, real dataset,
open-source reproducibility, fair-comparison dimensions, and implementation effort.
Current entries: `<secure-LoRA-candidate-1..N>` = `unverified` (see
`compatibility_matrix.csv`).

## Reference (not an external baseline)
**Plaintext LoRA** (L1/L3/L5) is the correctness + utility reference, NOT the paper's
only external baseline. The archived rank-masked-SGD (L9) is a frozen ablation isolating
adapter/loss protection from the full private-base system — not rerun.
