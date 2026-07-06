# Attack evaluation plan

Unified framework comparing the security of **plaintext_gpu**, **stip_qwen**,
**obfuscatune_qwen_orthogonal** and **ours_amulet_style** under one
AttackResult schema + threat-model taxonomy. `src/pllo/attacks/`,
`src/pllo/baselines/stip/`, `scripts/attacks/`, `tests/attacks/`. CPU-first;
never downloads; defenses untouched.

## 1. Attack matrix (this round)

Eight attacks, implemented per `docs/attack_definitions.md`. Implementation
levels: NN/KPA/permutation/ArrowMatch/frequency = **full**; EIA/BiSR-forward/PIA
= **best_effort**; BiSR-backward = **blocked** (needs split-FT gradients).

## 2. Threat models (Kerckhoffs)

Attacker knows the algorithm, never the TEE secrets (orthogonal matrices,
padding, permutations, fresh masks).
- **closed_model_no_weight_access** (default deployment): no weights/table.
- **public_model**: has the public table (sanity / upper bound proxy).
- **known_plaintext** (KPA): has `(X, X*)` pairs.
- **weight_leakage_worst_case** (ArrowMatch): has obfuscated + public weights.
- **split_inference** (PIA / BiSR): observes activations (+ white-box preceding
  layers); BiSR-backward also needs training gradients.

The public Qwen used in experiments is a stand-in; real deployment does NOT give
the attacker the private weights.

## 3. Local smoke (CPU, offline)

```bash
python scripts/attacks/run_all_attacks.py                 # full matrix, toy, CPU
python scripts/attacks/run_baseline_audit.py              # STIP/ObfuscaTune correctness
python scripts/attacks/run_security_table.py              # measured Table B
python scripts/attacks/run_security_table.py --qualitative-draft
python scripts/attacks/summarize_attack_results.py
python scripts/attacks/capture_qwen_representations.py    # tiny-Qwen hook capture
# per-attack: run_{nn_embedding_inversion,eia_optimization,bre_bisr_attack,
#   permutation_multiset_attack,arrowmatch_attack,pia_prompt_inversion,
#   kpa_known_plaintext,frequency_distribution_attack}.py  (all support --dry-run)
```

Expected (toy): NN plaintext top1≈1 vs ours≈0; KPA recovers global linear once
`num_pairs≳1.5H`, fresh-pad fails; multiset leakage high for STIP/plaintext, 0
for orthogonal/ours; ArrowMatch recovers STIP's permutation (worst case), blocked
elsewhere; frequency/norm leaks under orthogonal; BiSR-backward blocked.

## 4. Building Table B

`run_security_table.py` →
`outputs/paper_security/security_attack_table_measured.{json,csv,md}` (measured
only; `missing`/`blocked`/`failed` otherwise). `--qualitative-draft` reads
`configs/attack_qualitative_matrix.json` → a `*_qualitative_draft` table with the
"not measured" banner; never overwrites the measured table.

## 5. Local vs server

- **Local CPU (works now):** all toy attacks, tiny-Qwen smoke + capture, baseline
  audits, structural/KPA/ArrowMatch/frequency, security table.
- **Recommended single GPU:** tiny/real-Qwen multi-step EIA/BiSR/PIA;
  Qwen-7B representation capture; ObfuscaTune/STIP/ours Qwen correctness at scale;
  larger-hidden KPA.
- **Must be server:** Qwen-7B / Llama-8B full attacks, multi-seed optimization
  sweeps, long-prompt PIA, full ObfuscaTune/STIP/ours comparison, latency+security
  joint table. See `docs/server_run_plan.md`.

## 6. Interpreting ArrowMatch worst-case (e.g. 98.5%)
ArrowMatch (and our prior gram/self-Gram audit) recover a permutation/mask under
**full weight access** — `weight_leakage_worst_case`, `has_model_weights=True`.
It is NOT the default `closed_model_no_weight_access` deployment. The table keeps
it in the worst-case row; read it as a weight-leakage consequence, not a
default-deployment break.

## 7. Interpreting closed-model no-weight-access
The default deployment: the attacker sees obfuscated activations but has no
private weights and no embedding table. NN inversion is `blocked` here; the
optimization attacks lose their white-box `F`. Quantitative breaks require one of
the stronger threat models above, which must be stated explicitly.
