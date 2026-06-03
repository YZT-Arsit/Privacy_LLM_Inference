# Evaluation Sufficiency Review

Reviewer-risk pass focused on whether the evaluation is *sufficient* given the absence of full Qwen / TinyLlama / LLaMA fine-tunes and the synthetic-tile correctness path. Goal: ensure the synthetic-tile scope is restated in every RQ Result paragraph, not only in sec:limitations.

## Risk items

- **Risk ID:** `EVA-01`
  - Severity: **high**
  - Dimension: `evaluation_sufficiency_risk`
  - Location: sec:eval:rq1 / rq5 / rq8 / rq11 + sec:limitations item 5
  - Risky wording or missing explanation: Every correctness / LoRA / toy-task / stability RQ uses 'synthetic' or 'tiny-HF' configurations. The paper does name this as a limitation, but the evaluation section reads as 'we evaluated' rather than 'we evaluated on synthetic tiles only'.
  - Why a reviewer may object: Reviewer will say 'this is not a real LLM evaluation.' We cannot add new experiments under Stage 7.6c; we must surface the limitation in every RQ result paragraph rather than only in sec:limitations.
  - Recommended revision: Append 'on synthetic / tiny-HF configurations only' to the Result line of RQ1, RQ5, RQ8, RQ11 so a reviewer skimming Result paragraphs sees the scope immediately, not only in the Limitation paragraph.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P0`

- **Risk ID:** `EVA-02`
  - Severity: **medium**
  - Dimension: `evaluation_sufficiency_risk`
  - Location: sec:eval:rq3 (Workload profile)
  - Risky wording or missing explanation: RQ3 names 'tslp_trusted_nonlinear_baseline' and 'amulet_style_reference' as cost-model baselines but uses the word 'amulet_style' which is exactly the vague label the Stage 7.5c spec ruled out.
  - Why a reviewer may object: Stage 7.5c spec forbids 'Amulet-style' phrasing. The RQ3 cost-model table still uses the legacy label.
  - Recommended revision: Rename 'amulet_style_reference' to 'amulet_static_phq_cost_model' in the narrative paragraph (the artifact key may stay for backward compatibility, but the prose should be precise).
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `EVA-03`
  - Severity: **medium**
  - Dimension: `evaluation_sufficiency_risk`
  - Location: sec:eval:rq6 (Rank-inference and timing)
  - Risky wording or missing explanation: Reports four 'high' risk rows (gradient rank inference, stronger-dummy spectral, stronger-dummy gradient) and one 'medium' (dummy-strategy classifier). The Result paragraph states the numbers but does not tie each 'high' to a future-work item.
  - Why a reviewer may object: Reviewer who sees 'high' without a next-step plan may ask 'so why is this a contribution?'
  - Recommended revision: After each 'high' or 'medium' status add a sentence: 'This is reported as open; heterogeneous padded rank (sec:conclusion item c) is the proposed follow-up.'
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P1`

- **Risk ID:** `EVA-04`
  - Severity: **low**
  - Dimension: `evaluation_sufficiency_risk`
  - Location: sec:eval:rq2 + sec:limitations item 9 (sequence length leak)
  - Risky wording or missing explanation: RQ2 covers KV cache append invariant but does not measure prompt-length leakage in the decode trace.
  - Why a reviewer may object: Reviewer may ask 'what about prompt-length recovery?' We must point to sec:limitations not promise an experiment we did not run.
  - Recommended revision: Append 'we do not measure prompt-length or seq-length recovery; this is explicit in sec:limitations item 9' to RQ2 Limitation paragraph.
  - New experiment needed: **no** ; wording fix enough: **yes**
  - Priority: `P2`

