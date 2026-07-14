# Invalid Initial V2 Cohort Record

Status: **AUDIT_EVIDENCE_ONLY**  
Formal status: `INITIAL_COHORT_INVALID_FOR_FINAL_MIA`; `SOURCE_COHORT_CONFOUND_CONFIRMED`.

- Members: 500 seed-20260714 samples from the frozen official E2E train-derived population,
  captured as `e2e_train_member_v2_s1234.jsonl`.
- Nonmembers: 500 unique-MR official E2E test prompts, captured as `e2e_test_v2_s1234.jsonl`.
- Both used `capture_v2_prefill.py`, run ID `e2e_L12_s1234_full`, package root
  `bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1`, and adapter
  SHA-256 `1e29821f0b4cf4eb90bd748f171a14c2d2607b6e086b7d23bd2ad18b16e90cd2`.
- Exact-length matching retained 215 records per class and controlled sequence length completely.
- 446/803 features nevertheless had disjoint member/nonmember ranges; the classifier reached AUC 1.0.

Length matching could not remove official train/test semantic and construction differences. Because
membership and source split were identical, AUC 1.0 is source-cohort classification and is invalid as
final membership evidence. The corrective design uses 1,000 unique MR groups from one train-derived
pool, constrained-random 500/500 membership per shadow, identical C0 collection, and leave-shadow-run-out evaluation.
