# Limitations (v2)

- Corpus BLEU, corpus chrF, and ROUGE-L do not move uniformly; competitive BLEU must not be generalized to every quality metric.
- Three seeds characterize variability weakly and do not support a formal equivalence conclusion.
- The hierarchical seed-and-example intervals are exploratory, not equivalence tests.
- G2 seed-1234 uses a manifest-matched canonical cache because the ordinary workspace copy is truncated; both copies remain preserved.
- G1 BF16-matched, complete T2–T4, corrected MIA, deployable transformed-adapter, and ObfuscaTune system results are unavailable.
- TPR at 0.1% FPR is resolution-limited in small MIA folds; empirical negative counts and FPR resolution must accompany it.
- T1 retained trusted roundtrip timing but not a trusted-compute/transport-only profiling split; those fields remain MISSING.
- Projected TDX state and tensor bytes omit service, serialization, framing, and authentication overhead.
