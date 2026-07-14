# Statistical method

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample subset
Schema: common-semantics schema v2

- Per-feature comparison uses population SD (`ddof=0`), 5/50/95% empirical quantiles, paired absolute/relative differences, Pearson and Spearman correlations where non-degenerate, two-sample KS, Wasserstein distance, and exact min/max range overlap.
- Source models use five-fold `GroupKFold`; a simulated/real sample pair always remains in the same fold. Models are standardized L2 logistic regressions (`C=0.01`).
- The predeclared numerical sensitivity transform estimates source-specific mean/SD only from each training fold and maps both known collection paths to pooled moments. It uses no membership labels.
- Source AUC intervals use 2000 paired sample-group percentile bootstrap replicates over fixed out-of-fold scores.
- Shuffled control uses 25 pairwise source-label swaps, preserving one simulated/one real label per sample pair.
- The source gate is an engineering fidelity criterion, not a formal equivalence test. Phase-6 membership evaluation is forbidden unless it passes.
