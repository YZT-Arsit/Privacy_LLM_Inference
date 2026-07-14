# Limitations

Collection mode: SIMULATED_PROTOCOL_VIEW

- Only three shadow models are evaluated; cross-shadow uncertainty is exploratory.
- The primary LOSO folds reuse the same 1,000 sample identities across shadows. Semantic-group-disjoint sensitivity is therefore required and reported.
- The view is simulated and must not be described as real-TDX-backed evidence.
- TPR at 0.1% FPR is unsupported with 500 negative examples per held-out shadow.
- Hyperparameters and classifiers are deliberately small and frozen; the pilot does not establish optimal membership inference.
- Two selected secondary LinearSVC V0+V2 fits (held-out shadows 2 and 3, C=10) reached the iteration limit. All 21 selected primary logistic fits and all nine pooled group-sensitivity results converged; paper-facing claims use logistic regression only.
- A shadow-ID classifier can distinguish independently trained model states, but no shadow metadata is present in the matrix.
- Results do not establish zero leakage, formal privacy, equivalence of V0 and V2, or universal resistance.
