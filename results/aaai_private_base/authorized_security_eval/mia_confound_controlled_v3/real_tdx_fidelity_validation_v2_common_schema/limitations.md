# Limitations

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample subset
Schema: common-semantics schema v2

- This is a frozen 250-sample engineering validation, not formal or universal equivalence.
- Exact bit equality is not required; BF16/runtime and reduction-order differences remain.
- Source normalization is a predefined sensitivity control and cannot prove distributional identity.
- Phase-6 membership transfer status: not run because source fidelity did not pass.
- TPR at 0.1% FPR is not reported; TPR at 1% FPR would be exploratory for 115 nonmembers.
- No claim of zero leakage, formal privacy, information-theoretic security, or behavior beyond this subset is supported.
