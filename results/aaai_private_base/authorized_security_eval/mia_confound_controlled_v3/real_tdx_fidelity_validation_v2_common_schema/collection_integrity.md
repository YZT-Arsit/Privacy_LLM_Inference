# Collection integrity

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample subset
Schema: common-semantics schema v2

Status: **PASS**

- Derived from existing hash-verified v1 captures; real-TDX rerun: **not required / not performed**.
- Rows: simulated 250, real 250; ordered common columns: 610.
- Identical column names/order: True.
- Missing/non-finite values: 0; duplicate real rows: 0.
- Frozen sample index entries: 250.
- V1 fresh attestation and 250/250 exact V0 text checks remain referenced by SHA-256; no attestation claim is regenerated here.
- No labels or runtime metadata occur in either selected feature matrix.
