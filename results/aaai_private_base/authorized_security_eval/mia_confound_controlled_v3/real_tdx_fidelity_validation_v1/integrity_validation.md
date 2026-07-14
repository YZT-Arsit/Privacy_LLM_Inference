# Integrity validation

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample fidelity-validation subset

Status: **PASS with downstream semantic-mismatch hold**

- Frozen shadow: 2 (seed 1234).
- Source adapter SHA-256: `7c46d98316c5f09266cade307ecd2653f81c5fcfa021eef5536e326d45547ec9`; trusted-side fold probe max error `5.21540641784668e-08`.
- Masked adapter SHA-256: `5669c7d35c7dec9723ceab87e5019258f2979563fbdfeb7dfcb78f233f005c1b`.
- Transformed package root: `bfd578b809ef2313ed623c88893b1199f10be770f7678c1346ad473e36f0cde1`.
- Feature schema SHA-256: `7be799cbebf59d383c805e2195cf855bf22de3ade04c511dd2dd68069486406e`; 611 ordered columns verified.
- Sample-set SHA-256: `ea399513af254681b1fba253735915936399a2a14e020ede3eb023a7a262baef`; 250 ordered IDs verified.
- Schema/order/finite gate: True.
- Both collections report fresh attestation verified, DEBUG=false, report-data bound.
- No labels, loss, gradients, optimizer state, run/session IDs, or runtime metadata appear in either feature matrix.
- Handoff byte/hash integrity passes; fidelity conclusion is held only because the frozen simulated feature has no semantically matched real-runtime collection point.
