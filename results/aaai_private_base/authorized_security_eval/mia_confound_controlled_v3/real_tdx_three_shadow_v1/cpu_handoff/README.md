# CPU handoff: three real-TDX shadow collections

This handoff is complete and hash-verified. It contains the three terminal collection manifests, frozen adapters and membership maps by path and hash, V0/V2 matrices, sample joins, the 610-column schema, prohibited-field policy, fresh TDX attestation evidence, per-shadow validity reports, joint validation, environment metadata, and recorded collection commands.

Start with `handoff_manifest.json`, verify every entry in `artifact_hashes.json`, then run the frozen offline CPU pipeline. The intended next analysis is leave-one-shadow-out MIA with V0, V2, and V0+V2 comparisons plus the listed controls and intervals.

The collection session did **not** run the final paper-facing MIA classifier. Files `shadow_*/a10_session.json` and `shadow_*/tdx_session.json` are intentionally excluded because they contain historical session secrets and are not analysis inputs.

Collection mode: `REAL_TDX_BACKED_VIEW`. Scope: three 1,000-sample pools, balanced 500/500 per shadow. Schema: common-semantics schema v2, 610 frozen feature columns.
