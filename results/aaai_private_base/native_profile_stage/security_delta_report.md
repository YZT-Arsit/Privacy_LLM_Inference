# Profile N security-delta (PHASE 3.3, LOCAL)

Source: `security_delta.json`. Threat model: from-scratch private base (no plaintext weights, no
paired plaintext, masks secret). Profile N's GPU state is a **superset** of Profile E's: it adds
transformed AdamW m, v and per-step updates.

| Attack | Profile E surface {A~,B~} | Profile N surface {A~,B~,m,v,updates} |
|---|---|---|
| plaintext ΔW recovery (SVD of masked product) | rel-err **1.00** (not recovered) | rel-err **1.00** (unchanged) |
| does m,v reduce plaintext-ΔW rel-err below E? | — | **no** (rel-err 1.00) |
| mask recovery from m,v | n/a | needs paired plaintext (as S1); moment norm-preservation gap **0** (documented) |

**Measured delta:** exposing transformed m,v does **not** reduce plaintext-ΔW recovery below
Profile E (both rely on the secret masks / TEE). The **additional exposure** is the second-moment
gradient structure (m,v) in the transformed basis, whose norms/Grams are preserved (documented, as
for A~,B~). Cross-step linkability/membership from accumulated m,v is analogous to S4/S5 and is
**flagged, not fully characterized** locally.

**We do NOT claim Profile N has the same security surface as Profile E.** Profile N trades a lower
trusted footprint for an additional GPU-visible transformed-optimizer-state surface; the plaintext-
ΔW confidentiality is unchanged, but the extra m,v exposure is real and reported.
