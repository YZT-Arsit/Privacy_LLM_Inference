# Common V2 feature schema

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample subset
Schema: common-semantics schema v2

- Frozen ordered columns: **610**.
- Canonical definition hash: `311f79a5951906839a91725bc24abf9ed09dd59f5b1c1c586ba6adaf66156a67`.
- Families: attention 384, KV 192, hidden 24, masked-logit 10.
- Derivation: exact ordered column selection from existing hash-verified simulated and real captures; no recollection and no value synthesis.
- Exclusion: `hidden.final.last_l2` only, because no semantically identical GPU-visible real tensor exists.
- The three near-constant layer-0 K reductions are retained: their graph semantics match and their tiny differences are consistent with runtime precision.
