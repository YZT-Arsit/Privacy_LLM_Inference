# Joint real-TDX validation

Status: **PASS**

- Three 1000x610 matrices; identical schema/order/dtype: True.
- Shadow identity CV accuracy: 1.0000 (chance 0.3333); attributed to legitimate model behavior, not metadata.
- Mean shuffle/pseudo AUC: 0.5018/0.4714.
- Top joint univariate symmetric AUC: 0.5182 (`kv.k.layer_03.last_l2_by_head.std`).
- Disjoint ranges / forbidden columns / opposite-label exact row conflicts: 0/0/0.
