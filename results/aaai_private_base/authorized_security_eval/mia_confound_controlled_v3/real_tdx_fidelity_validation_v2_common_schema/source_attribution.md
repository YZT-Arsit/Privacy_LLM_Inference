# Source attribution

Collection mode: REAL_TDX_BACKED_VIEW
Scope: frozen 250-sample subset
Schema: common-semantics schema v2

- Raw full-schema source ROC-AUC: **0.8054**, 95% paired-bootstrap CI [0.7696, 0.8403].
- Predeclared train-fold source-conditional normalized ROC-AUC: **0.6722**, CI [0.6273, 0.7183].
- Shuffled-source-label mean AUC: 0.4888.
- Maximum univariate symmetric source AUC: 0.7600.
- Gate: **FAIL**.

The remaining source predictability is a multivariate numerical/runtime-domain effect among semantically matched features, not an explicit metadata or computational-quantity identifier. Score agreement alone is not used to override this gate.

| Top univariate feature | Symmetric source AUC | Mean absolute difference | KS |
|---|---:|---:|---:|
| `hidden.layer_00.last_l2` | 0.7600 | 3.099e-08 | 0.520 |
| `kv.v.layer_00.last_l2_by_head.std` | 0.5980 | 1.580e-09 | 0.196 |
| `kv.v.layer_00.last_l2_by_head.min` | 0.5900 | 2.682e-09 | 0.180 |
| `kv.v.layer_00.last_l2_by_head.mean` | 0.5820 | 1.580e-09 | 0.180 |
| `kv.k.layer_00.last_l2_by_head.mean` | 0.5354 | 4.211e-06 | 0.044 |
| `kv.k.layer_00.last_l2_by_head.min` | 0.5238 | 7.324e-06 | 0.032 |
| `kv.k.layer_00.last_l2_by_head.max` | 0.5180 | 1.099e-06 | 0.036 |
| `kv.v.layer_00.last_l2_by_head.max` | 0.5080 | 4.768e-10 | 0.016 |
| `kv.k.layer_12.last_l2_by_head.mean` | 0.5021 | 1.486e-05 | 0.008 |
| `kv.k.layer_11.last_l2_by_head.max` | 0.5021 | 2.146e-05 | 0.008 |
| `attention.layer_17.last_entropy_by_head.min` | 0.5020 | 1.100e-06 | 0.008 |
| `kv.k.layer_18.last_l2_by_head.max` | 0.5020 | 1.577e-05 | 0.008 |
| `hidden.layer_03.last_l2` | 0.5020 | 9.941e-06 | 0.008 |
| `hidden.layer_06.last_l2` | 0.5020 | 1.352e-05 | 0.008 |
| `kv.k.layer_05.last_l2_by_head.std` | 0.5020 | 9.274e-06 | 0.008 |
| `hidden.layer_04.last_l2` | 0.5020 | 1.334e-05 | 0.008 |
| `kv.v.layer_17.last_l2_by_head.min` | 0.5020 | 1.135e-05 | 0.008 |
| `kv.k.layer_12.last_l2_by_head.min` | 0.5020 | 1.385e-05 | 0.008 |
| `kv.k.layer_18.last_l2_by_head.mean` | 0.5020 | 1.253e-05 | 0.008 |
| `kv.k.layer_12.last_l2_by_head.max` | 0.5020 | 1.588e-05 | 0.008 |
