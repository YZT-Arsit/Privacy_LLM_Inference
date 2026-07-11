# D2 identifiability / capability-extraction -- go/no-go

positive controls all passed: **True** (3/3)

- T0 (package only) functional weight recovered: **False** (must be False)
- T0 tied-view absolute basis recovered: **False** (must be False)
- T3 (weight-leakage worst case) functional recovered: **True** (EXPECTED True -- masks are not claimed to protect a plaintext-holding attacker)
- T0 spectral leaks (gamma-free weights): ['model.layers.0.mlp.down_proj_tilde', 'model.layers.0.self_attn.o_proj_tilde', 'model.layers.12.mlp.down_proj_tilde', 'model.layers.12.self_attn.o_proj_tilde', 'model.layers.23.mlp.down_proj_tilde', 'model.layers.23.self_attn.o_proj_tilde']

**Empirical private-base weight-confidentiality claim holds (T0/T2, private weights): True**

Scope: holds ONLY under the private-weight (no plaintext reference) assumption; BROKEN at T3 (weight leakage) as expected and shown.

Unavailable families (recorded, NOT failures): ['fastica_second_bss', 'generalized_procrustes_unanchored', 'full_residual_graph_chain_alignment', 'functional_extraction_full_generation']
