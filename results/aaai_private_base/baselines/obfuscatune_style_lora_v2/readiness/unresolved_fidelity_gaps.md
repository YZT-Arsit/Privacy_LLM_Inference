# Unresolved fidelity gaps after Phase F

1. Paper does not specify LoRA coordinates or optimizer placement; Candidate 2
   is a conservative adaptation, not a recovered missing algorithm.
2. Transformed-coordinate AdamW is not canonical AdamW. Utility differences may
   reflect optimizer coordinates as well as system execution.
3. Paper experiments use GPT-2, rank 16/alpha 32/dropout .05, LR 3e-5, batch 1,
   ten epochs and a GPU-simulated TEE. Qwen, rank 8/alpha 16/dropout 0, matched
   WD001 recipe, 750 steps, and Intel TDX are explicit adaptations.
4. Full Qwen RMSNorm/RoPE/GQA/SwiGLU backward has not been implemented or tested.
5. Adapter export/handoff and transform refresh are absent from the paper and
   deferred from the minimum run.
6. The baseline exposes true Q/K/V and plaintext KV cache and uses a proprietary
   secret-backbone threat model, unlike G2.

These gaps do not make a carefully labeled adapted-design baseline impossible,
but they prevent an official-reproduction label and weaken direct superiority
claims in the main paper table.
