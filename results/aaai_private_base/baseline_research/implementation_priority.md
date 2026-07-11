# External baseline implementation priority

Deferred until after Gate 0. Priority when the literature pass runs:
1. TEE-based secure LoRA/PEFT training papers with open code + decoder-only support (closest threat-model match).
2. Split-learning / MPC LoRA training (different trust model; fair-dimension mapping needed).
3. HE/FHE fine-tuning (likely scale-incompatible; record but low priority).

Do NOT implement any until the original paper + code confirm matching training support.
