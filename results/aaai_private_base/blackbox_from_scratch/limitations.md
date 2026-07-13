# Limitations

1. The selected victim is a public pretrained Qwen2.5-0.5B checkpoint, not a private backbone trained from scratch.
2. No backbone initialization seed commitment, private pretraining-data manifest, or private pretraining log exists for the selected victim.
3. No immutable B0/B1/B2/B3 attack corpus was collected for this stage.
4. Package replay, model stealing, alignment, MIA, cross-phase and attention-leakage attacks were not run.
5. No B2-B0 equivalence/non-inferiority decision can be made.
6. Existing private-base security results use Qwen weights as a secret proxy and must remain scoped to that different access assumption.
