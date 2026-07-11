# Open questions for external baselines

- Which published secure-LoRA methods actually protect PRIVATE base weights (vs public-base)?
- Which support real backward + optimizer under the untrusted-GPU threat model?
- Fair comparison dimensions: trusted-invocations/step, VRAM, wall-clock, utility delta.
- Can any run decoder-only generation from the same protected package (our unification claim)?
- Scale: which report >=0.5B real models on real datasets (not toy)?
