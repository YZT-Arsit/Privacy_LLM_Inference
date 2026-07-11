# H800 unified dry run (H800_UNIFIED_DRY_RUN_PASS)

root hash matches pinned bfd578b8: True

- Check A base masked forward vs HF plaintext: top1 1.0000, cos 1.000320, max|Δ| 0.000 -> **True**
- Check B LoRA connectivity: 168/168 targets, base_grad=False, loss@init 0.4554 vs HF base 0.4539 -> **True**
- Check C operator LoRA equivalence (fp64, real folded base): **True**
- counters clean: silent_fallbacks 0, nonlinear_trusted_calls 0, plaintext_hidden 0

**H800_UNIFIED_DRY_RUN_PASS** (diagnostic non-TDX boundary; not a paper result)
