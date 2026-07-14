# CPU fidelity revalidation

Command:

```text
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider tests/obfuscatune tests/test_baselines_obfuscatune.py
```

Result: **52 passed in 5.10s**.

All 14 source hashes recorded in the earlier `pilot_config.json` were recomputed
and matched. This validates the existing CPU inference simulator only; it does
not validate Qwen LoRA training, a real TDX split, optimizer placement, adapter
export, or fresh-process handoff.

