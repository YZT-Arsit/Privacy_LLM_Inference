#!/usr/bin/env python3
"""Preserve the invalid-control attempt and promote corrected validation."""
from pathlib import Path
root=Path(__file__).resolve().parent
for suffix in ('json','md'):
    current=root/f'fixed_length_v2_validation.{suffix}'
    attempt=root/f'fixed_length_v2_validation_attempt1.{suffix}'
    corrected=root/f'fixed_length_v2_validation_v2.{suffix}'
    if attempt.exists(): raise RuntimeError(f'refusing to overwrite {attempt}')
    attempt.write_bytes(current.read_bytes())
    current.write_bytes(corrected.read_bytes())
print('promoted corrected split-local mapping validation; attempt1 preserved')
