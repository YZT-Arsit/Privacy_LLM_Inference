"""Boundary-artifact integrity guard.

Regression for the TDX folded_remote repeat-token bug: the trusted boundary
artifact's vocab scale had been hand-patched (bf16-quantised) so it no longer
matched the folded package head, silently mis-scaling the recovered logits and
producing degenerate (repeat-token) generation. ``load_embedding_artifact`` now
hard-checks the tensors file against the ``tensors_sha256`` recorded at build
time, so a modified/corrupted artifact fails loudly instead of generating garbage.

Run: python -m pytest tests/test_embedding_artifact_integrity.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.deployment.embedding_artifact import (  # noqa: E402
    ARTIFACT_META, build_embedding_artifact, load_embedding_artifact)
from pllo.ops.causal_lm_boundaries import VocabLogitMask  # noqa: E402


def _tiny_artifact(out_dir: Path, vocab: int = 8, hidden: int = 4):
    perm = torch.randperm(vocab)
    inv = torch.argsort(perm)
    scale = torch.rand(vocab) + 0.5
    vm = VocabLogitMask(permutation=perm, inverse_permutation=inv,
                        scale=scale, inverse_scale=1.0 / scale)
    build_embedding_artifact(
        out_dir, embed_tokens_weight=torch.randn(vocab, hidden),
        residual_mask_n0=torch.eye(hidden), vocab_mask=vm,
        meta={"rms_norm_eps": 1e-6, "seed": 7})


def test_clean_artifact_loads(tmp_path) -> None:
    _tiny_artifact(tmp_path)
    embed, n0, vm, meta = load_embedding_artifact(str(tmp_path))
    assert embed.shape[0] == 8 and meta["tensors_sha256"]


def test_tampered_tensors_file_is_rejected(tmp_path) -> None:
    _tiny_artifact(tmp_path)
    tpath = tmp_path / "boundary.safetensors"
    raw = bytearray(tpath.read_bytes())
    raw[-1] ^= 0xFF                       # flip the last byte (a stored scale)
    tpath.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="sha256"):
        load_embedding_artifact(str(tmp_path))


def test_verify_can_be_explicitly_disabled(tmp_path) -> None:
    _tiny_artifact(tmp_path)
    tpath = tmp_path / "boundary.safetensors"
    raw = bytearray(tpath.read_bytes())
    raw[-1] ^= 0xFF
    tpath.write_bytes(bytes(raw))
    # audited escape hatch: still loads (no raise) when verification is off
    embed, _, _, _ = load_embedding_artifact(
        str(tmp_path), verify_tensors_sha256=False)
    assert embed.shape[0] == 8


def test_stale_meta_sha_is_rejected(tmp_path) -> None:
    # the exact TDX failure mode: file body replaced, meta sha left unchanged
    _tiny_artifact(tmp_path)
    import json
    meta = json.loads((tmp_path / ARTIFACT_META).read_text())
    meta["tensors_sha256"] = "deadbeef" * 8     # wrong claim
    (tmp_path / ARTIFACT_META).write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="does NOT match"):
        load_embedding_artifact(str(tmp_path))
