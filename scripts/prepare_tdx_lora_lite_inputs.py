"""Prepare TDX-lite LoRA replay inputs + a ready-to-run decode command.

After a trusted H800 reference LoRA decode (the local/remote folded-LoRA probe,
which echoes the trusted ``input_ids`` and the generated ``package_token_ids``),
this builds everything the TDX guest needs to REPLAY that decode in lite mode and
check it bit-for-bit -- WITHOUT a full model, the 26GB base package, or the raw
LoRA on the TDX side. The folded base + folded LoRA live only on the H800 worker.

Inputs: the H800 reference JSON (``--reference-json``), the TDX-side embedding
artifact path (``--embedding-path``; recorded into the command, hashed if it
exists locally), the folded-LoRA package (``--lora-folded-package-path``; only its
PUBLIC metadata -- adapter_hash / rank / alpha / target_modules -- is read).

Outputs (under ``--output-dir``): ``tdx_lora_input_ids.json``,
``tdx_lora_expected_tokens.json``, a combined ``tdx_lora_replay.json`` (+ artifact
hashes), and the command file ``run_tdx_lora_lite_decode.sh``.

Example::

    python scripts/prepare_tdx_lora_lite_inputs.py \\
        --reference-json outputs/qwen7b_lora_folded_remote_decode_probe.json \\
        --embedding-path /root/.../qwen7b_boundary_artifact_cuda \\
        --lora-folded-package-path /root/.../qwen7b_lora_folded_synth_r4 \\
        --gpu-worker-url http://127.0.0.1:18083 \\
        --output-dir outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ids(data, *keys):
    for k in keys:
        v = data.get(k) if isinstance(data, dict) else None
        if v:
            return [int(x) for x in (v[0] if isinstance(v[0], list) else v)]
    return None


def _dir_hash(path):
    """sha256 over (relpath, sha256) of every file under a dir, or of a file.
    Returns None if the path does not exist locally (TDX path may differ)."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    files = [p] if p.is_file() else sorted(
        q for q in p.rglob("*") if q.is_file())
    for q in files:
        rel = q.name if p.is_file() else q.relative_to(p).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(hashlib.sha256(q.read_bytes()).hexdigest().encode("utf-8"))
    return h.hexdigest()


def _lora_public_meta(lora_pkg):
    """PUBLIC folded-LoRA metadata only (no raw A/B). Returns ({}, None) if absent."""
    if not lora_pkg:
        return {}, None
    try:
        from pllo.deployment.lora_folded_package import load_lora_meta
        meta = load_lora_meta(lora_pkg)
    except Exception:                                       # noqa: BLE001
        return {}, None
    pub = {k: meta.get(k) for k in (
        "adapter_hash", "rank", "alpha", "scaling", "target_modules",
        "base_package_manifest_hash", "package_type", "trusted_setup")}
    man = None
    try:
        from pllo.deployment import compute_manifest_hash, load_manifest
        man = compute_manifest_hash(load_manifest(lora_pkg))
    except Exception:                                       # noqa: BLE001
        man = None
    return pub, man


def build_replay(reference: dict, *, embedding_path, lora_pkg, gpu_worker_url,
                 expected_override=None, max_new_tokens=None, dtype="bfloat16",
                 device="cpu", audit="true") -> dict:
    """Build the TDX-lite replay descriptor (pure; no file I/O)."""
    input_ids = _ids(reference, "input_ids")
    expected = (expected_override if expected_override is not None
                else _ids(reference, "package_token_ids", "reference_token_ids",
                          "expected_token_ids"))
    if input_ids is None:
        raise SystemExit("reference JSON has no 'input_ids' (rerun the H800 "
                         "reference probe; it echoes trusted input_ids)")
    if expected is None:
        raise SystemExit("reference JSON has no token ids; pass "
                         "--expected-token-ids")
    seq_len = len(input_ids)
    n_new = int(max_new_tokens if max_new_tokens is not None else len(expected))
    lora_pub, lora_manifest_hash = _lora_public_meta(lora_pkg)
    return {
        "stage": "tdx_lora_lite_replay",
        "source_reference": reference.get("stage"),
        "gpu_worker_url": gpu_worker_url,
        "embedding_path": embedding_path,
        "input_ids": input_ids,
        "expected_token_ids": expected,
        "seq_len": seq_len, "max_new_tokens": n_new,
        "dtype": dtype, "device": device, "audit": audit,
        # PUBLIC folded-LoRA metadata for provenance (the boundary never loads the
        # LoRA package; the H800 worker holds + merges it).
        "lora_public_metadata": lora_pub,
        "artifact_hashes": {
            "embedding_artifact_sha256": _dir_hash(embedding_path),
            "lora_folded_package_manifest_hash": lora_manifest_hash,
            "base_package_manifest_hash": lora_pub.get(
                "base_package_manifest_hash"),
        },
        "tdx_constraints": {
            "no_full_model_on_tdx": True,
            "no_base_folded_package_on_tdx": True,
            "no_raw_lora_on_tdx": True,
            "skip_reference": True,
        },
    }


def _command(replay: dict, *, input_ids_file, expected_tokens_file,
             output_json) -> str:
    expected_csv = ",".join(str(t) for t in replay["expected_token_ids"])
    pub = replay.get("lora_public_metadata") or {}
    lora_id = pub.get("adapter_hash") or "n/a"
    return "\n".join([
        "#!/usr/bin/env bash",
        "# TDX-lite private-LoRA decode replay (generated).",
        "# Boundary holds ONLY the embedding artifact -- no full model, no base",
        "# folded package, no raw LoRA. The H800 worker holds base + folded LoRA.",
        "# folded-LoRA adapter_hash (public id) = %s" % lora_id,
        "# rank=%s alpha=%s target_modules=%s"
        % (pub.get("rank"), pub.get("alpha"), pub.get("target_modules")),
        "set -euo pipefail",
        "",
        "python scripts/run_tee_gpu_protocol_demo.py \\",
        "  --mode boundary_client \\",
        "  --gpu-backend qwen7b_folded_package \\",
        "  --gpu-worker-url %s \\" % replay["gpu_worker_url"],
        "  --embedding-path %s \\" % replay["embedding_path"],
        "  --skip-reference true \\",
        "  --input-ids-file %s \\" % input_ids_file,
        "  --expected-token-ids %s \\" % expected_csv,
        "  --seq-len %d --max-new-tokens %d \\"
        % (replay["seq_len"], replay["max_new_tokens"]),
        "  --dtype %s --device %s --audit %s \\"
        % (replay["dtype"], replay["device"], replay["audit"]),
        "  --output-json %s" % output_json,
        "",
        "# For the ATTESTED run, add (same flags) + regenerate the runtime hash",
        "# first (see docs/runbooks/REAL_H800_TDX_LORA_RUNBOOK.md):",
        "#   --attestation-evidence evidence.json --expected-mr-td <mr_td>",
        "",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-json", required=True)
    ap.add_argument("--embedding-path", required=True,
                    help="TDX-side boundary embedding artifact dir (recorded)")
    ap.add_argument("--lora-folded-package-path", default=None,
                    help="folded-LoRA package (PUBLIC metadata only is read)")
    ap.add_argument("--gpu-worker-url", default="http://127.0.0.1:18083")
    ap.add_argument("--expected-token-ids", default=None,
                    help="override expected ids (csv); default: from reference")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--audit", default="true")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--command-file", default=None,
                    help="default: <output-dir>/run_tdx_lora_lite_decode.sh")
    args = ap.parse_args()

    expected_override = None
    if args.expected_token_ids:
        expected_override = [int(x) for x in args.expected_token_ids.split(",")
                             if x.strip()]

    reference = _load(args.reference_json)
    replay = build_replay(
        reference, embedding_path=args.embedding_path,
        lora_pkg=args.lora_folded_package_path,
        gpu_worker_url=args.gpu_worker_url, expected_override=expected_override,
        max_new_tokens=args.max_new_tokens, dtype=args.dtype,
        device=args.device, audit=args.audit)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_file = out / "tdx_lora_input_ids.json"
    exp_file = out / "tdx_lora_expected_tokens.json"
    replay_file = out / "tdx_lora_replay.json"
    cmd_file = Path(args.command_file) if args.command_file \
        else out / "run_tdx_lora_lite_decode.sh"
    decode_out = (out / "tdx_lora_lite_decode.json").as_posix()

    ids_file.write_text(json.dumps(
        {"input_ids": replay["input_ids"], "seq_len": replay["seq_len"]},
        indent=2), encoding="utf-8")
    exp_file.write_text(json.dumps(
        {"expected_token_ids": replay["expected_token_ids"]}, indent=2),
        encoding="utf-8")
    replay_file.write_text(json.dumps(replay, indent=2, default=str),
                           encoding="utf-8")
    cmd = _command(replay, input_ids_file=ids_file.as_posix(),
                   expected_tokens_file=exp_file.as_posix(),
                   output_json=decode_out)
    cmd_file.write_text(cmd, encoding="utf-8")
    try:
        cmd_file.chmod(0o755)
    except OSError:
        pass

    print("=== TDX-lite LoRA replay prepared ===")
    print("input_ids (%d) -> %s" % (replay["seq_len"], ids_file))
    print("expected_token_ids=%s -> %s"
          % (replay["expected_token_ids"], exp_file))
    print("replay descriptor -> %s" % replay_file)
    print("command file -> %s" % cmd_file)
    print("artifact_hashes=%s" % json.dumps(replay["artifact_hashes"]))
    print("lora_public_metadata=%s" % json.dumps(replay["lora_public_metadata"],
                                                 default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
