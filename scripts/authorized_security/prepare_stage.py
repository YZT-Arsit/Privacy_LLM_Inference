#!/usr/bin/env python3
"""Create corrected stage records without mutating any frozen result."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from view_contracts import registry_json


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results/aaai_private_base/authorized_security_eval"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new_or_same(path: Path, content: str) -> None:
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"refusing to overwrite non-identical stage record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    write_new_or_same(out / "authorization_scope.md", """# Authorization scope

All evaluated code, models, artifacts, adapters, logs, datasets, A10/TDX hosts,
and runtime endpoints are researcher-owned or explicitly controlled fixtures.
The evaluation is limited to this repository and the isolated experimental
infrastructure. It does not probe third-party systems, use third-party
credentials, process third-party private communications, create persistence,
or exfiltrate artifacts.
""")
    write_new_or_same(out / "scenario_definition.md", """# Scenario definition

The target scenario is a fully private, independently trained backbone with no
public same-lineage checkpoint available to the primary adversary. The
untrusted accelerator observes only protocol-defined transformed artifacts and
tensors. TDX secrets and plaintext trusted state are unavailable to V0/V1/V2.
This is an empirical confidentiality evaluation, not a formal security claim.
""")
    write_new_or_same(out / "experimental_proxy_note.md", """# Experimental proxy

Qwen2.5-0.5B is a reproducible experimental proxy. Its plaintext checkpoint is
available only to V3, the trusted evaluator, for package construction, scoring,
and positive controls. The primary V2 evaluation is not granted plaintext Qwen
weights. Known-plaintext access is reported separately as a stress test and is
not mixed into the primary result. The preserved prior audit correctly records
that the proxy loads Qwen weights.
""")

    candidates = [
        REPO / "results/aaai_private_base/blackbox_from_scratch/provenance_audit.json",
        REPO / "results/aaai_private_base/generative_lora/monitor/e2e_L12_s1234_full/completion_manifest.json",
        REPO / "results/aaai_private_base/generative_lora/protected_runs/e2e_L12_s1234_full.json",
        REPO / "results/aaai_private_base/generative_lora/protected_runs/e2e_L12_s1234_full.training_tdx_counters.json",
        REPO / "results/aaai_private_base/generative_lora/protected_runs/e2e_L12_s1234_full.generation_tdx_counters.json",
    ]
    rows = []
    for path in candidates:
        if path.is_file():
            rows.append((str(path.relative_to(REPO)), sha256(path), path.stat().st_size, "reused_read_only"))
    inv = out / "reused_artifact_inventory.csv"
    text = "path,sha256,bytes,disposition\n" + "".join(
        f'{json.dumps(p)},{digest},{size},{disposition}\n' for p, digest, size, disposition in rows
    )
    write_new_or_same(inv, text)

    registry = json.dumps(registry_json(), indent=2, sort_keys=True) + "\n"
    write_new_or_same(out / "views/view_registry.json", registry)
    write_new_or_same(out / "views/allowed_fields.json", registry)
    write_new_or_same(out / "views/field_allowlists.json", registry)
    schemas = {
        "schema_version": "1.0",
        "record_type": "object",
        "additionalProperties": False,
        "required_for_all_views": ["sample_id"],
        "per_view_properties": registry_json()["views"],
    }
    write_new_or_same(out / "views/view_schema.json", json.dumps(schemas, indent=2, sort_keys=True) + "\n")
    write_new_or_same(out / "views/view_matrix.md", """# Enforced views

| View | Role | Plaintext weights | Final output | GPU-visible transformed tensors |
|---|---|---:|---:|---:|
| V0 | label-only primary baseline | no | yes | no |
| V1 | deployed API output | no | yes | no |
| V2 | untrusted GPU transformed view | no | no | yes |
| V3 | trusted evaluator / positive control | yes | yes | evaluator-defined |

Access is default-deny and checked both when records are constructed and when a
field is read. Live model/oracle objects are rejected.
""")
    write_new_or_same(out / "views/audit.md", """# View-enforcement audit

- Every attacker-facing record requires an explicit `V0`, `V1`, `V2`, or `V3`.
- Construction rejects keys outside the exact allowlist; field reads are checked again.
- V0 rejects logits, timing, hidden states, attention, package tensors, and gradients.
- V2 rejects plaintext weights/adapters, labels, references, raw evaluator inputs,
  mask/inverse/gamma/rank/TDX secrets, and live model/oracle objects.
- Corpus manifests bind ordered deterministic sample IDs, file hashes, byte sizes,
  and record counts; a changed frozen manifest is rejected rather than overwritten.
- V3 is restricted to trusted positive controls and is not mixed into primary V2 results.

The enforcement test result is recorded separately in `enforcement_tests.json`.
""")
    source_snapshot = out / "source_hashes_before.sha256"
    if not source_snapshot.exists():
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO, check=True, text=True, capture_output=True
        ).stdout.splitlines()
        hashes = []
        for rel in tracked:
            p = REPO / rel
            if p.is_file():
                hashes.append(f"{sha256(p)}  {rel}")
        write_new_or_same(source_snapshot, "\n".join(hashes) + "\n")
    write_new_or_same(out / "final_status.md", """# Final status

`AUTHORIZED_SECURITY_EVALUATION_PARTIAL`

Remaining experiment cells:

- immutable V0/V2 evaluation corpus;
- protected artifact misuse-resistance R0--R5;
- surrogate fidelity V0/V2 comparison;
- primary alignment and separated known-plaintext stress test;
- private LoRA functionality evaluation;
- membership evaluation;
- runtime/input leakage and attention ablation;
- cross-phase evaluation and final scoped statistics.
""")
    print(json.dumps({"output": str(out), "reused_artifacts": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
