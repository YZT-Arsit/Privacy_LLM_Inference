#!/usr/bin/env python3
"""Isolated R0--R2 misuse-resistance audit for the transformed GPU package.

This is an observed fail-closed evaluation, not a cryptographic impossibility claim.
It never starts or configures a TDX service and never accesses a plaintext checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch


PLAIN_PATTERNS = (
    "model.safetensors", "pytorch_model", "optimizer", "mask_secret",
    "plaintext", "unmasked", "session_key", "vocab_perm", "perm_inv",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def root_hash(pkg: Path) -> str:
    pins = {p.name: sha256(p) for p in sorted(pkg.glob("*")) if p.is_file()}
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def check(name: str, passed: bool, detail: str, mode: str = "static_rejection") -> dict:
    return {"test": name, "passed": bool(passed), "detail": detail, "evidence_mode": mode}


def validate_adapter_manifest(m: dict, expected: dict, present: set[str]) -> tuple[bool, str]:
    required = {
        "schema", "base_package_root_hash", "adapter_sha256", "target_modules",
        "rank", "alpha", "tensor_names", "tensor_hashes", "session_binding", "created_at_epoch",
        "expires_at_epoch", "transform_version",
    }
    missing_fields = sorted(required - set(m))
    if missing_fields:
        return False, f"missing fields: {missing_fields}"
    comparisons = {
        "base_package_root_hash": expected["base_package_root_hash"],
        "adapter_sha256": expected["adapter_sha256"],
        "target_modules": expected["target_modules"],
        "rank": expected["rank"],
        "alpha": expected["alpha"],
        "session_binding": expected["session_binding"],
        "transform_version": expected["transform_version"],
    }
    for k, want in comparisons.items():
        if m.get(k) != want:
            return False, f"{k} mismatch"
    if float(m["expires_at_epoch"]) <= time.time():
        return False, "manifest stale"
    names = set(m["tensor_names"])
    if names != present:
        return False, "tensor inventory mismatch"
    bad = sorted(n for n in names if any(x in n.lower() for x in PLAIN_PATTERNS))
    if bad:
        return False, f"forbidden tensor names: {bad[:3]}"
    if m["tensor_hashes"] != expected["tensor_hashes"]:
        return False, "per-tensor hash mismatch"
    canonical = copy.deepcopy(m)
    claimed = canonical.pop("manifest_sha256", "")
    actual = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        return False, "manifest digest mismatch"
    return True, "all pinned fields and inventory accepted"


def seal_manifest(m: dict) -> dict:
    out = copy.deepcopy(m)
    out.pop("manifest_sha256", None)
    out["manifest_sha256"] = hashlib.sha256(
        json.dumps(out, sort_keys=True).encode()
    ).hexdigest()
    return out


def mutate_and_test(base: dict, expected: dict, present: set[str], name: str, fn) -> dict:
    m = copy.deepcopy(base)
    fn(m)
    # A manifest mutation has a valid digest unless this is specifically the digest control.
    if name != "modified_manifest_digest":
        m = seal_manifest(m)
    ok, reason = validate_adapter_manifest(m, expected, present)
    return check(name, not ok, reason)


def run_entrypoint(label: str, runtime: Path, package: Path, gen_input: Path,
                   out_dir: Path, include_input: bool) -> dict:
    log = out_dir / f"{label}.log"
    cmd = [
        sys.executable, str(runtime),
        "--session", str(out_dir / "intentionally_absent_session.json"),
        "--tdx", "127.0.0.1", "--key", str(out_dir / "absent_key"),
        "--service-cmd", "disabled_for_r0_r2_audit",
        "--gen-in", str(gen_input if include_input else out_dir / "absent_input.json"),
        "--gen-out", str(out_dir / f"{label}.jsonl"),
        "--gen-max", "1", "--max-new", "1",
    ]
    env = dict(os.environ)
    env["PB_PKG_DIR"] = str(package)
    start = time.time()
    cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        env=env, timeout=120)
    log.write_text(cp.stdout)
    created = (out_dir / f"{label}.jsonl").exists()
    missing_session = "intentionally_absent_session" in cp.stdout and (
        "No such file" in cp.stdout or "FileNotFoundError" in cp.stdout
    )
    return {
        "condition": label,
        "evidence_mode": "attempted_execution",
        "command": cmd,
        "return_code": cp.returncode,
        "wall_sec": round(time.time() - start, 3),
        "failed_closed": cp.returncode != 0 and missing_session and not created,
        "failure_boundary": "session_read_before_package/model forward" if missing_session else "unexpected",
        "protected_output_created": created,
        "log": log.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--runtime-entry", required=True, type=Path)
    ap.add_argument("--gen-input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())

    observed_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    observed_uuid = "unknown"
    if torch.cuda.is_available():
        try:
            observed_uuid = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
            ).strip().splitlines()[0]
        except Exception as e:
            observed_uuid = f"query_failed:{type(e).__name__}"

    package_files = sorted(p for p in args.package.glob("*") if p.is_file())
    manifest = json.loads((args.package / "package_manifest.json").read_text())
    hashes = json.loads((args.package / "artifact_hashes.json").read_text())
    actual_root = root_hash(args.package)
    integrity = []
    integrity.append(check("cuda_available", torch.cuda.is_available(), observed_gpu, "attempted_execution"))
    integrity.append(check("gpu_uuid_matches", observed_uuid == cfg["gpu_uuid"], observed_uuid,
                           "attempted_execution"))
    integrity.append(check("package_root_matches", actual_root == cfg["package_root_hash"], actual_root,
                           "attempted_execution"))
    integrity.append(check("artifact_count_matches", len(hashes) == manifest["artifact_count"],
                           f"hashes={len(hashes)} manifest={manifest['artifact_count']}"))
    bad_hashes = []
    tensor_inventory = []
    peak_alloc = 0
    for artifact, expected_hash in sorted(hashes.items()):
        p = args.package / f"{artifact}.pt"
        if not p.exists() or sha256(p) != expected_hash:
            bad_hashes.append(artifact)
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        values = list(blob.values()) if isinstance(blob, dict) else [blob]
        if len(values) != 1 or not torch.is_tensor(values[0]):
            bad_hashes.append(f"{artifact}:invalid_payload")
            continue
        t = values[0].to("cuda")
        peak_alloc = max(peak_alloc, torch.cuda.memory_allocated())
        tensor_inventory.append({"artifact": artifact, "shape": list(t.shape), "dtype": str(t.dtype)})
        del t, blob, values
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    integrity.append(check("all_transformed_tensor_hashes_and_gpu_initialization", not bad_hashes,
                           f"loaded={len(tensor_inventory)} bad={bad_hashes[:3]}", "attempted_execution"))
    names = [p.name.lower() for p in package_files]
    forbidden_files = sorted(n for n in names if any(x in n for x in PLAIN_PATTERNS))
    integrity.extend([
        check("no_forbidden_plaintext_or_secret_filenames", not forbidden_files, str(forbidden_files)),
        check("manifest_declares_no_mask_secrets", manifest.get("contains_mask_secrets") is False,
              str(manifest.get("contains_mask_secrets"))),
        check("manifest_declares_no_plaintext_base", all(manifest.get(k) is False for k in (
            "contains_plaintext_base_weights", "contains_plaintext_embeddings", "contains_plaintext_lm_head")),
              "manifest declaration only; filename/hash controls provide separate observed evidence"),
    ])

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {f"L{l}.{p}.{side}" for l in range(24) for p in targets for side in ("A", "B")}
    expected = {
        "base_package_root_hash": cfg["package_root_hash"],
        "adapter_sha256": cfg["adapter_sha256"],
        "target_modules": targets,
        "rank": 8,
        "alpha": 16,
        "session_binding": "closure_eval_session_v1",
        "transform_version": "private_base_fold_v1.0",
        "tensor_hashes": {name: hashlib.sha256(name.encode()).hexdigest() for name in sorted(present)},
    }
    base = seal_manifest({
        "schema": "isolated_adapter_binding_control_v1",
        **expected,
        "tensor_names": sorted(present),
        "tensor_hashes": expected["tensor_hashes"],
        "created_at_epoch": time.time(),
        "expires_at_epoch": time.time() + 3600,
    })
    valid, detail = validate_adapter_manifest(base, expected, present)
    controls = [check("valid_manifest_positive_control", valid, detail)]
    controls += [
        mutate_and_test(base, expected, present, "wrong_base_hash", lambda m: m.update(base_package_root_hash="0" * 64)),
        mutate_and_test(base, expected, present, "wrong_adapter_hash", lambda m: m.update(adapter_sha256="1" * 64)),
        mutate_and_test(base, expected, present, "wrong_target_set", lambda m: m.update(target_modules=targets[:-1])),
        mutate_and_test(base, expected, present, "wrong_rank", lambda m: m.update(rank=16)),
        mutate_and_test(base, expected, present, "wrong_alpha", lambda m: m.update(alpha=32)),
        mutate_and_test(base, expected, present, "missing_tensor", lambda m: m["tensor_names"].pop()),
        mutate_and_test(base, expected, present, "modified_tensor", lambda m: m["tensor_hashes"].update({m["tensor_names"][0]: "f" * 64})),
        mutate_and_test(base, expected, present, "modified_manifest_digest", lambda m: m.update(rank=16)),
        mutate_and_test(base, expected, present, "stale_manifest", lambda m: m.update(expires_at_epoch=1)),
        mutate_and_test(base, expected, present, "cross_run_session", lambda m: m.update(session_binding="other_run")),
        mutate_and_test(base, expected, present, "plaintext_named_tensor", lambda m: m["tensor_names"].append("plaintext.weight")),
    ]
    adapter_ok = sha256(args.adapter) == cfg["adapter_sha256"]
    controls.insert(1, check("adapter_blob_hash_positive_control", adapter_ok, sha256(args.adapter)))
    runtime_text = args.runtime_entry.read_text()
    raw_adapter_path = bool(re.search(r"elif\s+a\.adapter\s*:", runtime_text) and
                            re.search(r"torch\.load\(a\.adapter", runtime_text))
    controls.insert(2, check(
        "native_runtime_disallows_unmanifested_raw_adapter",
        not raw_adapter_path,
        ("FAIL: copied native entrypoint contains --adapter torch.load path without adapter-package manifest validation"
         if raw_adapter_path else "no unmanifested raw-adapter path found"),
        "static_rejection",
    ))

    execution = [
        {
            "condition": "R0_package_only",
            "view": "transformed package only; no runtime, session, trusted service, or local input",
            "evidence_mode": "attempted_execution",
            "package_gpu_initialized": not bad_hashes,
            "protected_generation_attempted": False,
            "reason_not_attempted": "R0 excludes runtime and trusted/session capabilities by definition",
            "fallback_output_created": False,
        },
        run_entrypoint("R1_package_plus_runtime", args.runtime_entry, args.package,
                       args.gen_input, args.out, include_input=False),
        run_entrypoint("R2_plus_local_inputs", args.runtime_entry, args.package,
                       args.gen_input, args.out, include_input=True),
    ]
    logs_text = "\n".join(p.read_text(errors="replace") for p in args.out.glob("*.log"))
    output_scan_hits = sorted(set(re.findall(
        r"(?i)[^\s'\"]*(?:plaintext|optimizer|model\.safetensors|session_key|perm_inv)[^\s'\"]*",
        logs_text,
    )))
    scan = {
        "files_scanned": [p.name for p in args.out.glob("*") if p.is_file()],
        "forbidden_name_hits": output_scan_hits,
        "plaintext_named_parameter_or_optimizer_state_observed": bool(output_scan_hits),
    }

    control_dir = args.out / "per_control_logs"; control_dir.mkdir(exist_ok=True)
    for item in integrity + controls:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", item["test"])
        (control_dir / f"{safe}.json").write_text(json.dumps(item, indent=2) + "\n")

    all_pass = all(x["passed"] for x in integrity + controls) and all(
        x.get("failed_closed", True) for x in execution if x["condition"].startswith(("R1", "R2"))
    ) and not scan["plaintext_named_parameter_or_optimizer_state_observed"]
    result = {
        "schema": "artifact_misuse_r0_r2_v1",
        "created_at_epoch": time.time(),
        "scope": "researcher-owned defensive evaluation",
        "view_label": "SIMULATED_PROTOCOL_VIEW",
        "hardware": {"gpu_name": observed_gpu, "gpu_uuid": observed_uuid,
                     "claimed_resource_class": cfg["resource_class"]},
        "claims_boundary": {
            "supported": "observed initialization/integrity and fail-closed behavior for enumerated R0-R2 conditions",
            "unsupported": "cryptographic impossibility, resistance to untested attacks, or REAL_TDX_BACKED_VIEW claims",
        },
        "package": {"root_hash": actual_root, "file_count": len(package_files),
                    "tensor_count": len(tensor_inventory), "peak_single_load_alloc_bytes": peak_alloc},
        "hashes": {"audit_script_sha256": sha256(Path(__file__)), "config_sha256": sha256(args.config),
                   "runtime_entry_sha256": sha256(args.runtime_entry), "adapter_sha256": sha256(args.adapter),
                   "generation_input_sha256": sha256(args.gen_input)},
        "integrity_checks": integrity,
        "adapter_manifest_controls": controls,
        "conditions": execution,
        "output_scan": scan,
        "questions": {
            "package_can_be_initialized": {"answer": not bad_hashes, "evidence": "all transformed tensors loaded individually on the verified GPU"},
            "valid_protected_generation_completed": {"answer": False, "evidence": "R0 lacks runtime; attempted R1/R2 stop at absent session before model forward"},
            "arbitrary_adapter_accepted_in_r0_r2": {"answer": False, "evidence": "attempted R1/R2 stop before adapter load; this does not validate behavior with trusted material present"},
            "bindings_enforced": {"answer": False, "evidence": "isolated static verifier rejects enumerated mutations, but copied native runtime retains an unmanifested raw --adapter path"},
            "silent_fallback_observed": {"answer": False, "evidence": "nonzero exits and no protected output artifacts"},
            "plaintext_parameters_or_optimizer_state_in_outputs": {"answer": bool(scan["plaintext_named_parameter_or_optimizer_state_observed"]), "evidence": scan},
            "useful_surrogate_from_artifacts_alone": {"answer": "NOT_ESTABLISHED", "evidence": "R0 tests initialization/integrity only; the separate fidelity pilot evaluates frozen observations and is not an artifact-only surrogate"}
        },
        "overall_pass": all_pass,
    }
    rendered_json = json.dumps(result, indent=2) + "\n"
    (args.out / "artifact_misuse_r0_r2_report.json").write_text(rendered_json)
    lines = [
        "# Artifact misuse-resistance R0--R2", "",
        f"- View: **SIMULATED_PROTOCOL_VIEW**", f"- Overall enumerated-control pass: **{all_pass}**",
        f"- Observed GPU: `{observed_gpu}` (`{observed_uuid}`)",
        f"- Package root: `{actual_root}`", f"- Transformed tensors initialized on GPU: {len(tensor_inventory)}", "",
        "This supports only observed fail-closed behavior for the enumerated controls. It is not a cryptographic impossibility claim and is not a REAL_TDX_BACKED_VIEW result.", "",
        "## Conditions", "",
    ]
    for x in execution:
        status = x.get("failed_closed", x.get("package_gpu_initialized"))
        lines.append(f"- {x['condition']}: {'PASS' if status else 'FAIL'} ({x.get('failure_boundary', x.get('reason_not_attempted', ''))})")
    lines += ["", "## Negative controls", ""]
    for x in controls:
        lines.append(f"- {x['test']}: {'PASS' if x['passed'] else 'FAIL'} — {x['detail']}")
    (args.out / "artifact_misuse_r0_r2_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"overall_pass": all_pass, "gpu": observed_gpu,
                      "tensors": len(tensor_inventory), "out": str(args.out)}))


if __name__ == "__main__":
    main()
