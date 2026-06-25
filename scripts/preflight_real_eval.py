"""Preflight checks before an expensive real H800/TDX evaluation session.

Verifies the inputs a real run needs are present and consistent BEFORE the server
clock starts, so the session is a pure execution session. Reports blockers (must
fix), warnings (should review), and the commands to run next. Pure parsing /
filesystem checks -- no model load, no GPU, no real run.

Example::

    python scripts/preflight_real_eval.py \\
        --model-path /root/.../Qwen2___5-7B-Instruct \\
        --base-folded-package-path /root/.../qwen7b_folded_full \\
        --embedding-artifact-path /root/.../qwen7b_boundary_artifact_cuda \\
        --lora-folded-package-path /root/.../qwen7b_lora_folded_synth_r4 \\
        --gpu-worker-url http://127.0.0.1:18083 --backend tdx_attested_remote \\
        --attestation-evidence outputs/attestation_evidence.json \\
        --expected-mr-td <mr_td> \\
        --result-json outputs/tdx_attested_qwen7b_folded_remote_decode.json \\
        --output-json outputs/preflight.json --output-md outputs/preflight.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_ATTESTED_BACKENDS = ("tdx_attested_remote", "tdx_attested_folded_lora_remote")
_REMOTE_BACKENDS = ("folded_remote", "tdx_lite_remote", "tdx_attested_remote",
                    "folded_lora_remote", "tdx_attested_folded_lora_remote")
_LORA_BACKENDS = ("folded_lora_remote", "tdx_attested_folded_lora_remote")


def _exists_file(p):
    return bool(p) and Path(p).is_file()


def _exists_path(p):
    return bool(p) and Path(p).exists()


def _dir_nonempty(p):
    return bool(p) and Path(p).is_dir() and any(Path(p).iterdir())


def _norm_nonlinear(name):
    if not name:
        return None
    try:
        from pllo.experiments.nonlinear_designs import (
            normalize_nonlinear_backend)
        return normalize_nonlinear_backend(name)
    except Exception:
        return str(name)


def run_preflight(opts: dict) -> dict:
    backend = opts.get("backend") or "plaintext_local"
    attested = (backend in _ATTESTED_BACKENDS
                or bool(opts.get("require_attested"))
                or bool(opts.get("attestation_evidence")))
    is_remote = backend in _REMOTE_BACKENDS
    is_lora = backend in _LORA_BACKENDS
    nonlinear_backend = _norm_nonlinear(opts.get("nonlinear_backend"))

    checks = []
    blockers = []
    warnings = []

    def _chk(name, ok, detail, *, blocker=True):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            (blockers if blocker else warnings).append("%s: %s" % (name, detail))
        return bool(ok)

    # 1. model_path
    _chk("model_path_exists", _exists_path(opts.get("model_path")),
         opts.get("model_path") or "(not provided)")

    # 2-3. base folded package + manifest hash (only the private path needs it)
    base = opts.get("base_folded_package_path")
    _chk("base_folded_package_exists", _dir_nonempty(base),
         base or "(not provided)", blocker=is_remote)
    base_hash = None
    if base and Path(base).is_dir():
        try:
            from pllo.deployment import (
                check_nonlinear_backend, compute_manifest_hash, load_manifest)
            base_manifest = load_manifest(base)
            base_hash = compute_manifest_hash(base_manifest)
            _chk("base_manifest_hash_readable", bool(base_hash),
                 base_hash[:16] + "..." if base_hash else "unreadable",
                 blocker=is_remote)
            # nonlinear design of the package must match the expected design
            if nonlinear_backend is not None:
                ok, probs = check_nonlinear_backend(base_manifest,
                                                    nonlinear_backend)
                _chk("base_manifest_nonlinear_backend_matches", ok,
                     "manifest nonlinear_backend=%s expected=%s%s"
                     % (base_manifest.nonlinear_backend, nonlinear_backend,
                        ("" if ok else " :: " + "; ".join(probs))),
                     blocker=is_remote)
        except Exception as exc:                            # noqa: BLE001
            _chk("base_manifest_hash_readable", False, "error: %s" % exc,
                 blocker=is_remote)
    else:
        _chk("base_manifest_hash_readable", not is_remote,
             "base package missing", blocker=is_remote)

    # 4. embedding artifact + hash readable
    art = opts.get("embedding_artifact_path")
    _chk("embedding_artifact_exists", _dir_nonempty(art) or _exists_file(art),
         art or "(not provided)", blocker=is_remote)
    if art and Path(art).exists():
        try:
            import hashlib
            p = Path(art)
            files = [p] if p.is_file() else sorted(
                q for q in p.rglob("*") if q.is_file())
            h = hashlib.sha256()
            for q in files:
                h.update(q.read_bytes())
            _chk("boundary_artifact_hash_readable", bool(files),
                 h.hexdigest()[:16] + "...", blocker=is_remote)
        except Exception as exc:                            # noqa: BLE001
            _chk("boundary_artifact_hash_readable", False, "error: %s" % exc,
                 blocker=is_remote)
    else:
        _chk("boundary_artifact_hash_readable", not is_remote,
             "embedding artifact missing", blocker=is_remote)

    # 5. folded LoRA package exists OR build command available
    lora = opts.get("lora_folded_package_path")
    if is_lora:
        if _dir_nonempty(lora):
            _chk("folded_lora_package_exists", True, lora)
            # LoRA package design must match the base folded package design
            try:
                from pllo.deployment import (
                    check_lora_base_nonlinear_compatibility, load_manifest)
                if base and Path(base).is_dir():
                    okc, probs = check_lora_base_nonlinear_compatibility(
                        load_manifest(lora), load_manifest(base))
                    _chk("lora_base_nonlinear_compatible", okc,
                         "compatible" if okc else "; ".join(probs))
            except Exception as exc:                        # noqa: BLE001
                _chk("lora_base_nonlinear_compatible", False,
                     "could not check: %s" % exc, blocker=False)
        else:
            _chk("folded_lora_package_exists", False,
                 "missing -- build with scripts/build_qwen7b_lora_folded_package.py",
                 blocker=False)
            warnings.append(
                "folded LoRA package absent; build it first: "
                "python scripts/build_qwen7b_lora_folded_package.py "
                "--base-folded-package-path %s --output-dir %s --output-json ..."
                % (base or "<BASE>", lora or "<LORA>"))

    # 6. TDX evidence present if attested mode requested
    ev = opts.get("attestation_evidence")
    if attested:
        ev_ok = _exists_file(ev)
        _chk("attestation_evidence_exists", ev_ok, ev or "(not provided)")
        # 7. runtime_hash == evidence.report_data
        if ev_ok:
            try:
                from pllo.protocol.attestation import (
                    boundary_manifest_metadata, build_trusted_boundary_manifest,
                    compute_runtime_hash_from_manifest, verify_evidence)
                md = boundary_manifest_metadata(
                    "process", opts.get("hash_gpu_backend", "qwen7b"),
                    opts.get("expected_mr_td"),
                    nonlinear_backend=nonlinear_backend)
                expected_hex = compute_runtime_hash_from_manifest(
                    build_trusted_boundary_manifest(metadata=md))
                evidence = json.loads(Path(ev).read_text(encoding="utf-8"))
                res = verify_evidence(
                    evidence, bytes.fromhex(expected_hex),
                    expected_mr_td=opts.get("expected_mr_td"))
                bound = res.runtime_hash_bound is True
                detail = ("report_data matches runtime_hash" if bound else
                          "report_data != runtime_hash (stale/changed binding); "
                          "regenerate the runtime hash and re-bind the quote")
                _chk("runtime_hash_matches_evidence", bound, detail)
                if res.mr_td_match is False:
                    _chk("mr_td_matches", False,
                         "evidence mr_td != expected_mr_td")
            except Exception as exc:                        # noqa: BLE001
                _chk("runtime_hash_matches_evidence", False,
                     "could not verify: %s" % exc)
    elif ev:
        warnings.append("attestation-evidence supplied but backend %r is not "
                        "attested; it will be ignored" % backend)

    # 8. run_e9 --require-real will not fall back to stub (args sufficient)
    if backend == "plaintext_local":
        req_ok = _exists_path(opts.get("model_path"))
        req_detail = "needs --model-path" if not req_ok else "model_path present"
    elif is_remote:
        have = (_exists_path(opts.get("model_path"))
                and (_dir_nonempty(art) or _exists_file(art))
                and bool(opts.get("gpu_worker_url")))
        if attested:
            have = have and _exists_file(ev)
        req_ok = have
        req_detail = ("model_path + embedding + gpu_worker_url"
                      + (" + evidence" if attested else "") + " required")
    else:
        req_ok = False
        req_detail = "unknown backend %r" % backend
    _chk("require_real_will_not_fallback", req_ok, req_detail)

    # optional: best-effort worker health
    url = opts.get("gpu_worker_url")
    if url and is_remote:
        try:
            import urllib.request
            with urllib.request.urlopen(url.rstrip("/") + "/health",
                                        timeout=3.0) as resp:
                json.loads(resp.read().decode("utf-8"))
            _chk("gpu_worker_health", True, url, blocker=False)
        except Exception as exc:                            # noqa: BLE001
            _chk("gpu_worker_health", False,
                 "not reachable now (%s); start the worker before running"
                 % exc, blocker=False)

    # 9. deployment truth checker can parse prior result JSONs
    results = opts.get("result_json") or []
    if results:
        from pllo.experiments.deployment_truth import infer_deployment_truth
        bad = []
        for rp in results:
            try:
                rep = json.loads(Path(rp).read_text(encoding="utf-8"))
                infer_deployment_truth(rep)
            except Exception as exc:                        # noqa: BLE001
                bad.append("%s (%s)" % (rp, exc))
        _chk("deployment_truth_parses_results", not bad,
             "all parsed" if not bad else "unparseable: %s" % bad,
             blocker=False)

    # 10. output directory writable
    out_dir = opts.get("output_dir") or "outputs"
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        probe = Path(out_dir) / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        _chk("output_dir_writable", True, out_dir)
    except Exception as exc:                                # noqa: BLE001
        _chk("output_dir_writable", False, "%s: %s" % (out_dir, exc))

    # reminder: a changed nonlinear design needs its own runtime hash + quote
    if attested and nonlinear_backend is not None:
        warnings.append(
            "attested run for nonlinear design %r: the runtime hash binds the "
            "design, so regenerate it (write_tee_boundary_runtime_hash.py "
            "--nonlinear-backend %s) and re-bind the TD Quote if any measured "
            "boundary file or the design changed -- design A evidence cannot be "
            "reused for design B." % (nonlinear_backend, nonlinear_backend))

    passed = not blockers
    commands = _next_commands(opts, backend, attested, is_lora, base, art, lora,
                              url, nonlinear_backend)
    return {
        "stage": "preflight_real_eval", "backend": backend,
        "nonlinear_backend": nonlinear_backend,
        "attested": attested, "preflight_passed": passed,
        "blockers": blockers, "warnings": warnings,
        "commands_to_run_next": commands, "checks": checks,
        "base_manifest_hash": base_hash,
    }


def run_preflight_matrix(opts: dict) -> dict:
    """Run :func:`run_preflight` once per nonlinear design and aggregate the
    per-backend results. Paths can be auto-namespaced by appending
    ``_<design>`` to the directory/file names (the dual-matrix convention) when
    ``namespace_by_backend`` is set."""
    from pllo.experiments.nonlinear_designs import parse_nonlinear_backends
    designs = parse_nonlinear_backends(opts.get("nonlinear_backends")
                                       or "current,trusted_shortcut")
    ns = bool(opts.get("namespace_by_backend"))

    by = {}
    passed_by = {}
    blockers_by = {}
    warnings_by = {}
    commands_by = {}
    for d in designs:
        sub = dict(opts)
        sub["nonlinear_backend"] = d
        if ns:
            for k in ("base_folded_package_path", "embedding_artifact_path",
                      "lora_folded_package_path"):
                if sub.get(k):
                    sub[k] = _namespace_path(sub[k], d)
            if sub.get("attestation_evidence"):
                sub["attestation_evidence"] = _namespace_path(
                    sub["attestation_evidence"], d)
        r = run_preflight(sub)
        by[d] = r
        passed_by[d] = r["preflight_passed"]
        blockers_by[d] = r["blockers"]
        warnings_by[d] = r["warnings"]
        commands_by[d] = r["commands_to_run_next"]

    return {
        "stage": "preflight_real_eval_matrix",
        "nonlinear_backends": designs,
        "namespace_by_backend": ns,
        "preflight_passed": all(passed_by.values()),
        "preflight_passed_by_backend": passed_by,
        "blockers_by_backend": blockers_by,
        "warnings_by_backend": warnings_by,
        "commands_to_run_next_by_backend": commands_by,
        "per_backend": by,
    }


def _namespace_path(path, backend):
    """Append ``_<backend>`` to the final path component (before a file
    suffix), e.g. ``qwen7b_folded_full`` -> ``qwen7b_folded_full_current`` and
    ``evidence.json`` -> ``evidence_current.json``."""
    p = Path(path)
    if p.suffix:
        return str(p.with_name(p.stem + "_" + backend + p.suffix))
    return str(p.with_name(p.name + "_" + backend))


def _next_commands(opts, backend, attested, is_lora, base, art, lora, url,
                   nonlinear_backend=None):
    nb = (" --nonlinear-backend %s" % nonlinear_backend) if nonlinear_backend \
        else ""
    cmds = []
    if is_lora and not _dir_nonempty(lora):
        cmds.append("python scripts/build_qwen7b_lora_folded_package.py "
                    "--base-folded-package-path %s --output-dir %s%s "
                    "--output-json outputs/qwen7b_lora_folded_build.json"
                    % (base or "<BASE>", lora or "<LORA>", nb))
    if attested and nonlinear_backend:
        cmds.append("python scripts/write_tee_boundary_runtime_hash.py "
                    "--boundary-backend process --gpu-backend qwen7b "
                    "--nonlinear-backend %s --expected-mr-td %s "
                    "--output outputs/runtime_hash_%s.txt"
                    % (nonlinear_backend, opts.get("expected_mr_td") or "<MRTD>",
                       nonlinear_backend))
    cmds.append("python scripts/run_tee_gpu_protocol_demo.py --mode "
                "gpu_worker_server --gpu-backend qwen7b_folded_package "
                "--folded-package-path %s%s --device cuda --dtype bfloat16 "
                "--audit true" % (base or "<BASE>",
                                  (" --folded-lora-package-path %s" % lora)
                                  if is_lora else ""))
    e9 = ("python scripts/run_e9_task_utility_benchmark.py --require-real "
          "--backend %s%s --dataset-jsonl outputs/bench/<DS>.jsonl "
          "--model-path %s --gpu-worker-url %s --embedding-path %s "
          "--output-json outputs/e9_<ds>_%s.json"
          % (backend, nb, opts.get("model_path") or "<MODEL>", url or "<URL>",
             art or "<ART>", backend))
    if attested:
        e9 += (" --attestation-evidence %s --expected-mr-td %s"
               % (opts.get("attestation_evidence") or "<EVIDENCE>",
                  opts.get("expected_mr_td") or "<MRTD>"))
    cmds.append(e9)
    cmds.append("python scripts/run_e9_pairwise_utility_preservation.py "
                "--baseline-json outputs/e9_<ds>_plaintext_local.json "
                "--candidate-json outputs/e9_<ds>_%s.json "
                "--output-json outputs/e9_<ds>_pairwise.json" % backend)
    cmds.append("python scripts/validate_paper_claims.py --result-json ... "
                "--output-json outputs/paper_claim_validation.json")
    return cmds


def _render_md(r: dict) -> str:
    if r.get("stage") == "preflight_real_eval_matrix":
        L = ["# Preflight: real H800/TDX eval (nonlinear matrix)", "",
             "- designs: %s" % ", ".join(r["nonlinear_backends"]),
             "- namespace_by_backend: %s" % r["namespace_by_backend"],
             "- **preflight_passed: %s**" % r["preflight_passed"], ""]
        for d in r["nonlinear_backends"]:
            sub = r["per_backend"][d]
            L += ["## design `%s` (passed=%s)" % (d, sub["preflight_passed"]), "",
                  "| check | ok | detail |", "| --- | --- | --- |"]
            for c in sub["checks"]:
                L.append("| %s | %s | %s |"
                         % (c["name"], "yes" if c["ok"] else "**NO**",
                            c["detail"]))
            L += ["", "Blockers: " + ("; ".join(sub["blockers"]) or "none"),
                  "", "Warnings: " + ("; ".join(sub["warnings"]) or "none"), ""]
        return "\n".join(L)
    L = ["# Preflight: real H800/TDX eval", "",
         "- backend: `%s`  attested: %s" % (r["backend"], r["attested"]),
         "- **preflight_passed: %s**" % r["preflight_passed"], "",
         "## Checks", "", "| check | ok | detail |", "| --- | --- | --- |"]
    for c in r["checks"]:
        L.append("| %s | %s | %s |"
                 % (c["name"], "yes" if c["ok"] else "**NO**", c["detail"]))
    L += ["", "## Blockers", ""]
    L += (["- " + b for b in r["blockers"]] or ["- none"])
    L += ["", "## Warnings", ""]
    L += (["- " + w for w in r["warnings"]] or ["- none"])
    L += ["", "## Commands to run next", ""]
    L += ["```", *r["commands_to_run_next"], "```", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--base-folded-package-path", default=None)
    ap.add_argument("--embedding-artifact-path", default=None)
    ap.add_argument("--lora-folded-package-path", default=None)
    ap.add_argument("--gpu-worker-url", default=None)
    ap.add_argument("--backend", default="plaintext_local")
    ap.add_argument("--nonlinear-backend", default=None,
                    help="single nonlinear design to preflight against")
    ap.add_argument("--nonlinear-backends", default=None,
                    help="comma-separated designs -> per-backend matrix preflight "
                         "(e.g. current,trusted_shortcut)")
    ap.add_argument("--namespace-by-backend", action="store_true", default=False,
                    help="in matrix mode, append _<design> to package/evidence "
                         "paths (dual-matrix layout convention)")
    ap.add_argument("--attested", action="store_true", default=False)
    ap.add_argument("--attestation-evidence", default=None)
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--hash-gpu-backend", default="qwen7b")
    ap.add_argument("--result-json", action="append", default=[])
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--output-json", default="outputs/preflight.json")
    ap.add_argument("--output-md", default="outputs/preflight.md")
    args = ap.parse_args()

    base_opts = {
        "model_path": args.model_path,
        "base_folded_package_path": args.base_folded_package_path,
        "embedding_artifact_path": args.embedding_artifact_path,
        "lora_folded_package_path": args.lora_folded_package_path,
        "gpu_worker_url": args.gpu_worker_url, "backend": args.backend,
        "require_attested": args.attested,
        "attestation_evidence": args.attestation_evidence,
        "expected_mr_td": args.expected_mr_td,
        "hash_gpu_backend": args.hash_gpu_backend,
        "result_json": args.result_json, "output_dir": args.output_dir,
        "nonlinear_backend": args.nonlinear_backend,
    }

    matrix = bool(args.nonlinear_backends)
    if matrix:
        base_opts["nonlinear_backends"] = args.nonlinear_backends
        base_opts["namespace_by_backend"] = args.namespace_by_backend
        report = run_preflight_matrix(base_opts)
    else:
        report = run_preflight(base_opts)

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.output_md:
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_render_md(report), encoding="utf-8")

    if matrix:
        print("=== preflight: real eval (matrix) ===")
        for d in report["nonlinear_backends"]:
            print("  [%s] design=%s blockers=%d warnings=%d"
                  % ("OK" if report["preflight_passed_by_backend"][d] else "XX",
                     d, len(report["blockers_by_backend"][d]),
                     len(report["warnings_by_backend"][d])))
        print("\npreflight_passed=%s" % report["preflight_passed"])
    else:
        print("=== preflight: real eval ===")
        for c in report["checks"]:
            print("  [%s] %s -- %s" % ("OK" if c["ok"] else "XX", c["name"],
                                       c["detail"]))
        print("\npreflight_passed=%s blockers=%d warnings=%d"
              % (report["preflight_passed"], len(report["blockers"]),
                 len(report["warnings"])))
    return 0 if report["preflight_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
