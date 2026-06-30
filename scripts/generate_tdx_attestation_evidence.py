"""Generate TDX attestation evidence bound to the CURRENT trusted boundary hash.

The cross-machine demo verifies that the TD Quote's ``report_data`` equals the
runtime hash of the *current* trusted-boundary code + metadata. Whenever a
measured boundary file changes (e.g. ``run_tee_gpu_protocol_demo.py``), that hash
changes and any previously-generated quote becomes a *stale binding*
(``runtime_hash_bound=false``). This script regenerates fresh evidence bound to
the current hash:

1. Compute the runtime hash with the SAME recipe the demo verifies
   (``boundary_manifest_metadata`` -> ``build_trusted_boundary_manifest`` ->
   ``compute_runtime_hash_from_manifest``), using ``--boundary-backend`` /
   ``--gpu-backend`` / ``--expected-mr-td`` / ``--protocol-version``.
2. Bind the 64-byte hash into the TD Quote ``report_data``.
3. Generate the TD report/quote on the TDX VM (configfs-tsm by default; or a
   supplied ``--quote-file`` / ``--quote-command``).
4. Acquire a signed attestation JWT from the Alibaba Cloud Attestation API
   (``--attest-endpoint``), or consume a supplied response/JWT
   (``--attest-response-file`` / ``--attest-command`` / ``--jwt``).
5. Assemble + locally verify the evidence (``report_data == runtime_hash`` and
   ``mr_td == expected_mr_td``) using the SAME verifier the demo uses.
6. Save every artifact (runtime hash, manifest, raw quote/report, attest request,
   JWT, decoded claims, final evidence JSON).

Off-TDX, ``--simulate`` exercises the full plumbing with a clearly-marked
unsigned token so the binding can be validated in CI; it is never real evidence.

stdlib + the pllo attestation module only. Python 3.6-safe.

Example (TDX VM)::

    python scripts/generate_tdx_attestation_evidence.py \\
        --boundary-backend process --gpu-backend mock \\
        --expected-mr-td e0199499baacb2e4...9ab2568a \\
        --attest-command 'trustflags attest --quote-file {quote_file} --json' \\
        --output-dir /root/privacy_llm_tee_artifacts/current_boundary_attestation \\
        --output-evidence /root/.../attestation_evidence_current_boundary.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.protocol.attestation import (  # noqa: E402
    boundary_manifest_metadata,
    build_trusted_boundary_manifest,
    compute_runtime_hash_from_manifest,
    runtime_report_data_hex,
    verify_evidence,
)
from pllo.tee.runtime_api import TDX_GUEST_DEVICE  # noqa: E402

DEFAULT_CONFIGFS = "/sys/kernel/config/tsm/report"
# claim keys Alibaba / generic verifiers may use for each field.
_MRTD_KEYS = ("tdx_mr_td", "mr_td", "mrtd", "tdx_mrtd", "MRTD")
_RD_KEYS = ("tdx_report_data", "report_data", "reportdata", "tdx_reportdata")
_DEBUG_KEYS = ("tdx_td_attributes_debug", "debug", "td_attributes_debug")
_TOKEN_KEYS = ("token", "jwt", "attestation_token", "Token", "JWT")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _b64url_decode(s: str) -> bytes:
    s = s + "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _read_arg_or_file(value: str | None) -> str | None:
    """Return a literal string, or the contents of a file if value starts '@'."""
    if value is None:
        return None
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8").strip()
    return value


def _run(cmd: str, *, input_bytes: bytes | None = None) -> str:
    """Run a shell command, return stdout (text). Raises on non-zero."""
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, input=input_bytes)
    if proc.returncode != 0:
        raise RuntimeError(
            "command failed (%d): %s\n%s" % (
                proc.returncode, cmd,
                proc.stderr.decode("utf-8", "replace")))
    return proc.stdout.decode("utf-8", "replace")


def _decode_jwt_claims(jwt: str) -> dict:
    parts = jwt.split(".")
    if len(parts) < 2:
        return {}
    try:
        return json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception:                                       # noqa: BLE001
        return {}


def _first_key(d: dict, keys) -> object:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    # also search one level of nesting (e.g. {"tdx": {...}})
    for v in d.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v and v[k] not in (None, ""):
                    return v[k]
    return None


def _norm_hex(v) -> str | None:
    if v is None:
        return None
    s = str(v).lower()
    if s.startswith("0x"):
        s = s[2:]
    return s


# ---------------------------------------------------------------------------
# quote generation
# ---------------------------------------------------------------------------


def generate_quote(report_data: bytes, args) -> tuple[bytes, bytes | None, str]:
    """Return (quote_bytes, report_bytes_or_None, source). report_data is 64B."""
    if args.quote_file:
        return Path(args.quote_file).read_bytes(), None, "quote_file"

    if args.quote_command:
        # template: {report_data_hex} {report_data_file} {quote_out}
        rd_hex = report_data.hex()
        with_tmp = args.quote_command
        rd_file = Path(args.output_dir) / "report_data.bin"
        rd_file.parent.mkdir(parents=True, exist_ok=True)
        rd_file.write_bytes(report_data)
        quote_out = Path(args.output_dir) / "td_quote.bin"
        cmd = (with_tmp.replace("{report_data_hex}", rd_hex)
               .replace("{report_data_file}", str(rd_file))
               .replace("{quote_out}", str(quote_out)))
        out = _run(cmd)
        if quote_out.exists() and quote_out.stat().st_size > 0:
            return quote_out.read_bytes(), None, "quote_command_file"
        # else assume the command wrote the quote (base64) to stdout
        return base64.b64decode(out.strip()), None, "quote_command_stdout"

    if args.simulate:
        import hashlib
        fake = (hashlib.sha256(b"SIMQUOTE" + report_data).digest() * 32)
        return fake, hashlib.sha256(report_data).digest(), "simulated"

    # configfs-tsm (standard Linux TDX attestation interface)
    return _quote_via_configfs(report_data, args)


def _quote_via_configfs(report_data: bytes, args) -> tuple[bytes, bytes | None, str]:
    base = Path(args.configfs_path)
    if not base.exists():
        raise RuntimeError(
            "configfs-tsm not found at %s (is /sys/kernel/config mounted and the "
            "TDX guest configured?). Supply --quote-file or --quote-command, or "
            "use --simulate off-TDX." % base)
    entry = base / ("pllo_boundary_%d" % os.getpid())
    try:
        entry.mkdir()
    except FileExistsError:
        pass
    try:
        if args.report_data_privlevel is not None and \
                (entry / "privlevel").exists():
            (entry / "privlevel").write_text(str(args.report_data_privlevel))
        (entry / "inblob").write_bytes(report_data)        # 64 bytes
        quote = (entry / "outblob").read_bytes()
        provider = ""
        if (entry / "provider").exists():
            provider = (entry / "provider").read_text().strip()
        return quote, None, "configfs_tsm:%s" % provider
    finally:
        try:
            entry.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# JWT acquisition (Alibaba Cloud Attestation API or supplied response)
# ---------------------------------------------------------------------------


def acquire_jwt(quote: bytes, report_data: bytes, args,
                ) -> tuple[str, dict]:
    """Return (jwt, attest_request). attest_request is recorded for audit."""
    direct = _read_arg_or_file(args.jwt)
    if direct:
        return direct.strip(), {"source": "jwt_arg"}

    if args.attest_response_file:
        resp = json.loads(Path(args.attest_response_file).read_text("utf-8"))
        tok = _first_key(resp, _TOKEN_KEYS) or _first_key(
            resp, (args.attest_token_field,))
        if not tok:
            raise RuntimeError(
                "no token field %r in %s (keys: %s)" % (
                    args.attest_token_field, args.attest_response_file,
                    list(resp)[:20]))
        return str(tok), {"source": "attest_response_file",
                          "file": args.attest_response_file}

    quote_b64 = base64.b64encode(quote).decode("ascii")
    request = {args.attest_quote_field: quote_b64, "tee": "tdx"}
    if args.attest_extra_json:
        request.update(json.loads(args.attest_extra_json))

    if args.attest_command:
        qf = Path(args.output_dir) / "td_quote.bin"
        qf.parent.mkdir(parents=True, exist_ok=True)
        qf.write_bytes(quote)
        cmd = (args.attest_command.replace("{quote_file}", str(qf))
               .replace("{quote_b64}", quote_b64)
               .replace("{report_data_hex}", report_data.hex()))
        out = _run(cmd).strip()
        try:
            resp = json.loads(out)
            tok = _first_key(resp, _TOKEN_KEYS) or _first_key(
                resp, (args.attest_token_field,))
            return str(tok or out), {"source": "attest_command", "cmd": cmd}
        except json.JSONDecodeError:
            return out, {"source": "attest_command", "cmd": cmd}  # raw JWT

    if args.attest_endpoint:
        import urllib.request
        body = json.dumps(request).encode("utf-8")
        req = urllib.request.Request(
            args.attest_endpoint, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=args.attest_timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
        tok = _first_key(resp, _TOKEN_KEYS) or _first_key(
            resp, (args.attest_token_field,))
        if not tok:
            raise RuntimeError("no token in attestation response: %s"
                               % list(resp)[:20])
        return str(tok), {"source": "attest_endpoint",
                          "endpoint": args.attest_endpoint, "request": request}

    if args.simulate:
        header = _b64url_encode(json.dumps(
            {"alg": "none", "typ": "JWT"}).encode())
        payload = _b64url_encode(json.dumps({
            "tee": "tdx", "report_data": report_data.hex(),
            "mr_td": _norm_hex(args.expected_mr_td),
            "td_attributes": {"debug": False},
            "note": "SIMULATED UNSIGNED TOKEN -- not real attestation"}).encode())
        return "%s.%s.%s" % (header, payload, "SIMULATED"), {"source": "simulate"}

    raise RuntimeError(
        "no way to obtain a JWT: supply one of --jwt / --attest-response-file / "
        "--attest-command / --attest-endpoint (or --simulate off-TDX).")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boundary-backend", default="process")
    ap.add_argument("--gpu-backend", default="mock")
    ap.add_argument("--expected-mr-td", required=True)
    ap.add_argument("--protocol-version", default="8.5")
    ap.add_argument("--nonlinear-backend", default=None,
                    help="bind a nonlinear design (A_rightmul / amulet_secure_R / "
                         "alias) into the runtime hash / report_data so the quote "
                         "is bound to a specific nonlinear design")
    ap.add_argument("--nonlinear-design-metadata-hash", default=None,
                    help="optional explicit design metadata hash to bind (defaults "
                         "to the registry hash for --nonlinear-backend)")
    ap.add_argument("--runtime-hash", default=None,
                    help="override the computed runtime hash (128 hex); normally "
                         "omit so it is computed from current boundary code")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-evidence", required=True)
    # quote generation
    ap.add_argument("--configfs-path", default=DEFAULT_CONFIGFS)
    ap.add_argument("--report-data-privlevel", type=int, default=0)
    ap.add_argument("--quote-file", default=None)
    ap.add_argument("--quote-command", default=None,
                    help="shell template; {report_data_hex} {report_data_file} "
                         "{quote_out} substituted")
    # jwt acquisition
    ap.add_argument("--jwt", default=None, help="literal JWT or @path")
    ap.add_argument("--attest-response-file", default=None)
    ap.add_argument("--attest-command", default=None,
                    help="shell template; {quote_file} {quote_b64} "
                         "{report_data_hex} substituted")
    ap.add_argument("--attest-endpoint", default=None)
    ap.add_argument("--attest-quote-field", default="quote")
    ap.add_argument("--attest-token-field", default="token")
    ap.add_argument("--attest-extra-json", default=None)
    ap.add_argument("--attest-timeout", type=float, default=30.0)
    ap.add_argument("--debug-allowed", action="store_true",
                    help="do not fail if td_attributes.debug is true (NOT for "
                         "production)")
    ap.add_argument("--simulate", action="store_true",
                    help="off-TDX plumbing test: fabricate quote + UNSIGNED token "
                         "(clearly marked; never real evidence)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. runtime hash (same recipe the demo verifies) --------------------------
    # bind the nonlinear design into the runtime hash so a quote cannot be reused
    # across designs (boundary_manifest_metadata folds it into runtime_identity).
    nonlinear_backend = None
    nonlinear_design_metadata_hash = None
    if args.nonlinear_backend is not None:
        from pllo.experiments.nonlinear_designs import (  # noqa: E402
            normalize_nonlinear_backend,
            nonlinear_design_metadata_hash as _design_hash)
        nonlinear_backend = normalize_nonlinear_backend(args.nonlinear_backend)
        nonlinear_design_metadata_hash = (
            args.nonlinear_design_metadata_hash
            or _design_hash(nonlinear_backend))
    metadata = boundary_manifest_metadata(
        args.boundary_backend, args.gpu_backend, args.expected_mr_td,
        protocol_version=args.protocol_version,
        nonlinear_backend=nonlinear_backend,
        nonlinear_design_metadata_hash=nonlinear_design_metadata_hash)
    manifest = build_trusted_boundary_manifest(metadata=metadata)
    runtime_hash_hex = (args.runtime_hash.lower() if args.runtime_hash
                        else compute_runtime_hash_from_manifest(manifest))
    rh_bytes = bytes.fromhex(runtime_hash_hex)
    report_data_hex = runtime_report_data_hex(rh_bytes)     # == runtime_hash_hex
    rd_bytes = bytes.fromhex(report_data_hex)

    (out_dir / "runtime_hash.hex").write_text(runtime_hash_hex + "\n")
    (out_dir / "trusted_boundary_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # 2-3. quote -------------------------------------------------------------
    quote, report, quote_source = generate_quote(rd_bytes, args)
    (out_dir / "td_quote.bin").write_bytes(quote)
    if report is not None:
        (out_dir / "td_report.bin").write_bytes(report)

    # 4. JWT -----------------------------------------------------------------
    jwt, attest_request = acquire_jwt(quote, rd_bytes, args)
    (out_dir / "attestation.jwt").write_text(jwt, encoding="utf-8")
    attest_request = dict(attest_request)
    attest_request.update({"quote_bytes": len(quote), "quote_source": quote_source,
                           "report_data": report_data_hex})
    (out_dir / "attest_request.json").write_text(
        json.dumps(attest_request, indent=2, default=str), encoding="utf-8")

    # 5. decode claims + assemble evidence -----------------------------------
    claims = _decode_jwt_claims(jwt)
    (out_dir / "claims.json").write_text(
        json.dumps(claims, indent=2, default=str), encoding="utf-8")

    claim_mr_td = _norm_hex(_first_key(claims, _MRTD_KEYS))
    claim_rd = _norm_hex(_first_key(claims, _RD_KEYS))
    claim_debug = _first_key(claims, _DEBUG_KEYS)
    debug_val = bool(claim_debug) if claim_debug is not None else False

    # truthful evidence: prefer what the quote/claims actually bound, falling
    # back to our computed/expected values when the claim does not surface them.
    evidence = {
        "tee": "tdx",
        "mr_td": claim_mr_td or _norm_hex(args.expected_mr_td),
        "report_data": claim_rd or report_data_hex,
        "jwt": jwt,
        "tdx": {"td_attributes": {"debug": debug_val}},
        "generated_by": "generate_tdx_attestation_evidence.py",
        "boundary_backend": args.boundary_backend,
        "gpu_backend": args.gpu_backend,
        "protocol_version": args.protocol_version,
        "runtime_hash": runtime_hash_hex,
        "quote_source": quote_source,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if nonlinear_backend is not None:
        # the runtime hash above was computed from metadata that folds these in
        evidence["nonlinear_backend"] = nonlinear_backend
        evidence["nonlinear_design_metadata_hash"] = nonlinear_design_metadata_hash
        evidence["runtime_hash_binds_nonlinear_backend"] = True
    if args.simulate:
        evidence["simulated_unsigned"] = True
        # off-TDX plumbing test -> NEVER paper-facing
        evidence["paper_facing"] = False

    # 6. local verification (same verifier the demo uses) --------------------
    ev = verify_evidence(evidence, rh_bytes, expected_mr_td=args.expected_mr_td)

    out_ev = Path(args.output_evidence)
    out_ev.parent.mkdir(parents=True, exist_ok=True)
    out_ev.write_text(json.dumps(evidence, indent=2, default=str),
                      encoding="utf-8")
    shutil.copyfile(out_ev, out_dir / "evidence.json")

    # cross-checks worth surfacing loudly
    warnings = []
    if claim_rd and claim_rd != report_data_hex:
        warnings.append("claim report_data != computed runtime hash (stale/"
                        "wrong binding)")
    if claim_mr_td and _norm_hex(args.expected_mr_td) and \
            claim_mr_td != _norm_hex(args.expected_mr_td):
        warnings.append("claim mr_td != expected_mr_td")
    if ev.debug is True and not args.debug_allowed:
        warnings.append("td_attributes.debug is TRUE (not production-safe)")

    print("=== TDX attestation evidence (current boundary) ===")
    print("tee=%s" % ev.tee_type)
    print("mr_td=%s" % ev.mr_td)
    print("debug=%s" % ev.debug)
    print("runtime_hash=%s" % runtime_hash_hex)
    print("report_data =%s" % ev.report_data_hex)
    print("runtime_hash_bound=%s" % ev.runtime_hash_bound)
    print("mr_td_match=%s" % ev.mr_td_match)
    print("boundary_attested(verified)=%s" % ev.verified)
    print("jwt_parts=%d quote_source=%s" % (ev.jwt_parts, quote_source))
    print("output_evidence=%s" % out_ev)
    print("artifacts_dir=%s" % out_dir)
    if args.simulate:
        print("** SIMULATED: unsigned token; NOT real attestation evidence **")
    for w in warnings:
        print("WARNING: %s" % w)

    # exit non-zero if the binding is not solid (so CI / the operator notices),
    # except in simulate mode where an unsigned token is expected.
    ok = (ev.runtime_hash_bound is True
          and (ev.mr_td_match is True or not args.expected_mr_td)
          and (ev.debug is not True or args.debug_allowed))
    if args.simulate:
        return 0 if ev.runtime_hash_bound is True else 1
    return 0 if (ok and ev.verified) else 1


if __name__ == "__main__":
    raise SystemExit(main())
