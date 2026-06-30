"""Generate REAL TDX attestation evidence on an Alibaba Cloud TDX guest.

This is the Alibaba-Cloud-specific driver around the SAME runtime-hash recipe and
the SAME ``verify_evidence`` verifier used by
``scripts/generate_tdx_attestation_evidence.py``. It adapts the flow to Alibaba's
local quote-generation + quote-verification samples (which verify the TD Quote
*locally* with the relying-party verifier -- there is no remote JWT here):

1. **Preflight** (never destructive): ``lscpu | grep tdx_guest``,
   ``/dev/tdx_guest``, kernel version, and the presence of
   ``/opt/alibaba/tdx-quote-generation-sample/app`` and the
   ``tdx-quote-verification-sample`` verifier / relying_party. Missing pieces are
   reported with the install command to run by hand -- the script never modifies
   the system.
2. **Runtime hash** bound to ``--nonlinear-backend A_rightmul`` (same recipe the
   cross-machine demo verifies), folded into the 64-byte ``report_data``.
3. **Quote**: run the Alibaba generation sample
   ``<app> -d <report_data_hex>`` (writing ``quote.dat``; copied to the output
   dir). The quote is therefore bound to the current code + A_rightmul metadata.
4. **Verify**: run the Alibaba verification sample (verifier / relying_party) on
   the quote, then extract ``overall_appraisal_result``, ``tdx_reportdata``, the
   ``mr_td``, and ``td_attributes.debug``. We assert
   ``tdx_reportdata == report_data == runtime_hash``, ``debug == false``, and
   ``mr_td == --expected-mr-td`` (when supplied).
5. **Evidence JSON** with ``tee=tdx``,
   ``quote_source=alibaba_tdx_quote_generation_sample``,
   ``verifier_overall_appraisal_result``, ``tdx_reportdata``, ``runtime_hash``,
   ``report_data``, ``runtime_hash_binds_nonlinear_backend=true``,
   ``nonlinear_backend=A_rightmul``, ``td_attributes.debug=false``,
   ``paper_facing=true`` (only on a real, fully-verified run).

STALE QUOTES: a previously-generated quote binds the *previous* code hash. Any
code change to a measured boundary file changes the runtime hash, so the quote
must be regenerated before each formal experiment; an old quote fails the
``tdx_reportdata == runtime_hash`` check.

Off-TDX, ``--simulate`` exercises the full plumbing with a clearly-marked unsigned
flow (``paper_facing=false``); it is never real evidence. stdlib + the pllo
attestation module only. No passwords / SSH keys are read or written here.
"""

from __future__ import annotations

import argparse
import json
import re
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

ALIBABA_QGEN_APP = "/opt/alibaba/tdx-quote-generation-sample/app"
ALIBABA_QVERIFY_DIR = "/opt/alibaba/tdx-quote-verification-sample"
QUOTE_SOURCE = "alibaba_tdx_quote_generation_sample"

# Alibaba install command (printed, NEVER executed by this wrapper)
_ALIBABA_INSTALL_CMD = (
    "sudo yum install -y tdx-quote-generation-sample tee-appraisal-tool "
    "libsgx-dcap-ql-devel libsgx-dcap-quote-verify-devel "
    "libsgx-dcap-default-qpl-devel tdx-quote-verification-sample")
_PCCS_HINT = ("configure the DCAP quote-provider / PCCS_URL (e.g. "
              "/etc/sgx_default_qcnl.conf -> \"pccs_url\": \"https://<PCCS_HOST>:"
              "8081/sgx/certification/v4/\") so the verifier can fetch the TDX "
              "collateral")

# install hints (printed, NEVER executed)
_INSTALL_HINTS = {
    "tdx_guest_cpu": "ensure the VM is a TDX guest (Alibaba g8i TDX instance)",
    "tdx_guest_dev": "modprobe tdx_guest; ls -l /dev/tdx_guest",
    "qgen_app": "%s   # provides %s" % (_ALIBABA_INSTALL_CMD, ALIBABA_QGEN_APP),
    "qverify": "%s   # provides %s/{verifier,relying_party}; then %s"
               % (_ALIBABA_INSTALL_CMD, ALIBABA_QVERIFY_DIR, _PCCS_HINT),
}


def _norm_hex(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    return re.sub(r"[^0-9a-f]", "", s) or None


def _run(cmd, *, cwd=None, timeout=120):
    """Run a command (list or shell string); return (rc, stdout, stderr)."""
    shell = isinstance(cmd, str)
    proc = subprocess.run(cmd, shell=shell, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout)
    return (proc.returncode, proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def run_preflight(*, qgen_app=ALIBABA_QGEN_APP, qverify_dir=ALIBABA_QVERIFY_DIR,
                  min_kernel=(5, 19)) -> dict:
    """Non-destructive environment check. Returns a dict of booleans + hints."""
    checks: dict = {}
    missing: list = []

    # 1. TDX guest CPU flag
    try:
        rc, out, _ = _run(["lscpu"], timeout=20)
        checks["cpu_tdx_guest"] = (rc == 0 and "tdx_guest" in out.lower())
    except Exception:                                            # noqa: BLE001
        checks["cpu_tdx_guest"] = False
    if not checks["cpu_tdx_guest"]:
        missing.append(("cpu_tdx_guest", _INSTALL_HINTS["tdx_guest_cpu"]))

    # 2. /dev/tdx_guest device
    checks["dev_tdx_guest"] = Path("/dev/tdx_guest").exists()
    if not checks["dev_tdx_guest"]:
        missing.append(("dev_tdx_guest", _INSTALL_HINTS["tdx_guest_dev"]))

    # 3. kernel version (>= min_kernel for the configfs-tsm/TDX guest interface)
    kernel_ok = None
    try:
        rc, out, _ = _run(["uname", "-r"], timeout=10)
        m = re.match(r"(\d+)\.(\d+)", out.strip())
        if m:
            kv = (int(m.group(1)), int(m.group(2)))
            kernel_ok = kv >= min_kernel
            checks["kernel_release"] = out.strip()
    except Exception:                                            # noqa: BLE001
        kernel_ok = None
    checks["kernel_ok"] = kernel_ok

    # 4. Alibaba quote-generation sample app
    checks["qgen_app_present"] = Path(qgen_app).exists()
    if not checks["qgen_app_present"]:
        missing.append(("qgen_app", _INSTALL_HINTS["qgen_app"]))

    # 5. Alibaba quote-verification sample (verifier + relying_party)
    verifier = Path(qverify_dir) / "verifier"
    relying_party = Path(qverify_dir) / "relying_party"
    checks["qverify_verifier_present"] = verifier.exists()
    checks["qverify_relying_party_present"] = relying_party.exists()
    if not (verifier.exists() or relying_party.exists()):
        missing.append(("qverify", _INSTALL_HINTS["qverify"]))

    checks["all_ok"] = bool(
        checks.get("cpu_tdx_guest") and checks.get("dev_tdx_guest")
        and checks.get("qgen_app_present")
        and (checks.get("qverify_verifier_present")
             or checks.get("qverify_relying_party_present")))
    checks["missing"] = [{"item": k, "hint": h} for k, h in missing]
    return checks


# ---------------------------------------------------------------------------
# quote generation + verification (Alibaba samples)
# ---------------------------------------------------------------------------


def generate_quote_alibaba(report_data_hex, out_dir, *, qgen_app=ALIBABA_QGEN_APP,
                           quote_command=None, quote_out_name="td_quote.dat"):
    """Run the Alibaba generation sample to produce a quote bound to report_data.

    ``quote_command`` (testable override) is a shell template with
    ``{report_data_hex}`` / ``{quote_out}`` placeholders; if omitted the default
    ``<app> -d <report_data_hex>`` is used and ``quote.dat`` is copied to
    ``{quote_out}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quote_out = out_dir / quote_out_name
    if quote_command:
        cmd = (quote_command.replace("{report_data_hex}", report_data_hex)
               .replace("{quote_out}", str(quote_out)))
        rc, out, err = _run(cmd, cwd=str(out_dir))
        if rc != 0:
            raise RuntimeError("quote command failed (%d): %s" % (rc, err[:400]))
    else:
        rc, out, err = _run([qgen_app, "-d", report_data_hex], cwd=str(out_dir))
        if rc != 0:
            raise RuntimeError(
                "Alibaba quote app failed (%d): %s" % (rc, err[:400]))
    # the Alibaba sample writes quote.dat in cwd; copy it to the canonical name
    # whenever {quote_out} was not produced directly (works for the default app
    # AND for a custom command that still emits quote.dat).
    if not quote_out.exists() or quote_out.stat().st_size == 0:
        default = out_dir / "quote.dat"
        if default.exists() and default.resolve() != quote_out.resolve():
            shutil.copyfile(default, quote_out)
    if not quote_out.exists() or quote_out.stat().st_size == 0:
        raise RuntimeError("no quote produced at %s (expected the sample to write "
                           "quote.dat or {quote_out})" % quote_out)
    return quote_out


def _coerce_debug(v):
    """Return True/False if v clearly encodes a bool, else None (UNKNOWN)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def parse_verifier_output(text) -> dict:
    """Extract appraisal / reportdata / mr_td / debug from verifier output.

    Strict paper-facing evidence requires these fields to be *actually parsed*
    from the verifier/relying_party output. expected_mr_td is never used to
    fill missing mr_td.
    """
    text = text or ""

    def _find_key(obj, names):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if kl in names:
                    return v
            for v in obj.values():
                got = _find_key(v, names)
                if got is not None:
                    return got
        elif isinstance(obj, list):
            for v in obj:
                got = _find_key(v, names)
                if got is not None:
                    return got
        return None

    def _parse_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        sv = str(v).strip().lower()
        if sv in ("true", "1", "yes"):
            return True
        if sv in ("false", "0", "no"):
            return False
        return None

    # JSON first, including nested td_attributes.debug-like outputs.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {
                "overall_appraisal_result": (
                    _find_key(obj, {"overall_appraisal_result", "appraisal_result", "result"})
                ),
                "tdx_reportdata": _norm_hex(_find_key(
                    obj, {"tdx_reportdata", "tdx_report_data", "report_data", "reportdata"}
                )),
                "mr_td": _norm_hex(_find_key(
                    obj, {"tdx_mr_td", "mr_td", "mrtd", "mrt d"}
                )),
                "debug": _parse_bool(_find_key(
                    obj, {"debug", "td_attributes.debug"}
                )),
                "verifier_returncode": _find_key(
                    obj, {"verifier_returncode", "returncode", "return_code", "rc"}
                ),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    # Plain-text fallback.
    def _grab(pat):
        m = re.search(pat, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    appraisal = _grab(
        r"(?:overall[_ ]appraisal[_ ]result|appraisal[_ ]result|result)\s*[:=]\s*([A-Za-z0-9_\- ]+)"
    )
    rd = _norm_hex(_grab(
        r"(?:tdx[_ ]reportdata|tdx[_ ]report[_ ]data|report[_ ]data|report\s+data)\s*[:=]\s*(0x[0-9a-fA-F]+|[0-9a-fA-F]+)"
    ))
    mrtd = _norm_hex(_grab(
        r"(?:tdx[_ ]mr[_ ]td|mr[_ ]td|mrt\s*d|mrtd)\s*[:=]\s*(0x[0-9a-fA-F]+|[0-9a-fA-F]+)"
    ))
    debug = _grab(r"(?:td[_ ]attributes\.debug|td_attributes\.debug|debug)\s*[:=]\s*(true|false|0|1|yes|no)")
    rc = _grab(r"(?:verifier[_ ]returncode|returncode|return[_ ]code|rc)\s*[:=]\s*(-?\d+)")
    return {
        "overall_appraisal_result": appraisal,
        "tdx_reportdata": rd,
        "mr_td": mrtd,
        "debug": _parse_bool(debug),
        "verifier_returncode": int(rc) if rc is not None else None,
    }

def verify_quote_alibaba(quote_path, out_dir, *, qverify_dir=ALIBABA_QVERIFY_DIR,
                         verify_command=None):
    """Run the Alibaba verification sample; return the parsed appraisal dict."""
    if verify_command:
        cmd = verify_command.replace("{quote_file}", str(quote_path))
        rc, out, err = _run(cmd, cwd=str(out_dir))
    else:
        relying_party = Path(qverify_dir) / "relying_party"
        verifier = Path(qverify_dir) / "verifier"
        exe = relying_party if relying_party.exists() else verifier
        if not exe.exists():
            raise RuntimeError("no Alibaba verifier at %s" % qverify_dir)
        rc, out, err = _run([str(exe), str(quote_path)], cwd=str(out_dir))
    parsed = parse_verifier_output(out or err)
    parsed["verifier_returncode"] = rc
    parsed["raw_output_present"] = bool(out or err)
    return parsed


# ---------------------------------------------------------------------------
# evidence assembly + binding verification
# ---------------------------------------------------------------------------


def build_evidence(*, runtime_hash_hex, report_data_hex, nonlinear_backend,
                   nonlinear_design_metadata_hash, appraisal, expected_mr_td,
                   quote_source=QUOTE_SOURCE, simulated=False,
                   command_provenance=None) -> dict:
    """Assemble evidence JSON.

    Important: verifier-derived fields must stay verifier-derived. We do NOT use
    expected_mr_td to fill a missing mr_td, and we do NOT treat a missing debug
    field as debug=false.
    """
    debug_val = appraisal.get("debug")
    evidence = {
        "tee": "tdx",
        "quote_source": quote_source,
        "verifier_overall_appraisal_result":
            appraisal.get("overall_appraisal_result"),
        "verifier_returncode": appraisal.get("verifier_returncode"),
        "tdx_reportdata": appraisal.get("tdx_reportdata"),
        "report_data": report_data_hex,
        "runtime_hash": runtime_hash_hex,
        "mr_td": appraisal.get("mr_td"),
        "expected_mr_td": _norm_hex(expected_mr_td),
        "nonlinear_backend": nonlinear_backend,
        "nonlinear_design_metadata_hash": nonlinear_design_metadata_hash,
        "runtime_hash_binds_nonlinear_backend": True,
        "tdx": {"td_attributes": {"debug": debug_val if debug_val is not None else "unknown"}},
        "generated_by": "generate_alibaba_tdx_quote_evidence.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command_provenance": command_provenance or {},
    }
    if simulated:
        evidence["simulated_unsigned"] = True
        evidence["paper_facing"] = False
    return evidence


# a positive appraisal must be one of these tokens AND contain none of the
# negative tokens (so "SIMULATED_PASS" / "UNKNOWN" / "PASS but ERROR" all FAIL).
_APPRAISAL_POSITIVE = re.compile(r"\b(pass|passed|ok|success|succeeded|trusted|"
                                 r"trustworthy)\b", re.IGNORECASE)
_APPRAISAL_NEGATIVE = re.compile(r"simulat|unknown|fail|reject|error|untrust|"
                                 r"deny|denied|invalid|none", re.IGNORECASE)


def _appraisal_pass(appraisal) -> bool:
    s = str(appraisal or "").strip()
    if not s:
        return False
    if _APPRAISAL_NEGATIVE.search(s):
        return False
    return bool(_APPRAISAL_POSITIVE.search(s))


def verify_bindings(evidence, *, runtime_hash_hex, report_data_hex,
                    expected_mr_td) -> dict:
    """Strict binding checks for paper-facing Alibaba TDX evidence.

    Missing verifier fields fail closed:
    * mr_td must be actually parsed when expected_mr_td is provided;
    * debug must be actually parsed as False;
    * verifier return code must be 0 when present;
    * appraisal result must be an explicit PASS/OK/SUCCESS/TRUSTED value.
    """
    rd = _norm_hex(evidence.get("tdx_reportdata"))
    reportdata_present = rd is not None
    reportdata_binds = bool(
        reportdata_present
        and rd == _norm_hex(report_data_hex)
        and rd == _norm_hex(runtime_hash_hex)
    )

    parsed_mr = _norm_hex(evidence.get("mr_td"))
    if expected_mr_td:
        mr_ok = bool(parsed_mr is not None and parsed_mr == _norm_hex(expected_mr_td))
    else:
        mr_ok = parsed_mr is not None

    debug = (((evidence.get("tdx") or {}).get("td_attributes") or {}).get("debug"))
    debug_present = isinstance(debug, bool)
    debug_false = (debug is False)

    rc = evidence.get("verifier_returncode")
    verifier_returncode_ok = (rc == 0 or rc == "0")

    appraisal = str(evidence.get("verifier_overall_appraisal_result") or "").strip().upper()
    explicit_pass_values = {"PASS", "OK", "SUCCESS", "TRUSTED"}
    appraisal_ok = appraisal in explicit_pass_values

    return {
        "tdx_reportdata_present": reportdata_present,
        "tdx_reportdata_binds_runtime_hash": reportdata_binds,
        "mr_td_present": parsed_mr is not None,
        "mr_td_match": mr_ok,
        "debug_present": debug_present,
        "debug_false": debug_false,
        "verifier_returncode_ok": verifier_returncode_ok,
        "appraisal_ok": appraisal_ok,
        "all_bindings_ok": bool(
            reportdata_binds
            and mr_ok
            and debug_present
            and debug_false
            and verifier_returncode_ok
            and appraisal_ok
        ),
    }

def _diagnose_failure(exc, preflight, args) -> list[str]:
    """Map a quote/verify failure to actionable diagnoses (no system changes)."""
    pf = preflight or {}
    diag = []
    if pf.get("cpu_tdx_guest") is False:
        diag.append("CPU is not a TDX guest (lscpu has no tdx_guest) -> %s"
                    % _INSTALL_HINTS["tdx_guest_cpu"])
    if pf.get("dev_tdx_guest") is False:
        diag.append("/dev/tdx_guest missing -> %s"
                    % _INSTALL_HINTS["tdx_guest_dev"])
    if pf.get("kernel_ok") is False:
        diag.append("kernel %s too old for the TDX guest interface"
                    % pf.get("kernel_release"))
    if pf.get("qgen_app_present") is False:
        diag.append("quote-generation sample missing -> %s"
                    % _INSTALL_HINTS["qgen_app"])
    if not (pf.get("qverify_verifier_present")
            or pf.get("qverify_relying_party_present")):
        diag.append("quote-verification sample missing -> %s"
                    % _INSTALL_HINTS["qverify"])
    msg = str(exc).lower()
    if "pccs" in msg or "qcnl" in msg or "collateral" in msg:
        diag.append("verifier could not fetch collateral -> %s" % _PCCS_HINT)
    if "verify" in msg or "appraisal" in msg or "relying" in msg:
        diag.append("verifier/relying_party failed: check the quote is fresh and "
                    "the PCCS/QPL is reachable")
    if not diag:
        diag.append("re-run the preflight (--skip-preflight off) and confirm the "
                    "Alibaba samples + PCCS config; re-quote after any code change")
    return diag


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boundary-backend", default="process")
    ap.add_argument("--gpu-backend", default="qwen7b_folded_package")
    ap.add_argument("--protocol-version", default="8.5")
    ap.add_argument("--nonlinear-backend", default="A_rightmul")
    ap.add_argument("--expected-mr-td", default=None)
    ap.add_argument("--runtime-hash", default=None,
                    help="override the computed runtime hash (128 hex); normally "
                    "omit so it is computed from current boundary code")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-evidence", required=True)
    ap.add_argument("--qgen-app", default=ALIBABA_QGEN_APP)
    ap.add_argument("--qverify-dir", default=ALIBABA_QVERIFY_DIR)
    ap.add_argument("--quote-command", default=None,
                    help="override quote generation (shell; {report_data_hex} "
                    "{quote_out})")
    ap.add_argument("--verify-command", default=None,
                    help="override quote verification (shell; {quote_file})")
    ap.add_argument("--skip-preflight", action="store_true", default=False)
    ap.add_argument("--ignore-preflight", action="store_true", default=False,
                    help="run even if preflight reports missing dependencies "
                    "(prints the install hints; still never modifies the system)")
    ap.add_argument("--simulate", action="store_true", default=False,
                    help="off-TDX plumbing test: fabricate a passing-but-UNSIGNED "
                    "flow (paper_facing=false); never real evidence")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. preflight ----------------------------------------------------------
    pf = {}
    if not args.skip_preflight:
        pf = run_preflight(qgen_app=args.qgen_app, qverify_dir=args.qverify_dir)
        (out_dir / "preflight.json").write_text(json.dumps(pf, indent=2),
                                                encoding="utf-8")
        print("=== Alibaba TDX preflight ===")
        for k in ("cpu_tdx_guest", "dev_tdx_guest", "kernel_ok",
                  "qgen_app_present", "qverify_relying_party_present",
                  "qverify_verifier_present", "all_ok"):
            print("%s=%s" % (k, pf.get(k)))
        for m in pf.get("missing", []):
            print("MISSING %s -> %s" % (m["item"], m["hint"]))
        if not pf.get("all_ok") and not (args.ignore_preflight or args.simulate):
            print("ERROR: preflight failed; fix the items above or pass "
                  "--ignore-preflight (real evidence still requires a real TDX "
                  "guest).", file=sys.stderr)
            return 2

    # 2. runtime hash bound to A_rightmul -----------------------------------
    from pllo.experiments.nonlinear_designs import (
        normalize_nonlinear_backend,
        nonlinear_design_metadata_hash as _design_hash)
    nb = normalize_nonlinear_backend(args.nonlinear_backend)
    if nb != "A_rightmul":
        print("ERROR: this paper-facing wrapper requires --nonlinear-backend "
              "A_rightmul (got %r)" % nb, file=sys.stderr)
        return 3
    design_hash = _design_hash(nb)
    metadata = boundary_manifest_metadata(
        args.boundary_backend, args.gpu_backend, args.expected_mr_td,
        protocol_version=args.protocol_version, nonlinear_backend=nb,
        nonlinear_design_metadata_hash=design_hash)
    manifest = build_trusted_boundary_manifest(metadata=metadata)
    runtime_hash_hex = (args.runtime_hash.lower() if args.runtime_hash
                        else compute_runtime_hash_from_manifest(manifest))
    rh_bytes = bytes.fromhex(runtime_hash_hex)
    report_data_hex = runtime_report_data_hex(rh_bytes)
    (out_dir / "runtime_hash.hex").write_text(runtime_hash_hex + "\n")
    (out_dir / "trusted_boundary_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # 3-4. quote + verify ----------------------------------------------------
    if args.simulate:
        # fabricate a quote + a passing appraisal that binds report_data, but mark
        # it UNSIGNED so it can never masquerade as real evidence.
        quote_path = out_dir / "td_quote.dat"
        quote_path.write_bytes(b"SIMQUOTE" + rh_bytes)
        appraisal = {"overall_appraisal_result": "SIMULATED_PASS",
                     "tdx_reportdata": report_data_hex,
                     "mr_td": _norm_hex(args.expected_mr_td), "debug": False,
                     "verifier_returncode": 0, "raw_output_present": True}
        quote_source = "simulated_unsigned"
    else:
        try:
            quote_path = generate_quote_alibaba(
                report_data_hex, out_dir, qgen_app=args.qgen_app,
                quote_command=args.quote_command)
            appraisal = verify_quote_alibaba(
                quote_path, out_dir, qverify_dir=args.qverify_dir,
                verify_command=args.verify_command)
        except Exception as exc:                                # noqa: BLE001
            diag = _diagnose_failure(exc, pf, args)
            print("ERROR: quote generation/verification failed: %s" % exc,
                  file=sys.stderr)
            for d in diag:
                print("  DIAGNOSIS: %s" % d, file=sys.stderr)
            (out_dir / "failure_diagnosis.json").write_text(
                json.dumps({"error": str(exc)[:400], "diagnosis": diag},
                           indent=2), encoding="utf-8")
            return 1
        quote_source = QUOTE_SOURCE
    (out_dir / "appraisal.json").write_text(json.dumps(appraisal, indent=2),
                                            encoding="utf-8")

    command_provenance = {
        "quote_command": (args.quote_command
                          or ("%s -d <report_data_hex>" % args.qgen_app)),
        "verify_command": (args.verify_command
                           or ("%s/relying_party <quote_file>"
                               % args.qverify_dir)),
        "qgen_app": args.qgen_app, "qverify_dir": args.qverify_dir,
        "simulate": bool(args.simulate),
    }

    # 5. evidence + binding verification ------------------------------------
    evidence = build_evidence(
        runtime_hash_hex=runtime_hash_hex, report_data_hex=report_data_hex,
        nonlinear_backend=nb, nonlinear_design_metadata_hash=design_hash,
        appraisal=appraisal, expected_mr_td=args.expected_mr_td,
        quote_source=quote_source, simulated=args.simulate,
        command_provenance=command_provenance)
    verdict = verify_bindings(
        evidence, runtime_hash_hex=runtime_hash_hex,
        report_data_hex=report_data_hex, expected_mr_td=args.expected_mr_td)
    evidence.update(verdict)
    # the shared verifier (same one the demo uses) for the runtime-hash binding
    ev = verify_evidence(evidence, rh_bytes, expected_mr_td=args.expected_mr_td)
    evidence["runtime_hash_bound"] = ev.runtime_hash_bound
    # paper_facing is true ONLY on a real, fully-bound, debug-false run
    evidence["paper_facing"] = bool(
        not args.simulate and verdict["all_bindings_ok"]
        and ev.runtime_hash_bound is True)

    out_ev = Path(args.output_evidence)
    out_ev.parent.mkdir(parents=True, exist_ok=True)
    out_ev.write_text(json.dumps(evidence, indent=2, default=str),
                      encoding="utf-8")
    shutil.copyfile(out_ev, out_dir / "evidence.json")

    print("=== Alibaba TDX evidence (A_rightmul) ===")
    print("runtime_hash=%s" % runtime_hash_hex)
    print("report_data =%s" % report_data_hex)
    print("tdx_reportdata=%s" % evidence.get("tdx_reportdata"))
    print("quote_source=%s" % quote_source)
    print("overall_appraisal_result=%s"
          % evidence.get("verifier_overall_appraisal_result"))
    print("tdx_reportdata_binds_runtime_hash=%s"
          % verdict["tdx_reportdata_binds_runtime_hash"])
    print("mr_td_match=%s debug_false=%s appraisal_ok=%s"
          % (verdict["mr_td_match"], verdict["debug_false"],
             verdict["appraisal_ok"]))
    print("paper_facing=%s output_evidence=%s" % (evidence["paper_facing"],
                                                  out_ev))
    if args.simulate:
        print("** SIMULATED: unsigned; NOT real attestation evidence **")

    if args.simulate:
        return 0
    return 0 if evidence["paper_facing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
