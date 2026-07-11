"""Gate 2 §2 — REAL TDX quote generation + DCAP QVL appraisal (guest-side).

Runs INSIDE the TDX guest. Drives the Alibaba DCAP samples that are proven to work
on 39.96.4.252:

  * ``/opt/alibaba/tdx-quote-generation-sample/app -d <report_data>`` -> quote.dat
    (a REAL TD Quote, signed by the platform Quoting Enclave, bound to report_data);
  * ``/opt/alibaba/tdx-quote-verification-sample/verifier -quote <quote>`` -> a DCAP
    QVL appraisal JWT (``alg=none``: the cryptographic trust is the QVL verification
    of the quote against the Intel PCK certificate chain / PCCS collateral, NOT a JWS
    signature -- ``overall_appraisal_result == 1`` means the platform + TD measurement
    chain verified).

The measured fields (mr_td, rtmr0..3, td_attributes/DEBUG, report_data) are parsed
directly from the TD Quote binary at their fixed TD-report offsets, so they come from
the signed quote body -- not from scraping verifier text. Once QVL appraises SUCCESS,
those fields are trustworthy.

report_data byte order: the Alibaba qgen app takes report_data as two 32-byte halves,
each byte-reversed. ``_qgen_reportdata`` applies that; the embedded quote value is the
reversed form, and we check it against the reversed form of our expected report_data.

No secrets here: report_data is a public identity binding; the quote is public evidence.
"""

from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

QGEN_APP = "/opt/alibaba/tdx-quote-generation-sample/app"
QVERIFY_DIR = "/opt/alibaba/tdx-quote-verification-sample"

# TD Quote v4 layout: 48-byte header, then the TD10 report body. Offsets of the
# fields we need, measured from the start of the whole quote.
_HEADER = 48
_OFF_TD_ATTRIBUTES = _HEADER + 120
_OFF_MR_TD = _HEADER + 136
_OFF_RTMR0 = _HEADER + 328
_OFF_REPORT_DATA = _HEADER + 520


class QuoteError(Exception):
    """Quote generation / appraisal / parse failure. Fail-closed."""


def _reverse_hex(s: str) -> str:
    return "".join([s[i:i + 2] for i in range(0, len(s), 2)][::-1])


def _qgen_reportdata(report_data_hex: str) -> str:
    """The byte order the Alibaba qgen app expects (each 32-byte half reversed)."""
    s = report_data_hex.lower()
    if len(s) != 128:
        raise QuoteError("report_data must be 64 bytes / 128 hex chars")
    return _reverse_hex(s[64:]) + _reverse_hex(s[:64])


def _run(cmd, cwd=None, timeout=180):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def generate_quote(report_data_hex: str, workdir, *, qgen_app: str = QGEN_APP) -> bytes:
    """Generate a REAL TD Quote bound to ``report_data_hex`` (128 hex). Returns raw bytes."""
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    for stale in ("quote.dat", "report.dat"):
        try:
            (workdir / stale).unlink()
        except FileNotFoundError:
            pass
    rc, out = _run([qgen_app, "-d", _qgen_reportdata(report_data_hex)], cwd=str(workdir))
    quote = workdir / "quote.dat"
    if rc != 0 or not quote.exists() or quote.stat().st_size == 0:
        raise QuoteError(f"quote generation failed (rc={rc}): {out[:400]}")
    return quote.read_bytes()


def appraise_quote(quote_path, *, qverify_dir: str = QVERIFY_DIR) -> dict:
    """Two-stage DCAP QVL appraisal, per the Alibaba TDX doc:

      verifier -quote <q>          -> DCAP QVL appraisal JWT (quote checked against
                                      the Intel/Alibaba PCK cert chain + PCCS collateral)
      relying_party -v -a          -> VERIFIES the appraisal JWT's ES384 policy
                                      signature (a real signature check) and emits the
                                      "json payload" containing overall_appraisal_result
                                      + measurement (tdx_reportdata, tdx_mrtd, tdx_rtmr*,
                                      tdx_attributes).

    Returns overall_appraisal_result + the relying-party-verified measurement fields.
    Falls back to parsing the raw verifier JWT if relying_party is unavailable."""
    verifier = Path(qverify_dir) / "verifier"
    relying = Path(qverify_dir) / "relying_party"
    if not verifier.exists():
        raise QuoteError(f"no DCAP verifier at {verifier}")
    rc_v, jwt_out = _run([str(verifier), "-quote", str(quote_path)], cwd=str(qverify_dir))
    jwt = (jwt_out or "").strip().splitlines()[-1].strip() if jwt_out else ""
    result = {"verifier_returncode": rc_v, "appraisal_jwt": jwt,
              "jwt_parts": len([p for p in jwt.split(".") if p]),
              "signature_verified": None}
    if relying.exists() and jwt:
        rc_r, rp_out = _run_pipe([str(verifier), "-quote", str(quote_path)],
                                 [str(relying), "-v", "-a"], cwd=str(qverify_dir))
        payload = _extract_json_payload(rp_out)
        result["relying_party_returncode"] = rc_r
        result["signature_verified"] = (rc_r == 0)      # relying_party -v verifies sig
        if payload is not None:
            result.update(_parse_relying_payload(payload))
            result["appraisal_source"] = "relying_party_v_a"
            return result
    # fallback: parse the raw verifier JWT (no signature verification)
    result.update(_parse_appraisal_jwt(jwt))
    result["appraisal_source"] = "verifier_jwt"
    return result


def _run_pipe(cmd1, cmd2, cwd=None, timeout=180):
    """Run ``cmd1 | cmd2`` and return (cmd2_rc, cmd2_stdout)."""
    p1 = subprocess.Popen(cmd1, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(cmd2, cwd=cwd, stdin=p1.stdout, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    p1.stdout.close()
    out = p2.communicate(timeout=timeout)[0].decode("utf-8", "replace")
    p1.wait(timeout=5)
    return p2.returncode, out


def _extract_json_payload(text: str):
    """Pull the JSON after a 'json payload:' line (relying_party output)."""
    if not text:
        return None
    for line in text.splitlines():
        if "payload:" in line.lower():
            frag = line.split("payload:", 1)[1].strip()
            try:
                return json.loads(frag)
            except Exception:                              # noqa: BLE001
                pass
    # relying_party may print the JSON on following lines; try the whole tail
    idx = text.lower().find("payload:")
    if idx != -1:
        frag = text[idx + len("payload:"):].strip()
        try:
            return json.loads(frag)
        except Exception:                                  # noqa: BLE001
            return None
    return None


def _find_measurement(obj):
    """Depth-first search for a dict with tdx_reportdata (the TD report measurement)."""
    if isinstance(obj, dict):
        if "tdx_reportdata" in obj or "tdx_mrtd" in obj:
            return obj
        for v in obj.values():
            got = _find_measurement(v)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_measurement(v)
            if got is not None:
                return got
    return None


def _norm_hex(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return s[2:] if s.startswith("0x") else s


def _parse_relying_payload(payload) -> dict:
    overall = None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        overall = (payload[0].get("result") or {}).get("overall_appraisal_result")
    elif isinstance(payload, dict):
        overall = (payload.get("result") or {}).get("overall_appraisal_result")
    meas = _find_measurement(payload) or {}
    return {
        "overall_appraisal_result": overall,
        "appraisal_ok": overall in (1, "1", True, "SUCCESS", "success"),
        "rp_tdx_reportdata": _norm_hex(meas.get("tdx_reportdata")),
        "rp_mr_td": _norm_hex(meas.get("tdx_mrtd")),
        "rp_rtmr0": _norm_hex(meas.get("tdx_rtmr0")),
        "rp_rtmr1": _norm_hex(meas.get("tdx_rtmr1")),
        "rp_rtmr2": _norm_hex(meas.get("tdx_rtmr2")),
        "rp_rtmr3": _norm_hex(meas.get("tdx_rtmr3")),
        "rp_tdx_attributes": _norm_hex(meas.get("tdx_attributes")),
    }


def _parse_appraisal_jwt(jwt: str) -> dict:
    parts = jwt.split(".")
    if len(parts) < 2:
        return {"overall_appraisal_result": None, "appraisal_ok": False,
                "reason": "no JWT emitted by verifier"}
    try:
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
    except Exception as exc:                                   # noqa: BLE001
        return {"overall_appraisal_result": None, "appraisal_ok": False,
                "reason": f"bad JWT payload: {exc}"}
    ar = payload.get("appraisal_result")
    if isinstance(ar, str):
        try:
            ar = json.loads(ar)
        except Exception:                                      # noqa: BLE001
            ar = None
    overall = None
    if isinstance(ar, list) and ar and isinstance(ar[0], dict):
        overall = (ar[0].get("result") or {}).get("overall_appraisal_result")
    elif isinstance(ar, dict):
        overall = (ar.get("result") or {}).get("overall_appraisal_result")
    ok = overall in (1, "1", True, "SUCCESS", "success")
    return {"overall_appraisal_result": overall, "appraisal_ok": bool(ok)}


def parse_td_quote(quote: bytes) -> dict:
    """Extract measured fields from the signed TD Quote body (fixed offsets)."""
    if len(quote) < _OFF_REPORT_DATA + 64:
        raise QuoteError(f"quote too short ({len(quote)} bytes) to be a TD Quote")
    version = int.from_bytes(quote[0:2], "little")
    tee_type = int.from_bytes(quote[4:8], "little")
    td_attributes = quote[_OFF_TD_ATTRIBUTES:_OFF_TD_ATTRIBUTES + 8]
    mr_td = quote[_OFF_MR_TD:_OFF_MR_TD + 48].hex()
    rtmrs = [quote[_OFF_RTMR0 + 48 * i:_OFF_RTMR0 + 48 * (i + 1)].hex() for i in range(4)]
    report_data_embedded = quote[_OFF_REPORT_DATA:_OFF_REPORT_DATA + 64].hex()
    return {
        "quote_version": version,
        "tee_type": "tdx" if tee_type == 0x00000081 else f"0x{tee_type:08x}",
        "td_attributes": td_attributes.hex(),
        "debug": bool(td_attributes[0] & 0x01),      # TD_ATTRIBUTES bit 0 = DEBUG
        "mr_td": mr_td,
        "rtmr0": rtmrs[0], "rtmr1": rtmrs[1], "rtmr2": rtmrs[2], "rtmr3": rtmrs[3],
        "report_data_embedded": report_data_embedded,
    }


@dataclass
class QuoteBundle:
    report_data_hex: str                # our expected (normal-order) report_data
    quote_b64: str
    quote_hash: str
    overall_appraisal_result: object
    appraisal_ok: bool
    reportdata_binds: bool              # AUTHORITATIVE binding (relying-party verified)
    mr_td: str
    signature_verified: object = None   # relying_party -v (ES384 policy sig)
    appraisal_source: str = ""
    rp_reportdata_binds: object = None  # relying_party tdx_reportdata == expected (normal)
    binary_reportdata_binds: object = None   # v4-only binary cross-check (None if n/a)
    rtmr0: str = ""
    rtmr1: str = ""
    rtmr2: str = ""
    rtmr3: str = ""
    debug: bool = False
    td_attributes: str = ""
    quote_version: int = 0
    tee_type: str = ""
    appraisal_jwt: str = ""
    jwt_parts: int = 0
    verifier_returncode: int = -1
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("appraisal_jwt", None)     # kept separately; large
        return d


def generate_and_appraise(report_data_hex: str, workdir, *, qgen_app: str = QGEN_APP,
                          qverify_dir: str = QVERIFY_DIR) -> QuoteBundle:
    """Full guest-side flow: generate a real quote for report_data, appraise via QVL,
    parse the measured fields, and confirm the embedded report_data binds."""
    quote = generate_quote(report_data_hex, workdir, qgen_app=qgen_app)
    quote_path = Path(workdir) / "quote.dat"
    appraisal = appraise_quote(quote_path, qverify_dir=qverify_dir)
    fields = parse_td_quote(quote)
    # AUTHORITATIVE binding + measurements come from the DCAP-verified relying_party
    # (works for every TD Quote version). The fixed-offset binary parse is a v4-ONLY
    # cross-check; TD Quote v5 relocates the report body, so it is skipped for v5+.
    rp_rd = appraisal.get("rp_tdx_reportdata")
    rp_binds = (rp_rd == report_data_hex.lower()) if rp_rd else None
    expected_embedded = _qgen_reportdata(report_data_hex)
    binary_binds = (fields["report_data_embedded"].lower() == expected_embedded.lower()
                    if fields["quote_version"] == 4 else None)
    # relying_party fields take precedence; fall back to the binary parse
    mr_td = appraisal.get("rp_mr_td") or fields["mr_td"]
    rp_attr = appraisal.get("rp_tdx_attributes")
    debug = bool(int(rp_attr[:2], 16) & 0x01) if rp_attr else fields["debug"]
    td_attributes = rp_attr or fields["td_attributes"]
    rtmr = {i: appraisal.get(f"rp_rtmr{i}") or fields[f"rtmr{i}"] for i in range(4)}
    # authoritative reportdata_binds: relying-party when available, else v4 binary
    authoritative_binds = rp_binds if rp_binds is not None else binary_binds
    return QuoteBundle(
        report_data_hex=report_data_hex.lower(),
        quote_b64=base64.b64encode(quote).decode(),
        quote_hash=sha256(quote).hexdigest(),
        overall_appraisal_result=appraisal.get("overall_appraisal_result"),
        appraisal_ok=appraisal.get("appraisal_ok", False),
        reportdata_binds=bool(authoritative_binds),
        mr_td=mr_td, rtmr0=rtmr[0], rtmr1=rtmr[1], rtmr2=rtmr[2], rtmr3=rtmr[3],
        debug=debug, td_attributes=td_attributes, quote_version=fields["quote_version"],
        tee_type=fields["tee_type"], appraisal_jwt=appraisal.get("appraisal_jwt", ""),
        jwt_parts=appraisal.get("jwt_parts", 0),
        verifier_returncode=appraisal.get("verifier_returncode", -1),
        signature_verified=appraisal.get("signature_verified"),
        appraisal_source=appraisal.get("appraisal_source", ""),
        rp_reportdata_binds=rp_binds, binary_reportdata_binds=binary_binds,
        extra={"reason": appraisal.get("reason"),
               "relying_party_returncode": appraisal.get("relying_party_returncode"),
               "rp_tdx_attributes": rp_attr})
