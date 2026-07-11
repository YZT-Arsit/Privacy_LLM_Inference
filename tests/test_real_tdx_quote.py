"""Gate 2 §2 — TD Quote binary parsing + report_data byte-order (offline).

No real quote here (that is the guest). We validate the fixed-offset TD-report parser
and the qgen report_data byte-order transform against a synthetic quote frame.
"""

from __future__ import annotations

import pytest

from pllo.experiments.real_tdx_quote import (
    QuoteError,
    _qgen_reportdata,
    _reverse_hex,
    generate_and_appraise,      # noqa: F401 (import smoke)
    parse_td_quote,
)

_HEADER = 48
_OFF_TD_ATTRIBUTES = _HEADER + 120
_OFF_MR_TD = _HEADER + 136
_OFF_RTMR0 = _HEADER + 328
_OFF_REPORT_DATA = _HEADER + 520


def _synth_quote(*, mr_td=b"\x11" * 48, td_attributes=b"\x00" * 8,
                 report_data=b"\x22" * 64, rtmr0=b"\x33" * 48):
    buf = bytearray(_OFF_REPORT_DATA + 64)
    buf[0:2] = (4).to_bytes(2, "little")               # version 4
    buf[4:8] = (0x00000081).to_bytes(4, "little")      # TEE type = TDX
    buf[_OFF_TD_ATTRIBUTES:_OFF_TD_ATTRIBUTES + 8] = td_attributes
    buf[_OFF_MR_TD:_OFF_MR_TD + 48] = mr_td
    buf[_OFF_RTMR0:_OFF_RTMR0 + 48] = rtmr0
    buf[_OFF_REPORT_DATA:_OFF_REPORT_DATA + 64] = report_data
    return bytes(buf)


def test_parse_fields():
    q = _synth_quote()
    f = parse_td_quote(q)
    assert f["quote_version"] == 4 and f["tee_type"] == "tdx"
    assert f["mr_td"] == "11" * 48 and f["report_data_embedded"] == "22" * 64
    assert f["rtmr0"] == "33" * 48 and f["debug"] is False


def test_debug_bit_detected():
    q = _synth_quote(td_attributes=b"\x01" + b"\x00" * 7)   # DEBUG = bit 0
    assert parse_td_quote(q)["debug"] is True


def test_non_debug_seam_bit_not_confused():
    # td_attributes 00 00 00 00 10 00 00 00 (a non-DEBUG bit set) -> debug False
    q = _synth_quote(td_attributes=bytes.fromhex("0000000010000000"))
    assert parse_td_quote(q)["debug"] is False


def test_short_quote_rejected():
    with pytest.raises(QuoteError, match="too short"):
        parse_td_quote(b"\x00" * 100)


def test_report_data_byteorder_roundtrip():
    rd = "ab" * 32 + "cd" * 32           # two distinct 32-byte halves
    embedded = _qgen_reportdata(rd)      # each half byte-reversed, halves swapped
    # recover: split embedded into halves h1,h2; rd == reverse(h2)+reverse(h1)
    h1, h2 = embedded[:64], embedded[64:]
    assert _reverse_hex(h2) + _reverse_hex(h1) == rd


def test_qgen_reportdata_rejects_bad_length():
    with pytest.raises(QuoteError, match="64 bytes"):
        _qgen_reportdata("ab" * 16)
