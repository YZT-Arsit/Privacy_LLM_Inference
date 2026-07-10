"""Gate 1.5-A — safe tensor codec security tests (no pickle deserialization)."""

from __future__ import annotations

import json
import struct

import pytest
import torch

from pllo.experiments.real_tdx_safe_codec import (
    MAGIC,
    CodecError,
    Limits,
    decode_message,
    encode_message,
)


def test_valid_bf16_and_nested_round_trip():
    msg = {"run_id": "r", "step_id": 3, "loss": 1.25, "flag": True, "none": None,
           "grads": {"L0": {"gradA": torch.randn(4, 3, dtype=torch.bfloat16),
                            "gradB": torch.randn(3, 5, dtype=torch.float32)}},
           "list": [torch.zeros(2, 2, dtype=torch.int64)]}
    back = decode_message(encode_message(msg))
    assert back["run_id"] == "r" and back["step_id"] == 3 and back["loss"] == 1.25
    assert back["grads"]["L0"]["gradA"].dtype == torch.bfloat16
    assert (back["grads"]["L0"]["gradB"] - msg["grads"]["L0"]["gradB"]).abs().max() == 0
    assert back["list"][0].dtype == torch.int64


def test_malicious_pickle_payload_rejected_without_deserialization():
    import pickle

    class Boom:
        def __reduce__(self):
            raise AssertionError("pickle must never be executed by the codec")
    blob = b"\x80\x04" + pickle.dumps({"x": 1})       # a pickle stream, not our frame
    with pytest.raises(CodecError, match="bad magic"):
        decode_message(blob)
    # even a torch.save blob (which IS pickle) is rejected on magic, never loaded
    import io
    buf = io.BytesIO(); torch.save(torch.randn(3), buf)
    with pytest.raises(CodecError, match="bad magic"):
        decode_message(buf.getvalue())


def _frame(structure, tensors_meta, body: bytes):
    header = json.dumps({"structure": structure, "tensors": tensors_meta}).encode()
    return MAGIC + bytes([1]) + struct.pack("<I", len(header)) + header + body


def test_oversized_shape_rejected_before_allocation():
    # dims within max_dim, but product*itemsize exceeds max_tensor_bytes -> reject pre-alloc
    meta = [{"dtype": "float32", "shape": [1_000_000, 1000], "nbytes": 4}]
    frame = _frame({"__t__": 0}, meta, b"\x00" * 4)
    with pytest.raises(CodecError, match="max_tensor_bytes"):
        decode_message(frame)


def test_huge_dim_rejected_before_allocation():
    meta = [{"dtype": "float32", "shape": [10**9, 10**9], "nbytes": 4}]
    frame = _frame({"__t__": 0}, meta, b"\x00" * 4)
    with pytest.raises(CodecError, match="bad dim"):        # rejected, no allocation
        decode_message(frame)


def test_running_product_rejected_before_allocation():
    # 6 dims each within max_dim, but running numel*itemsize blows the byte cap
    meta = [{"dtype": "int64", "shape": [1_000_000] * 6, "nbytes": 8}]
    frame = _frame({"__t__": 0}, meta, b"\x00" * 8)
    with pytest.raises(CodecError, match="max_tensor_bytes"):
        decode_message(frame)


def test_dtype_not_whitelisted_rejected():
    meta = [{"dtype": "float64", "shape": [2], "nbytes": 16}]
    frame = _frame({"__t__": 0}, meta, b"\x00" * 16)
    with pytest.raises(CodecError, match="whitelist"):
        decode_message(frame)


def test_truncated_payload_rejected():
    t = torch.randn(4, 4, dtype=torch.float32)
    data = bytearray(encode_message({"a": t}))
    with pytest.raises(CodecError, match="length mismatch|truncated"):
        decode_message(bytes(data[:-4]))


def test_extra_trailing_bytes_rejected():
    t = torch.randn(4, 4, dtype=torch.float32)
    data = encode_message({"a": t}) + b"\xff\xff\xff\xff"
    with pytest.raises(CodecError, match="length mismatch"):
        decode_message(data)


def test_duplicate_tensor_ref_rejected():
    meta = [{"dtype": "float32", "shape": [2], "nbytes": 8}]
    frame = _frame({"a": {"__t__": 0}, "b": {"__t__": 0}}, meta, b"\x00" * 8)
    with pytest.raises(CodecError, match="duplicate tensor ref"):
        decode_message(frame)


def test_unexpected_tensor_payload_rejected():
    # two tensors in payload but structure references only one
    meta = [{"dtype": "float32", "shape": [1], "nbytes": 4},
            {"dtype": "float32", "shape": [1], "nbytes": 4}]
    frame = _frame({"a": {"__t__": 0}}, meta, b"\x00" * 8)
    with pytest.raises(CodecError, match="unused/unexpected"):
        decode_message(frame)


def test_nan_inf_policy_enforced():
    t = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    data = encode_message({"g": t})
    decode_message(data)                                   # default: allowed
    with pytest.raises(CodecError, match="NaN/Inf"):
        decode_message(data, limits=Limits(forbid_nan_inf=True))


def test_too_many_tensors_rejected():
    lim = Limits(max_tensors=2)
    msg = {"a": torch.zeros(1), "b": torch.zeros(1), "c": torch.zeros(1)}
    with pytest.raises(CodecError, match="too many tensors"):
        decode_message(encode_message(msg), limits=lim)


def test_real_tdx_mode_refuses_unsafe_codec():
    from pllo.experiments.real_tdx_attestation import digest_config
    from pllo.experiments.real_tdx_training_service import TrustedTrainingService, serve_http
    cfg = {"config_digest": digest_config({"x": 1})}
    # a require_real_tdx service must refuse to serve the pickle codec
    svc = TrustedTrainingService(run_id="r", config=cfg, mode="real_tdx",
                                 require_real_tdx=True)
    with pytest.raises(RuntimeError, match="unsafe torch_save codec"):
        serve_http(svc, codec="torch_save")
