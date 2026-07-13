"""Negative + positive tests for the secure tensor frame (PHASE 1.2).

Covers the required rejection cases: pickle payload, unknown tensor name, wrong dtype,
wrong rank, oversized shape, duplicate factor, incomplete trusted factor set, malformed frame.
Run: python3 scripts/test_secure_tensor_frame.py
"""
from __future__ import annotations
import io, json, struct, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from secure_tensor_frame import encode, decode, FrameSpec, FrameError, MAGIC  # noqa: E402

P = F = 0


def ok(name, cond):
    global P, F
    if cond:
        P += 1; print(f"  ok   {name}")
    else:
        F += 1; print(f"  FAIL {name}")


def rejects(name, fn):
    try:
        fn()
        ok(name, False)
    except FrameError:
        ok(name, True)
    except Exception as e:  # noqa: BLE001
        ok(name + f" (wrong exc {type(e).__name__})", False)


def main():
    # ---- positive round-trips ----
    spec = FrameSpec(op="ce_dlogits", allowed_names={"logits"}, max_rank=2, max_total_bytes=10_000_000)
    blob = encode("ce_dlogits", {"logits": torch.randn(8, 128, dtype=torch.bfloat16)})
    dec = decode(blob, spec)
    ok("roundtrip bf16 logits shape", tuple(dec["logits"].shape) == (8, 128) and dec["logits"].dtype == torch.bfloat16)

    nspec = FrameSpec(op="init_adamw", allowed_name_prefixes=("A/", "B/"),
                      expected_names={"A/0.q_proj", "B/0.q_proj"}, max_rank=2)
    nested = {"A": {"0.q_proj": torch.randn(16, 32)}, "B": {"0.q_proj": torch.randn(8, 16)}}
    dec2 = decode(encode("init_adamw", nested), nspec)
    ok("roundtrip nested unflatten", set(dec2) == {"A", "B"} and "0.q_proj" in dec2["A"])
    val = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    ok("roundtrip bit-exact", torch.equal(decode(encode("x", {"t": val}), FrameSpec(op="x", allowed_names={"t"}))["t"], val))

    # ---- 8 required negative cases ----
    # 1. pickle payload (torch.save/pickle) must be refused (bad magic)
    bio = io.BytesIO(); torch.save({"logits": torch.randn(4, 4)}, bio)
    rejects("1 pickle payload rejected", lambda: decode(bio.getvalue(), spec))

    # 2. unknown tensor name
    bad2 = encode("ce_dlogits", {"evil": torch.randn(2, 2, dtype=torch.bfloat16)})
    rejects("2 unknown tensor name rejected", lambda: decode(bad2, spec))

    # 3. wrong dtype (float64 when spec allows only bf16-ish)
    spec_dt = FrameSpec(op="ce_dlogits", allowed_names={"logits"}, allowed_dtypes={"bfloat16"}, max_rank=2)
    bad3 = encode("ce_dlogits", {"logits": torch.randn(2, 2, dtype=torch.float64)})
    rejects("3 wrong dtype rejected", lambda: decode(bad3, spec_dt))

    # 4. wrong rank (3D when max_rank=2)
    bad4 = encode("ce_dlogits", {"logits": torch.randn(2, 2, 2, dtype=torch.bfloat16)})
    rejects("4 wrong rank rejected", lambda: decode(bad4, spec))

    # 5. oversized shape (numel over cap)
    spec_small = FrameSpec(op="ce_dlogits", allowed_names={"logits"}, max_numel=10, max_rank=2)
    bad5 = encode("ce_dlogits", {"logits": torch.randn(100, 100, dtype=torch.bfloat16)})
    rejects("5 oversized shape rejected", lambda: decode(bad5, spec_small))

    # 6. duplicate factor (hand-craft a frame with two identical names)
    hdr = json.dumps({"op": "init_adamw", "tensors": [
        {"name": "A/0.q_proj", "dtype": "float32", "shape": [1]},
        {"name": "A/0.q_proj", "dtype": "float32", "shape": [1]}]}).encode()
    body = struct.pack("<f", 1.0) + struct.pack("<f", 2.0)
    dup = MAGIC + struct.pack(">I", len(hdr)) + hdr + body
    rejects("6 duplicate factor rejected", lambda: decode(dup, nspec))

    # 7. incomplete trusted factor set (expected A/0.q_proj + B/0.q_proj, send only A)
    bad7 = encode("init_adamw", {"A": {"0.q_proj": torch.randn(16, 32)}})
    rejects("7 incomplete factor set rejected", lambda: decode(bad7, nspec))

    # 8. malformed frame (truncated body)
    good = encode("ce_dlogits", {"logits": torch.randn(4, 8, dtype=torch.bfloat16)})
    rejects("8 malformed/truncated frame rejected", lambda: decode(good[:-5], spec))

    # bonus: forbidden substring (mask-secret hint) in a name
    fspec = FrameSpec(op="init_adamw", allowed_name_prefixes=("A/",), forbid_substrings=("n_res", "perm"))
    badf = encode("init_adamw", {"A": {"0.q_proj_perm": torch.randn(2, 2)}})
    rejects("bonus forbidden-substring name rejected", lambda: decode(badf, fspec))

    # bonus: extra header key
    hdr2 = json.dumps({"op": "x", "tensors": [], "evil": 1}).encode()
    rejects("bonus extra header key rejected",
            lambda: decode(MAGIC + struct.pack(">I", len(hdr2)) + hdr2 + b"", FrameSpec(op="x")))

    print(f"\n[test_secure_tensor_frame] PASS={P} FAIL={F}")
    sys.exit(1 if F else 0)


if __name__ == "__main__":
    main()
