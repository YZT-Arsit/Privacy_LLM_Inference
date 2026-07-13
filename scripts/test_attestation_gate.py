"""PHASE 1.1 negative gate test — attestation MUST gate execution (no optimizer init, no step).

Tests the pure gate predicate used by the TDX service (`attested_execution_allowed`) and the
equivalent A10-runner predicate. The contract under test:

    invalid_attestation -> execution NOT allowed -> (service refuses protected ops;
                            runner raises before init_adamw / any training/eval step).

Run: python3 scripts/test_attestation_gate.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdx_persistent_service import attested_execution_allowed as tdx_ok  # noqa: E402

P = F = 0


def ok(name, cond):
    global P, F
    if cond:
        P += 1; print(f"  ok   {name}")
    else:
        F += 1; print(f"  FAIL {name}")


# mirror of the A10-runner gate (scripts/a10_batch_runner.py, post-handshake)
def runner_refuses(attestation: dict, hs: dict, require_flag: bool) -> bool:
    att_verified = attestation.get("attestation_verified")
    att_skipped = bool(attestation.get("attestation_skipped", False))
    att_session_valid = bool(hs.get("attested_session_valid", att_verified))
    att_required = bool(require_flag) or (not att_skipped and att_verified is not None)
    return att_required and not att_session_valid          # True => runner raises, no init/step


def main():
    good = {"attestation_verified": True, "reportdata_bound": True, "debug_false": True}
    # ---- TDX service predicate ----
    ok("TDX: not required -> allowed", tdx_ok({"attestation_skipped": True}, False) is True)
    ok("TDX: required + verified -> allowed", tdx_ok(good, True) is True)
    ok("TDX: required + NOT verified -> refused",
       tdx_ok({"attestation_verified": False}, True) is False)
    ok("TDX: required + reportdata NOT bound -> refused",
       tdx_ok({"attestation_verified": True, "reportdata_bound": False, "debug_false": True}, True) is False)
    ok("TDX: required + debug TRUE -> refused",
       tdx_ok({"attestation_verified": True, "reportdata_bound": True, "debug_false": False}, True) is False)
    ok("TDX: required + attestation error -> refused",
       tdx_ok({"attestation_verified": False, "error": "no /dev/tdx_guest"}, True) is False)

    # ---- A10 runner predicate (refuse => raise before init_adamw) ----
    ok("A10: skipped attestation, no --require -> proceeds",
       runner_refuses({"attestation_skipped": True}, {}, False) is False)
    ok("A10: enclave reports verified session -> proceeds",
       runner_refuses(good, {"attested_session_valid": True}, False) is False)
    ok("A10: enclave reports INVALID session -> refuses (no init/step)",
       runner_refuses({"attestation_verified": False}, {"attested_session_valid": False}, False) is True)
    ok("A10: --require-attestation but session invalid -> refuses",
       runner_refuses({"attestation_verified": False}, {"attested_session_valid": False}, True) is True)
    ok("A10: --require-attestation but no attestation present -> refuses",
       runner_refuses({}, {}, True) is True)

    # the whole point: an invalid attestation blocks BOTH sides
    invalid = {"attestation_verified": False, "error": "verify failed"}
    ok("END-TO-END invalid attestation: TDX refuses AND A10 refuses",
       (tdx_ok(invalid, True) is False) and (runner_refuses(invalid, {"attested_session_valid": False}, True) is True))

    print(f"\n[test_attestation_gate] PASS={P} FAIL={F}")
    sys.exit(1 if F else 0)


if __name__ == "__main__":
    main()
