"""Tests for the private-base security harness (CPU, offline, fast).

Validates the properties the S1-S4 attacks depend on, and the attacker-wall discipline.
Run:  python3 scripts/security/test_security_harness.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_harness import PrivateBaseOracle  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def main():
    o = PrivateBaseOracle()
    H = torch.randn(32, o.H, dtype=torch.float64)

    # 1. residual mask orthogonal + exactly invertible
    check("Nr_orthogonal", float((o.Nr @ o.Nr.T - torch.eye(o.H, dtype=torch.float64)).abs().max()) < 1e-10)
    Ht = o.observe_residual(H)
    check("Nr_roundtrip_exact", float((Ht @ o.Nr_inv - H).abs().max()) < 1e-10)

    # 2. documented invariant leaks: norm + pairwise gram preserved exactly
    check("norm_preserved", float((H.norm(dim=1) - Ht.norm(dim=1)).abs().max()) < 1e-10)
    check("gram_preserved", float(((H @ H.T) - (Ht @ Ht.T)).abs().max()) < 1e-9)

    # 3. LoRA masking: A_tilde B_tilde == Nout^{-1} (A B) Nin, and hides plaintext product
    lora = o.make_lora(896, 896, 16, seed=1)
    masked_ref = lora["Nout"].T @ (lora["A"] @ lora["B"]) @ lora["Nin"]
    check("lora_masked_product_matches", float((lora["dW_tilde"] - masked_ref).abs().max()) < 1e-9)
    plaintext_gap = float((lora["dW_tilde"] - lora["dW"]).abs().max())
    check("lora_plaintext_hidden", plaintext_gap > 1e-2, f"(gap={plaintext_gap:.3f})")

    # 4. logit schemes: perm-only preserves multiset; monomial changes it
    lg = torch.randn(4, o.V, dtype=torch.float64)
    po = o.observe_logits(lg, "perm_only")
    mo = o.observe_logits(lg, "monomial")
    check("perm_only_multiset_exact",
          float((lg.sort(-1).values - po.sort(-1).values).abs().max()) < 1e-9)
    check("monomial_multiset_changed",
          float((lg.sort(-1).values - mo.sort(-1).values).abs().max()) > 1e-2)

    # 5. determinism: fresh oracle reproduces identical masks (fixed seeds)
    o2 = PrivateBaseOracle()
    check("masks_deterministic", torch.equal(o.Nr, o2.Nr) and torch.equal(o.vocab_perm, o2.vocab_perm))

    # 6. attacker-wall: the transformed embedding table is E@Nr (no plaintext table leak path
    #    in observe_* outputs) and equals a change of basis of the real table
    Et = o.observe_embed_table()
    E = o.embed_table_plain()
    check("embed_table_is_masked", float((Et - E @ o.Nr).abs().max()) < 1e-6,
          "(observe_embed_table returns E@Nr only)")
    check("embed_table_not_plaintext", float((Et - E).abs().max()) > 1e-3)

    print(f"\n[test_security_harness] PASS={PASS} FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
