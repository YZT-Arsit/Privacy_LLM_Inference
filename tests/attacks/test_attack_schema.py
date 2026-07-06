"""AttackResult v2 schema: roundtrip + honesty invariants."""

from __future__ import annotations

import pytest

from pllo.attacks.schema import AttackResult, blocked_result, empty_metrics, make_attacker_knowledge


def _r(**kw):
    base = dict(attack_id="x", attack_name="X", attack_family="structural",
                target_method="toy", threat_model="public_model")
    base.update(kw)
    return AttackResult(**base)


def test_json_roundtrip():
    r = _r()
    r.metrics["attack_success_rate"] = 0.5
    d = r.to_dict()
    r2 = AttackResult.from_dict(d)
    assert r2.metrics["attack_success_rate"] == 0.5
    assert set(r2.metrics) == set(empty_metrics())
    assert set(r2.attacker_knowledge) == set(make_attacker_knowledge())


def test_measured_requires_metric():
    r = _r(status="measured")
    assert any("measured status requires" in e for e in r.validate())
    r.metrics["mse"] = 0.1
    assert r.validate() == []


def test_blocked_requires_reason_and_no_metric():
    r = blocked_result("a", "A", "alignment", "stip_qwen", "weight_leakage_worst_case", "need weights")
    assert r.status == "blocked" and r.implementation_level == "blocked"
    assert r.validate() == []
    r.metrics["attack_success_rate"] = 0.9
    assert any("blocked status must not carry" in e for e in r.validate())


def test_weights_require_whitebox_threat_model():
    r = _r(threat_model="closed_model_no_weight_access",
           attacker_knowledge=make_attacker_knowledge(has_model_weights=True))
    assert any("has_model_weights" in e for e in r.validate())
    for tm in ("weight_leakage_worst_case", "split_inference", "public_model"):
        r.threat_model = tm
        assert not any("has_model_weights" in e for e in r.validate())


def test_kpa_pairs_require_known_plaintext():
    r = _r(threat_model="public_model",
           attacker_knowledge=make_attacker_knowledge(has_known_plaintext_pairs=True))
    assert any("has_known_plaintext_pairs" in e for e in r.validate())
    r.threat_model = "known_plaintext"
    assert not any("has_known_plaintext_pairs" in e for e in r.validate())


def test_implementation_level_validated():
    r = _r(implementation_level="bogus", status="blocked", error="x")
    assert any("implementation_level" in e for e in r.validate())
    for lvl in ("full", "best_effort", "partial", "blocked"):
        r.implementation_level = lvl
        assert not any("implementation_level" in e for e in r.validate())
