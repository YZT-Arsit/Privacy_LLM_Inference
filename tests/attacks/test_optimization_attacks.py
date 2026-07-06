"""EIA / BiSR-forward / PIA optimization attacks: loss decreases, metrics finite."""

from __future__ import annotations

import math

from pllo.attacks.bre_bisr_attack import run_bre_bisr_attack, run_bre_backward_gradient_matching
from pllo.attacks.eia_optimization import run_eia_optimization
from pllo.attacks.pia_prompt_inversion import run_pia_prompt_inversion
from pllo.attacks.representations import toy_representations


def _initial_final(notes):
    parts = notes.replace(";", " ").split()
    def grab(pfx):
        for p in parts:
            if p.startswith(pfx):
                return float(p.split("=")[1])
        return None
    return grab("initial="), grab("initial_loss="), grab("final="), grab("final_loss=")


def test_eia_loss_decreases():
    inp = toy_representations(vocab_size=64, hidden_size=16, num_samples=8, seed=0)
    r = run_eia_optimization(inp, target_method="toy", num_steps=100, lr=0.2)
    assert r.status == "measured" and not math.isnan(r.metrics["mse"])
    i1, i2, f1, f2 = _initial_final(r.notes)
    assert (f2 or f1) < (i2 or i1)


def test_bisr_forward_loss_decreases_and_not_full_bisr():
    inp = toy_representations(vocab_size=64, hidden_size=16, num_samples=8, seed=1)
    r = run_bre_bisr_attack(inp, target_method="toy", num_steps=120)
    assert r.status == "measured" and r.implementation_level == "best_effort"
    assert "NOT full BiSR" in r.notes
    i1, i2, f1, f2 = _initial_final(r.notes)
    assert (f1 or f2) < (i1 or i2)


def test_bisr_backward_blocked_with_reason():
    r = run_bre_backward_gradient_matching(toy_representations(seed=0), target_method="toy")
    assert r.status == "blocked" and r.implementation_level == "blocked"
    assert "grad" in r.error and r.validate() == []


def test_pia_runs_and_labels_omitted_oracle():
    inp = toy_representations(vocab_size=64, hidden_size=16, num_samples=6, seed=2)
    r = run_pia_prompt_inversion(inp, target_method="toy", num_steps=60)
    assert r.status == "measured" and r.implementation_level == "best_effort"
    assert "S_s OMITTED" in r.notes
    assert r.validate() == []
