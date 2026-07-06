"""Result IO (failed/blocked safe) + security table builder (measured vs qualitative)."""

from __future__ import annotations

import json
from pathlib import Path

from pllo.attacks.result_io import (failed_result, read_jsonl, results_to_csv,
                                    results_to_markdown, write_jsonl)
from pllo.attacks.schema import AttackResult, blocked_result
from pllo.attacks.table_builder import (build_measured_table, write_measured_table,
                                        write_qualitative_table)


def _measured(aid="nn_embedding_inversion", method="stip_qwen", succ=0.7):
    r = AttackResult(attack_id=aid, attack_name=aid, attack_family="structural",
                     target_method=method, threat_model="public_model")
    r.metrics["attack_success_rate"] = succ
    return r


def test_jsonl_roundtrip_with_failed_and_blocked(tmp_path):
    rs = [_measured(),
          failed_result("eia_optimization", "EIA", "optimization", "toy", "public_model", "oom"),
          blocked_result("bre_bisr_backward", "BiSR-b", "optimization", "stip_qwen",
                         "split_inference", "needs training gradients")]
    p = write_jsonl(rs, tmp_path / "r.jsonl")
    back = read_jsonl(p)
    assert len(back) == 3
    assert {r.status for r in back} == {"measured", "failed", "blocked"}


def test_malformed_line_skipped(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(_measured().to_dict()) + "\nNOT JSON\n")
    assert len(read_jsonl(p)) == 1


def test_csv_md_written(tmp_path):
    rs = [_measured(), failed_result("kpa", "KPA", "cryptanalysis", "toy", "known_plaintext", "boom")]
    assert results_to_csv(rs, tmp_path / "s.csv").exists()
    assert results_to_markdown(rs, tmp_path / "s.md").exists()
    assert "failed" in (tmp_path / "s.csv").read_text()


def test_measured_table_only_measured():
    rs = [_measured("kpa_known_plaintext", "obfuscatune_qwen_orthogonal", 1.0),
          _measured("kpa_known_plaintext", "ours_amulet_style", 0.0)]
    t = build_measured_table(rs)
    kpa = [r for r in t["rows"] if r["attack_probe"] == "kpa_known_plaintext"][0]
    assert kpa["obfuscatune_qwen_orthogonal"] == "1.000"
    assert kpa["ours_amulet_style"] == "0.000"
    assert kpa["plaintext_gpu"] == "missing"


def test_blocked_cell_shows_blocked():
    rs = [blocked_result("arrowmatch_weight_alignment", "Arrow", "alignment",
                         "obfuscatune_qwen_orthogonal", "weight_leakage_worst_case", "no weights")]
    t = build_measured_table(rs)
    row = [r for r in t["rows"] if r["attack_probe"] == "arrowmatch_weight_alignment"][0]
    assert row["obfuscatune_qwen_orthogonal"] == "blocked"


def test_qualitative_filename_and_banner(tmp_path):
    cfg = {"kpa_known_plaintext": {"obfuscatune_qwen_orthogonal": "high_expected"}}
    paths = write_qualitative_table(cfg, tmp_path)
    assert "qualitative_draft" in paths["md"].name
    assert "Qualitative draft; not measured" in paths["md"].read_text()


def test_qualitative_does_not_overwrite_measured(tmp_path):
    write_measured_table([_measured()], tmp_path)
    write_qualitative_table({"nn_embedding_inversion": {"stip_qwen": "medium"}}, tmp_path)
    assert (tmp_path / "security_attack_table_measured.json").exists()
    assert (tmp_path / "security_attack_table_qualitative_draft.json").exists()
