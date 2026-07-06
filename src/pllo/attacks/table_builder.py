"""Security main table (Table B) builder — measured and qualitative-draft modes.

measured mode (default): cells come ONLY from real AttackResults. Missing =>
'missing', blocked => 'blocked', failed => 'failed'. No subjective labels.

qualitative-draft mode: from configs/attack_qualitative_matrix.json; filenames
carry 'qualitative_draft' and the header states it is not measured. Never
overwrites the measured table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import AttackResult

DEFAULT_METHOD_COLUMNS = ["plaintext_gpu", "stip_qwen", "obfuscatune_qwen_orthogonal",
                          "ours_amulet_style"]
DEFAULT_ATTACK_ROWS = [
    "nn_embedding_inversion", "eia_optimization", "bre_bisr_forward", "bre_bisr_backward",
    "kpa_known_plaintext", "multiset_permutation_leakage", "arrowmatch_weight_alignment",
    "gram_weight_recovery", "pia_prompt_inversion", "frequency_distribution",
]
_FAMILY = {
    "nn_embedding_inversion": "structural", "eia_optimization": "optimization",
    "bre_bisr_forward": "optimization", "bre_bisr_backward": "optimization",
    "kpa_known_plaintext": "cryptanalysis", "multiset_permutation_leakage": "structural",
    "arrowmatch_weight_alignment": "alignment", "gram_weight_recovery": "alignment",
    "pia_prompt_inversion": "optimization", "frequency_distribution": "statistical",
}


def _cell(results: list[AttackResult], attack_id: str, method: str) -> str:
    matches = [r for r in results if r.attack_id == attack_id and r.target_method == method]
    if not matches:
        return "missing"
    for r in matches:
        if r.status == "measured":
            for key in ("attack_success_rate", "token_recovery_top1", "alignment_accuracy",
                        "multiset_leakage_score", "permutation_recovery_accuracy"):
                v = r.metrics.get(key)
                if isinstance(v, (int, float)):
                    return f"{v:.3f}"
            return "measured"
    st = matches[0].status
    if st == "blocked":
        return "blocked"
    if st == "failed":
        return f"failed: {(matches[0].error or '')[:30]}"
    return st


def build_measured_table(results, *, methods=None, attacks=None) -> dict[str, Any]:
    methods = methods or DEFAULT_METHOD_COLUMNS
    attacks = attacks or DEFAULT_ATTACK_ROWS
    rows = []
    for aid in attacks:
        rel = [r for r in results if r.attack_id == aid]
        # prefer a MEASURED result for the row metadata so a blocked cell in the
        # first method column (e.g. plaintext has no weights for ArrowMatch)
        # doesn't mislabel the whole row as blocked.
        rep = next((r for r in rel if r.status == "measured"), rel[0] if rel else None)
        row = {"attack_probe": aid, "attack_family": _FAMILY.get(aid, "?"),
               "implementation_level": rep.implementation_level if rep else "?",
               "threat_model": rep.threat_model if rep else "n/a",
               "attacker_knowledge": ("weights" if rep and rep.attacker_knowledge.get("has_model_weights")
                                      else ("kpa_pairs" if rep and rep.attacker_knowledge.get("has_known_plaintext_pairs")
                                            else "activations"))}
        for m in methods:
            row[m] = _cell(results, aid, m)
        row["notes"] = (rep.notes[:120] if rep and rep.notes else "")
        rows.append(row)
    return {"mode": "measured", "methods": methods, "rows": rows,
            "disclaimer": "Cells are measured attack results only; 'missing' = no data, "
                          "'blocked' = not implementable in this harness."}


def build_qualitative_table(config, *, methods=None) -> dict[str, Any]:
    methods = methods or DEFAULT_METHOD_COLUMNS
    rows = []
    for aid, per_method in config.items():
        if aid.startswith("_"):
            continue
        row = {"attack_probe": aid, "attack_family": _FAMILY.get(aid, "?"),
               "implementation_level": "n/a", "threat_model": "n/a",
               "attacker_knowledge": "n/a"}
        for m in methods:
            row[m] = per_method.get(m, "n/a")
        row["notes"] = "QUALITATIVE — not measured"
        rows.append(row)
    return {"mode": "qualitative_draft", "methods": methods, "rows": rows,
            "disclaimer": "Qualitative draft; not measured attack success rates."}


def _write(table, out_dir, stem, *, title) -> dict[str, Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    methods = table["methods"]
    cols = ["attack_probe", "attack_family", "implementation_level", "threat_model",
            "attacker_knowledge", *methods, "notes"]
    (out_dir / f"{stem}.json").write_text(json.dumps(table, indent=2, default=str))
    lines = [",".join(cols)]
    for r in table["rows"]:
        lines.append(",".join(str(r.get(c, "")).replace(",", ";") for c in cols))
    (out_dir / f"{stem}.csv").write_text("\n".join(lines) + "\n")
    md = [f"# {title}", "", f"> {table['disclaimer']}", ""]
    if table["mode"] == "qualitative_draft":
        md += ["> **Qualitative draft; not measured attack success rates.**", ""]
    md += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in table["rows"]:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n")
    return {"json": out_dir / f"{stem}.json", "csv": out_dir / f"{stem}.csv",
            "md": out_dir / f"{stem}.md"}


def write_measured_table(results, out_dir, **kw) -> dict[str, Path]:
    return _write(build_measured_table(results, **kw), out_dir, "security_attack_table_measured",
                  title="Security attack table (Table B) — measured")


def write_qualitative_table(config, out_dir, **kw) -> dict[str, Path]:
    return _write(build_qualitative_table(config, **kw), out_dir,
                  "security_attack_table_qualitative_draft",
                  title="Security attack table (Table B) — QUALITATIVE DRAFT")


__all__ = ["build_measured_table", "build_qualitative_table", "write_measured_table",
           "write_qualitative_table", "DEFAULT_METHOD_COLUMNS", "DEFAULT_ATTACK_ROWS"]
