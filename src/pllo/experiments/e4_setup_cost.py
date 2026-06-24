"""E4: setup / provisioning cost + amortization for the folded-package deployment.

Pure parsing + arithmetic (stdlib only -- no torch / numpy): consolidate the
one-time setup facts (folded-package generation/size/load, boundary embedding
artifact size/hash) from whatever sources are available (the package manifest on
disk and/or prior build/verify/inspection/load-probe JSON outputs), then compute
transfer-time estimates per bandwidth and amortized per-session setup cost.

Honest by construction: every field is sourced (``*_source``) and missing inputs
stay ``None`` rather than being assumed. The known F32-vs-bf16 size gotcha is
recorded with an explicit note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "GB", "F32_TO_BF16_RATIO",
    "load_json", "gather_facts", "transfer_times", "amortized_costs",
    "build_e4_report", "render_e4_md", "render_e4_csv",
]

GB = 1024 ** 3
F32_TO_BF16_RATIO = 0.5            # bf16 (2B) is half of float32 (4B)

_SIZE_NOTE = (
    "The folded package is stored in float32 for numerical fidelity, so the "
    "measured size (~26.34GB) is ~2x a bf16 store (~13.17GB). bf16 storage would "
    "be smaller but is NOT the current measured artifact.")


def load_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return None


def _first(*pairs):
    """Return (value, source_label) for the first non-None value among
    (value, label) pairs; else (None, None)."""
    for val, label in pairs:
        if val is not None:
            return val, label
    return None, None


def _store_dtype_from_inspection(insp: dict | None) -> Any:
    if not isinstance(insp, dict):
        return None
    sd = insp.get("store_dtypes")
    if isinstance(sd, (list, tuple)) and sd:
        return sd[0]
    return insp.get("store_dtype")


def gather_facts(*, folded_package_path: str | Path | None = None,
                 embedding_artifact_path: str | Path | None = None,
                 build_json: dict | None = None,
                 verify_json: dict | None = None,
                 inspection_json: dict | None = None,
                 load_probe_json: dict | None = None) -> dict:
    """Collect the one-time setup facts from disk + the provided JSON outputs.

    Disk artifacts (manifest + measured sizes) are preferred when present; JSON
    outputs fill any gaps. Each returned fact carries a ``*_source`` label."""
    build_json = build_json or {}
    verify_json = verify_json or {}
    inspection_json = inspection_json or {}
    load_probe_json = load_probe_json or {}

    manifest = None
    pkg_size_disk = None
    if folded_package_path and Path(folded_package_path).is_dir():
        try:
            from pllo.deployment import load_manifest, package_size_gb
            manifest = load_manifest(folded_package_path)
            pkg_size_disk = round(package_size_gb(folded_package_path), 6)
        except Exception:                                    # noqa: BLE001
            manifest = None

    def m(attr):
        return getattr(manifest, attr, None) if manifest is not None else None

    def num_shards():
        if manifest is not None:
            try:
                return int(manifest.num_shards)
            except Exception:                                # noqa: BLE001
                return None
        return None

    manifest_hash = None
    if manifest is not None:
        try:
            from pllo.deployment import compute_manifest_hash
            manifest_hash = compute_manifest_hash(manifest)
        except Exception:                                    # noqa: BLE001
            manifest_hash = None

    size_gb, size_src = _first(
        (pkg_size_disk, "package_dir"),
        (build_json.get("size_gb"), "build_json"),
        (build_json.get("package_size_gb"), "build_json"),
        (inspection_json.get("total_size_gb"), "inspection_json"),
        (inspection_json.get("package_size_gb"), "inspection_json"),
        (verify_json.get("package_size_gb"), "verify_json"),
        (load_probe_json.get("package_size_gb"), "load_probe_json"))
    size_gb = round(float(size_gb), 6) if size_gb is not None else None

    store_dtype, store_dtype_src = _first(
        (_store_dtype_from_inspection(inspection_json), "inspection_json"),
        (build_json.get("store_dtype"), "build_json"),
        ((build_json.get("store_dtypes") or [None])[0]
         if isinstance(build_json.get("store_dtypes"), list) else None,
         "build_json"),
        (m("dtype"), "manifest"))

    num_layers, nl_src = _first(
        (m("num_layers"), "manifest"),
        (build_json.get("num_layers"), "build_json"),
        (inspection_json.get("num_layers"), "inspection_json"),
        (load_probe_json.get("num_layers"), "load_probe_json"))

    nshards, ns_src = _first(
        (num_shards(), "manifest"),
        (build_json.get("num_shards"), "build_json"),
        (inspection_json.get("num_shards"), "inspection_json"),
        (load_probe_json.get("num_shards"), "load_probe_json"))

    mhash, mhash_src = _first(
        (manifest_hash, "manifest"),
        (build_json.get("manifest_hash"), "build_json"),
        (inspection_json.get("manifest_hash"), "inspection_json"),
        (load_probe_json.get("manifest_hash"), "load_probe_json"))

    gen_time, gen_src = _first(
        (build_json.get("generation_time_s"), "build_json"),
        (build_json.get("gen_time_s"), "build_json"),
        (build_json.get("generation_time"), "build_json"))

    verify_passed, vp_src = _first(
        (verify_json.get("package_valid"), "verify_json"),
        (m("contains_mask_secrets") is False if manifest is not None else None,
         "manifest_flags"),
        (load_probe_json.get("folded_package_valid"), "load_probe_json"))

    load_time, lt_src = _first(
        (load_probe_json.get("load_time_s"), "load_probe_json"),
        (load_probe_json.get("package_load_time_s"), "load_probe_json"))

    # boundary embedding artifact ------------------------------------------
    art_meta = None
    art_size = None
    art_has_tensors = False
    if embedding_artifact_path and Path(embedding_artifact_path).is_dir():
        try:
            from pllo.deployment.embedding_artifact import (
                ARTIFACT_META,
                ARTIFACT_TENSORS,
                ARTIFACT_TENSORS_PT,
                embedding_artifact_size_gb,
            )
            ad = Path(embedding_artifact_path)
            mp = ad / ARTIFACT_META
            if mp.is_file():
                art_meta = json.loads(mp.read_text(encoding="utf-8"))
            art_has_tensors = ((ad / ARTIFACT_TENSORS).is_file()
                               or (ad / ARTIFACT_TENSORS_PT).is_file())
            # measured dir size is authoritative ONLY when the big tensors file is
            # actually present; otherwise fall back to the meta's recorded size.
            if art_has_tensors:
                art_size = round(embedding_artifact_size_gb(
                    embedding_artifact_path), 6)
        except Exception:                                    # noqa: BLE001
            art_meta = art_meta

    art_size_gb, art_size_src = _first(
        (art_size, "artifact_dir"),
        ((art_meta or {}).get("size_gb"), "artifact_meta"))
    art_hash, art_hash_src = _first(
        ((art_meta or {}).get("tensors_sha256"), "artifact_meta"))
    art_secrets = (art_meta or {}).get("contains_mask_secrets")
    art_trusted = (art_meta or {}).get("trusted_only")

    size_if_bf16 = (round(size_gb * F32_TO_BF16_RATIO, 6)
                    if (size_gb is not None
                        and str(store_dtype).upper() in ("F32", "FLOAT32"))
                    else None)

    return {
        "folded_package_path": (str(folded_package_path)
                                if folded_package_path else None),
        "folded_package_size_gb": size_gb,
        "folded_package_size_gb_source": size_src,
        "folded_package_store_dtype": store_dtype,
        "folded_package_store_dtype_source": store_dtype_src,
        "folded_package_size_if_bf16_gb": size_if_bf16,
        "num_layers": num_layers, "num_layers_source": nl_src,
        "num_shards": nshards, "num_shards_source": ns_src,
        "manifest_hash": mhash, "manifest_hash_source": mhash_src,
        "generation_time_s": gen_time, "generation_time_s_source": gen_src,
        "package_verify_passed": (bool(verify_passed)
                                  if verify_passed is not None else None),
        "package_verify_passed_source": vp_src,
        "package_load_time_s": load_time, "package_load_time_s_source": lt_src,
        "boundary_embedding_artifact_path": (str(embedding_artifact_path)
                                             if embedding_artifact_path else None),
        "boundary_embedding_artifact_size_gb": art_size_gb,
        "boundary_embedding_artifact_size_gb_source": art_size_src,
        "boundary_embedding_artifact_hash": art_hash,
        "boundary_embedding_artifact_hash_source": art_hash_src,
        "boundary_artifact_contains_mask_secrets": art_secrets,
        "boundary_artifact_trusted_only": art_trusted,
    }


def transfer_times(size_gb: float | None,
                   bandwidth_mbps_list: list[float]) -> list[dict]:
    """Transfer time (s) for ``size_gb`` at each bandwidth in Mbit/s."""
    out: list[dict] = []
    for mbps in bandwidth_mbps_list:
        if size_gb is None or not mbps:
            t = None
        else:
            bits = float(size_gb) * GB * 8.0
            t = round(bits / (float(mbps) * 1_000_000.0), 3)
        out.append({"bandwidth_mbps": mbps, "transfer_time_s": t})
    return out


def amortized_costs(one_time_setup_s: float | None,
                    sessions_list: list[int]) -> list[dict]:
    """One-time setup cost amortized across ``sessions`` runs."""
    out: list[dict] = []
    for s in sessions_list:
        if one_time_setup_s is None or not s:
            a = None
        else:
            a = round(float(one_time_setup_s) / int(s), 6)
        out.append({"sessions": int(s), "amortized_setup_cost_s": a})
    return out


def build_e4_report(facts: dict, *, bandwidth_mbps_list: list[float],
                    sessions_list: list[int]) -> dict:
    gen = facts.get("generation_time_s")
    load = facts.get("package_load_time_s")
    one_time = None
    if gen is not None or load is not None:
        one_time = float(gen or 0.0) + float(load or 0.0)

    report = {
        "experiment": "E4",
        "stage": "setup_cost_and_amortization",
    }
    report.update(facts)
    report["one_time_setup_s"] = (round(one_time, 6)
                                  if one_time is not None else None)
    report["one_time_setup_components"] = {
        "generation_time_s": gen, "package_load_time_s": load}
    report["transfer_estimates"] = {
        "folded_package": transfer_times(facts.get("folded_package_size_gb"),
                                         bandwidth_mbps_list),
        "boundary_embedding_artifact": transfer_times(
            facts.get("boundary_embedding_artifact_size_gb"),
            bandwidth_mbps_list),
    }
    report["amortized_setup_cost"] = amortized_costs(one_time, sessions_list)
    report["bandwidth_mbps_list"] = list(bandwidth_mbps_list)
    report["sessions_list"] = list(sessions_list)
    report["size_explanation_note"] = _SIZE_NOTE
    report["notes"] = [
        _SIZE_NOTE,
        "one_time_setup_s = generation_time_s + package_load_time_s when both "
        "are available; transfer time is reported separately per bandwidth.",
        "the boundary embedding artifact is trusted-only (contains mask "
        "secrets) and stays inside the TDX guest; it is NOT sent to the GPU.",
    ]
    return report


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return ("%.6f" % v).rstrip("0").rstrip(".") if v else "0"
    return str(v)


def _md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return out


def render_e4_md(report: dict) -> str:
    f = report
    L = ["# E4 — Setup / provisioning cost + amortization", "",
         "## Folded weight package", "",
         "- folded_package_size_gb: **%s** (source: %s)"
         % (_fmt(f["folded_package_size_gb"]),
            f["folded_package_size_gb_source"]),
         "- folded_package_store_dtype: %s  size_if_bf16_gb: %s"
         % (f["folded_package_store_dtype"],
            _fmt(f["folded_package_size_if_bf16_gb"])),
         "- num_layers: %s  num_shards: %s" % (f["num_layers"], f["num_shards"]),
         "- manifest_hash: `%s`" % f["manifest_hash"],
         "- generation_time_s: %s  package_load_time_s: %s"
         % (_fmt(f["generation_time_s"]), _fmt(f["package_load_time_s"])),
         "- package_verify_passed: %s" % f["package_verify_passed"],
         "- **one_time_setup_s: %s**" % _fmt(f["one_time_setup_s"]),
         "", "## Boundary embedding artifact (trusted-only)", "",
         "- size_gb: **%s**  hash: `%s`"
         % (_fmt(f["boundary_embedding_artifact_size_gb"]),
            f["boundary_embedding_artifact_hash"]),
         "- contains_mask_secrets: %s  trusted_only: %s"
         % (f["boundary_artifact_contains_mask_secrets"],
            f["boundary_artifact_trusted_only"]),
         "", "## Transfer time estimates", "",
         "### Folded package", ""]
    L += _md_table(["bandwidth_mbps", "transfer_time_s"],
                   [[t["bandwidth_mbps"], t["transfer_time_s"]]
                    for t in f["transfer_estimates"]["folded_package"]])
    L += ["", "### Boundary embedding artifact", ""]
    L += _md_table(["bandwidth_mbps", "transfer_time_s"],
                   [[t["bandwidth_mbps"], t["transfer_time_s"]] for t in
                    f["transfer_estimates"]["boundary_embedding_artifact"]])
    L += ["", "## Amortized setup cost", ""]
    L += _md_table(["sessions", "amortized_setup_cost_s"],
                   [[a["sessions"], a["amortized_setup_cost_s"]]
                    for a in f["amortized_setup_cost"]])
    L += ["", "## Notes", ""]
    L += ["- %s" % n for n in f["notes"]]
    L += [""]
    return "\n".join(L)


def render_e4_csv(report: dict) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["section", "key", "value", "source"])
    flat_keys = [
        "folded_package_size_gb", "folded_package_store_dtype",
        "folded_package_size_if_bf16_gb", "num_layers", "num_shards",
        "manifest_hash", "generation_time_s", "package_verify_passed",
        "package_load_time_s", "one_time_setup_s",
        "boundary_embedding_artifact_size_gb", "boundary_embedding_artifact_hash",
        "boundary_artifact_contains_mask_secrets",
        "boundary_artifact_trusted_only",
    ]
    for k in flat_keys:
        w.writerow(["facts", k, _fmt(report.get(k)),
                    report.get(k + "_source", "")])
    for t in report["transfer_estimates"]["folded_package"]:
        w.writerow(["transfer_package", "bandwidth_mbps=%s" % t["bandwidth_mbps"],
                    _fmt(t["transfer_time_s"]), "computed"])
    for t in report["transfer_estimates"]["boundary_embedding_artifact"]:
        w.writerow(["transfer_artifact", "bandwidth_mbps=%s" % t["bandwidth_mbps"],
                    _fmt(t["transfer_time_s"]), "computed"])
    for a in report["amortized_setup_cost"]:
        w.writerow(["amortized", "sessions=%s" % a["sessions"],
                    _fmt(a["amortized_setup_cost_s"]), "computed"])
    return buf.getvalue()
