"""Stage 5.8 -- Lookup Nonlinear Cost Proxy.

Compares the current compatible SwiGLU nonlinear island against a
finite-domain lookup-style SwiGLU proxy. The goal is paper-grade
*cost* comparison only.

This is a lookup cost proxy, not a secure lookup implementation. No
garbled circuit, MPC, FHE, Tabula, FLUTE, or cryptographic lookup
protocol is implemented. Lookup-style nonlinear protection may
improve value hiding, but this stage evaluates only table-size and
memory-access costs. The current compatible island is faster and
lower-memory but preserves permutation-invariant activation
statistics. No formal, cryptographic, or semantic security is
claimed. No real TEE or GPU wall-time is measured.

CPU-only. Synthetic-by-default. Outputs only summary scalars,
shapes, and short labels; no raw tensors are exported.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


_REQUIRED_HONESTY_PHRASES: tuple[str, ...] = (
    "This is a lookup cost proxy, not a secure lookup "
    "implementation.",
    "No garbled circuit, MPC, FHE, Tabula, FLUTE, or cryptographic "
    "lookup protocol is implemented.",
    "Lookup-style nonlinear protection may improve value hiding, but "
    "this stage evaluates only table-size and memory-access costs.",
    "The current compatible island is faster and lower-memory but "
    "preserves permutation-invariant activation statistics.",
    "No formal, cryptographic, or semantic security is claimed.",
    "No real TEE or GPU wall-time is measured.",
)


_BYTES_PER_FLOAT32 = 4


@dataclass
class LookupNonlinearCostProxyConfig:
    batch_size: int = 1
    seq_len: int = 128
    intermediate_size: int = 11008
    bit_widths: tuple[int, ...] = (4, 6, 8)
    entry_bytes: int = 2
    num_layers: int = 1
    num_tables_policy: str = "per_layer_shared"  # or "per_session_fresh"
    run_microbench: bool = True
    microbench_intermediate_size: int = 1024
    microbench_seq_len: int = 128
    repeats: int = 10
    seed: int = 0
    dtype: str = "float32"
    device: str = "cpu"
    per_channel_proxy: bool = True  # report per-channel as impractical proxy
    method_labels: tuple[str, ...] = field(
        default_factory=lambda: (
            "compatible_swiglu_island_current",
            "compatible_swiglu_full_bundle",
        ),
    )


# ---------------------------------------------------------------------------
# Cost formulas
# ---------------------------------------------------------------------------


def _num_tables_for_policy(policy: str, num_layers: int) -> int:
    if policy not in ("per_layer_shared", "per_session_fresh"):
        raise ValueError(
            f"num_tables_policy must be 'per_layer_shared' or "
            f"'per_session_fresh', got {policy!r}"
        )
    return int(num_layers)


def lookup_table_costs(
    *, bit_width: int, entry_bytes: int, num_layers: int,
    policy: str, batch_size: int, seq_len: int,
    intermediate_size: int, per_channel_proxy: bool,
) -> dict[str, Any]:
    table_entries = 2 ** (2 * bit_width)
    table_bytes = int(table_entries) * int(entry_bytes)
    num_tables = _num_tables_for_policy(policy, num_layers)
    num_lookups = int(batch_size) * int(seq_len) * int(intermediate_size)
    online_lookup_bytes = num_lookups * int(entry_bytes)
    preprocessing_bytes = int(table_bytes) * int(num_tables)
    per_channel_bytes = int(table_bytes) * int(intermediate_size)
    return {
        "bit_width": int(bit_width),
        "table_entries": int(table_entries),
        "table_bytes": int(table_bytes),
        "num_tables": int(num_tables),
        "num_lookups": int(num_lookups),
        "online_lookup_bytes": int(online_lookup_bytes),
        "preprocessing_bytes": int(preprocessing_bytes),
        "per_channel_table_bytes": int(per_channel_bytes),
        "per_channel_status": (
            "impractical_proxy_only" if per_channel_proxy
            else "not_evaluated"
        ),
        "policy": policy,
    }


def compatible_island_costs(
    *, batch_size: int, seq_len: int, intermediate_size: int,
    dtype: str,
) -> dict[str, Any]:
    bpf = (
        _BYTES_PER_FLOAT32 if dtype == "float32"
        else 2 if dtype == "float16" else 8
    )
    n = int(batch_size) * int(seq_len) * int(intermediate_size)
    silu_ops = n
    multiply_ops = n
    # Reads G and U; writes A.
    read_bytes = 2 * n * bpf
    write_bytes = n * bpf
    return {
        "silu_ops": int(silu_ops),
        "multiply_ops": int(multiply_ops),
        "read_bytes_G_plus_U": int(read_bytes),
        "write_bytes_A": int(write_bytes),
        "online_memory_bytes": int(read_bytes + write_bytes),
        "table_entries": 0,
        "table_bytes": 0,
        "num_tables": 0,
        "preprocessing_bytes": 0,
        "known_leakage": "permutation_invariant_statistics_preserved",
    }


# ---------------------------------------------------------------------------
# Microbenchmark
# ---------------------------------------------------------------------------


def _stats(times_ms: list[float]) -> dict[str, float]:
    if not times_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "std_ms": 0.0}
    mean = float(statistics.mean(times_ms))
    median = float(statistics.median(times_ms))
    std = float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0
    return {"mean_ms": mean, "median_ms": median, "std_ms": std}


def _microbench_current(
    *, batch_size: int, seq_len: int, intermediate_size: int,
    repeats: int, seed: int,
) -> dict[str, float]:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    G = torch.randn(
        batch_size, seq_len, intermediate_size, dtype=torch.float32,
        generator=g,
    )
    U = torch.randn(
        batch_size, seq_len, intermediate_size, dtype=torch.float32,
        generator=g,
    )
    # Warmup
    _ = torch.nn.functional.silu(G) * U
    times: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        _ = torch.nn.functional.silu(G) * U
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return _stats(times)


def _microbench_lookup(
    *, bit_width: int, batch_size: int, seq_len: int,
    intermediate_size: int, repeats: int, seed: int,
) -> dict[str, float]:
    g = torch.Generator(device="cpu").manual_seed(int(seed) + int(bit_width))
    table_size = 2 ** (2 * int(bit_width))
    table = torch.randint(
        low=-32768, high=32767, size=(table_size,), dtype=torch.int32,
        generator=g,
    )
    levels = 2 ** int(bit_width)
    G_q = torch.randint(
        low=0, high=levels,
        size=(batch_size, seq_len, intermediate_size), dtype=torch.int64,
        generator=g,
    )
    U_q = torch.randint(
        low=0, high=levels,
        size=(batch_size, seq_len, intermediate_size), dtype=torch.int64,
        generator=g,
    )
    # Warmup
    idx = G_q * levels + U_q
    _ = table[idx]
    times: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        idx = G_q * levels + U_q
        _ = table[idx]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return _stats(times)


# ---------------------------------------------------------------------------
# Top-level experiment
# ---------------------------------------------------------------------------


def _security_profile_for(method: str) -> dict[str, str]:
    if method == "compatible_swiglu_island_current":
        return {
            "security_profile": (
                "lightweight_correctness_preserving_proxy_evaluated"
                "_not_formal"
            ),
            "implemented_security": "proxy_evaluated_not_formal",
            "security_potential": "as_currently_implemented_only",
        }
    if method == "compatible_swiglu_full_bundle":
        return {
            "security_profile": (
                "lightweight_correctness_preserving_proxy_evaluated"
                "_not_formal"
            ),
            "implemented_security": (
                "fresh_perm_plus_sandwich_plus_pad_proxy_evaluated"
                "_not_formal"
            ),
            "security_potential": "as_currently_implemented_only",
        }
    if method.startswith("lookup_swiglu_proxy_"):
        return {
            "security_profile": (
                "cost_proxy_only_not_a_security_implementation"
            ),
            "implemented_security": "none_cost_proxy_only",
            "security_potential": (
                "stronger_value_hiding_if_combined_with_secure_lookup"
                "_protocol"
            ),
        }
    return {
        "security_profile": "unspecified",
        "implemented_security": "unspecified",
        "security_potential": "unspecified",
    }


def run_lookup_nonlinear_cost_proxy(
    cfg: LookupNonlinearCostProxyConfig,
) -> dict[str, Any]:
    if cfg.entry_bytes <= 0:
        raise ValueError("entry_bytes must be > 0")
    if any(b <= 0 or b > 16 for b in cfg.bit_widths):
        raise ValueError("bit_widths must be in (0, 16]")

    methods: list[dict[str, Any]] = []

    current_costs = compatible_island_costs(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len,
        intermediate_size=cfg.intermediate_size, dtype=cfg.dtype,
    )
    methods.append({
        "method": "compatible_swiglu_island_current",
        "kind": "current_island",
        "cost": current_costs,
        **_security_profile_for("compatible_swiglu_island_current"),
        "notes": (
            "Stage 5.3e fresh_perm_only baseline. No lookup table; the "
            "GPU sees fresh masks per call but values themselves can "
            "leak through permutation-invariant statistics."
        ),
    })

    full_bundle_costs = dict(current_costs)
    methods.append({
        "method": "compatible_swiglu_full_bundle",
        "kind": "current_island_full_bundle",
        "cost": full_bundle_costs,
        **_security_profile_for("compatible_swiglu_full_bundle"),
        "notes": (
            "Stage 5.3e fresh_perm_plus_sandwich_plus_pad bundle. Same "
            "raw arithmetic shape as the current island; cost profile "
            "is dominated by the same SiLU + multiply path."
        ),
    })

    for b in cfg.bit_widths:
        lookup_costs = lookup_table_costs(
            bit_width=int(b),
            entry_bytes=int(cfg.entry_bytes),
            num_layers=int(cfg.num_layers),
            policy=cfg.num_tables_policy,
            batch_size=int(cfg.batch_size),
            seq_len=int(cfg.seq_len),
            intermediate_size=int(cfg.intermediate_size),
            per_channel_proxy=cfg.per_channel_proxy,
        )
        methods.append({
            "method": f"lookup_swiglu_proxy_{int(b)}bit",
            "kind": "lookup_proxy",
            "cost": lookup_costs,
            **_security_profile_for(f"lookup_swiglu_proxy_{int(b)}bit"),
            "notes": (
                f"Two-input ({int(b)}-bit, {int(b)}-bit) finite-domain "
                f"lookup proxy of SwiGLU. The table has "
                f"2^(2*{int(b)}) = {2 ** (2 * int(b))} entries. "
                "No secure lookup protocol is implemented."
            ),
        })

    microbench: dict[str, Any] = {
        "enabled": bool(cfg.run_microbench),
        "intermediate_size": int(cfg.microbench_intermediate_size),
        "seq_len": int(cfg.microbench_seq_len),
        "batch_size": int(cfg.batch_size),
        "repeats": int(cfg.repeats),
    }
    if cfg.run_microbench:
        current_bench = _microbench_current(
            batch_size=cfg.batch_size,
            seq_len=cfg.microbench_seq_len,
            intermediate_size=cfg.microbench_intermediate_size,
            repeats=cfg.repeats, seed=cfg.seed,
        )
        microbench["compatible_swiglu_island_current"] = current_bench
        for b in cfg.bit_widths:
            microbench[f"lookup_swiglu_proxy_{int(b)}bit"] = (
                _microbench_lookup(
                    bit_width=int(b),
                    batch_size=cfg.batch_size,
                    seq_len=cfg.microbench_seq_len,
                    intermediate_size=cfg.microbench_intermediate_size,
                    repeats=cfg.repeats, seed=cfg.seed,
                )
            )
    else:
        microbench["note"] = (
            "Microbenchmark disabled by config (run_microbench=False)."
        )

    table_size_scaling = [
        {
            "bit_width": int(b),
            "table_entries": 2 ** (2 * int(b)),
            "table_bytes": (2 ** (2 * int(b))) * int(cfg.entry_bytes),
        }
        for b in cfg.bit_widths
    ]

    return {
        "status": "ok",
        "stage": "5.8",
        "experiment": "lookup_nonlinear_cost_proxy",
        "config": asdict(cfg),
        "methods": methods,
        "table_size_scaling": table_size_scaling,
        "microbench": microbench,
        "formal_security_claim": False,
        "cryptographic_lookup_implemented": False,
        "recommended_use": (
            "cost-baseline-and-future-work-motivation"
        ),
        "honesty_phrases": list(_REQUIRED_HONESTY_PHRASES),
        "limitations": [
            "This is a CPU-only cost proxy; no real TEE or GPU "
            "wall-time is measured.",
            "No secure lookup, garbled circuit, MPC, FHE, Tabula, or "
            "FLUTE protocol is implemented; the microbenchmark uses "
            "ordinary CPU memory lookup.",
            "The current compatible island preserves permutation-"
            "invariant activation statistics; lookup-style protection "
            "could close this leakage channel but only if combined "
            "with a real cryptographic lookup protocol.",
            "Per-channel tables are reported as impractical_proxy_only "
            "to make the bandwidth cost explicit; they are not "
            "evaluated in the microbenchmark.",
            "Bit widths above 8 are not microbenchmarked because the "
            "table size grows as 2^(2b); a 12-bit table already has "
            "16M entries and is reported for cost only.",
            "No formal, cryptographic, or semantic security is "
            "claimed.",
        ],
        "next_stage_plan": (
            "Stage 5.8 produces a cost baseline. A future stage could "
            "(i) integrate a real secure lookup primitive (e.g. "
            "Tabula-style 2-PC) and validate correctness on a small "
            "model; (ii) explore mixed designs that keep the "
            "compatible island for hot paths and use lookup only at "
            "designated boundary tensors; (iii) extend the cost model "
            "to multi-layer / multi-head workloads. None of these are "
            "implemented in Stage 5.8."
        ),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _round(x: Any, digits: int = 6) -> Any:
    if isinstance(x, float):
        if x != x:
            return "NaN"
        return round(x, digits)
    return x


def _write_json(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)


def _flatten_methods(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in report["methods"]:
        cost = m["cost"]
        rows.append({
            "method": m["method"],
            "kind": m["kind"],
            "bit_width": cost.get("bit_width", ""),
            "table_entries": cost.get("table_entries", 0),
            "table_bytes": cost.get("table_bytes", 0),
            "num_tables": cost.get("num_tables", 0),
            "preprocessing_bytes": cost.get("preprocessing_bytes", 0),
            "online_lookup_bytes": cost.get("online_lookup_bytes", 0),
            "online_memory_bytes": cost.get("online_memory_bytes", 0),
            "silu_ops": cost.get("silu_ops", 0),
            "multiply_ops": cost.get("multiply_ops", 0),
            "per_channel_table_bytes": cost.get(
                "per_channel_table_bytes", 0,
            ),
            "per_channel_status": cost.get("per_channel_status", ""),
            "security_profile": m.get("security_profile", ""),
            "implemented_security": m.get("implemented_security", ""),
            "security_potential": m.get("security_potential", ""),
        })
    return rows


def _write_csv(report: dict[str, Any], path: str) -> None:
    rows = _flatten_methods(report)
    if not rows:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _format_bytes(b: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(b)
    i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    def w(line: str = "") -> None:
        lines.append(line)

    cfg = report["config"]
    w("# Stage 5.8 -- Lookup Nonlinear Cost Proxy")
    w()
    w("## 1. Experiment Scope")
    w()
    w(
        "We compare the current compatible SwiGLU nonlinear island "
        "against a finite-domain lookup-style SwiGLU proxy. The goal "
        "is paper-grade *cost* comparison only. No secure lookup, "
        "garbled circuit, MPC, FHE, Tabula, FLUTE, or cryptographic "
        "protocol is implemented. This is a lookup cost proxy, not a "
        "secure lookup implementation."
    )
    w()
    w(
        f"Workload: `batch_size={cfg['batch_size']}`, "
        f"`seq_len={cfg['seq_len']}`, "
        f"`intermediate_size={cfg['intermediate_size']}`, "
        f"`num_layers={cfg['num_layers']}`, "
        f"`num_tables_policy={cfg['num_tables_policy']}`."
    )
    w()
    w("## 2. Threat Model and Non-Claim")
    w()
    w(
        "Honest-but-curious cloud accelerator. The current compatible "
        "island is faster and lower-memory but preserves permutation-"
        "invariant activation statistics. Lookup-style nonlinear "
        "protection may improve value hiding, but this stage "
        "evaluates only table-size and memory-access costs. No formal, "
        "cryptographic, or semantic security is claimed. No real TEE "
        "or GPU wall-time is measured."
    )
    w()
    w("## 3. Current Compatible SwiGLU Island Cost")
    w()
    current = next(
        m for m in report["methods"]
        if m["method"] == "compatible_swiglu_island_current"
    )
    c = current["cost"]
    w(
        f"- SiLU ops: `{c['silu_ops']:,}`\n"
        f"- Multiply ops: `{c['multiply_ops']:,}`\n"
        f"- Online read bytes (G + U): "
        f"`{c['read_bytes_G_plus_U']:,}` ({_format_bytes(c['read_bytes_G_plus_U'])})\n"
        f"- Online write bytes (A): "
        f"`{c['write_bytes_A']:,}` ({_format_bytes(c['write_bytes_A'])})\n"
        f"- Online memory bytes: "
        f"`{c['online_memory_bytes']:,}` ({_format_bytes(c['online_memory_bytes'])})\n"
        f"- Table preprocessing bytes: `{c['preprocessing_bytes']:,}`\n"
        f"- Known leakage: `{c['known_leakage']}`"
    )
    w()
    w("## 4. Lookup-style SwiGLU Proxy Cost")
    w()
    w(
        "For a `b`-bit quantized binary SwiGLU lookup:\n"
        "```\n"
        "  table_entries        = 2^(2 * b)\n"
        "  table_bytes          = table_entries * entry_bytes\n"
        "  num_lookups          = batch_size * seq_len * intermediate_size\n"
        "  online_lookup_bytes  = num_lookups * entry_bytes\n"
        "  preprocessing_bytes  = table_bytes * num_tables\n"
        "  per_channel_table_bytes = table_bytes * intermediate_size"
        "  (impractical_proxy_only)\n"
        "```"
    )
    w()
    w("| method | bit_width | table_entries | table_bytes | preprocessing_bytes | online_lookup_bytes | per_channel_table_bytes |")
    w("|---|---|---|---|---|---|---|")
    for m in report["methods"]:
        if m["kind"] != "lookup_proxy":
            continue
        cc = m["cost"]
        w(
            f"| `{m['method']}` | {cc['bit_width']} | "
            f"{cc['table_entries']:,} | "
            f"{cc['table_bytes']:,} ({_format_bytes(cc['table_bytes'])}) | "
            f"{cc['preprocessing_bytes']:,} ({_format_bytes(cc['preprocessing_bytes'])}) | "
            f"{cc['online_lookup_bytes']:,} ({_format_bytes(cc['online_lookup_bytes'])}) | "
            f"{cc['per_channel_table_bytes']:,} ({_format_bytes(cc['per_channel_table_bytes'])}) |"
        )
    w()
    w("## 5. Table Size Scaling")
    w()
    w("| bit_width | table_entries | table_bytes |")
    w("|---|---|---|")
    for r in report["table_size_scaling"]:
        w(
            f"| {r['bit_width']} | {r['table_entries']:,} | "
            f"{r['table_bytes']:,} ({_format_bytes(r['table_bytes'])}) |"
        )
    w()
    w("## 6. Online Lookup Bandwidth")
    w()
    w(
        "Online lookup bandwidth grows linearly with "
        "`num_lookups = batch_size * seq_len * intermediate_size`. "
        "Table preprocessing bandwidth grows as "
        "`2^(2b) * entry_bytes * num_tables`. Per-channel tables are "
        "reported only as `impractical_proxy_only`."
    )
    w()
    w("| method | online_lookup_bytes | preprocessing_bytes |")
    w("|---|---|---|")
    for m in report["methods"]:
        if m["kind"] != "lookup_proxy":
            continue
        cc = m["cost"]
        w(
            f"| `{m['method']}` | "
            f"{cc['online_lookup_bytes']:,} | "
            f"{cc['preprocessing_bytes']:,} |"
        )
    w()
    w("## 7. CPU Microbenchmark")
    w()
    mb = report["microbench"]
    if not mb.get("enabled", False):
        w(f"Microbenchmark disabled: `{mb.get('note', '')}`")
    else:
        w(
            f"Microbench workload: "
            f"`batch_size={mb['batch_size']}`, "
            f"`seq_len={mb['seq_len']}`, "
            f"`intermediate_size={mb['intermediate_size']}`, "
            f"`repeats={mb['repeats']}`. "
            "This uses ordinary CPU memory lookup only -- not a secure "
            "lookup primitive."
        )
        w()
        w("| method | mean_ms | median_ms | std_ms |")
        w("|---|---|---|---|")
        for key, val in mb.items():
            if not isinstance(val, dict):
                continue
            if "mean_ms" not in val:
                continue
            w(
                f"| `{key}` | {val['mean_ms']:.4f} | "
                f"{val['median_ms']:.4f} | {val['std_ms']:.4f} |"
            )
    w()
    w("## 8. Security / Cost Interpretation")
    w()
    w(
        "- The current compatible island has zero table preprocessing "
        "bytes but preserves permutation-invariant activation "
        "statistics; its security profile is "
        "`lightweight_correctness_preserving_proxy_evaluated_not_formal`.\n"
        "- The lookup proxy has table-size cost growing as `2^(2b)`. "
        "Its `security_potential` is "
        "`stronger_value_hiding_if_combined_with_secure_lookup_protocol`. "
        "Its `implemented_security` is `none_cost_proxy_only`.\n"
        f"- `formal_security_claim` = `{report['formal_security_claim']}`\n"
        f"- `cryptographic_lookup_implemented` = "
        f"`{report['cryptographic_lookup_implemented']}`\n"
        f"- `recommended_use` = `{report['recommended_use']}`"
    )
    w()
    w("## 9. Limitations")
    w()
    for x in report["limitations"]:
        w(f"- {x}")
    w()
    w("## 10. Next Stage Plan")
    w()
    w(report["next_stage_plan"])
    w()
    w("## Honesty phrases (verbatim)")
    w()
    for phrase in report["honesty_phrases"]:
        w(f"- {phrase}")
    w()
    w(f"`formal_security_claim`: `{report['formal_security_claim']}`")
    w()
    w(
        f"`cryptographic_lookup_implemented`: "
        f"`{report['cryptographic_lookup_implemented']}`"
    )
    w()
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any], *, outputs_dir: str = "outputs",
    json_filename: str = "lookup_nonlinear_cost_proxy.json",
    csv_filename: str = "lookup_nonlinear_cost_proxy.csv",
    md_filename: str = "lookup_nonlinear_cost_proxy.md",
) -> tuple[str, str, str]:
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, json_filename)
    csv_path = os.path.join(outputs_dir, csv_filename)
    md_path = os.path.join(outputs_dir, md_filename)
    _write_json(report, json_path)
    _write_csv(report, csv_path)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return json_path, csv_path, md_path


__all__ = [
    "LookupNonlinearCostProxyConfig",
    "compatible_island_costs",
    "lookup_table_costs",
    "render_markdown",
    "run_lookup_nonlinear_cost_proxy",
    "write_reports",
]
