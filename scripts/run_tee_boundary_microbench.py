"""TEE boundary microbenchmark (Stage 8.3).

Times the trusted boundary operations (setup / embed+mask / recover / sample)
for the simulated or process backend across batch sizes and sequence lengths.
numpy + pandas only; no torch, no transformers, no GPU. Compact JSON/CSV/MD.

The untrusted decoder is NOT run here -- masked logits are synthesised to time
the trusted recovery/sampling boundary in isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pllo.tee.runtime_api import (  # noqa: E402
    MaskedLogitsPacket,
    TEEConfig,
    make_runtime,
)

METRIC_COLUMNS = [
    "backend", "batch_size", "seq_len", "hidden_size", "vocab_size", "repeats",
    "setup_latency_ms", "embed_and_mask_latency_ms", "recover_logits_latency_ms",
    "sampling_latency_ms", "total_latency_ms", "trusted_input_bytes",
    "trusted_output_bytes", "released_to_untrusted_bytes",
    "received_from_untrusted_bytes",
]


def _time_ms(fn, repeats: int) -> float:
    # one warmup (excluded), then mean of `repeats` timed calls
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return round((time.perf_counter() - t0) / repeats * 1000.0, 6)


def bench_one(runtime, backend: str, batch: int, seq: int, hidden: int,
              vocab: int, repeats: int, rng: np.random.Generator) -> dict:
    input_ids = rng.integers(0, vocab, size=(batch, seq), dtype=np.int64)
    masked_logits = rng.standard_normal((batch, vocab)).astype(np.float32)

    # warm the embedding table + masks before timing
    runtime.setup_masks(runtime.config.seed)
    emb = runtime.embed_and_mask(input_ids)
    pkt = MaskedLogitsPacket(masked_logits, batch, vocab, "float32",
                             int(masked_logits.nbytes))
    rec = runtime.recover_logits(pkt)
    samp = runtime.sample(rec)

    setup_ms = _time_ms(lambda: runtime.setup_masks(runtime.config.seed),
                        repeats)
    embed_ms = _time_ms(lambda: runtime.embed_and_mask(input_ids), repeats)
    recover_ms = _time_ms(lambda: runtime.recover_logits(pkt), repeats)
    sample_ms = _time_ms(lambda: runtime.sample(rec), repeats)
    total_ms = round(embed_ms + recover_ms + sample_ms, 6)

    return {
        "backend": backend, "batch_size": batch, "seq_len": seq,
        "hidden_size": hidden, "vocab_size": vocab, "repeats": repeats,
        "setup_latency_ms": setup_ms,
        "embed_and_mask_latency_ms": embed_ms,
        "recover_logits_latency_ms": recover_ms,
        "sampling_latency_ms": sample_ms,
        "total_latency_ms": total_ms,
        "trusted_input_bytes": int(input_ids.nbytes),       # client -> TEE
        "trusted_output_bytes": int(samp.nbytes),           # TEE -> client
        "released_to_untrusted_bytes": int(emb.nbytes),     # TEE -> untrusted
        "received_from_untrusted_bytes": int(pkt.nbytes),   # untrusted -> TEE
    }


def _write_md(path: Path, rows: list[dict], meta: dict) -> None:
    lines = ["# TEE boundary microbenchmark", ""]
    lines.append(f"- backend: **{meta['backend']}** | hidden_size: "
                 f"**{meta['hidden_size']}** | vocab_size: "
                 f"**{meta['vocab_size']}** | repeats: **{meta['repeats']}**")
    lines.append("- `total_latency_ms` = embed+recover+sample (per-step trusted "
                 "boundary cost); `setup_latency_ms` is a one-time cost.")
    lines.append("")
    cols = ["batch_size", "seq_len", "setup_latency_ms",
            "embed_and_mask_latency_ms", "recover_logits_latency_ms",
            "sampling_latency_ms", "total_latency_ms", "trusted_input_bytes",
            "trusted_output_bytes", "released_to_untrusted_bytes",
            "received_from_untrusted_bytes"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="simulated",
                    choices=["simulated", "process"])
    ap.add_argument("--batch-sizes", default="1,4,8")
    ap.add_argument("--seq-lens", default="64,128")
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=151936)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=8201)
    ap.add_argument("--output-json", default="outputs/tee_boundary.json")
    ap.add_argument("--output-csv", default="outputs/tee_boundary.csv")
    ap.add_argument("--output-md", default="outputs/tee_boundary.md")
    args = ap.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    cfg = TEEConfig(hidden_size=args.hidden_size, vocab_size=args.vocab_size,
                    seed=args.seed, backend=args.backend)
    runtime = make_runtime(cfg)
    rows: list[dict] = []
    try:
        attest = runtime.attest()
        for seq in seq_lens:
            for batch in batch_sizes:
                rows.append(bench_one(
                    runtime, args.backend, batch, seq, args.hidden_size,
                    args.vocab_size, args.repeats, rng))
                print(f"backend={args.backend} batch={batch} seq={seq} "
                      f"total_ms={rows[-1]['total_latency_ms']}")
    finally:
        runtime.close()

    report = {
        "stage": "8.3_tee_boundary_microbench",
        "backend": args.backend, "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size, "repeats": args.repeats,
        "attestation": {
            "tee_type": attest.tee_type,
            "tdx_guest_device_present": attest.tdx_guest_device_present,
            "tdreport_available": attest.tdreport_available,
            "quote_available": attest.quote_available,
            "quote_status": attest.quote_status,
            "attributes": attest.attributes,
        },
        "metric_columns": METRIC_COLUMNS,
        "rows": rows,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str),
                        encoding="utf-8")

    import pandas as pd
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    _write_md(Path(args.output_md), rows,
              {"backend": args.backend, "hidden_size": args.hidden_size,
               "vocab_size": args.vocab_size, "repeats": args.repeats})

    print(f"Wrote: {args.output_json}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
