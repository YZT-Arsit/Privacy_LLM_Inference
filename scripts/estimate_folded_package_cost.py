"""Estimate the setup/provisioning cost of a folded weight package.

Folding is a ONE-TIME setup cost; online decoding reuses provisioned folded
weights and never refolds or resends the base model per token. This script
estimates (or measures, given a built ``--package-dir``) the folded package size,
the file-level transfer cost at several bandwidths, and the per-session amortized
setup cost. It does NOT physically transfer the package.

The folded package holds the decoder-layer folded operators (q/k/v/o, gate/up,
down) + the folded LM head. Embeddings stay trusted-side, so they are NOT in the
package (raw model size is reported separately for context).

Examples::

    # from an HF config (no weights loaded):
    python scripts/estimate_folded_package_cost.py --model-path <MODEL_PATH> \\
        --num-layers 28 --dtype bfloat16
    # explicit dims (offline):
    python scripts/estimate_folded_package_cost.py --hidden-size 3584 \\
        --intermediate-size 18944 --vocab-size 152064 --num-kv-heads 4 \\
        --head-dim 128 --num-layers 28 --dtype bfloat16 --model-name Qwen2.5-7B
    # measure an already-built package:
    python scripts/estimate_folded_package_cost.py --package-dir packages/qwen7b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_DT_BYTES = {"float16": 2, "bf16": 2, "bfloat16": 2, "fp16": 2,
             "float32": 4, "fp32": 4, "float64": 8}
_GB = 1024 ** 3
# decimal byte rates for the requested bandwidths
_BANDWIDTHS = {"100MB/s": 100e6, "500MB/s": 500e6, "1GB/s": 1e9, "5GB/s": 5e9}
_SESSIONS = (1, 10, 100, 1000)


def _dtype_bytes(name: str) -> int:
    return _DT_BYTES.get((name or "").lower(), 4)


def _maybe_config(model_path):
    if not model_path:
        return None
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True,
                                          local_files_only=True)
    except Exception as exc:                              # noqa: BLE001
        print(f"NOTE: could not read HF config ({exc}); using explicit dims.")
        return None


def _dim(cfg, args, attr, cli, default=None):
    if cfg is not None and getattr(cfg, attr, None) is not None:
        return int(getattr(cfg, attr))
    v = getattr(args, cli)
    return int(v) if v is not None else default


def _folded_param_count(h, inter, vocab, n_kv_dim, num_layers):
    """Folded-operator parameter count (decoder layers + LM head)."""
    per_layer = (h * h          # q
                 + h * n_kv_dim  # k
                 + h * n_kv_dim  # v
                 + h * h         # o
                 + h * inter     # gate
                 + h * inter     # up
                 + inter * h)    # down
    return per_layer * num_layers + h * vocab     # + folded LM head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--hidden-size", default=None)
    ap.add_argument("--intermediate-size", default=None)
    ap.add_argument("--vocab-size", default=None)
    ap.add_argument("--num-kv-heads", default=None)
    ap.add_argument("--head-dim", default=None)
    ap.add_argument("--num-attention-heads", default=None)
    ap.add_argument("--num-layers", type=int, default=28)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--package-dir", default=None,
                    help="if given, measure folded size from a built package")
    ap.add_argument("--fold-time-s", type=float, default=None)
    ap.add_argument("--disk-write-time-s", type=float, default=None)
    ap.add_argument("--load-time-s", type=float, default=None)
    ap.add_argument("--amortize-bandwidth", default="1GB/s",
                    choices=list(_BANDWIDTHS))
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    cfg = _maybe_config(args.model_path)
    b = _dtype_bytes(args.dtype)
    h = _dim(cfg, args, "hidden_size", "hidden_size", 3584)
    inter = _dim(cfg, args, "intermediate_size", "intermediate_size", 18944)
    vocab = _dim(cfg, args, "vocab_size", "vocab_size", 152064)
    n_heads = _dim(cfg, args, "num_attention_heads", "num_attention_heads", 28)
    n_kv = _dim(cfg, args, "num_key_value_heads", "num_kv_heads", 4)
    head_dim = _dim(cfg, args, "head_dim", "head_dim", h // max(1, n_heads))
    n_kv_dim = n_kv * head_dim
    num_layers = args.num_layers

    folded_params = _folded_param_count(h, inter, vocab, n_kv_dim, num_layers)
    folded_bytes_est = folded_params * b
    # raw model also has embeddings (vocab*h) + small norms (context only)
    raw_bytes_est = folded_bytes_est + vocab * h * b

    measured_gb = None
    if args.package_dir:
        from pllo.deployment import package_size_gb
        measured_gb = round(package_size_gb(args.package_dir), 4)

    folded_gb = measured_gb if measured_gb is not None \
        else folded_bytes_est / _GB
    transfer_bytes = folded_gb * _GB
    transfer_time = {bw: round(transfer_bytes / rate, 2)
                     for bw, rate in _BANDWIDTHS.items()}

    # one-time setup cost = fold + disk write + transfer (at chosen bandwidth).
    setup_transfer_s = transfer_bytes / _BANDWIDTHS[args.amortize_bandwidth]
    one_time_s = (setup_transfer_s + (args.fold_time_s or 0.0)
                  + (args.disk_write_time_s or 0.0))
    amortized = {n: round(one_time_s / n, 4) for n in _SESSIONS}

    rep = {
        "stage": "folded_package_cost_estimate",
        "model_name": args.model_name,
        "model_path": args.model_path,
        "dtype": args.dtype,
        "num_layers": num_layers,
        "dims": {"hidden_size": h, "intermediate_size": inter,
                 "vocab_size": vocab, "num_kv_dim": n_kv_dim,
                 "head_dim": head_dim},
        "raw_model_size_gb": round(raw_bytes_est / _GB, 4),
        "folded_weight_size_gb": round(folded_gb, 4),
        "folded_weight_size_source": "measured" if measured_gb is not None
        else "estimated",
        "fold_time_s": args.fold_time_s,
        "disk_write_time_s": args.disk_write_time_s,
        "load_time_s": args.load_time_s,
        "transfer_size_gb": round(folded_gb, 4),
        "transfer_time_estimates_s": transfer_time,
        "amortize_bandwidth": args.amortize_bandwidth,
        "one_time_setup_cost_s": round(one_time_s, 4),
        "amortization_sessions": list(_SESSIONS),
        "amortized_setup_cost_s": amortized,
        "note": "folding is a one-time setup/provisioning cost; online decode "
                "reuses provisioned folded weights (not per-token, not per-request)",
    }

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"=== folded package cost estimate ({rep['model_name']}) ===")
    print(f"dtype={rep['dtype']} num_layers={num_layers} dims={rep['dims']}")
    print(f"raw_model_size_gb={rep['raw_model_size_gb']} "
          f"folded_weight_size_gb={rep['folded_weight_size_gb']} "
          f"({rep['folded_weight_size_source']})")
    print(f"transfer_size_gb={rep['transfer_size_gb']}")
    print(f"transfer_time_estimates_s={rep['transfer_time_estimates_s']}")
    print(f"one_time_setup_cost_s={rep['one_time_setup_cost_s']} "
          f"(@ {rep['amortize_bandwidth']})")
    print(f"amortized_setup_cost_s={rep['amortized_setup_cost_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
