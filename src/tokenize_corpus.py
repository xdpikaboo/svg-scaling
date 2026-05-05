"""Phase 3: tokenize the Phase-1 splits into uint16 memmap .bin files.

For each split, concatenates ``tokenizer.encode(svg).ids + [eot_id]`` across
all records and writes the flat stream to ``data/tokens/{split}.bin``.
This mirrors nanoGPT's prepare.py, the model trains by sampling random
1024-token windows from the mmap, so per-SVG sequence length matters for
the histogram (reported in the paper) but not for the training format.

Also produces ``data/splits/seqlen_hist.json`` with per-split summary stats
(mean / median / p95 / p99 / max in tokens) for the Data section of the
report.

CLI:
    python -m src.tokenize_corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from src.tokenizer import EOT, svg_iter


SPLITS = ("train", "val", "test")
TOKEN_TARGET_TRAIN = 100_000_000  # the project spec acceptance bar
DTYPE = np.uint16  # vocab <= 4096 << 65536


def _percentile(sorted_lens: list[int], p: float) -> int:
    if not sorted_lens:
        return 0
    k = max(0, min(len(sorted_lens) - 1, int(round(p * (len(sorted_lens) - 1)))))
    return int(sorted_lens[k])


def histogram(seqlens: list[int], step: int = 32, max_len: int = 2048) -> dict:
    """Token-length histogram for the report. ``max_len`` overflow goes in the last bin."""
    bins = list(range(0, max_len + step, step))
    counts = [0] * (len(bins) - 1)
    for n in seqlens:
        if n >= max_len:
            counts[-1] += 1
            continue
        idx = min(n // step, len(counts) - 1)
        counts[idx] += 1
    return {"bin_edges": bins, "counts": counts}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 3: tokenize splits to uint16 .bin files.")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--output-dir", default="data/tokens")
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args(argv)

    tok_path = Path(args.tokenizer)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tok_path.exists():
        print(f"ERROR: tokenizer not found: {tok_path}", file=sys.stderr)
        return 1

    tok = Tokenizer.from_file(str(tok_path))
    eot_id = tok.token_to_id(EOT)
    if eot_id is None:
        print(f"ERROR: tokenizer has no '{EOT}' special token", file=sys.stderr)
        return 1
    print(f"Loaded tokenizer (vocab={tok.get_vocab_size()}, eot_id={eot_id})")

    per_split: dict[str, dict] = {}
    all_seqlens: dict[str, list[int]] = {}
    for split in SPLITS:
        split_path = splits_dir / f"{split}.jsonl"
        if not split_path.exists():
            print(f"WARNING: {split_path} not found, skipping", file=sys.stderr)
            continue
        out_path = out_dir / f"{split}.bin"
        stats, sl = tokenize_split(tok, eot_id, split_path, out_path, args.batch_size)
        per_split[split] = stats
        all_seqlens[split] = sl
        print(
            f"  {split}: {stats['n_records']} records, {stats['n_tokens']:,} tokens "
            f"(mean={stats['mean_seqlen']:.0f}, p95={stats['p95_seqlen']}, p99={stats['p99_seqlen']}, max={stats['max_seqlen']})"
        )

    # Round-trip sanity: decode first 200 ids of train.bin.
    train_bin = out_dir / "train.bin"
    if train_bin.exists() and train_bin.stat().st_size > 0:
        m = np.memmap(train_bin, mode="r", dtype=DTYPE)
        head = m[:200].tolist()
        decoded = tok.decode(head)
        print(f"\nFirst 200 train.bin tokens decode to (truncated to 240 chars):")
        print(f"  {decoded[:240]}{' ...' if len(decoded) > 240 else ''}")
        del m

    # Token-target check.
    train_tokens = per_split.get("train", {}).get("n_tokens", 0)
    if train_tokens < TOKEN_TARGET_TRAIN:
        shortfall = TOKEN_TARGET_TRAIN - train_tokens
        ratio = TOKEN_TARGET_TRAIN / max(1, train_tokens)
        print(
            f"\nNOTE: train.bin has {train_tokens:,} tokens, "
            f"{shortfall:,} short of the {TOKEN_TARGET_TRAIN:,} target ({ratio:.1f}x).\n"
            f"  Remediation options:\n"
            f"   - Bump --fonts-subsample (Phase 1) toward the full ~1.7M, re-run scripts 01..03\n"
            f"   - Document a reduced training-token budget in the report\n"
        )
    else:
        print(f"\ntrain.bin: {train_tokens:,} tokens, clears the {TOKEN_TARGET_TRAIN:,} target.")

    # Seqlen histogram -> data/splits/seqlen_hist.json
    hist_path = splits_dir / "seqlen_hist.json"
    hist_payload = {
        "step": 32,
        "max_len": 2048,
        "per_split": {
            split: {
                "stats": per_split[split],
                "histogram": histogram(all_seqlens[split], step=32, max_len=2048),
            }
            for split in per_split
        },
    }
    hist_path.write_text(json.dumps(hist_payload, indent=2), encoding="utf-8")
    print(f"\nSeqlen histogram: {hist_path}")
    print("Phase 3 done.")
    return 0


def tokenize_split(
    tok: Tokenizer,
    eot_id: int,
    split_path: Path,
    out_path: Path,
    batch_size: int,
) -> tuple[dict, list[int]]:
    """Tokenize one split, write the .bin, return (summary_stats, per_record_seqlens)."""
    seqlens: list[int] = []
    all_ids: list[int] = []
    batch: list[str] = []

    def _flush() -> None:
        if not batch:
            return
        encs = tok.encode_batch(batch)
        for enc in encs:
            ids = enc.ids
            seqlens.append(len(ids))
            all_ids.extend(ids)
            all_ids.append(eot_id)
        batch.clear()

    for svg in tqdm(svg_iter(split_path), desc=f"tokenize:{split_path.name}"):
        batch.append(svg)
        if len(batch) >= batch_size:
            _flush()
    _flush()

    arr = np.asarray(all_ids, dtype=DTYPE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if arr.size == 0:
        arr.tofile(out_path)
        return ({
            "n_records": 0, "n_tokens": 0,
            "mean_seqlen": 0.0, "median_seqlen": 0,
            "p95_seqlen": 0, "p99_seqlen": 0, "max_seqlen": 0,
            "n_eot": 0, "bin_path": str(out_path),
        }, [])
    fp = np.memmap(out_path, mode="w+", dtype=DTYPE, shape=arr.shape)
    fp[:] = arr
    fp.flush()
    del fp
    s = sorted(seqlens)
    stats = {
        "n_records": len(seqlens),
        "n_tokens": int(arr.size),
        "mean_seqlen": float(np.mean(seqlens)),
        "median_seqlen": _percentile(s, 0.50),
        "p95_seqlen": _percentile(s, 0.95),
        "p99_seqlen": _percentile(s, 0.99),
        "max_seqlen": int(max(seqlens)),
        "n_eot": int(np.sum(arr == eot_id)),
        "bin_path": str(out_path),
    }
    return stats, seqlens


if __name__ == "__main__":
    raise SystemExit(main())
