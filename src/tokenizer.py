"""Phase 2 tokenizer: train a byte-level BPE on data/splits/train.jsonl,
save to data/tokenizer/tokenizer.json, and dump a sanity check. - HF tokenizers (not sentencepiece)
    - Byte-level BPE; UNK is impossible by construction
    - Vocab size 4096
    - Special tokens: ["<|endoftext|>"]
    - Train on train.jsonl only (no leakage)

CLI:
    python -m src.tokenizer --input data/splits/train.jsonl --output-dir data/tokenizer
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDec
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.trainers import BpeTrainer

from src.utils import set_seed


EOT = "<|endoftext|>"


# ---------------------------------------------------------------------------
# Streaming corpus iterator
# ---------------------------------------------------------------------------

def svg_iter(jsonl_path: Path, max_records: int | None = None) -> Iterator[str]:
    """Yield SVG strings from a Phase-1 splits JSONL, line by line.

    If ``max_records`` is set, stop after that many records, useful for
    training the tokenizer on a representative subset (BPE vocabulary is
    dominated by char n-gram stats; subset is enough)."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if line:
                yield json.loads(line)["svg"]


def count_lines_and_bytes(jsonl_path: Path) -> tuple[int, int]:
    """Single pass for stats: count records and total SVG bytes."""
    n_lines = 0
    n_bytes = 0
    for svg in svg_iter(jsonl_path):
        n_lines += 1
        n_bytes += len(svg.encode("utf-8"))
    return n_lines, n_bytes


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tokenizer(
    input_path: Path,
    vocab_size: int,
    min_frequency: int,
    max_records: int | None = None,
) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None))
    # use_regex=False is critical for SVG/code content. With the default GPT-2
    # regex enabled, pre-tokens split on letter/digit/punct boundaries, so a
    # path like "M3 12L8" becomes [M, 3, 12, L, 8] and BPE (which merges only
    # *within* a pre-token) can never form merges across those splits. Result:
    # ~1.8 chars/token, near-character-level. Disabling the regex lets BPE see
    # the whole byte stream and discover SVG-natural merges like '<path d="M'
    # and 'viewBox="0 0'. Byte-level encoding still makes UNK impossible.
    tok.pre_tokenizer = ByteLevelPre(add_prefix_space=False, use_regex=False)
    tok.decoder = ByteLevelDec()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=ByteLevelPre.alphabet(),
        min_frequency=min_frequency,
        show_progress=True,
    )
    tok.train_from_iterator(svg_iter(input_path, max_records=max_records), trainer=trainer)
    return tok


# ---------------------------------------------------------------------------
# Sanity dump
# ---------------------------------------------------------------------------

def _decode_individual(tok: Tokenizer, ids: list[int]) -> list[str]:
    """Decode each id individually so we can show what merges look like."""
    out = []
    for i in ids:
        s = tok.decode([i])
        # The decoder collapses some byte-level marks; if a single id decodes
        # to empty (rare for byte-level), fall back to id_to_token to keep
        # output informative.
        if not s:
            s = tok.id_to_token(i) or ""
        out.append(s)
    return out


def sanity_dump(tok: Tokenizer, input_path: Path, n: int, seed: int) -> float:
    """Pick n random SVGs, encode each, print compression + first ~30 tokens.

    Returns average chars-per-token across the sample (sanity metric).
    """
    set_seed(seed)
    # Reservoir sample n records from train.jsonl without loading all into RAM.
    # 132k strings is fine to load briefly though; reservoir sample for cleanliness.
    chosen: list[str] = []
    for i, svg in enumerate(svg_iter(input_path)):
        if len(chosen) < n:
            chosen.append(svg)
        else:
            j = random.randint(0, i)
            if j < n:
                chosen[j] = svg

    print()
    print("=" * 70)
    print(f"SANITY DUMP, {len(chosen)} random train SVGs")
    print("=" * 70)
    total_chars = 0
    total_tokens = 0
    for idx, svg in enumerate(chosen):
        enc = tok.encode(svg)
        ids = enc.ids
        n_chars = len(svg)
        n_tok = len(ids)
        total_chars += n_chars
        total_tokens += n_tok
        ratio = n_chars / max(1, n_tok)
        print(f"\n[{idx}] chars={n_chars} tokens={n_tok} ratio={ratio:.2f} chars/token")
        head = svg[:200].replace("\n", " ")
        if len(svg) > 200:
            head += " ..."
        print(f"    src:  {head}")
        first_tokens = _decode_individual(tok, ids[:30])
        # Display tokens with explicit bars to make boundaries readable.
        tok_display = " | ".join(repr(t) for t in first_tokens)
        print(f"    toks: {tok_display}")
    overall = total_chars / max(1, total_tokens)
    print()
    print(f"Average compression: {overall:.2f} chars/token across {len(chosen)} samples")
    print("=" * 70)
    return overall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 2: train byte-level BPE on cleaned SVGs.")
    p.add_argument("--input", default="data/splits/train.jsonl",
                   help="Phase-1 train.jsonl (default: data/splits/train.jsonl)")
    p.add_argument("--output-dir", default="data/tokenizer",
                   help="output dir for tokenizer.json and stats.json")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-records", type=int, default=None,
                   help="Train BPE on at most this many records (subsample). "
                        "Vocabulary converges fast on char n-gram stats; 100k is plenty.")
    p.add_argument("--sanity-n", type=int, default=5)
    p.add_argument("--no-sanity", dest="sanity", action="store_false", default=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"
    stats_path = out_dir / "stats.json"

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 1

    # Pre-pass for corpus stats (fast: just decoding JSON lines).
    print(f"Counting records and bytes in {input_path} ...")
    n_lines, n_bytes = count_lines_and_bytes(input_path)
    print(f"  {n_lines} records, {n_bytes} bytes ({n_bytes / 1e6:.1f} MB)")

    # Train.
    if args.max_records:
        print(f"Training byte-level BPE: vocab={args.vocab_size} min_freq={args.min_frequency} "
              f"max_records={args.max_records:,} (subset)")
    else:
        print(f"Training byte-level BPE: vocab={args.vocab_size} min_freq={args.min_frequency}")
    tok = train_tokenizer(input_path, args.vocab_size, args.min_frequency,
                          max_records=args.max_records)
    actual_vocab = tok.get_vocab_size()
    eot_id = tok.token_to_id(EOT)
    print(f"  trained: vocab_size={actual_vocab}, eot_id={eot_id}")

    # Save tokenizer.
    tok.save(str(tok_path))
    print(f"  saved: {tok_path}")

    # Sanity dump.
    sanity_ratio: float | None = None
    if args.sanity:
        sanity_ratio = sanity_dump(tok, input_path, args.sanity_n, args.seed)
        if sanity_ratio < 3.0:
            print(
                f"\nWARNING: average compression {sanity_ratio:.2f} chars/token is "
                "below the ~3.0 threshold, tokenizer may be mis-trained.",
                file=sys.stderr,
            )

    # Stats.
    stats = {
        "vocab_size": actual_vocab,
        "min_frequency": args.min_frequency,
        "n_train_lines": n_lines,
        "n_train_bytes": n_bytes,
        "alphabet_size": len(ByteLevelPre.alphabet()),
        "special_tokens": [EOT],
        "eot_id": eot_id,
        "sanity_compression_chars_per_token": sanity_ratio,
        "input": str(input_path),
        "seed": args.seed,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nStats written: {stats_path}")
    print("Phase 2 done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
