"""Phase 1 data pipeline: download SVG corpora, clean, length-filter,
optionally render-validate, then split into train/val/test JSONL.

CLI:
    python -m src.data --output-dir data --render --render-workers 8

Outputs (under ``--output-dir``, default ``data``):
    raw/                       # HF cache (managed by datasets)
    clean/{icons,emoji,fonts}.jsonl   # post-clean intermediates (cached)
    splits/train.jsonl
    splits/val.jsonl
    splits/test.jsonl
    splits/stats.json
    splits/samples.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from lxml import etree
from tqdm import tqdm


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ALLOWED_NS = {SVG_NS, XLINK_NS}

# SVG number: optional sign, integer/decimal, optional exponent.
# Matches "1", "1.5", ".5", "1.", "1e10", "-1.5e-3".
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

# Numeric attributes whose values get rounded. ``d`` and ``points`` are handled
# the same way (free-form number streams), the regex doesn't care about
# command letters since it only matches numeric literals.
_NUMERIC_ATTRS = {
    "d", "points", "x", "y", "x1", "y1", "x2", "y2",
    "cx", "cy", "r", "rx", "ry",
    "width", "height", "viewBox",
    "stroke-width", "stroke-dasharray", "stroke-dashoffset",
    "transform",  # transform contains numbers inside matrix(...)/translate(...)/etc.
    "offset", "fx", "fy", "dx", "dy",
}

DROP_TAGS = {"metadata", "title", "desc"}

SOURCES: dict[str, dict] = {
    "icons": {"hf_id": "starvector/svg-icons-simple", "subsample": None},
    "emoji": {"hf_id": "starvector/svg-emoji-simple", "subsample": None},
    "fonts": {"hf_id": "starvector/svg-fonts-simple", "subsample": 50000},
}


# ---------------------------------------------------------------------------
# Number rounding
# ---------------------------------------------------------------------------

def _fmt_rounded(v: float, ndp: int = 1) -> str:
    """Format a rounded float compactly: '1.5', '0', '123456.7', '-2'."""
    rounded = round(v, ndp)
    if rounded == 0:  # also collapses -0.0 to "0"
        return "0"
    if rounded == int(rounded):
        return str(int(rounded))
    # Fixed-point at ndp, then strip trailing zeros (but keep at least one digit).
    s = f"{rounded:.{ndp}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _round_numbers_in_str(s: str, ndp: int = 1) -> str:
    """Round every numeric literal inside ``s`` to ``ndp`` decimal places.
    Safe for path ``d`` strings, ``points``, ``transform``, viewBox, etc. ,
    the regex only matches numbers, never command letters or punctuation.
    """
    def _sub(m: re.Match) -> str:
        try:
            return _fmt_rounded(float(m.group(0)), ndp)
        except ValueError:
            return m.group(0)
    return _NUM_RE.sub(_sub, s)


# ---------------------------------------------------------------------------
# SVG cleaning
# ---------------------------------------------------------------------------

def _strip_disallowed_namespaces(root: etree._Element) -> None:
    """Remove elements/attributes outside SVG and XLink namespaces."""
    # Remove elements first (iterate over a list, we mutate the tree).
    for el in list(root.iter()):
        tag = el.tag
        if not isinstance(tag, str):  # comments, PIs
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
            continue
        if tag.startswith("{"):
            ns = tag[1:].split("}", 1)[0]
            if ns not in ALLOWED_NS:
                parent = el.getparent()
                if parent is not None:
                    parent.remove(el)
                continue
        # Strip attributes in foreign namespaces.
        for attr_name in list(el.attrib):
            if attr_name.startswith("{"):
                ns = attr_name[1:].split("}", 1)[0]
                if ns not in ALLOWED_NS:
                    del el.attrib[attr_name]


def _drop_tags(root: etree._Element, local_names: Iterable[str]) -> None:
    """Drop elements whose local name is in ``local_names`` (any namespace)."""
    targets = set(local_names)
    for el in list(root.iter()):
        if not isinstance(el.tag, str):
            continue
        local = el.tag.split("}", 1)[-1] if el.tag.startswith("{") else el.tag
        if local in targets:
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)


def _round_numeric_attrs(root: etree._Element, ndp: int = 1) -> None:
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for attr in list(el.attrib):
            local = attr.split("}", 1)[-1] if attr.startswith("{") else attr
            if local in _NUMERIC_ATTRS:
                el.attrib[attr] = _round_numbers_in_str(el.attrib[attr], ndp)


def clean_svg(svg_text: str, ndp: int = 1) -> str | None:
    """Parse, normalize, and re-serialize an SVG. Returns None on parse failure."""
    if not svg_text:
        return None
    try:
        parser = etree.XMLParser(
            remove_comments=True,
            remove_blank_text=True,
            recover=False,
            huge_tree=False,
            resolve_entities=False,
        )
        # Encode to bytes; fromstring on bytes lets lxml handle BOMs / encodings.
        root = etree.fromstring(svg_text.encode("utf-8"), parser=parser)
    except etree.XMLSyntaxError:
        return None
    except ValueError:
        return None

    # Drop comments/PIs and unwanted tags.
    _drop_tags(root, DROP_TAGS)
    # Drop elements/attrs in foreign namespaces (Inkscape, Sodipodi, etc.).
    _strip_disallowed_namespaces(root)
    # Round numeric attribute values (path d, points, viewBox, transform, ...).
    _round_numeric_attrs(root, ndp)
    # Remove orphan namespace declarations after deletions.
    etree.cleanup_namespaces(root, top_nsmap={None: SVG_NS, "xlink": XLINK_NS})

    # Verify the root is actually an SVG element.
    root_local = root.tag.split("}", 1)[-1] if root.tag.startswith("{") else root.tag
    if root_local != "svg":
        return None

    out = etree.tostring(root, encoding="unicode")
    return out


# ---------------------------------------------------------------------------
# Length filter
# ---------------------------------------------------------------------------

def length_ok(svg: str, min_chars: int = 50, max_chars: int = 8000) -> bool:
    n = len(svg)
    return min_chars <= n <= max_chars


# ---------------------------------------------------------------------------
# Render validation (process pool, best-effort)
# ---------------------------------------------------------------------------

def _render_one(svg: str) -> bool:
    """Worker: True iff cairosvg renders ``svg`` without error."""
    try:
        import cairosvg  # imported in worker so the master process can run without libcairo
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=64, output_height=64)
        return True
    except Exception:
        return False


def _probe_cairo() -> None:
    """Fail fast if libcairo isn't loadable (otherwise the pool silently rejects
    every record). Raises RuntimeError with a clear remedy."""
    probe_svg = b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='10' height='10'/></svg>"
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=probe_svg, output_width=8, output_height=8)
    except Exception as e:
        raise RuntimeError(
            f"cairosvg probe failed ({type(e).__name__}: {e}).\n"
            "  - On Linux: apt-get install libcairo2\n"
            "  - On macOS: brew install cairo\n"
            "  - On Windows: install GTK runtime, or run from WSL, or pass --no-render\n"
        ) from e


def filter_renderable(svgs: list[str], workers: int = 8, chunksize: int = 64) -> list[bool]:
    """Return a parallel list[bool]: True iff cairosvg renders the SVG.
    Uses a process pool. Probes libcairo upfront so a missing system library
    fails in <1 second instead of after a full pass with everything rejected.
    Per-task hard timeouts are not enforced cross-platform, if a worker
    hangs, re-run with ``--no-render``.
    """
    results = [False] * len(svgs)
    if not svgs:
        return results
    _probe_cairo()  # raises with remedy if libcairo missing
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_render_one, s): i for i, s in enumerate(svgs)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="render-validate"):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = False
    return results


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _find_svg_field(column_names: list[str]) -> str:
    for name in ("Svg", "svg", "svg_string", "text", "code"):
        if name in column_names:
            return name
    raise RuntimeError(
        f"No SVG text column found. Got columns: {column_names}. "
        "Expected one of: Svg, svg, svg_string, text, code."
    )


def load_source(
    hf_id: str,
    cache_dir: str,
    subsample: int | None,
    seed: int,
    max_per_source: int | None = None,
) -> list[str]:
    """Load raw SVG strings from a HuggingFace dataset, optionally subsampled."""
    from datasets import load_dataset

    ds = load_dataset(hf_id, cache_dir=cache_dir)
    # Pick the train split (these datasets typically have only "train").
    split_name = "train" if "train" in ds else next(iter(ds.keys()))
    split = ds[split_name]
    field_name = _find_svg_field(split.column_names)

    n = len(split)
    if max_per_source is not None:
        n = min(n, max_per_source)
    indices: list[int]
    if subsample is not None and subsample < n:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(n), subsample))
    else:
        indices = list(range(n))

    out: list[str] = []
    for i in tqdm(indices, desc=f"load:{hf_id}"):
        s = split[i][field_name]
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="replace")
        if isinstance(s, str):
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    seed: int = 0
    fonts_subsample_n: int = 0
    raw_counts: dict[str, int] = field(default_factory=dict)
    post_parse_counts: dict[str, int] = field(default_factory=dict)
    post_length_filter_counts: dict[str, int] = field(default_factory=dict)
    post_render_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    char_length_histogram: dict = field(default_factory=dict)
    source_mix_train: dict[str, float] = field(default_factory=dict)
    render_validation_run: bool = False


def char_length_histogram(svgs: Iterable[str], step: int = 250, max_len: int = 8000) -> dict:
    bins = list(range(0, max_len + step, step))
    counts = [0] * (len(bins) - 1)
    for s in svgs:
        n = len(s)
        if n >= max_len:
            counts[-1] += 1
            continue
        idx = min(n // step, len(counts) - 1)
        counts[idx] += 1
    return {"bin_edges": bins, "counts": counts}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def clean_source(
    raw_svgs: list[str],
    source: str,
    out_path: Path,
    min_chars: int,
    max_chars: int,
) -> tuple[list[dict], dict]:
    """Run parse + clean + length filter on ``raw_svgs``. Returns (records, counts)."""
    parsed_ok = 0
    after_length = 0
    records: list[dict] = []
    for svg in tqdm(raw_svgs, desc=f"clean:{source}"):
        cleaned = clean_svg(svg)
        if cleaned is None:
            continue
        parsed_ok += 1
        if not length_ok(cleaned, min_chars, max_chars):
            continue
        after_length += 1
        records.append({"svg": cleaned, "source": source})
    counts = {
        "raw": len(raw_svgs),
        "post_parse": parsed_ok,
        "post_length": after_length,
    }
    write_jsonl(records, out_path)
    return records, counts


def split_records(
    records: list[dict],
    seed: int,
    ratios: tuple[float, float, float] = (0.98, 0.01, 0.01),
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def write_samples(
    out_path: Path,
    splits: dict[str, list[dict]],
    n: int = 5,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    lines: list[str] = []
    for name, records in splits.items():
        lines.append(f"=== {name} ({len(records)} records) ===")
        if not records:
            lines.append("(empty)")
            continue
        sample = rng.sample(records, min(n, len(records)))
        for i, r in enumerate(sample):
            preview = r["svg"]
            if len(preview) > 600:
                preview = preview[:600] + " ...[truncated]"
            lines.append(f"--- {name}[{i}] source={r['source']} len={len(r['svg'])} ---")
            lines.append(preview)
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 1: SVG corpus preparation.")
    p.add_argument("--output-dir", default="data", help="root output dir (default: data)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fonts-subsample", type=int, default=50000,
                   help="N rows to subsample from svg-fonts-simple (default 50000)")
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--max-chars", type=int, default=8000)
    p.add_argument("--render", dest="render", action="store_true", default=True)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--render-workers", type=int, default=8)
    p.add_argument("--max-per-source", type=int, default=None,
                   help="cap per source (mainly for smoke tests)")
    p.add_argument("--force-clean", action="store_true",
                   help="re-run cleaning even if data/clean/{source}.jsonl exists")
    p.add_argument("--sources", nargs="+", default=list(SOURCES.keys()),
                   choices=list(SOURCES.keys()))
    args = p.parse_args(argv)

    out_root = Path(args.output_dir)
    raw_dir = out_root / "raw"
    clean_dir = out_root / "clean"
    splits_dir = out_root / "splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    stats = Stats(seed=args.seed, fonts_subsample_n=args.fonts_subsample)

    # ---- Per-source: download + clean + length filter, with caching ----
    per_source_records: dict[str, list[dict]] = {}
    for source in args.sources:
        meta = SOURCES[source]
        clean_path = clean_dir / f"{source}.jsonl"
        if clean_path.exists() and not args.force_clean:
            print(f"[{source}] using cached {clean_path}")
            recs = read_jsonl(clean_path)
            stats.raw_counts[source] = -1  # unknown after cache hit
            stats.post_parse_counts[source] = -1
            stats.post_length_filter_counts[source] = len(recs)
        else:
            subsample = meta["subsample"] if source == "fonts" else None
            if source == "fonts":
                subsample = args.fonts_subsample
            raw = load_source(
                meta["hf_id"],
                cache_dir=str(raw_dir),
                subsample=subsample,
                seed=args.seed,
                max_per_source=args.max_per_source,
            )
            recs, c = clean_source(raw, source, clean_path,
                                   args.min_chars, args.max_chars)
            stats.raw_counts[source] = c["raw"]
            stats.post_parse_counts[source] = c["post_parse"]
            stats.post_length_filter_counts[source] = c["post_length"]
        per_source_records[source] = recs

    # ---- Optional render validation (across all sources together) ----
    if args.render:
        all_records: list[dict] = []
        for source, recs in per_source_records.items():
            all_records.extend(recs)
        ok_mask = filter_renderable(
            [r["svg"] for r in all_records],
            workers=args.render_workers,
        )
        kept_by_source: dict[str, list[dict]] = {s: [] for s in per_source_records}
        for r, ok in zip(all_records, ok_mask):
            if ok:
                kept_by_source[r["source"]].append(r)
        for s, recs in kept_by_source.items():
            stats.post_render_counts[s] = len(recs)
        per_source_records = kept_by_source
        stats.render_validation_run = True
    else:
        for s, recs in per_source_records.items():
            stats.post_render_counts[s] = len(recs)

    # ---- Combine + split ----
    merged: list[dict] = []
    for recs in per_source_records.values():
        merged.extend(recs)
    train, val, test = split_records(merged, seed=args.seed)
    write_jsonl(train, splits_dir / "train.jsonl")
    write_jsonl(val,   splits_dir / "val.jsonl")
    write_jsonl(test,  splits_dir / "test.jsonl")
    stats.split_counts = {"train": len(train), "val": len(val), "test": len(test)}

    # ---- Stats: histogram + source mix on train ----
    stats.char_length_histogram = char_length_histogram(
        (r["svg"] for r in train), step=250, max_len=args.max_chars,
    )
    if train:
        from collections import Counter
        c = Counter(r["source"] for r in train)
        total = sum(c.values())
        stats.source_mix_train = {k: round(v / total, 4) for k, v in c.items()}

    # ---- Write stats + samples ----
    stats_path = splits_dir / "stats.json"
    stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    write_samples(
        splits_dir / "samples.txt",
        {"train": train, "val": val, "test": test},
        n=5, seed=args.seed,
    )

    # ---- Summary ----
    print("\nPhase 1 done.")
    print(f"  splits: train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  stats:  {stats_path}")
    print(f"  samples: {splits_dir / 'samples.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
