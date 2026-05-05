"""Render training-data examples + generated samples for the report.

Three things this script does:

1. Picks 6 training-data SVGs at varying complexity (short / medium / long
   token counts) and renders each to ``report/figures/data_examples/``.

2. Re-renders all generated samples in ``runs/samples/{best_run_id}/png/``
   (overwriting placeholder PNGs from the original Phase-10 run, which
   were written when libcairo was unavailable).

3. Re-runs ``src.eval_svg`` so the render-rate metric in
   ``runs/eval_{best_run_id}.json`` is populated with real numbers instead
   of ``null``.

Requires either ``cairosvg`` (Linux/macOS) or ``svglib + reportlab``
(Windows-friendly) installed. See requirements.txt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.render_utils import available_renderer, render_svg_to_png, renderer_diagnostics  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def render_training_examples(out_dir: Path, n: int = 6, seed: int = 0) -> dict:
    """Pick ``n`` training SVGs at varying complexities, render to PNG."""
    train_jsonl = ROOT / "data" / "splits" / "train.jsonl"
    if not train_jsonl.exists():
        print(f"  skipping: {train_jsonl} not found")
        return {}
    print(f"  reading {train_jsonl} ...")
    records = _read_jsonl(train_jsonl)
    # Sort by length, then pick uniformly across the percentile range.
    records.sort(key=lambda r: len(r["svg"]))
    picks = []
    for i in range(n):
        # Pick at the (i+0.5)/n quantile to span the distribution.
        q = (i + 0.5) / n
        idx = min(len(records) - 1, int(q * len(records)))
        picks.append((idx, records[idx]))

    out_dir.mkdir(parents=True, exist_ok=True)
    info = []
    for k, (idx, rec) in enumerate(picks):
        svg = rec["svg"]
        png_path = out_dir / f"example_{k}.png"
        ok = render_svg_to_png(svg, png_path, size=128)
        info.append({
            "k": k,
            "source": rec.get("source"),
            "n_chars": len(svg),
            "rank_in_train": idx,
            "png": str(png_path.relative_to(ROOT)),
            "render_ok": ok,
        })
        flag = "OK" if ok else "fail"
        print(f"    [{flag}] example_{k}.png  source={rec.get('source')} chars={len(svg)}")
    (out_dir / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {"n": n, "picks": info}


def rerender_generated_samples(samples_dir: Path) -> dict:
    """Re-render every .svg in ``samples_dir`` to ``samples_dir/png/``."""
    if not samples_dir.exists():
        print(f"  skipping: {samples_dir} not found")
        return {"n_samples": 0}
    svgs = sorted(samples_dir.glob("*.svg"))
    if not svgs:
        return {"n_samples": 0}
    png_dir = samples_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for p in svgs:
        text = p.read_text(encoding="utf-8")
        out_png = png_dir / f"{p.stem}.png"
        if render_svg_to_png(text, out_png, size=128):
            n_ok += 1
    print(f"  re-rendered {len(svgs)} SVGs ({n_ok} successful, {len(svgs) - n_ok} placeholder)")
    return {"n_samples": len(svgs), "n_render_ok": n_ok}


def rerun_eval(run_id: str) -> int:
    """Re-runs Phase 11 eval so the JSON's render_ok rate is fresh."""
    from src.eval_svg import main as eval_main
    ckpt = ROOT / "checkpoints" / run_id / "model.pt"
    samples = ROOT / "runs" / "samples" / run_id
    if not ckpt.exists():
        print(f"  skipping eval rerun: {ckpt} not found")
        return 1
    return eval_main([
        "--checkpoint", str(ckpt),
        "--samples-dir", str(samples),
        "--test-bin", str(ROOT / "data" / "tokens" / "test.bin"),
        "--output", str(ROOT / "runs" / f"eval_{run_id}.json"),
    ])


def main() -> int:
    renderer = available_renderer()
    if renderer is None:
        print(
            "ERROR: no SVG renderer available.\n"
            "  Diagnostics:\n"
            f"{renderer_diagnostics()}\n"
            "  Recommended fix on Windows (pure Rust, no system libs):\n"
            "    pip install resvg-py\n"
            "  Linux / macOS:\n"
            "    apt-get install libcairo2 && pip install cairosvg\n",
            file=sys.stderr,
        )
        return 1
    print(f"renderer: {renderer}")

    print("\n[1/3] Training-data examples ...")
    render_training_examples(ROOT / "report" / "figures" / "data_examples", n=6)

    print("\n[2/3] Re-render generated samples ...")
    run_id_file = ROOT / "runs" / "best_run_id.txt"
    if run_id_file.exists():
        run_id = run_id_file.read_text(encoding="utf-8").strip()
        rerender_generated_samples(ROOT / "runs" / "samples" / run_id)

        print("\n[3/3] Re-run eval (refresh render_ok rate) ...")
        rc = rerun_eval(run_id)
        if rc != 0:
            print(f"  eval rerun returned rc={rc}", file=sys.stderr)
    else:
        print(f"  skipping: {run_id_file} not found")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
