"""Best-of-N sampling for the xl_long model.

For each of the 45 sample slots (10 unconditional + 5 prefix-conditioned at
each of 3 temperatures), generate N candidates and keep the best one.
"Best" is preferred in this order: renders OK, then XML-valid, then longest.

At the baseline 11% per-attempt render rate, expected best-of-N rates:
    N=3 -> ~29%, N=5 -> ~44%, N=10 -> ~69%

Default is N=10 (~75 min wall time on RTX 3060 for xl_long).

Promotes the result to the canonical xl_long sample dir + re-runs eval if the
new render rate beats the baseline.

Usage:
    python scripts/18_best_of_n.py             # N=10 (default)
    python scripts/18_best_of_n.py --n 5       # faster, ~37 min
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from src.generate import (  # noqa: E402
    EOT,
    MAX_NEW_TOKENS_DEFAULT,
    N_UNCONDITIONAL,
    TEMPERATURES_DEFAULT,
    TOP_P_DEFAULT,
    generate,
    load_checkpoint,
    render_to_png,
)
from src.eval_svg import structural_valid, xml_valid  # noqa: E402
from src.utils import set_seed  # noqa: E402


def pick_best(svgs: list[str], png_dir: Path, stem: str) -> tuple[str, bool, bool]:
    """Score each candidate. Prefer (renders, struct-valid, xml-valid, longer).

    Returns (best_svg, render_ok, xml_valid_flag).
    """
    scored = []
    for i, svg in enumerate(svgs):
        xv = xml_valid(svg)
        sv = structural_valid(svg) if xv else False
        # Try render to a temp PNG, save and check return value.
        tmp_png = png_dir / f"_tmp_{stem}_{i}.png"
        rv = render_to_png(svg, tmp_png) if sv else False
        scored.append({
            "svg": svg, "png": tmp_png, "i": i,
            "renders": rv, "struct": sv, "xml": xv, "len": len(svg),
        })
    scored.sort(
        key=lambda d: (d["renders"], d["struct"], d["xml"], d["len"]),
        reverse=True,
    )
    best = scored[0]
    # Keep the best PNG, delete the rest.
    final_png = png_dir / f"{stem}.png"
    if best["png"].exists():
        if final_png.exists():
            final_png.unlink()
        best["png"].rename(final_png)
    for d in scored[1:]:
        if d["png"].exists():
            try:
                d["png"].unlink()
            except OSError:
                pass
    return best["svg"], best["renders"], best["xml"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Best-of-N sampling for xl_long.")
    p.add_argument("--n", type=int, default=10,
                   help="number of candidates per slot (default 10)")
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--temperatures", nargs="+", type=float,
                   default=list(TEMPERATURES_DEFAULT))
    p.add_argument("--run-id-file", default="runs/xl_long_run_id.txt")
    args = p.parse_args(argv)

    rid_file = ROOT / args.run_id_file
    if not rid_file.exists():
        sys.exit(f"ERROR: {rid_file} not found")
    run_id = rid_file.read_text(encoding="utf-8").strip()
    ckpt = ROOT / "checkpoints" / run_id / "model.pt"
    if not ckpt.exists():
        sys.exit(f"ERROR: checkpoint not found at {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"run_id={run_id}")
    print(f"best-of-N: N={args.n}, top-p={args.top_p}, T={args.temperatures}\n")

    model, cfg = load_checkpoint(ckpt, device)
    tok = Tokenizer.from_file(str(ROOT / "data/tokenizer/tokenizer.json"))
    eot_id = tok.token_to_id(EOT)

    out_dir = ROOT / "runs/samples" / f"{run_id}__bestof{args.n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    prefixes_file = ROOT / "configs/prefixes.txt"
    prefixes = [
        l for l in prefixes_file.read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    manifest_samples = []
    n_render_ok = 0
    n_xml = 0
    t_start = time.monotonic()

    def _do_slot(prefix: str, kind: str, idx: int, temp: float, stem: str):
        nonlocal n_render_ok, n_xml
        cands = []
        for n in range(args.n):
            set_seed(hash((stem, n)) & 0xFFFFFFFF)
            svg = generate(
                model, tok, prefix,
                temperature=temp, top_p=args.top_p,
                max_new_tokens=MAX_NEW_TOKENS_DEFAULT,
                eot_id=eot_id, device=device,
            )
            cands.append(svg)
        best_svg, ok, xv = pick_best(cands, png_dir, stem)
        (out_dir / f"{stem}.svg").write_text(best_svg, encoding="utf-8")
        if xv:
            n_xml += 1
        if ok:
            n_render_ok += 1
        manifest_samples.append({
            "kind": kind, "stem": stem, "temperature": temp,
            "idx": idx, "n_chars": len(best_svg),
            "render_ok": ok, "xml_valid": xv, "n_candidates": args.n,
        })

    for temp in args.temperatures:
        for k in range(N_UNCONDITIONAL):
            stem = f"uncond_T{temp:.1f}_{k:02d}"
            elapsed = (time.monotonic() - t_start) / 60
            print(f"  [{elapsed:5.1f} min] {stem}", flush=True)
            _do_slot("<svg", "unconditional", k, temp, stem)

    for p_i, prefix in enumerate(prefixes):
        for temp in args.temperatures:
            stem = f"prefix{p_i:02d}_T{temp:.1f}"
            elapsed = (time.monotonic() - t_start) / 60
            print(f"  [{elapsed:5.1f} min] {stem}", flush=True)
            _do_slot(prefix, "prefix", p_i, temp, stem)

    n_total = len(manifest_samples)
    pct_render = n_render_ok / max(1, n_total)
    pct_xml = n_xml / max(1, n_total)
    elapsed = (time.monotonic() - t_start) / 60

    print()
    print("=" * 70)
    print(f"BEST-OF-{args.n} RESULTS  ({elapsed:.1f} min)")
    print("=" * 70)
    print(f"  samples:   {n_total}")
    print(f"  render OK: {n_render_ok} / {n_total}  ({pct_render:.1%})")
    print(f"  xml valid: {n_xml} / {n_total}  ({pct_xml:.1%})")
    print()

    (out_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "best_of": args.n, "top_p": args.top_p,
        "temperatures": args.temperatures,
        "render_ok": n_render_ok, "xml_valid": n_xml,
        "samples": manifest_samples,
    }, indent=2), encoding="utf-8")

    # Promote if it beat the baseline.
    canonical = ROOT / "runs/samples" / run_id
    canonical_eval = ROOT / f"runs/eval_{run_id}.json"

    BASELINE = 0.18
    if pct_render > BASELINE:
        print(f"  beats baseline ({BASELINE:.0%}) -> promoting to canonical xl_long dir")
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.copytree(out_dir, canonical)
        rc = subprocess.run([
            sys.executable, "-m", "src.eval_svg",
            "--checkpoint", str(ckpt),
            "--samples-dir", str(canonical),
            "--test-bin", str(ROOT / "data/tokens/test.bin"),
            "--output", str(canonical_eval),
        ]).returncode
        if rc != 0:
            print(f"  WARNING: eval re-run rc={rc}")
        print()
        print("Re-execute the report to pick up the new render rate:")
        print("  jupyter nbconvert --to notebook --execute report/report.ipynb --output report.ipynb")
        print("  jupyter nbconvert --to html --no-input report/report.ipynb")
        print("  python scripts/12_html_to_pdf.py")
    else:
        print(f"  did NOT beat baseline ({BASELINE:.0%}); samples kept at {out_dir}")
        print(f"  consider true constrained decoding as the next step")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
