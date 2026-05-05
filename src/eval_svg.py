"""Evaluation: test perplexity + per-sample SVG validity rates.

What it computes:
    - Test perplexity on data/tokens/test.bin (sliding-window NLL).
    - XML validity rate over generated samples (lxml parse).
    - Structural validity (parse OK + root is <svg> + has drawing children).
    - Render rate (cairosvg / resvg-py / svglib fallback chain).

Output: runs/eval_{run_id}.json

CLI:
    python -m src.eval_svg --checkpoint checkpoints/.../model.pt --samples-dir runs/samples/...
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Test perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_test_perplexity(
    model: torch.nn.Module,
    test_bin: Path,
    block_size: int,
    batch_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> dict:
    """Non-overlapping sliding window over test.bin. Returns dict with mean NLL,
    perplexity, and counts."""
    data = np.memmap(test_bin, mode="r", dtype=np.uint16)
    n_tokens = int(data.shape[0])
    # Need s+1+block_size <= n_tokens, so largest valid start is n_tokens - block_size - 1.
    starts = list(range(0, max(0, n_tokens - block_size - 1), block_size))
    if not starts:
        return {"n_tokens": n_tokens, "n_windows": 0, "mean_nll": None, "perplexity": None}

    if device.type == "cuda" and autocast_dtype != torch.float32:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        autocast_ctx = nullcontext()

    total_loss = 0.0
    total_tokens = 0
    for batch_start in range(0, len(starts), batch_size):
        chunk = starts[batch_start:batch_start + batch_size]
        x = np.stack([np.asarray(data[s:s + block_size], dtype=np.int64) for s in chunk])
        y = np.stack([np.asarray(data[s + 1:s + 1 + block_size], dtype=np.int64) for s in chunk])
        x_t = torch.from_numpy(x).to(device, non_blocking=True)
        y_t = torch.from_numpy(y).to(device, non_blocking=True)
        with autocast_ctx:
            out = model(x_t)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            y_t.view(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += int(y_t.numel())

    mean_nll = total_loss / max(1, total_tokens)
    return {
        "n_tokens": n_tokens,
        "n_windows": len(starts),
        "block_size": block_size,
        "tokens_evaluated": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
    }


# ---------------------------------------------------------------------------
# SVG validity
# ---------------------------------------------------------------------------

_DRAW_TAGS = {"path", "circle", "rect", "ellipse", "line", "polygon",
              "polyline", "g", "use", "image"}


def xml_valid(svg: str) -> bool:
    """True iff lxml strict parse succeeds."""
    try:
        from lxml import etree
        etree.fromstring(svg.encode("utf-8"), parser=etree.XMLParser(recover=False))
        return True
    except Exception:
        return False


def structural_valid(svg: str) -> bool:
    """Stricter than xml_valid: requires lxml parse, root is <svg> (any
    namespace), and at least one drawing child element."""
    try:
        from lxml import etree
        root = etree.fromstring(svg.encode("utf-8"), parser=etree.XMLParser(recover=False))
    except Exception:
        return False
    local = root.tag.split("}", 1)[-1] if isinstance(root.tag, str) and root.tag.startswith("{") else root.tag
    if local != "svg":
        return False
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        t = el.tag.split("}", 1)[-1] if el.tag.startswith("{") else el.tag
        if t in _DRAW_TAGS:
            return True
    return False


def render_ok(svg: str) -> bool:
    """Try to render the SVG through any available backend (cairosvg, resvg,
    svglib). Writes to a tempfile and discards, we just want a True/False
    signal. Delegates to ``src.render_utils.render_svg_to_png`` so the renderer
    chain matches what was used to render samples for the report."""
    import tempfile, os
    from src.render_utils import render_svg_to_png
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        ok = render_svg_to_png(svg, Path(tmp_path), size=64)
    finally:
        try: os.remove(tmp_path)
        except OSError: pass
    return ok


def cairosvg_available() -> bool:
    """True iff *any* renderer is usable (cairosvg or svglib fallback)."""
    from src.render_utils import available_renderer
    return available_renderer() is not None


def eval_samples(samples_dir: Path) -> dict:
    """Walk samples_dir for .svg files, compute per-sample validity, aggregate."""
    svgs = sorted(samples_dir.glob("*.svg"))
    if not svgs:
        return {"n_samples": 0, "xml_valid_rate": None,
                "structural_valid_rate": None, "render_ok_rate": None,
                "per_sample": []}

    can_render = cairosvg_available()
    per_sample = []
    n_xml = 0
    n_struct = 0
    n_render = 0
    for path in svgs:
        text = path.read_text(encoding="utf-8")
        xv = xml_valid(text)
        sv = structural_valid(text)
        rv = render_ok(text) if can_render else None
        n_xml += int(xv)
        n_struct += int(sv)
        if rv is True:
            n_render += 1
        per_sample.append({
            "file": path.name,
            "n_chars": len(text),
            "xml_valid": xv,
            "structural_valid": sv,
            "render_ok": rv,
        })

    n = len(svgs)
    return {
        "n_samples": n,
        "xml_valid_rate": n_xml / n,
        "structural_valid_rate": n_struct / n,
        "render_ok_rate": (n_render / n) if can_render else None,
        "render_evaluated": can_render,
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Checkpoint reload (mirrors src/generate.py)
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path, device: torch.device):
    raw = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = raw["config"]
    from src.config import (
        BatchConfig, EvalConfig, ModelConfig, OptimConfig,
        PathsConfig, ScheduleConfig, TrainConfig,
    )
    cfg = TrainConfig(
        seed=cfg_dict.get("seed", 0),
        name=cfg_dict.get("name", "best"),
        parameterization=cfg_dict.get("parameterization", "sp"),
        precision=cfg_dict.get("precision", "bfloat16"),
        model=ModelConfig(**cfg_dict["model"]),
        optimizer=OptimConfig(**{k: v for k, v in cfg_dict["optimizer"].items()
                                 if k != "betas"} | {"betas": tuple(cfg_dict["optimizer"]["betas"])}),
        schedule=ScheduleConfig(**cfg_dict["schedule"]),
        batch=BatchConfig(**cfg_dict["batch"]),
        eval=EvalConfig(**cfg_dict["eval"]),
        paths=PathsConfig(**cfg_dict["paths"]),
    )
    if cfg.parameterization == "mup":
        from src.mup_model import build_mup_gpt
        model = build_mup_gpt(cfg.model)
    else:
        from src.model import GPT
        model = GPT(cfg.model)
    model.load_state_dict(raw["model"])
    model.to(device).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 11 evaluation.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--samples-dir", required=True)
    p.add_argument("--test-bin", default="data/tokens/test.bin")
    p.add_argument("--output", default=None,
                   help="default: runs/eval_{run_id}.json")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1
    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = load_checkpoint(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model={cfg.name}  parameterization={cfg.parameterization}  n_params={n_params:,}")

    autocast_dtype = (torch.bfloat16
                      if device.type == "cuda" and torch.cuda.is_bf16_supported()
                      and cfg.precision == "bfloat16"
                      else torch.float16 if device.type == "cuda" else torch.float32)

    print("\n[1/2] Test perplexity ...")
    ppl = eval_test_perplexity(
        model, Path(args.test_bin), cfg.model.block_size, args.batch_size,
        device, autocast_dtype,
    )
    print(f"  test n_tokens={ppl['n_tokens']:,}  windows={ppl['n_windows']}  "
          f"mean_nll={ppl['mean_nll']:.4f}  ppl={ppl['perplexity']:.2f}")

    print("\n[2/2] SVG validity over generated samples ...")
    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        print(f"  WARNING: {samples_dir} not found, skipping", file=sys.stderr)
        sample_eval = {"n_samples": 0}
    else:
        sample_eval = eval_samples(samples_dir)
        n = sample_eval["n_samples"]
        xv = sample_eval["xml_valid_rate"]
        sv = sample_eval["structural_valid_rate"]
        rv = sample_eval["render_ok_rate"]
        rv_str = f"{rv:.1%}" if rv is not None else "skipped (libcairo unavailable)"
        print(f"  n_samples={n}  xml_valid={xv:.1%}  structural={sv:.1%}  render={rv_str}")

    run_id = ckpt_path.parent.name
    out_path = Path(args.output or f"runs/eval_{run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "parameterization": cfg.parameterization,
        "n_params": n_params,
        "test_perplexity": ppl,
        "samples": sample_eval,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
