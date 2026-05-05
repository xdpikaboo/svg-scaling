"""Generation: load a trained checkpoint, sample SVGs at temperature and
top-p, save .svg files. Optionally renders PNGs via cairosvg / resvg-py.

What it produces:
    - 10 unconditional SVGs (from a "<svg" prefix) per temperature
    - 5 prefix-conditioned SVGs per temperature (prefixes from configs/prefixes.txt)
    - Temperatures: {0.5, 0.8, 1.0}; top-p: 0.9
    - Raw .svg files at runs/samples/{run_id}/
    - Rendered PNGs at runs/samples/{run_id}/png/ (placeholder if render fails)

CLI:
    python -m src.generate --checkpoint checkpoints/best_*/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.config import ModelConfig, TrainConfig
from src.model import GPT
from src.utils import set_seed


TEMPERATURES_DEFAULT = (0.5, 0.8, 1.0)
N_UNCONDITIONAL = 10
TOP_P_DEFAULT = 0.9
MAX_NEW_TOKENS_DEFAULT = 1024
EOT = "<|endoftext|>"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _build_model_from_cfg(model_cfg: ModelConfig, parameterization: str) -> torch.nn.Module:
    if parameterization == "mup":
        from src.mup_model import build_mup_gpt
        return build_mup_gpt(model_cfg)
    return GPT(model_cfg)


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, TrainConfig]:
    """Load a torch.save(...) checkpoint produced by src/train.py."""
    raw = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = raw["config"]
    # Rebuild dataclasses from the saved dict.
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
    model = _build_model_from_cfg(cfg.model, cfg.parameterization).to(device)
    model.load_state_dict(raw["model"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    eot_id: int,
    device: torch.device,
) -> str:
    """Top-p (nucleus) + temperature sampling. Stops at EOT or max_new_tokens."""
    ids = tokenizer.encode(prompt).ids
    if not ids:
        ids = [tokenizer.encode("<").ids[0]]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    block_size = model.cfg.block_size

    for _ in range(max_new_tokens):
        x_in = x if x.size(1) <= block_size else x[:, -block_size:]
        out = model(x_in)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Top-p (nucleus) filter.
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = cum > top_p
        # Always keep at least the top token.
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        probs = F.softmax(sorted_logits, dim=-1)
        sample = torch.multinomial(probs, num_samples=1)
        next_id = sorted_idx.gather(-1, sample)

        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == eot_id:
            break

    return tokenizer.decode(x[0].tolist())


# ---------------------------------------------------------------------------
# Rendering (best-effort)
# ---------------------------------------------------------------------------

def render_to_png(svg: str, png_path: Path, size: int = 128) -> bool:
    """Render via the multi-backend helper in src.render_utils (tries
    cairosvg first, then svglib+reportlab, then placeholder)."""
    from src.render_utils import render_svg_to_png
    return render_svg_to_png(svg, png_path, size=size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_prefixes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 10 generation.")
    p.add_argument("--checkpoint", required=True, help="path to model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--prefixes", default="configs/prefixes.txt")
    p.add_argument("--output-dir", default=None,
                   help="default: runs/samples/{run_id}")
    p.add_argument("--n-unconditional", type=int, default=N_UNCONDITIONAL)
    p.add_argument("--temperatures", nargs="+", type=float, default=list(TEMPERATURES_DEFAULT))
    p.add_argument("--top-p", type=float, default=TOP_P_DEFAULT)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-render", action="store_true",
                   help="skip cairosvg rendering even if available")
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

    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id(EOT)
    if eot_id is None:
        print(f"ERROR: tokenizer missing {EOT}", file=sys.stderr)
        return 1

    # Output dir.
    run_id = ckpt_path.parent.name  # checkpoints/{run_id}/model.pt
    out_dir = Path(args.output_dir or f"runs/samples/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    set_seed(args.seed)
    prefixes = _read_prefixes(Path(args.prefixes))
    print(f"Loaded {len(prefixes)} prefixes from {args.prefixes}")

    manifest: list[dict] = []
    render_ok_count = 0
    render_fail_count = 0

    # ---- Unconditional samples from "<svg" ----
    for t_i, temp in enumerate(args.temperatures):
        for k in range(args.n_unconditional):
            print(f"  uncond  T={temp}  k={k}", flush=True)
            svg = generate(
                model, tok, "<svg",
                temperature=temp, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, eot_id=eot_id,
                device=device,
            )
            stem = f"uncond_T{temp:.1f}_{k:02d}"
            (out_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")
            ok = False if args.no_render else render_to_png(svg, png_dir / f"{stem}.png")
            render_ok_count += int(ok)
            render_fail_count += int(not ok)
            manifest.append({
                "kind": "unconditional", "prefix": "<svg", "temperature": temp,
                "idx": k, "stem": stem, "render_ok": ok, "n_chars": len(svg),
            })

    # ---- Prefix-conditioned samples ----
    for p_i, prefix in enumerate(prefixes):
        for temp in args.temperatures:
            print(f"  prefix  T={temp}  p={p_i}", flush=True)
            svg = generate(
                model, tok, prefix,
                temperature=temp, top_p=args.top_p,
                max_new_tokens=args.max_new_tokens, eot_id=eot_id,
                device=device,
            )
            stem = f"prefix{p_i:02d}_T{temp:.1f}"
            (out_dir / f"{stem}.svg").write_text(svg, encoding="utf-8")
            ok = False if args.no_render else render_to_png(svg, png_dir / f"{stem}.png")
            render_ok_count += int(ok)
            render_fail_count += int(not ok)
            manifest.append({
                "kind": "prefix", "prefix": prefix, "temperature": temp,
                "idx": p_i, "stem": stem, "render_ok": ok, "n_chars": len(svg),
            })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "parameterization": cfg.parameterization,
        "n_params": n_params,
        "temperatures": list(args.temperatures),
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "render_ok": render_ok_count,
        "render_fail": render_fail_count,
        "samples": manifest,
    }, indent=2), encoding="utf-8")

    n_samples = len(manifest)
    print()
    print(f"Generated {n_samples} samples ({render_ok_count} rendered, "
          f"{render_fail_count} placeholder).")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
