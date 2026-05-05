"""SVG to PNG rendering utilities with cross-platform fallbacks.

Tries renderers in order:
    1. cairosvg           - preferred on Linux/macOS (needs system libcairo)
    2. resvg-py           - pure-Rust fallback, works on Windows
    3. svglib + reportlab - pure-Python last resort

Each call returns True on success. On total failure, writes a small placeholder
PNG so analysis-notebook grids stay shape-uniform.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

# Lazy imports so this module loads even when none of the renderers are
# installed, callers detect availability via ``available_renderer()``.

_CAIRO_OK: bool | None = None
_SVGLIB_OK: bool | None = None
_RESVG_OK:  bool | None = None
_CAIRO_ERR: str | None = None
_SVGLIB_ERR: str | None = None
_RESVG_ERR:  str | None = None


def _check_resvg() -> bool:
    """resvg-py: pure-Rust binding, no system libs needed. Recommended on Windows."""
    global _RESVG_OK, _RESVG_ERR
    if _RESVG_OK is not None:
        return _RESVG_OK
    try:
        import resvg_py
        png = resvg_py.svg_to_bytes(
            svg_string="<svg xmlns='http://www.w3.org/2000/svg'><rect width='4' height='4'/></svg>",
            width=4, height=4,
        )
        _RESVG_OK = bool(png) and len(png) > 0
        if not _RESVG_OK:
            _RESVG_ERR = "svg_to_bytes returned empty"
    except Exception as e:
        _RESVG_OK = False
        _RESVG_ERR = f"{type(e).__name__}: {e}"
    return _RESVG_OK


def _check_cairosvg() -> bool:
    global _CAIRO_OK, _CAIRO_ERR
    if _CAIRO_OK is not None:
        return _CAIRO_OK
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='4' height='4'/></svg>",
            output_width=4, output_height=4,
        )
        _CAIRO_OK = True
    except Exception as e:
        _CAIRO_OK = False
        _CAIRO_ERR = f"{type(e).__name__}: {e}"
    return _CAIRO_OK


def _check_svglib() -> bool:
    global _SVGLIB_OK, _SVGLIB_ERR
    if _SVGLIB_OK is not None:
        return _SVGLIB_OK
    try:
        from svglib.svglib import svg2rlg  # noqa: F401
        from reportlab.graphics import renderPM  # noqa: F401
        # Smoke-test an actual render, Windows reportlab installs sometimes
        # import OK but fail on the first drawToString call (missing freetype
        # / _renderPM binary).
        from io import BytesIO
        d = svg2rlg(BytesIO(b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='4' height='4'/></svg>"))
        renderPM.drawToString(d, fmt="PNG")
        _SVGLIB_OK = True
    except Exception as e:
        _SVGLIB_OK = False
        _SVGLIB_ERR = f"{type(e).__name__}: {e}"
    return _SVGLIB_OK


def renderer_diagnostics() -> str:
    """Returns a multi-line description of which renderers are available and
    why each is or isn't usable. Call this when no renderer is found."""
    _check_cairosvg(); _check_resvg(); _check_svglib()
    out = []
    out.append(f"  cairosvg: {'OK' if _CAIRO_OK  else 'unavailable - ' + (_CAIRO_ERR  or 'unknown')}")
    out.append(f"  resvg:    {'OK' if _RESVG_OK  else 'unavailable - ' + (_RESVG_ERR  or 'unknown')}")
    out.append(f"  svglib:   {'OK' if _SVGLIB_OK else 'unavailable - ' + (_SVGLIB_ERR or 'unknown')}")
    return "\n".join(out)


def available_renderer() -> str | None:
    """Preference: cairosvg > resvg > svglib. Returns the first that works."""
    if _check_cairosvg():
        return "cairosvg"
    if _check_resvg():
        return "resvg"
    if _check_svglib():
        return "svglib"
    return None


def _render_with_resvg(svg: str, png_path: Path, size: int) -> bool:
    try:
        import resvg_py
        png_bytes = resvg_py.svg_to_bytes(svg_string=svg, width=size, height=size)
        if not png_bytes:
            return False
        png_path.write_bytes(bytes(png_bytes))
        return True
    except Exception:
        return False


def _render_with_cairosvg(svg: str, png_path: Path, size: int) -> bool:
    try:
        import cairosvg
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(png_path),
            output_width=size,
            output_height=size,
        )
        return True
    except Exception:
        return False


def _render_with_svglib(svg: str, png_path: Path, size: int) -> bool:
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(BytesIO(svg.encode("utf-8")))
        if drawing is None:
            return False
        # Scale the drawing to roughly target_size on the longer side.
        if drawing.width and drawing.height:
            scale = size / max(drawing.width, drawing.height)
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)
        renderPM.drawToFile(drawing, str(png_path), fmt="PNG")
        return True
    except Exception:
        return False


def _placeholder(png_path: Path, size: int, label: str = "render fail") -> None:
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (size, size), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        # Best-effort small text via default font.
        draw.text((size // 6, size // 2 - 4), label, fill=(120, 120, 120))
        img.save(png_path)
    except Exception:
        # Give up silently, caller still gets render_ok=False.
        pass


def render_svg_to_png(svg: str, png_path: Path, size: int = 128) -> bool:
    """Render ``svg`` to ``png_path`` at ``size``x``size``. Returns True on
    success. On failure writes a placeholder PNG (gray with label)."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if _check_cairosvg() and _render_with_cairosvg(svg, png_path, size):
        return True
    if _check_resvg() and _render_with_resvg(svg, png_path, size):
        return True
    if _check_svglib() and _render_with_svglib(svg, png_path, size):
        return True
    _placeholder(png_path, size)
    return False
