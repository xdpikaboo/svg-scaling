"""Convert report/report.html to report/report.pdf via playwright.

Bypasses ``jupyter nbconvert --to webpdf``, which on Windows dispatches the
playwright call from a thread where ``asyncio`` falls back to the
``WindowsSelectorEventLoop`` and can no longer spawn subprocesses (the
NotImplementedError users hit). We just call playwright directly from the
main thread with the Proactor policy.

Prerequisites:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# On Windows, force the Proactor policy so subprocess_exec works.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def _render(html_path: Path, pdf_path: Path) -> None:
    from playwright.async_api import async_playwright
    url = html_path.resolve().as_uri()
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            await page.pdf(
                path=str(pdf_path),
                format="Letter",
                print_background=True,
                margin={"top": "0.5in", "bottom": "0.5in",
                        "left": "0.6in", "right": "0.6in"},
            )
        finally:
            await browser.close()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    html = root / "report" / "report.html"
    pdf  = root / "report" / "report.pdf"
    if not html.exists():
        print(f"ERROR: {html} not found.\n  Run first:\n"
              f"    jupyter nbconvert --to html --no-input report/report.ipynb",
              file=sys.stderr)
        return 1
    print(f"  source: {html}")
    print(f"  target: {pdf}")
    asyncio.run(_render(html, pdf))
    print(f"\nWrote {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
