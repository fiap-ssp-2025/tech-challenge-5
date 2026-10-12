"""Builds docs/relatorio_tecnico.html and docs/relatorio_tecnico.pdf from docs/relatorio_tecnico.md.

Markdown → HTML with python-markdown (tables, attr_list, md_in_html); PDF with headless Chrome,
which is the only renderer that handles the inline SVG diagrams and the print CSS without extra packages.

    uv run --with markdown scripts/relatorio_pdf.py [--no-pdf]
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "relatorio_tecnico.md"
HTML_OUT = ROOT / "docs" / "relatorio_tecnico.html"
PDF_OUT = ROOT / "docs" / "relatorio_tecnico.pdf"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser", "chrome",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
:root { --purple: #925C98; --purple-d: #5E3A63; --ink: #2B1F2E; --muted: #6B5E6E; --line: #D9CFDB; --lilac: #F3EDF6; }
* { box-sizing: border-box; }
html { font-size: 10.6pt; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--ink); line-height: 1.45;
       max-width: 180mm; margin: 0 auto; padding: 0; }
h1 { font-size: 24pt; color: var(--purple-d); margin: 0 0 6pt; line-height: 1.15; }
h2 { font-size: 15pt; color: var(--purple-d); margin: 22pt 0 8pt; padding-bottom: 3pt; border-bottom: 1.5px solid var(--purple);
     break-after: avoid; }
h3 { font-size: 12pt; color: var(--purple-d); margin: 14pt 0 6pt; break-after: avoid; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
p.lead { color: var(--muted); font-size: 10pt; }
hr { border: 0; border-top: 1px solid var(--line); margin: 10pt 0; }
a { color: var(--purple-d); text-decoration: none; }
strong { color: var(--purple-d); }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt; background: var(--lilac); padding: 0 3px; border-radius: 3px; }
pre { background: var(--lilac); border: 1px solid var(--line); border-radius: 6px; padding: 8pt 10pt; font-size: 9pt; line-height: 1.4;
      overflow: hidden; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt; font-size: 9.3pt; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
th, td { border: 1px solid var(--line); padding: 4pt 6pt; vertical-align: top; text-align: left; }
th { background: var(--lilac); color: var(--purple-d); font-weight: 600; }
td[align=right], th[align=right] { text-align: right; }
tr:nth-child(even) td { background: #FBF9FC; }
figure { margin: 10pt 0 14pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; }
figure.pair { display: flex; flex-wrap: wrap; gap: 8pt; justify-content: center; }
figure.pair img { width: calc(50% - 4pt); object-fit: contain; }
figcaption { flex-basis: 100%; font-size: 9pt; color: var(--muted); margin-top: 5pt; text-align: left; }
blockquote { margin: 0 0 8pt; padding: 6pt 10pt; border-left: 3px solid var(--purple); background: var(--lilac); }
.cover { margin-bottom: 14pt; }
.cover p { margin: 2pt 0; }
@media print {
  h2 { break-before: auto; }
  figure img { max-height: 100mm; }
  figure.pair img { max-height: 96mm; }
}
"""


def build_html(md_text: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "attr_list", "md_in_html", "sane_lists", "fenced_code"])
    # the first lines after the title are the cover block
    body = re.sub(r"(</h1>\s*)<p>", r"\1<div class='cover'><p>", body, count=1)
    body = re.sub(r"</p>\n(<h2>)", r"</p></div>\n\1", body, count=1)
    # one or two image paragraphs followed by an italic "Figura N" paragraph → <figure> (two images = side by side)
    def figure(m: re.Match) -> str:
        imgs = re.findall(r"<img[^>]*>", m.group(1))
        cls = " class='pair'" if len(imgs) == 2 else ""
        return f"<figure{cls}>{''.join(imgs)}<figcaption>{m.group(2)}</figcaption></figure>"
    body = re.sub(r"((?:<p><img[^>]*></p>\s*){1,2})<p><em>(Figura \d+[^<]*)</em></p>", figure, body)
    title = re.search(r"<h1>(.*?)</h1>", body)
    return (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<title>{html.escape(re.sub('<.*?>', '', title.group(1)) if title else 'Relatório')}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


def main() -> int:
    HTML_OUT.write_text(build_html(SRC.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"html → {HTML_OUT.relative_to(ROOT)}")
    if "--no-pdf" in sys.argv:
        return 0
    chrome = find_chrome()
    if not chrome:
        print("Chrome/Chromium não encontrado: abra o HTML no navegador e imprima em PDF.", file=sys.stderr)
        return 1
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=10000", f"--print-to-pdf={PDF_OUT}", HTML_OUT.resolve().as_uri()]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"pdf  → {PDF_OUT.relative_to(ROOT)} ({PDF_OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
