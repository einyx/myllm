#!/usr/bin/env python3
"""
Build the GitHub Pages site from WRITEUP.md.

    python scripts/12_build_site.py

Writes docs/index.html. GitHub Pages serves it from
https://einyx.github.io/myllm/ (the repo's Pages source is main/docs).

Keeping this a script rather than hand-written HTML means the write-up
and the site can never drift: WRITEUP.md is the single source of truth.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "WRITEUP.md"
OUT_DIR = ROOT / "docs"

CSS = """
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #5b6570;
  --rule: #e3e6ea;
  --accent: #0b6bcb;
  --code-bg: #f6f8fa;
  --quote-bg: #f6f8fa;
  --maxw: 44rem;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --fg: #e6edf3;
    --muted: #9198a1;
    --rule: #2b313a;
    --accent: #6cb6ff;
    --code-bg: #161b22;
    --quote-bg: #161b22;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
}
.wrap { max-width: var(--maxw); margin: 0 auto; padding: 3rem 1.25rem 5rem; }
header.top {
  border-bottom: 1px solid var(--rule);
  padding-bottom: 1.5rem;
  margin-bottom: 2.5rem;
}
header.top .kicker {
  font-size: .8rem; letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 .4rem;
}
header.top h1 { margin: 0 0 .5rem; font-size: 2rem; line-height: 1.25; }
header.top .sub { color: var(--muted); margin: 0; }
nav.links { margin-top: 1.25rem; display: flex; gap: 1.25rem; flex-wrap: wrap; }
nav.links a { color: var(--accent); text-decoration: none; font-size: .95rem; }
nav.links a:hover { text-decoration: underline; }
h2 { margin: 3rem 0 1rem; font-size: 1.45rem; line-height: 1.3;
     padding-top: .5rem; border-top: 1px solid var(--rule); }
h3 { margin: 2rem 0 .75rem; font-size: 1.12rem; }
p { margin: 0 0 1.1rem; }
a { color: var(--accent); }
ul, ol { margin: 0 0 1.1rem; padding-left: 1.4rem; }
li { margin-bottom: .4rem; }
code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
               "Liberation Mono", monospace;
  font-size: .875em;
  background: var(--code-bg);
  padding: .15em .35em;
  border-radius: 4px;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 1rem 1.1rem;
  overflow-x: auto;
  line-height: 1.5;
  margin: 0 0 1.3rem;
}
pre code { background: none; padding: 0; font-size: .84rem; }
blockquote {
  margin: 0 0 1.3rem;
  padding: .1rem 1.1rem;
  border-left: 3px solid var(--accent);
  background: var(--quote-bg);
  color: var(--muted);
}
blockquote p:last-child { margin-bottom: 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0 0 1.4rem;
  font-size: .94rem;
  display: block;
  overflow-x: auto;
}
th, td { border: 1px solid var(--rule); padding: .45rem .7rem; text-align: left; }
th { background: var(--code-bg); font-weight: 600; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.5rem 0; }
footer {
  margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: .88rem;
}
footer a { color: var(--accent); }
"""


def build() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found")

    text = SRC.read_text()

    # Pull the first H1 and the first paragraph out for the page header,
    # so the write-up file itself stays a plain, readable markdown doc.
    title = "Building a language model from scratch, on a CPU, in 4,400 lines"
    m = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if m:
        title = m.group(1).strip()

    body = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "attr_list", "sane_lists"],
    )

    # Drop the leading H1 from the body (the header renders it already).
    body = re.sub(r"^\s*<h1[^>]*>.*?</h1>\s*", "", body, flags=re.DOTALL)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="Building a byte-level BPE tokenizer, a
decoder-only transformer, and a training loop from scratch in PyTorch on a
CPU: what breaks, and what it teaches.">
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="wrap">
<header class="top">
  <p class="kicker">Learning project &middot; PyTorch &middot; no GPU</p>
  <h1>{title}</h1>
  <nav class="links">
    <a href="https://github.com/einyx/myllm">Source code</a>
    <a href="https://github.com/einyx/myllm#readme">README</a>
    <a href="https://huggingface.co/einyx/myllm-small">Model on HuggingFace</a>
  </nav>
</header>
<main>
{body}
</main>
<footer>
  Built from scratch by <a href="https://github.com/einyx">einyx</a>.
  Source, training data, and model card:
  <a href="https://github.com/einyx/myllm">github.com/einyx/myllm</a>
</footer>
</div>
</body>
</html>
"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.html").write_text(html)
    (OUT_DIR / "style.css").write_text(CSS)
    print(f"[site] wrote {OUT_DIR / 'index.html'} ({len(html) / 1024:.1f} KB)")
    print(f"[site] wrote {OUT_DIR / 'style.css'}")


if __name__ == "__main__":
    sys.exit(build())
