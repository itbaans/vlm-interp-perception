"""Build an index.html linking every generated logit-lens page."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiment import load_config, resolve  # noqa: E402

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Logit lens overview</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 40px; max-width: 900px; }}
 h1 {{ font-size: 22px; }} p {{ color: #444; line-height: 1.6; }}
 ul {{ columns: 2; }} li {{ margin: 4px 0; }} a {{ color: #0b62d0; }}
</style></head><body>
<h1>Logit lens &mdash; {count} images</h1>
<p>Each page decodes the residual stream at every token position through
<code>lm_head(embedding_norm(h))</code>. Hover the image to inspect a visual token.</p>
<ul>{items}</ul>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--folder", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    folder = resolve(args.folder or config["paths"]["logit_lens_dir"])
    pages = sorted(p for p in folder.glob("*.html") if p.name != "index.html")
    if not pages:
        raise SystemExit(f"no logit-lens pages in {folder}")

    items = "".join(
        f'<li><a href="{html.escape(p.name)}">{html.escape(p.stem)}</a></li>' for p in pages
    )
    index = folder / "index.html"
    index.write_text(TEMPLATE.format(count=len(pages), items=items), encoding="utf-8")
    print(f"Wrote {index} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
