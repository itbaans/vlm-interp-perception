"""Compact interactive logit-lens widget for the project page.

The generated lens pages are around 2 MB each because they carry the top five tokens and
probabilities for every position at every layer. Embedding several of those would make the
project page unusable. This extracts only what an at-a-glance widget needs: the image, the
grid shape, and the single top token per visual-token cell per layer. That is roughly
50 KB per example rather than 2 MB, so several fit comfortably.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def extract(path: Path) -> dict | None:
    """Pull the image, grid and per-layer top-1 tokens out of a generated lens page."""
    html = path.read_text(encoding="utf-8")

    grid = re.search(r"const gridRows = (\d+), gridCols = (\d+);", html)
    data_match = re.search(r"const data = (\[.*?\]);\n", html, re.S)
    flags_match = re.search(r"const isImageToken = (\[.*?\]);\n", html, re.S)
    image_match = re.search(r'src="(data:image/png;base64,[^"]+)"', html)
    if not all((grid, data_match, flags_match, image_match)):
        return None

    rows, cols = int(grid.group(1)), int(grid.group(2))
    data = json.loads(data_match.group(1))
    flags = json.loads(flags_match.group(1))

    # Positions of the visual tokens, in reading order.
    visual = [i for i, is_image in enumerate(flags) if is_image]
    if len(visual) != rows * cols:
        return None

    # data[layer][position][0][0] is the top token string at that position and layer.
    top1 = [[data[layer][pos][0][0] for pos in visual] for layer in range(len(data))]

    return {
        "id": path.stem.replace("_logit_lens", ""),
        "rows": rows,
        "cols": cols,
        "layers": len(data),
        "image": image_match.group(1),
        "top1": top1,
    }


def pick_examples(lens_dir: Path, results: dict | None, limit: int = 3) -> list[dict]:
    """Extract a few examples, preferring images where the effect is visible.

    ``results`` is the logit-lens results JSON. Images whose object tokens decode to the
    class name at some layer are chosen first, since an example that shows nothing would
    illustrate nothing.
    """
    pages = {p.stem.replace("_logit_lens", ""): p for p in lens_dir.glob("*_logit_lens.html")}
    if not pages:
        return []

    order = list(pages)
    if results:
        scored = []
        for image_id, record in results.get("images", {}).items():
            key = str(image_id).zfill(12)
            if key in pages:
                scored.append((record.get("best_rate_strict", 0.0), key))
        scored.sort(reverse=True)
        chosen = {key for _, key in scored}
        order = [key for _, key in scored] + [k for k in pages if k not in chosen]

    out = []
    for key in order:
        if len(out) >= limit:
            break
        extracted = extract(pages[key])
        if extracted:
            out.append(extracted)
    return out


WIDGET_CSS = """
.lens{border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:20px 0;
      background:var(--bg)}
.lens-controls{display:flex;align-items:center;gap:14px;margin-bottom:14px;flex-wrap:wrap}
.lens-controls input[type=range]{flex:1;min-width:220px;accent-color:var(--accent)}
.lens-layer{font-variant-numeric:tabular-nums;font-size:14px;color:var(--ink);
            min-width:118px}
.lens-tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
.lens-tabs button{font:inherit;font-size:13px;padding:4px 11px;border:1px solid var(--line);
                  background:var(--panel);color:var(--ink2);border-radius:6px;cursor:pointer}
.lens-tabs button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.lens-stage{position:relative;width:100%;max-width:560px;margin:0 auto}
.lens-stage img{width:100%;height:auto;display:block;border-radius:6px}
/* The overlay must not swallow events, but the cells themselves must receive them,
   otherwise nothing can be hovered. */
.lens-grid{position:absolute;inset:0;display:grid;pointer-events:none}
.lens-cell{pointer-events:auto;cursor:crosshair;display:flex;align-items:center;
           justify-content:center;overflow:hidden;font-size:8px;line-height:1;color:#fff;
           text-align:center;text-shadow:0 0 3px rgba(0,0,0,.95),0 0 6px rgba(0,0,0,.8);
           border:.5px solid rgba(255,255,255,.10)}
.lens-cell:hover{border:1.5px solid #eb6834;background:rgba(235,104,52,.22)}
.lens-readout{margin:12px auto 0;max-width:560px;min-height:60px;background:var(--panel);
              border:1px solid var(--line);border-radius:6px;padding:9px 12px;font-size:13px;
              color:var(--ink2)}
.lens-readout .hint{color:var(--ink3)}
.lens-readout .where{color:var(--ink);font-weight:640}
.lens-trace{margin-top:6px;display:flex;flex-wrap:wrap;gap:3px 7px;
            font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11.5px}
.lens-trace span{color:var(--ink3)}
.lens-trace span.now{color:#fff;background:var(--accent);border-radius:3px;padding:0 4px}
.lens-trace span b{color:var(--ink2);font-weight:400}
.lens-note{color:var(--ink3);font-size:13px;margin-top:10px;line-height:1.6}
"""

WIDGET_JS = """
(function(){
  const EX = __LENS_DATA__;
  const root = document.getElementById('lens-widget');
  if(!root || !EX.length) return;
  let current = 0, hovered = -1;

  const tabs = root.querySelector('.lens-tabs');
  const slider = root.querySelector('input[type=range]');
  const label = root.querySelector('.lens-layer');
  const stage = root.querySelector('.lens-stage');
  const readout = root.querySelector('.lens-readout');

  EX.forEach(function(ex,i){
    const b = document.createElement('button');
    b.textContent = 'image ' + (i+1);
    b.onclick = function(){ current = i; hovered = -1; draw(); };
    tabs.appendChild(b);
  });

  function esc(s){
    return String(s).replace(/[&<>"]/g, function(c){
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];
    });
  }

  function showReadout(){
    const ex = EX[current];
    if(hovered < 0){
      readout.innerHTML = '<span class="hint">Hover a square to see the token it decodes '+
        'to, and how that token changes across every layer.</span>';
      return;
    }
    const layer = +slider.value;
    const r = Math.floor(hovered / ex.cols), c = hovered % ex.cols;
    let trace = '';
    for(let L = 0; L < ex.layers; L++){
      const t = (ex.top1[L][hovered] || '').trim() || '_';
      trace += '<span class="' + (L === layer ? 'now' : '') + '">' +
               '<b>' + (L === 0 ? 'emb' : 'L'+L) + '</b> ' + esc(t) + '</span>';
    }
    readout.innerHTML =
      '<span class="where">row ' + r + ', column ' + c + '</span>' +
      ' decodes to <b>' + esc((ex.top1[layer][hovered]||'').trim() || '_') + '</b>' +
      ' at ' + (layer === 0 ? 'the embeddings' : 'layer ' + layer) +
      '<div class="lens-trace">' + trace + '</div>';
  }

  function draw(){
    const ex = EX[current];
    slider.max = ex.layers - 1;
    const layer = Math.min(+slider.value, ex.layers - 1);
    label.textContent = layer === 0
      ? 'embeddings (before layer 1)'
      : 'layer ' + layer + ' of ' + (ex.layers - 1);
    Array.prototype.forEach.call(tabs.children, function(b,i){
      b.className = i === current ? 'on' : '';
    });

    stage.innerHTML =
      '<img alt="COCO image with decoded tokens" src="' + ex.image + '">' +
      '<div class="lens-grid" style="grid-template-columns:repeat(' + ex.cols + ',1fr);' +
      'grid-template-rows:repeat(' + ex.rows + ',1fr)"></div>';
    const grid = stage.querySelector('.lens-grid');
    const row = ex.top1[layer];
    for(let i = 0; i < row.length; i++){
      const cell = document.createElement('div');
      cell.className = 'lens-cell';
      const t = (row[i] || '').trim();
      cell.textContent = t.length > 7 ? t.slice(0,7) : t;
      (function(index){
        cell.addEventListener('mouseenter', function(){ hovered = index; showReadout(); });
      })(i);
      grid.appendChild(cell);
    }
    showReadout();
  }

  stage.addEventListener('mouseleave', function(){ hovered = -1; showReadout(); });
  slider.addEventListener('input', draw);
  draw();
})();
"""


def widget_html(examples: list[dict]) -> str:
    """The markup for the widget. Data is injected as JSON, so nothing is fetched."""
    if not examples:
        return '<p class="missing">No logit-lens pages were found.</p>'
    default_layer = max(0, examples[0]["layers"] - 5)
    return (
        '<div class="lens" id="lens-widget">'
        '<div class="lens-tabs"></div>'
        '<div class="lens-controls">'
        f'<input type="range" min="0" max="{examples[0]["layers"] - 1}" '
        f'value="{default_layer}" step="1">'
        '<span class="lens-layer"></span>'
        "</div>"
        '<div class="lens-stage"></div>'
        '<div class="lens-readout"></div>'
        '<p class="lens-note">Each square is one visual token, showing the single token it '
        "decodes to at the selected layer. Move the slider from the embeddings (0) to the "
        "final layer. Hovering a square lists what it decodes to at every layer, so a single "
        "position can be followed through the network.</p>"
        "</div>"
    )
