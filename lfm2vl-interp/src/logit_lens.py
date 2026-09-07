"""Logit lens for LFM2.5-VL (paper Section 4.1).

Decode the residual stream at every position and layer through the unembedding:
``lm_head(embedding_norm(h))``. On LLaVA the paper found that visual-token positions
increasingly decode to text tokens describing the patch's contents, peaking in the
mid-to-late layers.

Two things differ from ``llava-interp-main/src/lvlm_lens.py``:

* **The final norm is already applied.** ``Lfm2Model`` runs ``embedding_norm`` after its
  layer loop, and the last entry of ``output_hidden_states`` is the *normed* result --
  verified by ``lm_head(hidden_states[-1]) == logits``. Applying ``embedding_norm``
  to it again, as the original does uniformly, silently double-norms the final layer.
  ``decode_positions`` skips the norm for that entry.
* **No fixed grid.** The original hard-codes token id 32000, 576 image tokens and a
  24x24 overlay. Here labels come from the real ``input_ids`` and the overlay is
  ``n_rows x n_cols`` on the resized, uncropped image.

This module also implements the paper's *quantitative* Section 4.1 result -- the
fraction of object-patch positions that decode to the object's class token -- which the
original repo never shipped.
"""

from __future__ import annotations

import base64
import html
import json
from io import BytesIO
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from geometry import TokenGrid, annotation_to_cells


def layer_logits(
    hidden_state: torch.Tensor,
    final_norm,
    lm_head,
    is_final_layer: bool,
    positions: Sequence[int] | None = None,
) -> torch.Tensor:
    """Unembed one layer's residual stream, returning ``(n_positions, vocab)`` logits.

    ``is_final_layer`` must be True for the last entry of ``output_hidden_states``,
    which ``Lfm2Model`` has already passed through ``embedding_norm``.
    """
    # no_grad matters here: lm_head's weights require grad, so without it every layer
    # builds a graph over a (positions x 128k vocab) matmul that is never backpropagated.
    with torch.no_grad():
        states = hidden_state[0]
        if positions is not None:
            states = states[list(positions)]
        if not is_final_layer:
            states = final_norm(states)
        return lm_head(states).float()


def decode_topk(
    hidden_states: Sequence[torch.Tensor],
    final_norm,
    lm_head,
    tokenizer,
    positions: Sequence[int] | None = None,
    top_k: int = 5,
) -> list[list[list[tuple[str, float]]]]:
    """``result[layer][position]`` -> list of ``(token_string, probability)``.

    Done one layer at a time so the full ``layers x positions x vocab`` tensor is never
    materialised (that would be ~4 GB for the real model).
    """
    n_layers = len(hidden_states)
    out: list[list[list[tuple[str, float]]]] = []

    for layer, state in enumerate(hidden_states):
        logits = layer_logits(
            state, final_norm, lm_head, is_final_layer=(layer == n_layers - 1), positions=positions
        )
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=top_k, dim=-1)

        layer_out = []
        for row in range(indices.shape[0]):
            layer_out.append(
                [
                    (tokenizer.decode([int(idx)]), float(prob))
                    for idx, prob in zip(indices[row], values[row])
                ]
            )
        out.append(layer_out)
        del logits, probs

    return out


def top1_token_ids(
    hidden_states: Sequence[torch.Tensor],
    final_norm,
    lm_head,
    positions: Sequence[int],
) -> torch.Tensor:
    """``(n_layers, n_positions)`` tensor of argmax token ids. Cheaper than ``decode_topk``."""
    n_layers = len(hidden_states)
    rows = []
    for layer, state in enumerate(hidden_states):
        logits = layer_logits(
            state, final_norm, lm_head, is_final_layer=(layer == n_layers - 1), positions=positions
        )
        rows.append(logits.argmax(dim=-1).cpu())
        del logits
    return torch.stack(rows)


# --------------------------------------------------------------------------------------
# Quantitative metric (paper Section 4.1)
# --------------------------------------------------------------------------------------


def _normalise(token: str) -> str:
    return token.strip().strip("Ġ▁").lower()


def class_match_ids(tokenizer, class_name: str) -> set[int]:
    """Token ids that count as a strict match for ``class_name``.

    Both the space-prefixed and bare forms of the class name and of its first word,
    since a byte-level BPE gives ``"dog"`` and ``" dog"`` different ids.
    """
    candidates = {class_name.lower(), class_name.lower().split()[0]}
    ids = set()
    for candidate in list(candidates):
        for form in (candidate, f" {candidate}"):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
    return ids


def is_lenient_match(token: str, class_name: str, min_prefix: int = 3) -> bool:
    """Whether a decoded token counts as naming the class under the loose criterion.

    The paper's Figure 3 credits partial word-pieces -- "swe"(ater), "diam"(ond) -- so a
    prefix of at least ``min_prefix`` characters counts. Reported alongside the strict
    criterion because the paper does not state which it used for the 23.7% figure.
    """
    normalised = _normalise(token)
    if len(normalised) < min_prefix:
        return False
    target = class_name.lower()
    return target.startswith(normalised) or target.split()[0].startswith(normalised)


def object_token_hit_rate(
    hidden_states: Sequence[torch.Tensor],
    final_norm,
    lm_head,
    tokenizer,
    grid: TokenGrid,
    annotation: dict,
    class_name: str,
    threshold: float = 0.0,
) -> dict:
    """Per-layer fraction of the object's visual tokens that decode to its class.

    This is the measurement behind the paper's "an average of 23.7% of the object patch
    token positions correspond to the correct object class token" at a best layer of
    25.7 out of 33.

    Returns a dict with per-layer strict and lenient rates, the best layer under each,
    and the number of object tokens the rates are computed over.
    """
    cells = annotation_to_cells(annotation, grid, threshold=threshold)
    positions = sorted(grid.cell_to_index(r, c) for r, c in cells)
    if not positions:
        return {"n_object_tokens": 0, "strict": [], "lenient": [], "best_layer_strict": None}

    ids = top1_token_ids(hidden_states, final_norm, lm_head, positions)
    match_ids = class_match_ids(tokenizer, class_name)

    strict_rates, lenient_rates = [], []
    for layer_ids in ids:
        strict = sum(1 for tid in layer_ids.tolist() if tid in match_ids)
        lenient = sum(
            1
            for tid in layer_ids.tolist()
            if tid in match_ids or is_lenient_match(tokenizer.decode([tid]), class_name)
        )
        strict_rates.append(strict / len(positions))
        lenient_rates.append(lenient / len(positions))

    return {
        "n_object_tokens": len(positions),
        "n_layers": len(hidden_states),
        "strict": strict_rates,
        "lenient": lenient_rates,
        "best_layer_strict": int(max(range(len(strict_rates)), key=strict_rates.__getitem__)),
        "best_rate_strict": max(strict_rates),
        "best_layer_lenient": int(max(range(len(lenient_rates)), key=lenient_rates.__getitem__)),
        "best_rate_lenient": max(lenient_rates),
    }


# --------------------------------------------------------------------------------------
# Interactive HTML
# --------------------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  body { margin: 0; padding: 0; font-family: -apple-system, Segoe UI, Arial, sans-serif; }
  .container { display: flex; align-items: flex-start; }
  .image-container { flex: 0 0 auto; margin: 20px; position: relative; }
  .image-container img { display: block; }
  .highlight-box { position: absolute; border: 2px solid red; pointer-events: none; display: none; }
  .table-container { flex: 1 1 auto; position: relative; max-height: 92vh; overflow: auto; margin: 20px; }
  table { border-collapse: separate; border-spacing: 0; }
  th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: center; min-width: 76px;
           font-size: 13px; white-space: nowrap; }
  th { background: #f2f2f2; font-weight: 600; }
  .corner-header { position: sticky; top: 0; left: 0; z-index: 3; background: #f2f2f2; }
  .row-header { position: sticky; left: 0; z-index: 2; background: #f2f2f2; text-align: right; }
  .col-header { position: sticky; top: 0; z-index: 1; background: #f2f2f2; }
  .img-row .row-header { background: #eef5ff; }
  #tooltip { display: none; position: fixed; background: #fff; border: 1px solid #333;
             padding: 6px 8px; z-index: 1000; pointer-events: none; font-size: 13px;
             box-shadow: 0 2px 8px rgba(0,0,0,.2); }
  .highlighted-row { background: #ffff99; }
  .meta { margin-top: 12px; font-size: 13px; max-width: 420px; line-height: 1.5; }
  .meta code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }
</style>
</head>
<body>
<div class="container">
  <div class="image-container">
    <img src="data:image/png;base64,__IMAGE__" alt="input" style="width:__DISPLAY_W__px;height:__DISPLAY_H__px;">
    <div class="highlight-box"></div>
    <div class="meta">
      <p><strong>Prompt:</strong> <code>__PROMPT__</code></p>
      <p><strong>Visual tokens:</strong> __ROWS__ x __COLS__ = __NTOKENS__ &nbsp;|&nbsp;
         <strong>Resized:</strong> __RESIZED_H__x__RESIZED_W__ px</p>
      <p>__MISC__</p>
      <p><em>Hover the image or a cell to inspect; click the image to lock.</em></p>
    </div>
  </div>
  <div class="table-container"><table id="lens"></table></div>
</div>
<div id="tooltip"></div>
<script>
const data = __DATA__;
const tokenLabels = __LABELS__;
const isImageToken = __ISIMG__;
const gridRows = __ROWS__, gridCols = __COLS__;
const displayW = __DISPLAY_W__, displayH = __DISPLAY_H__;
const firstImageRow = isImageToken.indexOf(true);

const tooltip = document.getElementById('tooltip');
const highlightBox = document.querySelector('.highlight-box');
const image = document.querySelector('.image-container img');
const table = document.getElementById('lens');
let isLocked = false, highlightedRow = null;

const header = table.createTHead().insertRow();
const corner = header.insertCell();
corner.textContent = 'Token / Layer';
corner.className = 'corner-header';
for (let i = 0; i < data.length; i++) {
  const th = document.createElement('th');
  th.textContent = (i === 0) ? 'embed' : ('L' + i);
  th.className = 'col-header';
  header.appendChild(th);
}

for (let pos = 0; pos < tokenLabels.length; pos++) {
  const row = table.insertRow();
  if (isImageToken[pos]) row.classList.add('img-row');
  const rh = row.insertCell();
  rh.textContent = tokenLabels[pos];
  rh.className = 'row-header';
  for (let layer = 0; layer < data.length; layer++) {
    const cell = row.insertCell();
    cell.textContent = data[layer][pos][0][0];
    cell.addEventListener('mouseover', (e) => { if (!isLocked) showTooltip(e, layer, pos, false); });
    cell.addEventListener('mousemove', moveTooltip);
    cell.addEventListener('mouseout', () => { if (!isLocked) hideTooltip(); });
  }
}

function showTooltip(e, layer, pos, scroll) {
  tooltip.innerHTML = data[layer][pos]
    .map(([t, p]) => escapeHtml(t) + ': ' + p.toFixed(4)).join('<br>');
  tooltip.style.display = 'block';
  moveTooltip(e);
  if (isImageToken[pos]) { highlightCell(pos - firstImageRow); highlightRow(pos, scroll); }
  else { highlightBox.style.display = 'none'; unhighlightRow(); }
}
function escapeHtml(s) {
  return s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function hideTooltip() {
  tooltip.style.display = 'none';
  if (!isLocked) { highlightBox.style.display = 'none'; unhighlightRow(); }
}
function moveTooltip(e) {
  const r = tooltip.getBoundingClientRect();
  let x = e.clientX + 12, y = e.clientY + 12;
  if (x + r.width > window.innerWidth) x = e.clientX - r.width - 12;
  if (y + r.height > window.innerHeight) y = e.clientY - r.height - 12;
  tooltip.style.left = Math.max(0, x) + 'px';
  tooltip.style.top = Math.max(0, y) + 'px';
}
function highlightCell(tokenIndex) {
  const r = Math.floor(tokenIndex / gridCols), c = tokenIndex % gridCols;
  highlightBox.style.left = (c * displayW / gridCols) + 'px';
  highlightBox.style.top = (r * displayH / gridRows) + 'px';
  highlightBox.style.width = (displayW / gridCols) + 'px';
  highlightBox.style.height = (displayH / gridRows) + 'px';
  highlightBox.style.display = 'block';
}
function highlightRow(pos, scroll) {
  unhighlightRow();
  highlightedRow = table.rows[pos + 1];
  if (!highlightedRow) return;
  highlightedRow.classList.add('highlighted-row');
  if (scroll) highlightedRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
}
function unhighlightRow() {
  if (highlightedRow) { highlightedRow.classList.remove('highlighted-row'); highlightedRow = null; }
}
function tokenIndexFromEvent(e) {
  const rect = image.getBoundingClientRect();
  const c = Math.floor((e.clientX - rect.left) / (rect.width / gridCols));
  const r = Math.floor((e.clientY - rect.top) / (rect.height / gridRows));
  if (r < 0 || r >= gridRows || c < 0 || c >= gridCols) return -1;
  return r * gridCols + c;
}
image.addEventListener('mousemove', (e) => {
  if (isLocked) return;
  const idx = tokenIndexFromEvent(e);
  if (idx < 0) return;
  highlightCell(idx);
  showTooltip(e, data.length - 1, firstImageRow + idx, true);
});
image.addEventListener('mouseout', () => { if (!isLocked) hideTooltip(); });
image.addEventListener('click', (e) => {
  isLocked = !isLocked;
  if (isLocked) {
    const idx = tokenIndexFromEvent(e);
    if (idx >= 0) { highlightCell(idx); highlightRow(firstImageRow + idx, true); }
  } else hideTooltip();
});
table.addEventListener('click', () => { isLocked = false; hideTooltip(); });
</script>
</body>
</html>
"""


def create_interactive_logit_lens(
    hidden_states: Sequence[torch.Tensor],
    final_norm,
    lm_head,
    tokenizer,
    input_ids: Sequence[int],
    image: Image.Image,
    grid: TokenGrid,
    prompt: str,
    output_path: str | Path,
    top_k: int = 5,
    max_display: int = 560,
    misc_text: str = "",
) -> Path:
    """Write a self-contained interactive logit-lens HTML for one image.

    The image is shown at its **resized** aspect ratio (what the vision tower sees),
    with an ``n_rows x n_cols`` overlay rather than LLaVA's fixed 24x24 square.
    """
    decoded = decode_topk(hidden_states, final_norm, lm_head, tokenizer, positions=None, top_k=top_k)

    labels, is_image = [], []
    image_seen = 0
    for position, token_id in enumerate(input_ids):
        if grid.start <= position < grid.end:
            row, col = divmod(image_seen, grid.n_cols)
            labels.append(f"IMG r{row:02d}c{col:02d}")
            is_image.append(True)
            image_seen += 1
        else:
            labels.append(tokenizer.decode([int(token_id)]))
            is_image.append(False)

    # Render at the resized aspect ratio, scaled to fit max_display.
    scale = max_display / max(grid.resized_w, grid.resized_h)
    display_w = int(round(grid.resized_w * scale))
    display_h = int(round(grid.resized_h * scale))
    preview = image.convert("RGB").resize((grid.resized_w, grid.resized_h), Image.LANCZOS)
    buffer = BytesIO()
    preview.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    replacements = {
        "__TITLE__": html.escape(Path(str(output_path)).stem),
        "__IMAGE__": encoded,
        "__DATA__": json.dumps(decoded),
        "__LABELS__": json.dumps(labels),
        "__ISIMG__": json.dumps(is_image),
        "__ROWS__": str(grid.n_rows),
        "__COLS__": str(grid.n_cols),
        "__NTOKENS__": str(grid.n_tokens),
        "__RESIZED_H__": str(grid.resized_h),
        "__RESIZED_W__": str(grid.resized_w),
        "__DISPLAY_W__": str(display_w),
        "__DISPLAY_H__": str(display_h),
        "__PROMPT__": html.escape(prompt),
        "__MISC__": html.escape(misc_text),
    }
    content = _HTML_TEMPLATE
    for key, value in replacements.items():
        content = content.replace(key, value)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
