"""Per-sample inspection viewer for the perception experiments.

A synthetic-data experiment fails silently if the scenes are not what you think they are,
or if the shape-to-token mapping is off by one cell. This builds a single self-contained
HTML file that walks every sample through the pipeline, and it works **before any GPU time
is spent** -- generate locally, open the viewer, confirm the scenes and the token mapping,
then run the model.

Two modes, same file:

* **Dataset mode** (``results=None``) -- the minimal pair side by side with the changed
  shape outlined, a toggleable 16x16 token-grid overlay, per-shape token cells, the
  patching token groups, the exact prompt, and the tokenized answer id.
* **Results mode** -- the same cards enriched with whatever stages have run: the baseline
  prediction and answer distribution, the per-layer answer-emergence curve, and the
  patching recovery heatmap.

Overlays are drawn as SVG on top of the image rather than baked into the PNG, so toggling
is instant and each scene is embedded exactly once.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

import synthetic as S

# Same design tokens as src/report.py and src/plots.py.
GROUP_COLOURS = {
    "changed": "#eb6834",
    "targets": "#1baf7a",
    "distractors": "#2a78d6",
    "background": "#8a8983",
}


def _png_data_uri(image: Image.Image, max_side: int = 512) -> str:
    if max(image.size) > max_side:
        image = image.resize((max_side, max_side), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _cells(cells: Iterable[tuple[int, int]]) -> list[list[int]]:
    return sorted([int(r), int(c)] for r, c in cells)


def token_groups(sample: S.Sample, grid) -> dict[str, list[list[int]]]:
    """The token groups activation patching splices, as grid cells.

    Kept here rather than in the patching module so the viewer can show exactly what the
    experiment will patch, before the experiment runs.
    """
    per_shape = S.scene_cells(sample.clean, grid)
    changed = per_shape[sample.changed_index]

    query = sample.query or {}
    if sample.task == "count":
        target_indices = sample.clean.indices_of(
            colour=query.get("colour"), kind=query.get("kind")
        )
    elif sample.task == "colour":
        target_indices = sample.clean.indices_of(kind=query.get("kind"))
    elif sample.task == "shape":
        target_indices = sample.clean.indices_of(colour=query.get("colour"))
    elif sample.task == "relation":
        target_indices = [query.get("subject_index"), query.get("object_index")]
    else:
        target_indices = [sample.changed_index]
    target_indices = [i for i in target_indices if i is not None]

    targets: set[tuple[int, int]] = set()
    for index in target_indices:
        if index != sample.changed_index:
            targets |= per_shape[index]

    distractors: set[tuple[int, int]] = set()
    for index, cells in enumerate(per_shape):
        if index != sample.changed_index and index not in target_indices:
            distractors |= cells

    return {
        "changed": _cells(changed),
        "targets": _cells(targets - changed),
        "distractors": _cells(distractors - changed - targets),
        "background": _cells(S.background_cells(sample.clean, grid)),
    }


def sample_record(sample: S.Sample, grid, processor=None, result: dict | None = None) -> dict:
    """Everything one card needs, JSON-serialisable."""
    import tasks as T

    task = T.TASKS[sample.task]
    per_shape = S.scene_cells(sample.clean, grid)

    record = {
        "id": sample.sample_id,
        "task": sample.task,
        "level": sample.level,
        "answer": sample.answer,
        "answer_cf": sample.answer_cf,
        "changed_index": sample.changed_index,
        "question": task.question(sample),
        "prefill": task.prefill(sample),
        "answer_style": task.answer_style,
        "clean_png": _png_data_uri(sample.clean.render()),
        "cf_png": _png_data_uri(sample.counterfactual.render()),
        "grid": [grid.n_rows, grid.n_cols],
        "groups": token_groups(sample, grid),
        "shapes": [
            {
                "kind": s.kind,
                "colour": s.colour,
                "cx": s.cx,
                "cy": s.cy,
                "size": s.size,
                "direction": s.direction,
                "changed": i == sample.changed_index,
                "bbox": list(s.bbox),
                "cells": _cells(per_shape[i]),
            }
            for i, s in enumerate(sample.clean.shapes)
        ],
    }

    if processor is not None:
        try:
            record["answer_token_id"] = task.answer_token_id(processor, sample.answer)
            record["answer_text"] = task.answer_text(sample.answer)
        except ValueError as exc:  # surfaced on the card rather than crashing the build
            record["answer_token_error"] = str(exc)

    if result:
        record["result"] = result
    return record


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --surface:#fcfcfb; --surface-2:#f5f4f1; --line:#e6e5e1;
        --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#8a8983;
        --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--surface); color:var(--ink);
       font:14px/1.6 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1200px; margin:0 auto; padding:32px 24px 80px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }
.sub { color:var(--ink-3); font-size:13.5px; margin:0 0 20px; }
code { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12.5px;
       background:var(--surface-2); padding:1px 5px; border-radius:3px; }

.controls { position:sticky; top:0; z-index:20; background:var(--surface);
            border-bottom:1px solid var(--line); padding:12px 0 14px; margin-bottom:20px;
            display:flex; flex-wrap:wrap; gap:16px; align-items:center; }
.controls label { font-size:12.5px; color:var(--ink-2); display:flex; gap:6px;
                  align-items:center; }
select { font:inherit; font-size:13px; padding:4px 8px; border:1px solid var(--line);
         border-radius:6px; background:var(--surface); color:var(--ink); }
.count { color:var(--ink-3); font-size:12.5px; margin-left:auto; }

.card { border:1px solid var(--line); border-radius:12px; padding:18px 20px;
        margin-bottom:20px; background:var(--surface); }
.card h2 { font-size:15px; margin:0 0 2px; letter-spacing:0; }
.card .meta { color:var(--ink-3); font-size:12.5px; margin-bottom:14px; }
.pill { display:inline-block; padding:1px 8px; border-radius:99px; font-size:11.5px;
        border:1px solid var(--line); background:var(--surface-2); margin-right:5px; }
.pill.ok { border-color:var(--aqua); color:#137a55; }
.pill.bad { border-color:var(--orange); color:#a8431d; }

.panes { display:flex; gap:18px; flex-wrap:wrap; }
.pane { position:relative; width:340px; }
.pane .label { font-size:12px; color:var(--ink-3); margin-bottom:5px; }
.stage { position:relative; width:340px; height:340px; }
.stage img { width:340px; height:340px; display:block; border:1px solid var(--line);
             border-radius:8px; }
.stage svg { position:absolute; inset:0; width:340px; height:340px; pointer-events:none; }

.info { flex:1 1 320px; min-width:300px; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th,td { padding:4px 8px; text-align:left; border-bottom:1px solid var(--line);
        white-space:nowrap; }
th { color:var(--ink-3); font-weight:600; font-size:11px; text-transform:uppercase;
     letter-spacing:.04em; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
tr.changed td { background:#fdf0ea; font-weight:600; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px;
          margin-right:5px; vertical-align:-1px; border:1px solid rgba(0,0,0,.15); }
.legend { font-size:12px; color:var(--ink-2); margin:10px 0 0; }
.legend span { margin-right:12px; }
.bars { display:flex; align-items:flex-end; gap:3px; height:46px; margin-top:6px; }
.bars div { flex:1; background:var(--blue); border-radius:2px 2px 0 0; min-height:1px; }
.bars div.true { background:var(--aqua); }
.barlab { display:flex; gap:3px; font-size:10px; color:var(--ink-3); }
.barlab span { flex:1; text-align:center; }
.empty { color:var(--ink-3); font-style:italic; padding:40px 0; text-align:center; }
</style></head><body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <p class="sub">__SUBTITLE__</p>

  <div class="controls">
    <label>Task <select id="f-task"><option value="">all</option></select></label>
    <label>Level <select id="f-level"><option value="">all</option></select></label>
    <label>Result <select id="f-correct">
      <option value="">all</option><option value="ok">correct</option>
      <option value="bad">incorrect</option></select></label>
    <label>Overlay <select id="f-overlay">
      <option value="none">none</option>
      <option value="grid">token grid</option>
      <option value="shapes">shape cells</option>
      <option value="groups">patch groups</option>
    </select></label>
    <span class="count" id="count"></span>
  </div>

  <div id="cards"></div>
</div>
<script>
const DATA = __DATA__;
const GROUP_COLOURS = __GROUPS__;

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

for (const [id, key] of [['f-task','task'], ['f-level','level']]) {
  [...new Set(DATA.map(d => d[key]))].sort().forEach(v => {
    const o = document.createElement('option'); o.value = v; o.textContent = v;
    $(id).appendChild(o);
  });
}

const SIZE = 340;

function overlaySvg(d, mode) {
  if (mode === 'none') return '';
  const [rows, cols] = d.grid, cw = SIZE / cols, ch = SIZE / rows;
  let out = '';
  if (mode === 'grid') {
    for (let c = 1; c < cols; c++)
      out += `<line x1="${c*cw}" y1="0" x2="${c*cw}" y2="${SIZE}"
               stroke="#0b0b0b" stroke-opacity=".16" stroke-width="1"/>`;
    for (let r = 1; r < rows; r++)
      out += `<line x1="0" y1="${r*ch}" x2="${SIZE}" y2="${r*ch}"
               stroke="#0b0b0b" stroke-opacity=".16" stroke-width="1"/>`;
  } else if (mode === 'shapes') {
    d.shapes.forEach(s => {
      const fill = s.changed ? GROUP_COLOURS.changed : GROUP_COLOURS.distractors;
      s.cells.forEach(([r, c]) => {
        out += `<rect x="${c*cw}" y="${r*ch}" width="${cw}" height="${ch}"
                 fill="${fill}" fill-opacity=".35"/>`;
      });
    });
  } else if (mode === 'groups') {
    for (const [name, cells] of Object.entries(d.groups)) {
      if (name === 'background') continue;
      cells.forEach(([r, c]) => {
        out += `<rect x="${c*cw}" y="${r*ch}" width="${cw}" height="${ch}"
                 fill="${GROUP_COLOURS[name]}" fill-opacity=".38"/>`;
      });
    }
  }
  return out;
}

function changedOutline(d) {
  const s = d.shapes[d.changed_index];
  if (!s) return '';
  const k = SIZE / 512, [x0, y0, x1, y1] = s.bbox;
  return `<rect x="${x0*k-3}" y="${y0*k-3}" width="${(x1-x0)*k+6}"
           height="${(y1-y0)*k+6}" fill="none" stroke="${GROUP_COLOURS.changed}"
           stroke-width="2.5" rx="4"/>`;
}

function distributionBars(r) {
  if (!r || !r.distribution) return '';
  const entries = Object.entries(r.distribution);
  const max = Math.max(...entries.map(([, v]) => v), 1e-9);
  const bars = entries.map(([k, v]) =>
    `<div class="${k === String(r.expected) ? 'true' : ''}"
      style="height:${Math.max(1, 100*v/max)}%" title="${esc(k)}: ${v.toFixed(3)}"></div>`
  ).join('');
  const labs = entries.map(([k]) => `<span>${esc(k)}</span>`).join('');
  return `<div class="legend">Answer distribution (green = correct)</div>
          <div class="bars">${bars}</div><div class="barlab">${labs}</div>`;
}

function emergence(r) {
  if (!r || !r.answer_by_layer) return '';
  const v = r.answer_by_layer, max = Math.max(...v, 1e-9);
  const w = 300, h = 44;
  const pts = v.map((y, i) =>
    `${(i/(v.length-1))*w},${h - (y/max)*h}`).join(' ');
  const peak = v.indexOf(Math.max(...v));
  return `<div class="legend">P(answer) by layer — peak L${peak}</div>
    <svg width="${w}" height="${h+6}" style="position:static;display:block">
      <polyline points="${pts}" fill="none" stroke="${GROUP_COLOURS.distractors}"
        stroke-width="2"/>
      <circle cx="${(peak/(v.length-1))*w}" cy="${h - (v[peak]/max)*h}" r="3.5"
        fill="${GROUP_COLOURS.distractors}"/>
    </svg>`;
}

function card(d) {
  const r = d.result;
  let verdict = '';
  if (r && r.predicted !== undefined) {
    const ok = r.correct;
    verdict = `<span class="pill ${ok ? 'ok' : 'bad'}">predicted
      <code>${esc(r.predicted)}</code> ${ok ? 'correct' : 'wrong'}</span>`;
  }
  const tokenInfo = d.answer_token_error
    ? `<span class="pill bad">${esc(d.answer_token_error)}</span>`
    : (d.answer_token_id !== undefined
        ? `<span class="pill">answer <code>${esc(d.answer_text)}</code>
           id ${d.answer_token_id}</span>` : '');

  const rows = d.shapes.map((s, i) => `
    <tr class="${s.changed ? 'changed' : ''}">
      <td>${i}</td>
      <td><span class="swatch" style="background:${colourHex(s.colour)}"></span>${esc(s.colour)}</td>
      <td>${esc(s.kind)}${s.kind === 'arrow' || s.kind === 'triangle'
            ? ' ' + esc(s.direction) : ''}</td>
      <td class="num">${s.cx},${s.cy}</td>
      <td class="num">${s.size}</td>
      <td class="num">${s.cells.length}</td>
    </tr>`).join('');

  return `<div class="card" data-task="${esc(d.task)}" data-level="${esc(d.level)}"
            data-correct="${r ? (r.correct ? 'ok' : 'bad') : ''}">
    <h2>${esc(d.id)}</h2>
    <div class="meta">
      <span class="pill">${esc(d.task)}</span><span class="pill">${esc(d.level)}</span>
      <span class="pill">answer <code>${esc(d.answer)}</code> →
        cf <code>${esc(d.answer_cf)}</code></span>${tokenInfo}${verdict}
    </div>
    <div class="panes">
      <div class="pane">
        <div class="label">clean</div>
        <div class="stage"><img src="${d.clean_png}" alt="clean scene">
          <svg viewBox="0 0 ${SIZE} ${SIZE}">${overlaySvg(d, OVERLAY)}${changedOutline(d)}</svg>
        </div>
      </div>
      <div class="pane">
        <div class="label">counterfactual — one shape changed</div>
        <div class="stage"><img src="${d.cf_png}" alt="counterfactual scene">
          <svg viewBox="0 0 ${SIZE} ${SIZE}">${overlaySvg(d, OVERLAY)}${changedOutline(d)}</svg>
        </div>
      </div>
      <div class="info">
        <p style="margin:0 0 8px"><strong>Q:</strong> ${esc(d.question)}<br>
          <strong>prefill:</strong> <code>${esc(d.prefill)}</code></p>
        <table><thead><tr><th>#</th><th>colour</th><th>kind</th><th>centre</th>
          <th>size</th><th>cells</th></tr></thead><tbody>${rows}</tbody></table>
        <p class="legend">
          <span><span class="swatch" style="background:${GROUP_COLOURS.changed}"></span>changed</span>
          <span><span class="swatch" style="background:${GROUP_COLOURS.targets}"></span>other targets</span>
          <span><span class="swatch" style="background:${GROUP_COLOURS.distractors}"></span>distractors</span>
        </p>
        ${distributionBars(r)}
        ${emergence(r)}
      </div>
    </div>
  </div>`;
}

const HEX = __COLOUR_HEX__;
function colourHex(name) { return HEX[name] || '#888'; }

let OVERLAY = 'none';
function render() {
  OVERLAY = $('f-overlay').value;
  const task = $('f-task').value, level = $('f-level').value,
        correct = $('f-correct').value;
  const shown = DATA.filter(d =>
    (!task || d.task === task) && (!level || d.level === level) &&
    (!correct || (d.result && (d.result.correct ? 'ok' : 'bad') === correct)));
  $('cards').innerHTML = shown.length
    ? shown.map(card).join('')
    : '<p class="empty">No samples match these filters.</p>';
  $('count').textContent = `${shown.length} of ${DATA.length} samples`;
}
['f-task','f-level','f-correct','f-overlay'].forEach(id =>
  $(id).addEventListener('change', render));
render();
</script></body></html>
"""


def build(
    samples: Sequence[S.Sample],
    output: str | Path,
    grid,
    processor=None,
    results: dict[str, dict] | None = None,
    title: str = "Perception dataset viewer",
    subtitle: str = "",
) -> Path:
    """Write the viewer. ``results`` maps sample_id -> per-sample result dict."""
    results = results or {}
    records = [
        sample_record(s, grid, processor=processor, result=results.get(s.sample_id))
        for s in samples
    ]

    stages = "dataset only (no model run yet)" if not results else f"{len(results)} scored"
    content = _TEMPLATE
    for key, value in {
        "__TITLE__": title,
        "__SUBTITLE__": subtitle
        or f"{len(records)} samples · {grid.n_rows}x{grid.n_cols} token grid · {stages}",
        "__DATA__": json.dumps(records),
        "__GROUPS__": json.dumps(GROUP_COLOURS),
        "__COLOUR_HEX__": json.dumps(
            {name: "#%02x%02x%02x" % rgb for name, rgb in S.COLOURS.items()}
        ),
    }.items():
        content = content.replace(key, value)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output
