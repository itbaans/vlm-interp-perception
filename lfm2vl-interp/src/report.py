"""Build the self-contained HTML report that ships inside the results bundle.

Every figure is embedded as a data URI and every figure has a table view beside it, so
the file opens correctly from a local disk with no network and no sibling assets. It
commits to a light surface, matching the PNGs.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path

import aggregate as A

CSS = """
:root {
  --surface: #fcfcfb; --surface-2: #f5f4f1; --line: #e6e5e1;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8983;
  --blue: #2a78d6; --orange: #eb6834; --aqua: #1baf7a;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.65 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 48px 28px 96px; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -.02em; }
h2 { font-size: 21px; margin: 56px 0 6px; letter-spacing: -.01em; }
h3 { font-size: 15px; margin: 28px 0 8px; color: var(--ink-2); font-weight: 600; }
p  { color: var(--ink-2); margin: 10px 0; }
.sub { color: var(--ink-3); font-size: 14px; margin: 0 0 4px; }
.rule { height: 1px; background: var(--line); border: 0; margin: 8px 0 0; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px; margin: 26px 0 8px; }
.tile { background: var(--surface-2); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 18px; }
.tile .k { font-size: 12px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--ink-3); margin-bottom: 6px; }
.tile .v { font-size: 27px; line-height: 1.15; font-weight: 600; letter-spacing: -.02em; }
.tile .n { font-size: 12.5px; color: var(--ink-3); margin-top: 4px; }

figure { margin: 20px 0 8px; }
figure img { width: 100%; height: auto; display: block; border: 1px solid var(--line);
  border-radius: 10px; background: var(--surface); }
figcaption { color: var(--ink-3); font-size: 13px; margin-top: 10px; }

.scroll { overflow-x: auto; margin: 14px 0; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { padding: 7px 12px; text-align: right; border-bottom: 1px solid var(--line);
  font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
thead th { color: var(--ink-3); font-weight: 600; font-size: 12px;
  text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--ink-3); }
tbody tr.group td { background: var(--surface-2); }
tbody tr.rule-after td { border-bottom: 1px solid var(--ink-3); }

.note { background: var(--surface-2); border-left: 3px solid var(--orange);
  border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 18px 0; font-size: 14px;
  color: var(--ink-2); }
.note.info { border-left-color: var(--blue); }
.pill { display: inline-block; padding: 2px 9px; border-radius: 99px; font-size: 12px;
  background: var(--surface-2); border: 1px solid var(--line); color: var(--ink-2);
  margin-right: 6px; }
ul { color: var(--ink-2); } li { margin: 5px 0; }
a { color: var(--blue); }
.missing { color: var(--ink-3); font-style: italic; }
"""


def _embed(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _esc(value) -> str:
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list], rule_after: set[int] | None = None) -> str:
    rule_after = rule_after or set()
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for index, row in enumerate(rows):
        cells = "".join(f"<td>{c if isinstance(c, str) else _esc(c)}</td>" for c in row)
        cls = ' class="rule-after"' if index in rule_after else ""
        body.append(f"<tr{cls}>{cells}</tr>")
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _tiles(items: list[tuple[str, str, str]]) -> str:
    cards = "".join(
        f'<div class="tile"><div class="k">{_esc(k)}</div>'
        f'<div class="v">{_esc(v)}</div><div class="n">{_esc(n)}</div></div>'
        for k, v, n in items
    )
    return f'<div class="tiles">{cards}</div>'


def _figure(path: Path | None, caption: str) -> str:
    if not path or not path.exists():
        return '<p class="missing">Figure not generated — the underlying results are missing.</p>'
    return (
        f'<figure><img alt="{_esc(caption)}" src="data:image/png;base64,{_embed(path)}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def _ablation_section(ablation: dict | None, vqa: dict | None, figure: Path | None) -> str:
    if not ablation and not vqa:
        return ""

    parts = [
        "<h2>1 · Token ablation</h2>",
        '<hr class="rule">',
        "<p>Visual-token embeddings are replaced with the mean visual token. Each number "
        "is the share of <em>initially correct</em> object identifications the ablation "
        "<strong>destroyed</strong>, so <strong>higher means it mattered more</strong> -- "
        "the same units as the paper's Table 1. The claim under test is that object "
        "tokens beat the random and integrated-gradients baselines <em>at a matched "
        "token count</em>.</p>",
        _figure(figure, "Degradation against the number of visual tokens ablated. "
                        "Object conditions are the labelled diamonds; the two lines are "
                        "the baselines sweeping token count."),
    ]

    if ablation:
        rows = A.ablation_rows(ablation, ("generative", "polling"))
        stats = A.visual_token_stats(ablation)
        if stats:
            parts.append(
                f'<p class="sub">Visual tokens per image: mean {stats["mean"]:.0f} '
                f'(min {stats["min"]}, max {stats["max"]}) — the paper\'s LLaVA had a fixed 576.</p>'
            )
        table_rows, rule_after = [], set()
        for index, row in enumerate(rows):
            table_rows.append([
                row.label,
                f"{row.avg_tokens:.1f}",
                "—" if row.decrease("generative") is None else f"{row.decrease('generative'):.2f}%",
                "—" if row.decrease("polling") is None else f"{row.decrease('polling'):.2f}%",
            ])
            if row.condition == "register":
                rule_after.add(index)
        parts.append("<h3>Generative and polling settings</h3>")
        parts.append(_table(
            ["Ablation type", "Avg tokens", "Generative dec.", "Polling dec."],
            table_rows, rule_after
        ))

    if vqa:
        rows, summary = A.vqa_rows(vqa)
        parts.append("<h3>VQA setting</h3>")
        parts.append(
            f'<p class="sub">{summary["n_correct_before"]} of {summary["n_total"]} images '
            f"answered correctly before ablation.</p>"
        )
        if rows:
            parts.append(_table(
                ["Ablation type", "Avg tokens", "VQA dec.", "Correct-token prob."],
                [[r.label, f"{r.avg_tokens:.1f}", f"{r.decrease('vqa'):.2f}%",
                  f"{r.retained['correct_token_prob']:.4f}"] for r in rows],
            ))
            if summary["ambiguous"]["n"]:
                parts.append(
                    f'<div class="note">{summary["ambiguous"]["n"]} scored images '
                    f'({summary["ambiguous"]["share"]:.0f}%) have a class whose first token is too '
                    f'short to identify it alone ({_esc(", ".join(summary["ambiguous"]["classes"]))}). '
                    f"Prefer the correct-token probability column for those.</div>"
                )
        else:
            parts.append('<p class="missing">No image was answered correctly before ablation.</p>')

    parts.append(
        '<div class="note info"><strong>Paper reference (LLaVA-1.5).</strong> Object 33.33 / '
        "15.38 / 33.33 and +1 Buffer 71.79 / 51.28 / 86.67 across generative / polling / VQA, "
        "with +1 Buffer averaging 33.4 of 576 tokens.</div>"
    )
    return "".join(parts)


def _lens_section(results: dict | None, figure: Path | None) -> str:
    if not results:
        return ""
    summary = A.logit_lens_summary(results)
    parts = [
        "<h2>2 · Logit lens</h2>",
        '<hr class="rule">',
        "<p>Each visual-token position is decoded through "
        "<code>lm_head(embedding_norm(h))</code> at every layer. The paper's claim is that "
        "visual tokens drift toward interpretable text embeddings, peaking in the "
        "mid-to-late layers — so the <em>shape</em> of this curve matters more than its "
        "height.</p>",
        _figure(figure, "Share of the object's visual tokens whose top-1 decoded token names "
                        "the object class, per layer. Peaks are direct-labelled."),
    ]
    if summary["n_images"]:
        num_layers = summary["num_layers"] or summary["n_entries"] - 1
        best = summary["best"]["strict"]
        parts.append(_tiles([
            ("Best-layer hit rate", f"{best['rate'] * 100:.1f}%",
             f"strict, over the {best['n_with_signal']} images showing the effect"),
            ("Curve peaks at", f"layer {summary['peak_layer']} / {num_layers}",
             f"{summary['peak_rate'] * 100:.1f}% averaged across images"),
            ("Images showing the effect", f"{best['n_with_signal']} / {summary['n_images']}",
             f"{best['share_with_signal']:.0f}% decode to the class at some layer"),
        ]))
        parts.append(
            f'<p class="sub">Median best layer {best["median_layer"]:.0f} of {num_layers}. '
            f"Images whose object tokens never decode to the class at any layer are excluded "
            f"from the best-layer statistics: their argmax is a tie at zero, which would "
            f"otherwise report a peak near layer 0 that does not exist.</p>"
        )
        parts.append(_table(
            ["Layer", "Strict", "Lenient"],
            [["embeddings" if i == 0 else f"L{i}",
              f"{summary['per_layer']['strict'][i] * 100:.1f}%",
              f"{summary['per_layer']['lenient'][i] * 100:.1f}%"]
             for i in range(summary["n_entries"])],
        ))
    parts.append(
        '<div class="note info"><strong>Paper reference.</strong> LLaVA-1.5 reaches 23.7% at '
        "an average best layer of 25.7 of 33; Qwen2VL-2B reaches 6.5% at 25.1 of 29. The paper "
        "does not state whether it counted exact class tokens or partial word-pieces, so both "
        "criteria are reported here.</div>"
    )
    return "".join(parts)


def _knockout_section(results: dict | None, figure: Path | None) -> str:
    if not results:
        return ""
    summary = A.knockout_summary(results)
    meta = summary["meta"]
    parts = [
        "<h2>3 · Attention knockout</h2>",
        '<hr class="rule">',
        "<p>Attention between chosen token groups is masked to −∞ over a window of layers. "
        "1.00 means no impact, 0.00 means every answer became wrong.</p>",
    ]

    attention = meta.get("attention_layers") or []
    conv = meta.get("conv_layers") or []
    if attention:
        parts.append(
            f'<div class="note"><strong>This model is a hybrid, so knockout is a partial cut.</strong> '
            f"Only {len(attention)} of {meta.get('num_layers')} LM layers have attention "
            f"(<span class='mono'>{_esc(attention)}</span>); the other {len(conv)} are "
            f"short-conv layers that mix ~2 positions backwards each, ignore token-pair masking "
            f"and cannot be blocked. The distance control below separates genuine attention "
            f"routing from conv leakage.</div>"
        )

    parts.append(_figure(figure, "Relative accuracy per token-group pair and layer window, "
                                 "with the per-layer sweep beneath."))

    if summary["n_usable"] and summary["grid"]:
        windows = summary["windows"]
        parts.append(_table(
            ["From → To"] + [w.replace("_", "-") for w in windows],
            [[f"{s} → {t}"] + [f"{values[w]:.2f}" if w in values else "—" for w in windows]
             for (s, t), values in summary["grid"].items()],
        ))
        parts.append(
            '<p class="sub">O = object tokens, O+n = with an n-cell buffer, '
            "I−(O+1) = all visual tokens except those, LTP = last token position, "
            "LVR = last visual-token row.</p>"
        )
    else:
        parts.append('<p class="missing">No image was answered correctly before blocking.</p>')

    control = summary["control"]
    if control and control["retained"] is not None:
        parts.append("<h3>Distance control</h3>")
        parts.append(_tiles([
            ("Accuracy, all attention blocked", f"{control['retained']:.2f}",
             f"padded prompt, n={control['n']}"),
            ("Object→last distance", f"{control['mean_distance']:.0f} tokens",
             "with padding applied"),
            ("Short-conv reach", f"{control['conv_reach']} tokens",
             "2 × number of conv layers"),
        ]))
        parts.append(
            "<p>If this figure matches the unpadded <em>all</em> column, the effect is genuine "
            "attention routing. If it is markedly higher, part of what the main table measured "
            "was information leaking through the short-conv layers.</p>"
        )

    parts.append(
        '<div class="note info"><strong>Paper reference (LLaVA-1.5).</strong> Object+1 → last '
        "token falls to 0.82 at mid-late layers and 0.67 across all layers, while any visual "
        "group → last visual row stays at 1.00 everywhere.</div>"
    )
    return "".join(parts)


# --------------------------------------------------------------------------------------


def build(
    output: Path,
    manifest: dict,
    results: dict[str, dict | None],
    figures: dict[str, Path],
    lens_pages: list[str],
) -> Path:
    """Write the report. ``results`` maps ablation/vqa/knockout/logit_lens to loaded JSON."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    preset = manifest.get("preset", "?")
    model = manifest.get("model_id", "?")

    header_tiles = []
    ablation = results.get("ablation")
    if ablation:
        rows = {r.condition: r for r in A.ablation_rows(ablation, ("generative", "polling"))}
        best = rows.get("object_1") or rows.get("object_0")
        if best and best.decrease("generative") is not None:
            header_tiles.append((
                "Object+buffer ablation",
                f"{best.decrease('generative'):.0f}%",
                f"destroyed, {best.avg_tokens:.0f} tokens (generative)",
            ))
    lens = results.get("logit_lens")
    if lens:
        summary = A.logit_lens_summary(lens)
        if summary["n_images"]:
            header_tiles.append((
                "Logit-lens peak",
                f"{summary['best']['strict']['rate'] * 100:.1f}%",
                f"at layer {summary['best']['strict']['layer']:.1f} "
                f"of {summary['num_layers'] or '?'}",
            ))
    knockout = results.get("knockout")
    if knockout:
        summary = A.knockout_summary(knockout)
        values = summary["grid"].get(("O+1", "LTP"), {})
        if "all" in values:
            header_tiles.append((
                "Attention knockout",
                f"{values['all']:.2f}",
                "object+1 → last token, all layers",
            ))

    stages = manifest.get("stages", {})
    pills = " ".join(
        f'<span class="pill">{_esc(name)} · {_esc(info.get("status", "?"))}</span>'
        for name, info in stages.items()
    )

    lens_links = ""
    if lens_pages:
        items = "".join(f'<li><a href="logit_lens/{_esc(p)}">{_esc(p)}</a></li>'
                        for p in lens_pages[:40])
        lens_links = (
            f"<h3>Interactive logit-lens pages</h3>"
            f"<p>Hover any image region to see what that visual token decodes to, layer by "
            f"layer. Open them from the <code>logit_lens/</code> folder beside this file.</p>"
            f"<ul>{items}</ul>"
        )

    body = f"""
<div class="wrap">
  <h1>Visual information processing in LFM2.5-VL</h1>
  <p class="sub">A replication of Neo et al. (ICLR 2025),
     <em>Towards Interpreting Visual Information Processing in Vision-Language Models</em>,
     on <code>{_esc(model)}</code>.</p>
  <p class="sub">Generated {_esc(generated)} · preset <strong>{_esc(preset)}</strong> ·
     split <code>{_esc(manifest.get('split', '?'))}</code></p>
  <div style="margin-top:14px">{pills}</div>

  {_tiles(header_tiles) if header_tiles else ''}

  <div class="note info">
    <strong>Read the pattern, not the absolute numbers.</strong> LFM2.5-VL has roughly 234
    visual tokens instead of LLaVA's 576, a 4:1 pixel-unshuffle projector, and a language
    model that is mostly short-convolution rather than attention. The replication targets
    are qualitative: object ablation beating the baselines at matched token counts, visual
    tokens becoming vocabulary-interpretable in mid-to-late layers, and object→last-token
    attention mattering while visual→last-row does not.
  </div>

  {_ablation_section(results.get('ablation'), results.get('vqa'), figures.get('ablation'))}
  {_lens_section(results.get('logit_lens'), figures.get('logit_lens'))}
  {lens_links}
  {_knockout_section(results.get('knockout'), figures.get('knockout'))}

  <h2>Run details</h2>
  <hr class="rule">
  {_table(["Stage", "Status", "Duration", "Notes"],
          [[name, info.get("status", "?"), info.get("duration", "—"), info.get("note", "")]
           for name, info in stages.items()])}
  <p class="sub">Raw results are in <code>raw/</code>, printed tables in
     <code>tables/</code>, the full log in <code>run.log</code>.</p>
</div>
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>LFM2.5-VL interpretability report</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )
    return output
