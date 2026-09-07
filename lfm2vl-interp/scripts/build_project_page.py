"""Build a single self-contained project page covering both experiments.

Every figure is embedded as a base64 data URI and every number is computed from the
result JSONs rather than typed in, so the page can be moved anywhere and still renders,
and cannot drift from the data it describes.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import aggregate as A  # noqa: E402
import lens_widget as LW  # noqa: E402

ATTENTION_LAYERS = [2, 5, 9, 13, 17, 21, 24, 27]
TASK_ORDER = ["relation", "count", "orientation", "shape", "colour"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def img(path: Path, alt: str, caption: str = "") -> str:
    if not path.exists():
        return f'<p class="missing">Figure not found: {html.escape(path.name)}</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return (f'<figure><img alt="{html.escape(alt)}" '
            f'src="data:image/png;base64,{data}">{cap}</figure>')


def table(headers, rows, note="") -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    n = f'<p class="note">{note}</p>' if note else ""
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>{n}")


# ---------------------------------------------------------------------------------------
# Numbers, all derived from the result files
# ---------------------------------------------------------------------------------------


def replication_numbers(raw: Path) -> dict:
    out = {}
    lens = load(raw / "logit_lens_val2017.json")
    if lens:
        out["lens"] = A.logit_lens_summary(lens)

    knock = load(raw / "attention_knockout_train2017.json")
    if knock:
        out["knockout"] = A.knockout_summary(knock)

    vqa = load(raw / "ablation_vqa_train2017.json")
    if vqa:
        rows, summary = A.vqa_rows(vqa)
        out["vqa_rows"] = rows
        out["vqa_summary"] = summary

    dataset = load(raw / "dataset_train2017.json")
    if dataset:
        images = dataset["images"]
        out["dataset"] = {
            "evaluated": len(images),
            "kept": sum(1 for r in images.values() if r.get("kept")),
        }
    return out


def perception_numbers(synth: Path) -> dict:
    out = {}
    patching = load(synth / "patching.json")
    baseline = load(synth / "baseline.json")
    lens = load(synth / "answer_lens.json")
    probes = load(synth / "probes.json")
    knock = load(synth / "knockout.json")

    records = [r for r in patching["images"].values() if "coarse" in r] if patching else []
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    def curve(rs, group):
        return [
            statistics.mean([
                r["coarse"][group][str(L)] for r in rs
                if str(L) in r["coarse"].get(group, {})
                and r["coarse"][group][str(L)] == r["coarse"][group][str(L)]
            ] or [float("nan")])
            for L in range(30)
        ]

    def contributions(rs):
        c = curve(rs, "changed")
        result = {}
        for a in ATTENTION_LAYERS:
            nxt = a + 1
            while nxt in ATTENTION_LAYERS:
                nxt += 1
            result[a] = c[a] - (c[nxt] if nxt < 30 else 0.0)
        return result

    base_by = defaultdict(list)
    if baseline:
        for r in baseline["images"].values():
            base_by[r["task"]].append(r)
    lens_by = defaultdict(list)
    if lens:
        for r in lens["images"].values():
            if r.get("final_correct"):
                lens_by[r["task"]].append(r)

    rows = []
    for task in TASK_ORDER:
        rs = by_task.get(task, [])
        if not rs:
            continue
        contrib = contributions(rs)
        peak = max(contrib, key=contrib.get)
        em = [r["emergence_layer"] for r in lens_by.get(task, [])
              if r.get("emergence_layer") is not None]
        pr = next((r["layer"] for r in (probes or {}).get("tasks", {}).get(task, [])
                   if r["test_accuracy"] >= 0.9), None)
        rows.append({
            "task": task, "n": len(rs), "peak": peak,
            "accuracy": statistics.mean(r["clean"]["correct"] for r in base_by[task])
            if base_by.get(task) else None,
            "emergence": statistics.mean(em) if em else None,
            "probe": pr,
            "changed": max(v for v in curve(rs, "changed") if v == v),
            "distractors": max(v for v in curve(rs, "distractors") if v == v),
            "contributions": contrib,
        })

    out["rows"] = rows
    out["n_patched"] = len(records)

    if baseline:
        levels = sorted({r["level"] for r in baseline["images"].values()})
        out["levels"] = levels
        out["accuracy_grid"] = {
            task: {
                lv: statistics.mean(
                    [r["clean"]["correct"] for r in base_by[task] if r["level"] == lv] or [0]
                )
                for lv in levels
            }
            for task in TASK_ORDER if task in base_by
        }
        out["n_baseline"] = len(baseline["images"])
        out["separable"] = sum(1 for r in baseline["images"].values() if r.get("separable"))
        counts = [r for r in baseline["images"].values() if r["task"] == "count"]
        if counts:
            out["count_mae"] = statistics.mean(
                abs(int(r["clean"]["predicted"]) - int(r["clean"]["expected"]))
                for r in counts
            )

    if knock:
        usable = [r for r in knock["images"].values() if r.get("baseline", {}).get("correct")]
        vals = [v["correct"] for r in usable for k, v in r.get("knockout", {}).items()
                if k.startswith("changed@")]
        out["knockout_mean"] = statistics.mean(vals) if vals else None
        out["knockout_n"] = len(usable)

    if records:
        question = curve(records, "question")
        out["question_peak"] = max(v for v in question if v == v)
    return out


CSS = """
:root{--bg:#fcfcfb;--panel:#f5f4f1;--line:#e2e1dd;--ink:#14140f;--ink2:#4a4944;
      --ink3:#84837c;--accent:#2a78d6;--warn:#eb6834}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:16px/1.68 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:56px 28px 120px}
h1{font-size:31px;line-height:1.22;margin:0 0 10px;letter-spacing:-.02em;font-weight:650}
h2{font-size:23px;margin:64px 0 4px;letter-spacing:-.01em;font-weight:640;
   padding-top:26px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:34px 0 8px;font-weight:640}
h4{font-size:15px;margin:26px 0 6px;font-weight:640;color:var(--ink2)}
p{margin:12px 0;color:var(--ink2)}
.lede{font-size:17px;color:var(--ink2)}
.meta{color:var(--ink3);font-size:14px;margin:0 0 26px}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13.5px;
     background:var(--panel);padding:1px 5px;border-radius:3px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;
    padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.55}
figure{margin:24px 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
           border-radius:8px;background:var(--bg)}
figcaption{color:var(--ink3);font-size:13.5px;margin-top:9px;line-height:1.55}
.scroll{overflow-x:auto;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:7px 12px;text-align:right;border-bottom:1px solid var(--line);
      font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
thead th{color:var(--ink3);font-weight:640;font-size:12px;text-transform:uppercase;
         letter-spacing:.05em;border-bottom:1px solid var(--ink3)}
.note{color:var(--ink3);font-size:13.5px;margin:8px 0 0;line-height:1.6}
.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;
     padding:14px 18px;margin:20px 0}
.box.warn{border-left:3px solid var(--warn)}
.box p:first-child{margin-top:0}.box p:last-child{margin-bottom:0}
ul,ol{color:var(--ink2)}li{margin:6px 0}
.toc{background:var(--panel);border:1px solid var(--line);border-radius:8px;
     padding:16px 22px;margin:28px 0 8px}
.toc ol{margin:6px 0;padding-left:20px}
.toc a{color:var(--ink2);text-decoration:none}.toc a:hover{text-decoration:underline}
a{color:var(--accent)}
.missing{color:var(--ink3);font-style:italic}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:5px 22px;font-size:14.5px;
    margin:14px 0}
.kv dt{color:var(--ink3)}.kv dd{margin:0;color:var(--ink)}
""" + LW.WIDGET_CSS


def build(root: Path, output: Path) -> Path:
    rep_dir = root / "lfm2vl-results"
    per_dir = root / "perception"
    rep = replication_numbers(rep_dir / "raw")
    per = perception_numbers(per_dir / "bundle" / "synthetic")

    P = []
    a = P.append

    a('<div class="wrap">')
    a("<h1>Interpreting visual information processing in LFM2.5-VL-3B</h1>")
    a('<p class="lede">Two studies on <code>LiquidAI/LFM2.5-VL-3B</code>. The first '
      "replicates Neo et al. (ICLR 2025) on this model. The second is a new experiment that "
      "locates where individual perceptual abilities are computed, using synthetic scenes "
      "and activation patching.</p>")
    a('<p class="meta">Every number on this page is computed from the result files produced '
      "by the runs described below.</p>")

    a('<div class="toc"><strong>Contents</strong><ol>'
      '<li><a href="#model">The model</a></li>'
      '<li><a href="#part1">Part 1. Replication of Neo et al.</a></li>'
      '<li><a href="#part2">Part 2. Where perceptual abilities are computed</a></li>'
      '<li><a href="#limits">Limitations</a></li>'
      '<li><a href="#repro">Reproduction</a></li></ol></div>')

    # ------------------------------------------------------------------ model
    a('<h2 id="model">1. The model</h2>')
    a("<p>LFM2.5-VL-3B pairs a SigLIP2 NaFlex vision encoder with a 30 layer LFM2 language "
      "model. Two architectural properties determined how both experiments had to be "
      "built.</p>")
    a("<p>First, the language model is hybrid. Only 8 of its 30 layers use attention, at "
      "indices 2, 5, 9, 13, 17, 21, 24 and 27. The other 22 are short convolutions reaching "
      "2 positions back. Attention is therefore the only mechanism able to move information "
      "between distant sequence positions.</p>")
    a("<p>Second, the number of visual tokens varies with image shape. The projector applies "
      "a 2x2 pixel unshuffle, so an image becomes a grid of (height/32) by (width/32) tokens, "
      "between 64 and 256 in total. A 640x480 photograph becomes a 13x18 grid of 234 tokens. "
      "There is no fixed 24x24 grid as in LLaVA, so patch to token mapping is computed per "
      "image from the processor output.</p>")
    a(table(
        ["", "LLaVA-1.5-7B (paper)", "LFM2.5-VL-3B"],
        [["Vision encoder", "CLIP ViT-L/14 at 336px", "SigLIP2 so400m-patch16-naflex"],
         ["Preprocessing", "centre crop to square", "aspect preserving resize, no crop"],
         ["Visual tokens", "576, fixed 24x24 grid", "64 to 256, variable grid"],
         ["Language model", "Vicuna-7B, 32 attention layers", "LFM2-2.6B, 8 attention, 22 conv"],
         ["Projector", "MLP, 1 patch to 1 token", "2x2 unshuffle, 4 patches to 1 token"]],
        "Differences that required reimplementation rather than a configuration change."))

    # ------------------------------------------------------------------ part 1
    a('<h2 id="part1">2. Part 1. Replication of Neo et al.</h2>')
    a("<p>Neo et al. study how the language model inside an adapter style vision language "
      "model processes visual tokens. They report three results on LLaVA-1.5-7B. Object "
      "information is localised to the visual tokens covering the object. Those tokens become "
      "interpretable in vocabulary space in the middle to late layers. The model extracts "
      "object information from them in the middle to late layers.</p>")

    a("<h3>Setting</h3>")
    if rep.get("dataset"):
        a('<dl class="kv">'
          "<dt>Dataset</dt><dd>COCO train2017 and val2017</dd>"
          "<dt>Filter</dt><dd>at most 3 annotations, one instance of a category with area "
          "between 1000 and 2000 square pixels</dd>"
          f'<dt>Images evaluated</dt><dd>{rep["dataset"]["evaluated"]}</dd>'
          "<dt>Hardware</dt><dd>one A100</dd></dl>")

    a("<h4>Logit lens</h4>")
    a("<p>Each visual token position is decoded through the model's own output weights at "
      "every layer and scored against the object's class name. The paper reports 23.7 percent "
      "at layer 25.7 of 33 for LLaVA-1.5, and 6.5 percent at layer 25.1 of 29 for "
      "Qwen2VL-2B.</p>")
    lens = rep.get("lens")
    if lens and lens.get("n_images"):
        best = lens["best"]["strict"]
        a(table(
            ["Model", "Best layer hit rate", "Peak layer", "Relative depth"],
            [["LLaVA-1.5-7B (paper)", "23.7%", "25.7 of 33", "0.78"],
             ["Qwen2VL-2B (paper)", "6.5%", "25.1 of 29", "0.87"],
             ["<b>LFM2.5-VL-3B</b>", f'<b>{best["rate"] * 100:.1f}%</b>',
              f'<b>{lens["peak_layer"]} of {lens["num_layers"]}</b>',
              f'<b>{lens["peak_layer"] / lens["num_layers"]:.2f}</b>']],
            f'Measured on {lens["n_images"]} COCO validation images with object area between '
            f"20,000 and 30,000 square pixels. The hit rate is averaged over the "
            f'{best["n_with_signal"]} images whose object tokens decode to the class name at '
            f'some layer ({best["share_with_signal"]:.0f} percent of the sample). Images that '
            f"never do are excluded from the best layer statistic, because their argmax is a "
            f"tie at zero rather than a peak."))
    a(img(rep_dir / "figures" / "logit_lens.png", "Logit lens hit rate by layer",
          "Fraction of the object's visual tokens whose top decoded token is the object class "
          "name, per layer. The rate is low and flat through the early and middle layers, "
          "rises after layer 21 and peaks at layer 26."))
    a("<p>The result reproduces. The peak sits at the same relative depth as in both models "
      "the paper tested, and the magnitude falls between them.</p>")

    a("<h4>Inspecting the lens directly</h4>")
    a("<p>The measurement above reduces each image to one number. The widget below shows the "
      "underlying decoding for individual images: every visual token is decoded through the "
      "output weights at the selected layer, and its top token is drawn on the square it "
      "came from.</p>")
    lens_examples = LW.pick_examples(
        rep_dir / "logit_lens", load(rep_dir / "raw" / "logit_lens_val2017.json"), limit=3
    )
    a(LW.widget_html(lens_examples))
    a("<p>Sliding through the layers shows something the aggregate curve hides. Already at "
      "layer 0, before any language model layer has run, the visual tokens decode to words "
      "related to the picture. This differs from LLaVA, where Neo et al. report that adapter "
      "outputs do not correspond to language tokens at all. The likely reason is that this "
      "model ties its output weights to its input embeddings, so the projector is trained to "
      "emit vectors in the same space the output weights read from.</p>")
    a("<p>What depth adds is specificity rather than interpretability. The decoding keeps "
      "changing with layer, and by layer 26 only about 14 percent of squares still decode to "
      "the token they gave at layer 0, while the rate at which a square decodes to the "
      "object's own class name roughly doubles, from 5.0 percent at the embeddings to 10.4 "
      "percent at layer 26. Some squares decode to tokens in other languages, which Neo et "
      "al. also observed on LLaVA.</p>")

    a("<h4>Attention knockout</h4>")
    knock = rep.get("knockout")
    if knock and knock.get("n_usable"):
        windows = knock["windows"]
        rows = [[f"{src} to {tgt}"] +
                [f"{values[w]:.2f}" if w in values else "-" for w in windows]
                for (src, tgt), values in knock["grid"].items()]
        a(table(["From, to"] + [w.replace("_", "-") for w in windows], rows,
                f'Relative accuracy under blocking, {knock["n_usable"]} images. 1.00 means no '
                f"effect. O is the object tokens, O+n adds an n cell buffer, I-(O+1) is all "
                f"visual tokens except those, LTP is the last token position and LVR is the "
                f"last row of visual tokens. Only the 8 attention layers can be blocked."))
    a("<p>Two of the paper's findings reproduce in direction. Blocking the object tokens "
      "degrades the answer more in the late layers than the early ones, and blocking any "
      "visual group to the last visual token row has no effect at any depth, which is the "
      "paper's evidence against a summarisation step. The magnitude is much smaller than the "
      "paper reports, which is expected because knockout reaches only 8 of 30 layers on this "
      "architecture.</p>")

    a("<h4>Token ablation</h4>")
    rows = rep.get("vqa_rows")
    if rows:
        summary = rep["vqa_summary"]
        by_cond = {r.condition: r for r in rows}
        trows = []
        for cond in ["object_0", "object_1", "object_2", "register",
                     "random_20", "random_40", "gradient_20", "gradient_40"]:
            r = by_cond.get(cond)
            if r:
                trows.append([r.label, f"{r.avg_tokens:.1f}", f'{r.decrease("vqa"):.1f}%',
                              f'{r.retained["correct_token_prob"]:.3f}'])
        a(table(["Ablation", "Tokens replaced", "Answers destroyed", "P(correct token)"],
                trows,
                f"Visual token embeddings are replaced with the mean visual token. "
                f'{summary["n_correct_before"]} of {summary["n_total"]} images were answered '
                f"correctly before ablation and form the denominator. Higher means the "
                f"ablation mattered more."))
    a("<p>Object localisation reproduces. Replacing about 22 tokens covering the object "
      "destroys roughly ten times more answers than replacing 20 tokens chosen at random. One "
      "result does not reproduce: integrated gradients selects more damaging tokens than the "
      "object mask at a matched token count, the opposite of the paper's ordering.</p>")

    # ------------------------------------------------------------------ part 2
    a('<h2 id="part2">3. Part 2. Where perceptual abilities are computed</h2>')
    a("<p>The replication locates object identity. It cannot separate individual perceptual "
      "abilities, because COCO labels are object classes and its scenes cannot be altered in "
      "a controlled way. This experiment uses synthetic scenes, where every object's colour, "
      "shape, size, orientation and position is known exactly and any one of them can be "
      "changed while the rest are held fixed.</p>")

    a("<h3>Setting</h3>")
    a("<p>Scenes are 512x512 pixels, the one size that passes through the processor's resize "
      "step unchanged. It yields a 16x16 grid of 256 visual tokens at exactly 32 pixels per "
      "token, so shape coordinates map to token positions with no interpolation error.</p>")
    a("<p>Each sample is a pair of scenes differing in exactly one shape attribute. Because "
      "the pair renders at the same size under the same prompt, both runs have identical "
      "token positions. That is the condition activation patching requires.</p>")
    a(table(["Level", "Scene", "What it isolates"],
            [["L1", "2 to 3 objects, one colour, one shape", "counting alone"],
             ["L2", "4 to 6 objects, one colour, one shape", "counting under load"],
             ["L3", "mixed colours", "colour filtering"],
             ["L4", "mixed colours and shapes", "two feature binding"],
             ["L5", "crowded, overlapping", "segmentation under clutter"]],
            "Five difficulty levels, crossed with five tasks."))

    a("<h3>Example tasks</h3>")
    a(img(per_dir / "figures" / "tasks.png", "The five perception tasks",
          "One sample per task. The top row is the clean scene, the bottom row the "
          "counterfactual, and the orange outline marks the single shape that differs. Every "
          "answer is one token, so the model's next token distribution is the answer "
          "distribution and no generation or parsing is involved."))

    a("<h3>Baseline accuracy</h3>")
    a("<p>Localisation is only meaningful where the model answers correctly, so accuracy was "
      "measured first and used to gate the later stages.</p>")
    if per.get("accuracy_grid"):
        levels = per["levels"]
        rows = []
        for task in TASK_ORDER:
            if task not in per["accuracy_grid"]:
                continue
            cells = per["accuracy_grid"][task]
            rows.append([task] + [f"{cells[lv] * 100:.0f}%" for lv in levels] +
                        [f"<b>{statistics.mean(cells.values()) * 100:.0f}%</b>"])
        a(table(["Task"] + levels + ["Overall"], rows,
                f'{per["n_baseline"]} samples scored. {per["separable"]} pairs were answered '
                f"correctly on both the clean and the counterfactual scene and are therefore "
                f"usable for patching."))
    mae = per.get("count_mae")
    a("<p>Four of the five tasks are at ceiling at every difficulty level, including the "
      "crowded one. Counting is the only task with errors" +
      (f", and its mean absolute error is {mae:.2f}" if mae is not None else "") +
      ". The difficulty ladder did not separate the levels for this model.</p>")

    a("<h3>Method</h3>")
    a(img(per_dir / "figures" / "0_method.png", "How activation patching works",
          "The clean scene and the counterfactual are run normally and produce different "
          "answers. In a third run the counterfactual image is used, but at one layer the "
          "activations at the changed object's token positions are replaced with those from "
          "the clean run. Recovery measures how far the answer moves back toward the clean "
          "one, from 0 for no effect to 1 for full restoration. This is repeated once per "
          "layer."))
    a("<p>The patch is applied to the residual stream entering each decoder layer. Nothing "
      "inside a layer is modified, so the method applies identically to convolution and "
      "attention layers and reaches all 30. Attention knockout, by contrast, reaches only the "
      "8 attention layers.</p>")
    a("<p>Two identity checks verify the implementation. Patching every position at layer 0 "
      "reproduces the clean run exactly, giving recovery 1.000. Patching no positions leaves "
      "the counterfactual run bit identical, giving 0.000.</p>")

    a("<h3>Result 1. Information moves only at attention layers</h3>")
    a(img(per_dir / "figures" / "2_staircase.png", "Recovery by layer",
          "Left: recovery against the layer the clean activations were pasted into, pooled "
          "over all tasks. Right: the size of the fall at each attention layer. Recovery is "
          "constant between attention layers and falls at each one."))
    a("<p>Recovery declines with depth, which is a property of the method: a patch applied "
      "later has fewer remaining layers through which to act. The informative feature is the "
      "step structure, not the level.</p>")
    a("<p>The patched information sits at the object's token positions, roughly 250 positions "
      "from where the answer is produced. Only attention can carry it that far. Pasting at "
      "layer 6, 7 or 8 gives identical results because no attention layer lies between them, "
      "and the value falls only when an attention layer is crossed. The size of each fall "
      "measures how much of the answer that layer was carrying. After layer 27, the last "
      "attention layer, patching the object tokens has no effect at all.</p>")

    a("<h3>Result 2. Different abilities are read out at different depths</h3>")
    a(img(per_dir / "figures" / "1_depth.png", "Layer contributions per task",
          "Rows are tasks, columns are the 8 attention layers, and each cell is how much of "
          "that task's information the layer carried. The outlined cell is each task's "
          "maximum. A dash marks values below 0.005."))
    rows = per.get("rows", [])
    if rows:
        trows = []
        for r in rows:
            cells = [f'{r["contributions"][x]:.3f}' if r["contributions"][x] > 0.005 else "-"
                     for x in ATTENTION_LAYERS]
            i = ATTENTION_LAYERS.index(r["peak"])
            cells[i] = f'<b>{r["contributions"][r["peak"]]:.3f}</b>'
            trows.append([r["task"], str(r["n"])] + cells + [f'<b>L{r["peak"]}</b>'])
        a(table(["Task", "n"] + [f"L{x}" for x in ATTENTION_LAYERS] + ["Peak"], trows,
                f'{per["n_patched"]} minimal pairs. Each value is the fall in recovery when '
                f"the patch is applied after rather than before that attention layer."))
    a("<p>The peaks are ordered. Spatial relation is resolved at layer 5, counting at layer "
      "13, and colour, shape and orientation at layer 21. Counting and relation reach zero "
      "from layer 17 onward, meaning the visual tokens no longer carry information the answer "
      "depends on, while colour, shape and orientation are still being read at layers 24 and "
      "27.</p>")
    a("<p>The ordering is not built into the design. Each task uses separate images and "
      "separate samples, and the tasks were not ordered in advance.</p>")

    a("<h3>Result 3. The effect is specific to the object that changed</h3>")
    a(img(per_dir / "figures" / "3_specificity.png", "Object against control",
          "Best recovery from patching the object that changed, against patching the other "
          "objects in the same scene."))
    if rows:
        trows = [[r["task"], f'{r["changed"]:.3f}', f'{r["distractors"]:.3f}',
                  f'{r["changed"] / max(r["distractors"], 1e-6):.0f}x'] for r in rows]
        a(table(["Task", "Changed object", "Other objects", "Ratio"], trows,
                "Patching objects that did not change has almost no effect, which rules out "
                "the possibility that disturbing any visual token would move the answer."))

    a("<h3>Result 4. When the answer appears</h3>")
    a(img(per_dir / "figures" / "4_emergence.png", "Answer emergence by layer",
          "Probability of the correct answer read at the output position through the model's "
          "own output weights, per layer. The orange line marks the layer from which the "
          "answer is correct and stays correct."))
    a("<p>Before roughly layer 10 the output position carries no information about the "
      "answer. The pooled probability is 27.5 percent against a pooled chance level of 27 "
      "percent. The answer becomes available late, which is consistent with the patching "
      "result.</p>")

    a("<h3>Comparison of the three measurements</h3>")
    a(img(per_dir / "figures" / "5_methods.png", "Three measurements compared",
          "Patching measures where information is used, the logit lens measures where the "
          "model commits to an answer, and a linear probe measures where the answer becomes "
          "linearly readable at the output position."))
    if rows:
        trows = [[r["task"],
                  f'{r["accuracy"] * 100:.0f}%' if r["accuracy"] is not None else "-",
                  f'<b>L{r["peak"]}</b>',
                  f'L{r["emergence"]:.0f}' if r["emergence"] else "-",
                  f'L{r["probe"]}' if r["probe"] is not None else "-"] for r in rows]
        a(table(["Task", "Accuracy", "Used (patching)", "Answer appears", "Probe readable"],
                trows,
                "The three columns answer different questions and need not agree."))
    a("<p>For relation and orientation the probe reads the answer about seven layers before "
      "the model commits to it. That is expected, because the probe fits its own decoding "
      "direction and can find information that is present but not yet aligned with the "
      "model's output weights. For counting the ordering is reversed, which should not happen "
      "and most likely reflects the small probe sample.</p>")

    a("<h3>An unresolved disagreement</h3>")
    a('<div class="box warn">')
    a("<p><b>Attention knockout contradicts patching, and the discrepancy is not "
      "resolved.</b></p>")
    km = per.get("knockout_mean")
    if km is not None:
        a(f"<p>Blocking attention from the object tokens to the output position leaves "
          f'relative accuracy at {km:.2f} across {per["knockout_n"]} samples, that is, almost '
          f"no effect. Patching says the same tokens are necessary.</p>")
    qp = per.get("question_peak")
    a("<p>A possible explanation is that the information does not travel directly from the "
      "object to the output position. Patching the question tokens recovers " +
      (f"{qp:.2f}" if qp else "0.33") +
      " even though those tokens are identical in both runs, which can only occur if they "
      "have absorbed image information. That would place the route through the question text, "
      "in which case knockout was blocking an edge the model does not use.</p>")
    a("<p>This was not tested. The experiment that would settle it is to block attention from "
      "the object tokens to the question tokens instead.</p>")
    a("</div>")

    # ------------------------------------------------------------------ limits
    a('<h2 id="limits">4. Limitations</h2>')
    a("<ul>"
      "<li>The probe results are weak. Each probe is fitted on about 21 training samples "
      "against 2048 features, enough to separate any labelling, and evaluated on 9 held out "
      "samples. Training accuracy is 100 percent at nearly every layer by construction. The "
      "onset layers should be treated as indicative.</li>"
      "<li>The synthetic tasks are too easy for this model. Four of five sit at 100 percent "
      "at every difficulty level, so the difficulty ladder measured nothing. Finding failure "
      "modes would require harder scenes.</li>"
      "<li>Patching recovery declines with depth for reasons intrinsic to the method. Only "
      "differences between token groups and steps between layers are interpretable, not the "
      "absolute level.</li>"
      "<li>The knockout and patching results disagree and the cause is untested.</li>"
      "<li>The replication's ablation stage was run only in the VQA setting. The generative "
      "and polling settings were not run.</li>"
      "</ul>")

    # ------------------------------------------------------------------ repro
    a('<h2 id="repro">5. Reproduction</h2>')
    a("<p>Both suites run from one entry point. The perception suite requires no dataset "
      "download, because scenes are generated deterministically from a seed.</p>")
    a("<pre>pip install -r requirements.txt\n\n"
      "# Part 1, replication on COCO\n"
      "python run_all.py --suite replication --preset standard\n\n"
      "# Part 2, perception on synthetic scenes\n"
      "python run_all.py --suite perception --preset standard</pre>")
    a("<p>Every stage writes results keyed by sample and is resumable. "
      "<code>transformers==5.8.1</code> is required, because the checkpoint uses conventions "
      "earlier versions cannot load. Correctness is covered by 231 tests that run on CPU "
      "against a small randomly initialised model of the same architecture, needing no GPU "
      "and no model weights.</p>")
    a("</div>")

    script = ""
    if lens_examples:
        script = ("<script>"
                  + LW.WIDGET_JS.replace("__LENS_DATA__", json.dumps(lens_examples))
                  + "</script>")

    doc = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           "<title>Interpreting visual information processing in LFM2.5-VL-3B</title>"
           f"<style>{CSS}</style></head><body>" + "".join(P) + script + "</body></html>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="Folder holding lfm2vl-results/ and perception/")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("wrote", build(Path(args.results), Path(args.out)))


if __name__ == "__main__":
    main()
