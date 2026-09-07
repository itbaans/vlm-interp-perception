"""Figures for the perception experiments.

Reads the result JSONs and writes PNGs. Runs anywhere, no GPU and no model needed, so
figures can be regenerated from a downloaded bundle.

Design follows src/plots.py: light surface, thin marks, solid hairline grid, and the same
validated palette. Where more than three series would be needed the chart is faceted into
small multiples instead of adding hues.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_RAMP = [
    "#f2f7fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)

ATTENTION_LAYERS = [2, 5, 9, 13, 17, 21, 24, 27]
TASK_ORDER = ["relation", "count", "orientation", "shape", "colour"]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
        "axes.linewidth": 0.8, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
        "legend.fontsize": 9, "legend.frameon": False,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.grid": True, "axes.axisbelow": True, "figure.dpi": 150,
    })


def despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------------------------------
# Shared computation
# ---------------------------------------------------------------------------------------


def recovery_by_layer(records, group, n_layers=30):
    """Mean recovery per layer for one token group."""
    out = []
    for layer in range(n_layers):
        values = [
            r["coarse"][group][str(layer)] for r in records
            if str(layer) in r["coarse"].get(group, {})
            and r["coarse"][group][str(layer)] == r["coarse"][group][str(layer)]
        ]
        out.append(statistics.mean(values) if values else float("nan"))
    return out


def layer_contributions(records, group="changed"):
    """How much each attention layer carried: the drop in recovery just after it."""
    curve = recovery_by_layer(records, group)
    contributions = {}
    for layer in ATTENTION_LAYERS:
        nxt = layer + 1
        while nxt in ATTENTION_LAYERS:
            nxt += 1
        contributions[layer] = curve[layer] - (curve[nxt] if nxt < len(curve) else 0.0)
    return contributions


def by_task(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["task"]].append(record)
    return grouped


def ordered_tasks(grouped):
    return [t for t in TASK_ORDER if t in grouped] + [
        t for t in sorted(grouped) if t not in TASK_ORDER
    ]


# ---------------------------------------------------------------------------------------
# Figure 1: where each ability is read out
# ---------------------------------------------------------------------------------------


def figure_depth(patching, output: Path) -> Path:
    """The headline: which attention layer carries each ability's information."""
    style()
    records = [r for r in patching["images"].values() if "coarse" in r]
    grouped = by_task(records)
    tasks = ordered_tasks(grouped)

    matrix = [list(layer_contributions(grouped[t]).values()) for t in tasks]
    peak = [max(range(len(row)), key=row.__getitem__) for row in matrix]

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    top = max(max(row) for row in matrix)
    image = ax.imshow(matrix, cmap=SEQ, vmin=0, vmax=top, aspect="auto")

    ax.set_xticks(range(len(ATTENTION_LAYERS)), [f"L{a}" for a in ATTENTION_LAYERS])
    ax.set_yticks(range(len(tasks)), [f"{t}  (n={len(grouped[t])})" for t in tasks])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for row in range(len(tasks)):
        for col in range(len(ATTENTION_LAYERS)):
            value = matrix[row][col]
            dark = value > 0.55 * top
            ax.text(col, row, f"{value:.2f}" if value > 0.005 else "-",
                    ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK,
                    fontweight="bold" if col == peak[row] else "normal")
        # Ring the peak so the ordering reads at a glance.
        ax.add_patch(plt.Rectangle((peak[row] - 0.5, row - 0.5), 1, 1, fill=False,
                                   edgecolor=ORANGE, linewidth=2.2))

    bar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.015)
    bar.set_label("information carried", fontsize=9, color=INK_2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)

    ax.set_xlabel("attention layer (the only 8 of 30 layers that move information "
                  "between positions)")
    fig.suptitle("Where each ability is read out", x=0.005, ha="left", fontsize=14, y=1.17)
    fig.text(0.005, 1.06,
             "Outlined cell = the layer carrying most of that ability's information. "
             "Tasks ordered by depth.",
             ha="left", fontsize=9, color=INK_3)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------------------
# Figure 2: the staircase, and what it means
# ---------------------------------------------------------------------------------------


def figure_staircase(patching, output: Path) -> Path:
    style()
    records = [r for r in patching["images"].values() if "coarse" in r]
    layers = list(range(30))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [3, 2]})

    ax = axes[0]
    changed_curve = recovery_by_layer(records, "changed")

    # Shade the conv stretches: nothing can travel there, so nothing changes.
    for start, end in zip([0] + ATTENTION_LAYERS, ATTENTION_LAYERS + [29]):
        if end - start > 1:
            ax.axvspan(start, end, color=GRID, alpha=0.45, zorder=0, linewidth=0)
    for layer in ATTENTION_LAYERS:
        ax.axvline(layer, color=ORANGE, linewidth=1.1, alpha=0.7, zorder=1)

    for group, colour, label in (
        ("changed", BLUE, "the object that changed"),
        ("background", "#9ec5f4", "background (223 tokens)"),
        ("distractors", AQUA, "other objects (control)"),
    ):
        ax.plot(layers, recovery_by_layer(records, group), color=colour, linewidth=2,
                linestyle="-", label=label, zorder=3)

    # Name the two things the reader has to understand: a flat run and a drop.
    ax.annotate("flat: no attention layer here,\nso pasting later changes nothing",
                xy=(7.5, changed_curve[7]), xytext=(1.5, 0.20),
                fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_3, linewidth=1))
    biggest = max(ATTENTION_LAYERS, key=lambda a: changed_curve[a] - changed_curve[a + 1])
    ax.annotate(f"drop at L{biggest}: this layer was\ncarrying the most information",
                xy=(biggest + 0.6, (changed_curve[biggest] + changed_curve[biggest + 1]) / 2),
                xytext=(9.0, 0.66), fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_3, linewidth=1))

    ax.set_xlabel("layer the clean activations were pasted into  "
                  "(orange lines = attention layers)")
    ax.set_ylabel("how much of the answer came back")
    ax.set_xlim(-0.5, 29.5)
    ax.set_ylim(-0.03, 0.88)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    despine(ax)
    ax.set_title("Paste early and it works; paste late and it does not",
                 loc="left", fontsize=11, pad=8)

    ax = axes[1]
    contributions = layer_contributions(records)
    names = [f"L{a}" for a in ATTENTION_LAYERS]
    values = list(contributions.values())
    ax.bar(names, values, color=BLUE, width=0.62, zorder=3)
    ax.set_ylabel("information carried")
    ax.set_xlabel("attention layer")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="x", visible=False)
    despine(ax)
    ax.set_title("Size of each drop = what that layer carried",
                 loc="left", fontsize=11, pad=8)

    fig.suptitle("How the object's information reaches the answer", x=0.005, ha="left",
                 fontsize=14, y=1.12)
    fig.text(0.005, 1.03,
             "Only 8 of the 30 layers can move information between positions. Between "
             "them nothing travels, so nothing changes.",
             ha="left", fontsize=9, color=INK_3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------------------
# Figure 3: when the answer appears
# ---------------------------------------------------------------------------------------


def figure_emergence(answer_lens, output: Path) -> Path:
    style()
    records = [r for r in answer_lens["images"].values() if r.get("final_correct")]
    grouped = by_task(records)
    tasks = ordered_tasks(grouped)
    n_layers = max(len(r["p_answer"]) for r in records)

    fig, axes = plt.subplots(1, len(tasks), figsize=(2.5 * len(tasks), 3.2), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        rows = grouped[task]
        curve = [statistics.mean(r["p_answer"][i] * 100 for r in rows) for i in range(n_layers)]
        ax.plot(range(n_layers), curve, color=BLUE, linewidth=2, zorder=3)

        emergence = [r["emergence_layer"] for r in rows if r["emergence_layer"] is not None]
        if emergence:
            mean_layer = statistics.mean(emergence)
            ax.axvline(mean_layer, color=ORANGE, linewidth=1.6, zorder=2)
            ax.text(mean_layer + 0.6, 8, f"L{mean_layer:.0f}", fontsize=8.5, color=ORANGE)

        ax.set_title(f"{task}\n(n={len(rows)})", fontsize=10, color=INK, pad=6)
        ax.set_xlabel("layer")
        ax.set_xlim(0, n_layers - 1)
        ax.set_ylim(0, 104)
        ax.grid(axis="x", visible=False)
        despine(ax)

    axes[0].set_ylabel("chance of the right answer (%)")
    fig.suptitle("When the answer appears at the output position", x=0.005, ha="left",
                 fontsize=14, y=1.20)
    fig.text(0.005, 1.09,
             "Orange line = the layer the answer becomes correct and stays correct. "
             "Flat and near chance until then.",
             ha="left", fontsize=9, color=INK_3)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------------------
# Figure 4: the three methods side by side
# ---------------------------------------------------------------------------------------


def figure_methods(patching, answer_lens, probes, output: Path) -> Path:
    style()
    records = [r for r in patching["images"].values() if "coarse" in r]
    grouped = by_task(records)
    lens = by_task([r for r in answer_lens["images"].values() if r.get("final_correct")])
    tasks = ordered_tasks(grouped)

    used, emerges, decodable = [], [], []
    for task in tasks:
        contributions = layer_contributions(grouped[task])
        used.append(max(contributions, key=contributions.get))
        rows = [r["emergence_layer"] for r in lens.get(task, [])
                if r.get("emergence_layer") is not None]
        emerges.append(statistics.mean(rows) if rows else float("nan"))
        probe_rows = probes["tasks"].get(task, []) if probes else []
        first = next((r["layer"] for r in probe_rows if r["test_accuracy"] >= 0.9), None)
        decodable.append(first if first is not None else float("nan"))

    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    y = list(range(len(tasks)))

    # The three measures often land on the same layer, so nudge each onto its own lane;
    # drawn on one row they would simply hide each other.
    offset = 0.16
    for i in y:
        points = [v for v in (used[i], emerges[i], decodable[i]) if v == v]
        if len(points) > 1:
            ax.plot([min(points), max(points)], [i, i], color=INK_3, linewidth=1,
                    alpha=0.45, zorder=2)

    ax.scatter(used, [v - offset for v in y], s=110, color=BLUE, edgecolor=SURFACE,
               linewidth=2, zorder=5, label="used (patching)")
    ax.scatter(emerges, y, s=110, color=ORANGE, marker="D", edgecolor=SURFACE,
               linewidth=2, zorder=5, label="answer appears")
    ax.scatter(decodable, [v + offset for v in y], s=110, color=AQUA, marker="s",
               edgecolor=SURFACE, linewidth=2, zorder=5, label="linearly readable (probe)")

    ax.set_yticks(y, tasks)
    ax.set_xlabel("layer (of 30)")
    ax.set_xlim(0, 33)  # headroom so the legend never sits on the data
    ax.set_ylim(len(tasks) - 0.5, -0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.09))
    despine(ax)

    fig.suptitle("Three ways of asking 'where'", x=0.005, ha="left", fontsize=14, y=1.18)
    fig.text(0.005, 1.11,
             "Probe values come from ~30 samples and overfit easily; treat them as "
             "indicative only.",
             ha="left", fontsize=9, color=INK_3)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# ---------------------------------------------------------------------------------------
# Figure 5: is the effect specific to the object that changed?
# ---------------------------------------------------------------------------------------


def figure_specificity(patching, output: Path) -> Path:
    style()
    records = [r for r in patching["images"].values() if "coarse" in r]
    grouped = by_task(records)
    tasks = ordered_tasks(grouped)

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    width = 0.38
    positions = range(len(tasks))

    changed, distractors = [], []
    for task in tasks:
        rows = grouped[task]
        changed.append(max(v for v in recovery_by_layer(rows, "changed") if v == v))
        distractors.append(max(v for v in recovery_by_layer(rows, "distractors") if v == v))

    ax.bar([p - width / 2 for p in positions], changed, width, color=BLUE, zorder=3,
           label="the object that changed")
    ax.bar([p + width / 2 for p in positions], distractors, width, color=AQUA, zorder=3,
           label="other objects (control)")

    for i, (a, b) in enumerate(zip(changed, distractors)):
        ax.text(i - width / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=8.5, color=INK)
        ax.text(i + width / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=8.5, color=INK_2)

    ax.set_xticks(list(positions), tasks)
    ax.set_ylabel("best recovery")
    ax.set_ylim(0, max(changed) * 1.22)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    despine(ax)

    fig.suptitle("The effect is specific to the object that changed", x=0.005, ha="left",
                 fontsize=14, y=1.12)
    fig.text(0.005, 1.03,
             "Pasting clean activations into other objects does essentially nothing.",
             ha="left", fontsize=9, color=INK_3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="Folder holding patching.json, answer_lens.json, probes.json")
    parser.add_argument("--out", default=None, help="Where to write PNGs")
    args = parser.parse_args()

    results = Path(args.results)
    out = Path(args.out) if args.out else results.parent / "figures"

    patching = load(results / "patching.json")
    answer_lens = load(results / "answer_lens.json")
    probes = load(results / "probes.json")
    if not patching:
        raise SystemExit(f"no patching.json in {results}")

    written = [figure_depth(patching, out / "1_depth.png"),
               figure_staircase(patching, out / "2_staircase.png"),
               figure_specificity(patching, out / "3_specificity.png")]
    if answer_lens:
        written.append(figure_emergence(answer_lens, out / "4_emergence.png"))
        written.append(figure_methods(patching, answer_lens, probes, out / "5_methods.png"))

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
