"""Figures for the replication report.

Design notes, so later edits do not quietly undo them:

* **Three categorical hues, assigned in fixed order** -- blue (Random baseline),
  orange (Integrated Gradients baseline), aqua (object-based conditions). Validated
  all-pairs on the light surface: worst CVD deltaE 9.2, worst normal-vision 24.0.
  Aqua sits below 3:1 contrast, so every aqua mark carries a visible direct label and
  the report ships a table view beside each figure.
* **Object conditions share one hue and are told apart by direct labels**, not by a
  fourth and fifth colour. Adding hues there would push the palette past what
  validates for scatter marks, and the labels read better anyway.
* **One y-axis per plot, ever.** Degradation-% is the only y measure; token count is
  the x axis.
* Thin marks, solid hairline grid, no dashed rules, no value printed on every point.

Committed to a light surface: these are PNGs inside a downloadable report, so there is
no viewer theme to follow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import aggregate as A  # noqa: E402

# --- design tokens ---------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e6e5e1"

SERIES_RANDOM = "#2a78d6"  # categorical slot 1
SERIES_GRADIENT = "#eb6834"  # slot 2
SERIES_OBJECT = "#1baf7a"  # slot 3

# Sequential blue ramp, steps 100 -> 700, for magnitude encoding.
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)

FONT = ["DejaVu Sans", "Segoe UI", "Helvetica", "sans-serif"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # never dashed
            "axes.grid": True,
            "axes.axisbelow": True,
            "figure.dpi": 140,
        }
    )


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _no_data(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=TEXT_MUTED, fontsize=10)
    ax.set_axis_off()


# --------------------------------------------------------------------------------------
# Figure 1 -- ablation
# --------------------------------------------------------------------------------------


def _ablation_panel(ax, rows, setting: str, title: str, token_budget: float | None) -> None:
    """One setting: degradation against the number of tokens ablated."""
    baselines = {"random": [], "gradient": []}
    objects = []
    for row in rows:
        # Plot degradation, the paper's Table 1 units: higher = the ablation mattered
        # more, so a strong object effect sits visibly ABOVE the baseline curves.
        value = row.decrease(setting)
        if value is None:
            continue
        if row.family in baselines:
            baselines[row.family].append((row.avg_tokens, value))
        else:
            objects.append((row.avg_tokens, value, A.SHORT.get(row.condition, row.label)))

    # Every condition ablates at least one token, so a plain log axis works and avoids
    # symlog's meaningless "0" tick and its compressed low-count region.
    ax.set_xscale("log")

    if token_budget:
        # Conditions at the token budget ablate essentially the whole image, so their
        # collapse is not a "baseline at a matched count" -- mark it rather than let it
        # read as a finding.
        ax.axvline(token_budget, color=TEXT_MUTED, linewidth=0.9, zorder=1)
        ax.annotate("all visual\ntokens", (token_budget, 4), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=TEXT_MUTED, zorder=2)

    for key, colour, name in (
        ("random", SERIES_RANDOM, "Random"),
        ("gradient", SERIES_GRADIENT, "Int. gradients"),
    ):
        points = sorted(baselines[key])
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=colour, linewidth=2, linestyle="-", marker="o", markersize=5,
                markerfacecolor=colour, markeredgecolor=SURFACE, markeredgewidth=2,
                label=name, zorder=3)

    if objects:
        xs, ys, names = zip(*objects)
        ax.scatter(xs, ys, s=95, color=SERIES_OBJECT, edgecolor=SURFACE, linewidth=2,
                   zorder=5, label="Object-based", marker="D")
        # Direct labels: the object conditions are told apart by text, not by hue.
        # Always above the marker -- placing them below runs them into the x-axis tick
        # labels for the low-lying points.
        for x, y, name in sorted(objects):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=8.5, color=TEXT_PRIMARY, zorder=6)

    ax.set_title(title, color=TEXT_PRIMARY, pad=10, loc="left")
    ax.set_xlabel("visual tokens ablated")
    ax.set_ylim(-6, 112)
    ax.grid(axis="x", visible=False)
    _despine(ax)


def figure_ablation(
    ablation: dict | None, vqa: dict | None, output: Path, token_budget: float | None = None
) -> Path | None:
    """Table 1 as a figure: the object conditions should sit *above* both baselines.

    Reading it: x is how many visual tokens were replaced, y is the share of
    initially-correct identifications destroyed. A point high on the y axis at a small x
    is a strong result -- little was ablated, yet the model lost the object.
    """
    _style()
    panels = []
    if ablation:
        rows = A.ablation_rows(ablation, ("generative", "polling"))
        panels.append((rows, "generative", "Generative"))
        panels.append((rows, "polling", "Polling"))
    if vqa:
        rows, _ = A.vqa_rows(vqa)
        panels.append((rows, "vqa", "VQA"))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (rows, setting, title) in zip(axes, panels):
        if rows:
            _ablation_panel(ax, rows, setting, title, token_budget)
        else:
            _no_data(ax, f"no {title.lower()} data")

    axes[0].set_ylabel("correct identifications destroyed (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))

    subtitle = "higher is a stronger effect"
    if token_budget:
        subtitle += f"  ·  ~{token_budget:.0f} visual tokens per image"
    # One left-aligned title block, so nothing can overlap a corner annotation. The
    # title and subtitle are spaced apart explicitly; suptitle's own padding is not
    # enough to clear a separate text line beneath it.
    fig.suptitle("Token ablation", x=0.005, ha="left", fontsize=14,
                 color=TEXT_PRIMARY, y=1.12)
    fig.text(0.005, 1.035, subtitle, ha="left", fontsize=9.5, color=TEXT_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.99))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# --------------------------------------------------------------------------------------
# Figure 2 -- logit lens
# --------------------------------------------------------------------------------------


def figure_logit_lens(results: dict, output: Path) -> Path | None:
    """Per-layer share of object tokens decoding to the class name.

    The paper's claim is a rise through the middle layers peaking mid-to-late, so the
    shape matters more than the height.
    """
    summary = A.logit_lens_summary(results)
    if not summary["n_images"]:
        return None

    _style()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    layers = list(range(summary["n_entries"]))

    for index, (criterion, colour, name) in enumerate((
        ("strict", SERIES_RANDOM, "Strict (exact class token)"),
        ("lenient", SERIES_GRADIENT, "Lenient (word-piece prefix)"),
    )):
        values = [v * 100 for v in summary["per_layer"][criterion]]
        ax.plot(layers, values, color=colour, linewidth=2, linestyle="-", label=name, zorder=3)

        peak = max(range(len(values)), key=values.__getitem__)
        ax.scatter([peak], [values[peak]], s=70, color=colour, edgecolor=SURFACE,
                   linewidth=2, zorder=5)
        # Direct-label the peak only -- never a number on every point. The two curves
        # peak at the same layer, so the labels are offset left/right of the marker to
        # keep them from overlapping each other and the marker.
        left = index == 0
        ax.annotate(f"L{peak}: {values[peak]:.1f}%", (peak, values[peak]),
                    textcoords="offset points", xytext=(-12 if left else 12, 10),
                    ha="right" if left else "left",
                    fontsize=9, color=colour, zorder=6)

    num_layers = summary["num_layers"] or summary["n_entries"] - 1
    ax.set_xlabel(f"layer (0 = input embeddings, {num_layers} = final)")
    ax.set_ylabel("object tokens decoding to the class (%)")
    ax.set_xlim(-0.5, summary["n_entries"] - 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    _despine(ax)

    ax.set_title(
        f"Logit lens across layers  ({summary['n_images']} images)",
        loc="left", fontsize=13, color=TEXT_PRIMARY, pad=12,
    )
    fig.text(0.005, -0.02, "Paper reference: LLaVA-1.5 peaks at 23.7% around layer 25.7 of 33.",
             fontsize=8.5, color=TEXT_MUTED)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


# --------------------------------------------------------------------------------------
# Figure 3 -- attention knockout
# --------------------------------------------------------------------------------------


def figure_knockout(results: dict, output: Path) -> Path | None:
    """Heatmap of relative accuracy per token group and layer window, plus the sweep."""
    summary = A.knockout_summary(results)
    if not summary["n_usable"] or not summary["grid"]:
        return None

    _style()
    has_sweep = bool(summary["sweep"])
    fig, axes = plt.subplots(
        2 if has_sweep else 1, 1,
        figsize=(9.0, 6.8 if has_sweep else 4.2),
        gridspec_kw={"height_ratios": [3, 2]} if has_sweep else None,
    )
    axes = axes if has_sweep else [axes]

    windows = summary["windows"]
    pairs = list(summary["grid"].keys())
    matrix = [[summary["grid"][p].get(w, float("nan")) for w in windows] for p in pairs]

    ax = axes[0]
    # Colour by accuracy *lost* (1 - value), which has a true zero and spans the range
    # the data actually occupies. Colouring relative accuracy on a fixed 0-1 scale made
    # every cell the same dark navy, because the effects here live in 0.89-1.00.
    # The printed numbers stay as relative accuracy, matching the paper's Table 2.
    lost = [[1.0 - v for v in row] for row in matrix]
    finite = [v for row in lost for v in row if v == v]
    vmax = max(max(finite), 0.02) if finite else 1.0
    image = ax.imshow(lost, cmap=SEQUENTIAL, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(windows)), [w.replace("_", "-") for w in windows])
    ax.set_yticks(range(len(pairs)), [f"{s} → {t}" for s, t in pairs])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Cell values are the table view in-figure; they fit, so they are safe to draw.
    for row in range(len(pairs)):
        for col in range(len(windows)):
            value = matrix[row][col]
            if value != value:  # NaN
                continue
            # Contrast follows the cell's colour, which is driven by accuracy lost.
            dark_cell = (1.0 - value) > 0.55 * vmax
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark_cell else TEXT_PRIMARY)

    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label("accuracy lost", fontsize=9, color=TEXT_SECONDARY)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)

    ax.set_title(
        f"Attention knockout  ({summary['n_usable']} images)\n"
        "numbers are relative accuracy (1.00 = no impact); colour is accuracy lost",
        loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12,
    )

    if has_sweep:
        ax = axes[1]
        names = sorted(summary["sweep"], key=lambda n: (n[0], int(n[1:].split("-")[0])))
        # Bars show accuracy *lost*, so they grow from a true zero and the differences
        # are visible. Plotting relative accuracy would need a truncated y axis to show
        # any variation at all, which would misrepresent the magnitudes.
        values = [1.0 - summary["sweep"][n] for n in names]
        ax.bar(names, values, color=SERIES_RANDOM, width=0.62, zorder=3)
        ax.set_ylim(0, max(max(values) * 1.25, 0.02))
        ax.set_ylabel("accuracy lost")
        ax.set_xlabel("blocked attention layer (L) or sliding window (W)")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        _despine(ax)
        ax.set_title("Per-layer sweep: object+1 → last token position",
                     loc="left", fontsize=10.5, color=TEXT_SECONDARY, pad=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def build_all(paths: dict[str, Path], figures_dir: Path) -> dict[str, Path]:
    """Generate every figure whose inputs exist. Returns the ones written."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    ablation = A.load(paths["ablation"]) if paths.get("ablation", Path()).exists() else None
    vqa = A.load(paths["vqa"]) if paths.get("vqa", Path()).exists() else None

    if ablation or vqa:
        stats = A.visual_token_stats(ablation or vqa)
        figure = figure_ablation(
            ablation, vqa, figures_dir / "ablation.png",
            token_budget=stats["mean"] if stats else None,
        )
        if figure:
            written["ablation"] = figure

    if paths.get("logit_lens", Path()).exists():
        figure = figure_logit_lens(A.load(paths["logit_lens"]), figures_dir / "logit_lens.png")
        if figure:
            written["logit_lens"] = figure

    if paths.get("knockout", Path()).exists():
        figure = figure_knockout(A.load(paths["knockout"]), figures_dir / "knockout.png")
        if figure:
            written["knockout"] = figure

    return written
