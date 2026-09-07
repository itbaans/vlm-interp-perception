"""Schematic of the activation-patching method.

Explains what the experiment does, using a real generated sample so the pictures are the
actual inputs rather than cartoons. Needs no GPU and no model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import geometry as G  # noqa: E402
import synthetic as S  # noqa: E402
import tasks as T  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
LINE = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

ATTENTION_LAYERS = [2, 5, 9, 13, 17, 21, 24, 27]
N_LAYERS = 30
PATCH_LAYER = 13


def layer_stack(ax, x0, x1, y, height, patched_at=None, label_layers=False):
    """Draw the 30 decoder layers as a strip, attention layers picked out."""
    width = (x1 - x0) / N_LAYERS
    for layer in range(N_LAYERS):
        attention = layer in ATTENTION_LAYERS
        ax.add_patch(Rectangle(
            (x0 + layer * width, y - height / 2), width * 0.86, height,
            facecolor=ORANGE if attention else "#dfe6ee",
            edgecolor="none", zorder=3,
        ))
        if patched_at == layer:
            ax.add_patch(Rectangle(
                (x0 + layer * width - width * 0.12, y - height / 2 - 0.012),
                width * 1.1, height + 0.024,
                facecolor="none", edgecolor=BLUE, linewidth=2, zorder=6,
            ))
    if label_layers:
        for layer in (0, PATCH_LAYER, N_LAYERS - 1):
            ax.text(x0 + layer * width + width * 0.43, y - height / 2 - 0.028,
                    f"L{layer}", ha="center", va="top", fontsize=7.5, color=INK_3, zorder=6)


def arrow(ax, start, end, colour=INK_3, width=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, color=colour,
                                 linewidth=width, mutation_scale=11, zorder=5))


def answer_box(ax, x, y, text, colour, note=""):
    ax.add_patch(Rectangle((x, y - 0.032), 0.075, 0.064, facecolor=SURFACE,
                           edgecolor=colour, linewidth=1.8, zorder=4))
    ax.text(x + 0.037, y, text, ha="center", va="center", fontsize=15,
            color=colour, zorder=5, fontweight="bold")
    if note:
        ax.text(x + 0.037, y - 0.048, note, ha="center", va="top", fontsize=8,
                color=INK_3, zorder=5)


def build(output: Path, seed: int = 7) -> Path:
    sample = T.generate("count", "L3", n=1, seed=seed)[0]
    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)
    changed_cells = S.shape_cells(sample.changed_shape, grid)
    question = T.TASKS["count"].question(sample)

    fig = plt.figure(figsize=(12.4, 7.4), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rows = [
        (0.755, sample.clean.render(), "clean image", sample.answer, AQUA,
         "the model is right"),
        (0.505, sample.counterfactual.render(), "one square recoloured",
         sample.answer_cf, ORANGE, "still right, different answer"),
        (0.255, sample.counterfactual.render(), "same changed image, but patched",
         "?", BLUE, ""),
    ]

    image_x, stack_x0, stack_x1 = 0.045, 0.30, 0.74
    for y, image, caption, answer, colour, note in rows:
        thumb = fig.add_axes([image_x, y - 0.075, 0.115, 0.155])
        thumb.imshow(image)
        thumb.set_xticks([])
        thumb.set_yticks([])
        for spine in thumb.spines.values():
            spine.set_edgecolor(LINE)
        # Ring the square that differs between the two images.
        cell = S.CANVAS / grid.n_cols
        rows_c = [r for r, _ in changed_cells]
        cols_c = [c for _, c in changed_cells]
        thumb.add_patch(Rectangle(
            (min(cols_c) * cell - 4, min(rows_c) * cell - 4),
            (max(cols_c) - min(cols_c) + 1) * cell + 8,
            (max(rows_c) - min(rows_c) + 1) * cell + 8,
            fill=False, edgecolor=ORANGE, linewidth=2.2))

        ax.text(image_x + 0.058, y + 0.093, caption, ha="center", fontsize=9, color=INK_2)
        arrow(ax, (image_x + 0.125, y), (stack_x0 - 0.012, y))
        layer_stack(ax, stack_x0, stack_x1, y, 0.052,
                    patched_at=PATCH_LAYER if answer == "?" else None,
                    label_layers=(answer == "?"))
        arrow(ax, (stack_x1 + 0.004, y), (stack_x1 + 0.045, y))
        answer_box(ax, stack_x1 + 0.052, y, answer, colour, note)

    # The copy arrow: clean activations at layer 13 spliced into the patched run.
    width = (stack_x1 - stack_x0) / N_LAYERS
    patch_x = stack_x0 + PATCH_LAYER * width + width * 0.43
    ax.plot([patch_x, patch_x], [0.755 - 0.032, 0.505 + 0.052], color=BLUE, linewidth=2.0,
            zorder=5, solid_capstyle="butt")
    arrow(ax, (patch_x, 0.505 - 0.052), (patch_x, 0.255 + 0.036), colour=BLUE, width=2.0)
    ax.text(patch_x + 0.012, 0.38,
            "copy the ~7 numbers-per-position\n"
            "sitting at the changed square,\n"
            "at this layer only",
            ha="left", va="center", fontsize=9, color=BLUE, zorder=7,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=3))

    ax.text(0.045, 0.975, "How the patching experiment works", fontsize=16, color=INK,
            va="top")
    ax.text(0.045, 0.940,
            f'Every run is asked:  "{question}"  '
            f'— cut off so the next word is the answer',
            fontsize=9.5, color=INK_3, va="top")

    ax.text(0.045, 0.900, "the 30 language-model layers:", fontsize=9, color=INK_2,
            va="center")
    ax.add_patch(Rectangle((0.245, 0.892), 0.013, 0.017, facecolor=ORANGE,
                           edgecolor="none"))
    ax.text(0.263, 0.900, "attention (8) — the only layers that move information between "
            "positions", va="center", fontsize=8.5, color=INK_3)
    ax.add_patch(Rectangle((0.245, 0.866), 0.013, 0.017, facecolor="#dfe6ee",
                           edgecolor="none"))
    ax.text(0.263, 0.874, "short convolution (22) — only see immediate neighbours",
            va="center", fontsize=8.5, color=INK_3)

    ax.text(0.045, 0.115,
            "Repeat for all 30 layers, one run each.\n"
            "Recovery = how far the answer moved back toward the clean one:\n"
            "0 means the patch did nothing, 1 means it fully restored it.",
            fontsize=9.5, color=INK_2, va="top")

    ax.text(0.52, 0.115,
            "Controls that make it meaningful:\n"
            "  paste into the squares that did NOT change  ->  0.001\n"
            "  paste into every position at layer 0        ->  1.000 (sanity check)",
            fontsize=9.5, color=INK_2, va="top", family="monospace")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print("wrote", build(Path(args.out), seed=args.seed))


if __name__ == "__main__":
    main()
