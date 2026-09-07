"""One worked example per perception task: the scene, the question, the answer.

Renders real generated samples so the page shows the actual model inputs.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import geometry as G  # noqa: E402
import synthetic as S  # noqa: E402
import tasks as T  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
LINE = "#e6e5e1"
ORANGE = "#eb6834"

TASK_ORDER = ["count", "colour", "shape", "orientation", "relation"]
SEEDS = {"count": 7, "colour": 42, "shape": 42, "orientation": 42, "relation": 42}


def build(output: Path) -> Path:
    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)
    samples = [T.generate(name, "L3", n=1, seed=SEEDS[name])[0] for name in TASK_ORDER]

    fig = plt.figure(figsize=(14.0, 6.4), facecolor=SURFACE)
    n = len(samples)
    left, gap = 0.035, 0.012
    width = (1 - 2 * left - (n - 1) * gap) / n

    for index, sample in enumerate(samples):
        task = T.TASKS[sample.task]
        x = left + index * (width + gap)

        for row, (image, label) in enumerate((
            (sample.clean.render(), "clean"),
            (sample.counterfactual.render(), "one attribute changed"),
        )):
            ax = fig.add_axes([x, 0.575 - row * 0.295, width, 0.255])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(LINE)
            cells = S.shape_cells(sample.changed_shape, grid)
            cell = S.CANVAS / grid.n_cols
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            ax.add_patch(Rectangle(
                (min(cs) * cell - 5, min(rs) * cell - 5),
                (max(cs) - min(cs) + 1) * cell + 10,
                (max(rs) - min(rs) + 1) * cell + 10,
                fill=False, edgecolor=ORANGE, linewidth=2))
            ax.set_ylabel(label, fontsize=8, color=INK_3, labelpad=4)

        fig.text(x + width / 2, 0.888, sample.task, ha="center", fontsize=13, color=INK)
        question = "\n".join(textwrap.wrap(task.question(sample), 30))
        fig.text(x + width / 2, 0.861, question, ha="center", va="top",
                 fontsize=8.5, color=INK_2)

        answer = task.answer_text(sample.answer)
        answer_cf = task.answer_text(sample.answer_cf)
        # Wrapped to the column width: unwrapped it runs into the next task.
        prefill = chr(10).join(textwrap.wrap(chr(34) + '...' + task.prefill(sample) + chr(34), 32))
        fig.text(x + width / 2, 0.250, prefill, ha='center', va='top',
                 fontsize=7.5, color=INK_3, family='monospace')
        fig.text(x + width / 2, 0.150,
                 f'answer:  {answer.strip()}\nafter the change:  {answer_cf.strip()}',
                 ha='center', va='top', fontsize=9.5, color=INK)

    fig.text(left, 0.988, "The five perception tasks", fontsize=16, color=INK, va="top")
    fig.text(left, 0.950,
             "Each task is a pair of 512x512 scenes differing in exactly one shape "
             "attribute, outlined in orange. The prompt is cut off so the next token is "
             "the answer.",
             fontsize=9.5, color=INK_3, va="top")
    fig.text(left, 0.040,
             "Every answer is a single token. Counting uses a bare digit (the prompt ends "
             "with a space); the other tasks use a space-prefixed word.",
             fontsize=9, color=INK_3, va="top")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("wrote", build(Path(args.out)))


if __name__ == "__main__":
    main()
