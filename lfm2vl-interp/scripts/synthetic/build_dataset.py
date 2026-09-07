"""Generate the synthetic perception dataset and its inspection viewer.

Needs no GPU and no model weights -- scenes are drawn deterministically from a seed, so
this runs locally in seconds and the whole suite needs no dataset download at all.

It also asserts, against the real tokenizer, that every answer in every task vocabulary is
exactly one token *before* writing anything. That check is the reason ' 3' (two tokens)
never silently reaches a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import geometry as G  # noqa: E402
import synthetic as S  # noqa: E402
import tasks as T  # noqa: E402
from experiment import add_common_args, load_config, resolve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--tasks", nargs="+", default=None, help=f"default: {list(T.TASKS)}")
    parser.add_argument("--levels", nargs="+", default=None, help=f"default: {list(S.LEVELS)}")
    parser.add_argument("--per-cell", type=int, default=None,
                        help="samples per (task, level) cell")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    synth = config["synthetic"]
    task_names = args.tasks or synth["tasks"]
    levels = args.levels or synth["levels"]
    per_cell = args.per_cell or synth["per_cell"]
    seed = args.seed if args.seed is not None else synth["seed"]

    unknown = set(task_names) - set(T.TASKS)
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(sorted(unknown))}")

    # Fail before writing anything if an answer is not a single token.
    print("verifying single-token answers against the real tokenizer")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id or config["model_id"])
    vocab = T.verify_single_token(processor, task_names)
    for name, ids in vocab.items():
        preview = ", ".join(f"{a}={i}" for a, i in list(ids.items())[:4])
        print(f"  {name:<12} {len(ids)} answers  ({preview}, ...)")

    samples: list[S.Sample] = []
    for task_name in task_names:
        for level in levels:
            samples.extend(T.generate(task_name, level, per_cell, seed=seed))
    # Stored interleaved so that any --limit downstream takes a stratified sample rather
    # than a prefix of one task.
    samples = T.interleave(samples)
    print(f"\ngenerated {len(samples)} samples "
          f"({len(task_names)} tasks x {len(levels)} levels x {per_cell})")

    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)
    if (grid.n_rows, grid.n_cols) != (16, 16):
        raise SystemExit(f"expected a 16x16 token grid, got {grid.n_rows}x{grid.n_cols}")
    print(f"token grid {grid.n_rows}x{grid.n_cols} = {grid.n_tokens} tokens, "
          f"{S.CANVAS // grid.n_cols}px per cell, no resampling")

    out_dir = resolve(config["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "synthetic_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "meta": {
                    "seed": seed, "per_cell": per_cell,
                    "tasks": task_names, "levels": levels,
                    "canvas": S.CANVAS, "grid": [grid.n_rows, grid.n_cols],
                    "answer_token_ids": vocab,
                },
                "samples": [s.to_dict() for s in samples],
            }
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {dataset_path}")

    if not args.no_viewer:
        import dataviewer

        viewer_path = resolve(config["paths"]["viewer"])
        dataviewer.build(
            samples, viewer_path, grid, processor=processor,
            title="Perception dataset",
            subtitle=f"{len(samples)} samples · {grid.n_rows}x{grid.n_cols} token grid · "
                     f"dataset mode (no model run yet)",
        )
        print(f"wrote {viewer_path}")
        print("\nOpen the viewer and check the scenes, the minimal-pair diffs and the")
        print("shape-to-token-cell mapping BEFORE spending any GPU time.")


if __name__ == "__main__":
    main()
