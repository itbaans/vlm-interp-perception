"""Baseline accuracy per (task x level) -- the gate for everything downstream.

Localizing "where counting happens" is meaningless if the model cannot count, so this runs
first and decides which cells are worth investigating. It is one forward pass per sample:
the answer is a single token, so the next-token distribution *is* the answer distribution.

Two things are recorded beyond correct/incorrect:

* the full distribution over the task's answer vocabulary, so a model that is confidently
  wrong is distinguishable from one that is spreading mass evenly;
* for counting, the predicted count itself, so "off by one" is distinguishable from
  "guessing" -- the digits are contiguous token ids, which makes this free.

Both the clean and the counterfactual scene are scored, because activation patching later
needs samples where the model gets *both* right; a pair it cannot separate carries no
signal to recover.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import geometry as G  # noqa: E402
import synthetic as S  # noqa: E402
import tasks as T  # noqa: E402
from experiment import (  # noqa: E402
    ResultsStore,
    add_common_args,
    build_model,
    load_config,
    resolve,
)


def score(model, task, sample, variant: str, vocab_ids: dict[str, int]) -> dict:
    """Next-token distribution restricted to this task's answer vocabulary."""
    prompt = T.build_prompt(model.processor, task, sample)
    image = sample.scene(variant).render()
    probs = model.next_token_distribution(image, prompt)

    ids = torch.tensor(list(vocab_ids.values()), device=probs.device)
    answers = list(vocab_ids)
    restricted = probs[ids]
    predicted = answers[int(restricted.argmax())]

    expected = sample.answer if variant == "clean" else sample.answer_cf
    return {
        "predicted": predicted,
        "expected": expected,
        "correct": predicted == expected,
        # Probability mass the model put on the answer space at all -- a low value means
        # it wanted to say something outside the vocabulary, which the argmax would hide.
        "in_vocab_mass": float(restricted.sum()),
        "p_expected": float(probs[vocab_ids[expected]]),
        "p_predicted": float(probs[vocab_ids[predicted]]),
        "distribution": {a: float(probs[i]) for a, i in vocab_ids.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = resolve(config["paths"]["results_dir"])
    dataset_path = Path(args.dataset) if args.dataset else out_dir / "synthetic_dataset.json"
    if not dataset_path.exists():
        raise SystemExit(f"no dataset at {dataset_path}; run build_dataset.py first")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = [S.Sample.from_dict(r) for r in dataset["samples"]]
    if args.limit:
        samples = samples[: args.limit]
    print(f"{len(samples)} samples from {dataset_path}")

    model = build_model(args, config)
    vocab = {name: T.TASKS[name].vocabulary_ids(model.processor) for name in T.TASKS}

    store = ResultsStore(
        out_dir / "baseline.json",
        meta={
            "model_id": args.model_id or config["model_id"],
            "dataset": str(dataset_path),
            "stage": "baseline",
        },
    )

    for sample in tqdm(samples, desc="baseline"):
        if sample.sample_id in store:
            continue
        task = T.TASKS[sample.task]
        record = {
            "task": sample.task,
            "level": sample.level,
            "clean": score(model, task, sample, "clean", vocab[sample.task]),
            "counterfactual": score(model, task, sample, "counterfactual", vocab[sample.task]),
        }
        # Patching needs both halves right: the pair must actually be separated.
        record["separable"] = record["clean"]["correct"] and record["counterfactual"]["correct"]
        store.set(sample.sample_id, record)
        store.save()

    store.save()
    report(store, samples)

    if not args.no_viewer:
        import dataviewer

        grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)
        viewer_results = {
            sid: {
                **r["clean"],
                "answer_by_layer": r.get("answer_by_layer"),
            }
            for sid, r in store.images.items()
        }
        viewer_path = resolve(config["paths"]["viewer"])
        dataviewer.build(
            samples, viewer_path, grid, processor=model.processor, results=viewer_results,
            title="Perception dataset",
            subtitle=f"{len(samples)} samples · baseline scored",
        )
        print(f"\nviewer updated: {viewer_path}")


def report(store, samples) -> None:
    records = store.images
    if not records:
        print("no results")
        return

    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records.values():
        by_cell[(record["task"], record["level"])].append(record)

    tasks = sorted({t for t, _ in by_cell})
    levels = sorted({l for _, l in by_cell})

    header = f"{'Task':<14}" + "".join(f"{l:>12}" for l in levels) + f"{'overall':>12}"
    print("\n" + "=" * len(header))
    print(f"BASELINE ACCURACY on the clean scene ({len(records)} samples)")
    print("Localization only makes sense in cells the model actually gets right.")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for task in tasks:
        line = f"{task:<14}"
        overall = []
        for level in levels:
            cell = by_cell.get((task, level), [])
            if not cell:
                line += f"{'-':>12}"
                continue
            accuracy = sum(r["clean"]["correct"] for r in cell) / len(cell)
            overall.extend(r["clean"]["correct"] for r in cell)
            line += f"{accuracy * 100:>11.0f}%"
        line += f"{statistics.mean(overall) * 100:>11.0f}%" if overall else f"{'-':>12}"
        print(line)

    print("\n" + "-" * len(header))
    print("SEPARABLE PAIRS -- both clean and counterfactual correct (usable for patching)")
    print("-" * len(header))
    for task in tasks:
        line = f"{task:<14}"
        overall = []
        for level in levels:
            cell = by_cell.get((task, level), [])
            if not cell:
                line += f"{'-':>12}"
                continue
            rate = sum(r["separable"] for r in cell) / len(cell)
            overall.extend(r["separable"] for r in cell)
            line += f"{rate * 100:>11.0f}%"
        line += f"{statistics.mean(overall) * 100:>11.0f}%" if overall else f"{'-':>12}"
        print(line)

    # Counting deserves a closer look: is it off by one, or guessing?
    counts = [r for r in records.values() if r["task"] == "count"]
    if counts:
        errors = [
            abs(int(r["clean"]["predicted"]) - int(r["clean"]["expected"])) for r in counts
        ]
        print(f"\ncount: mean absolute error {statistics.mean(errors):.2f}, "
              f"within 1: {100 * sum(e <= 1 for e in errors) / len(errors):.0f}%")

    mass = [r["clean"]["in_vocab_mass"] for r in records.values()]
    print(f"answer-vocabulary mass: mean {statistics.mean(mass):.2f} "
          f"(low means the model wanted to say something else entirely)")


if __name__ == "__main__":
    main()
