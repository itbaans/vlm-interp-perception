"""Answer-emergence lens: at which depth does the answer appear?

Logit-lens the **last token** at every layer and read the probability of the correct
answer. One forward pass per sample -- cheap, and it gives the depth axis that patching
gives causally.

Two curves are recorded per sample:

* ``p_answer`` -- P(correct answer) restricted to the task's answer vocabulary, so it
  measures "which answer" rather than "is the model ready to answer at all";
* ``p_answer_full`` -- the same probability over the whole vocabulary, which climbs only
  once the model has also decided to emit an answer token here.

The emergence layer is the first layer where the correct answer becomes the argmax within
the answer vocabulary and stays so to the end -- a transient early crossing is not
emergence.
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
import logit_lens as LL  # noqa: E402
import synthetic as S  # noqa: E402
import tasks as T  # noqa: E402
from experiment import (  # noqa: E402
    ResultsStore,
    add_common_args,
    build_model,
    load_config,
    resolve,
)


def emergence_layer(is_correct: list[bool]) -> int | None:
    """First layer from which the answer is the argmax and stays so through the end."""
    stable = None
    for layer in range(len(is_correct) - 1, -1, -1):
        if is_correct[layer]:
            stable = layer
        else:
            break
    return stable


def analyse(model, sample, task, vocab_ids: dict[str, int]) -> dict:
    prompt = T.build_prompt(model.processor, task, sample)
    outputs = model.forward(sample.clean.render(), prompt, output_hidden_states=True)
    hidden = outputs.hidden_states
    n_layers = len(hidden)

    answers = list(vocab_ids)
    ids = torch.tensor(list(vocab_ids.values()))
    expected_index = answers.index(sample.answer)

    p_answer, p_full, correct = [], [], []
    for layer, state in enumerate(hidden):
        logits = LL.layer_logits(
            state, model.final_norm, model.lm_head,
            is_final_layer=(layer == n_layers - 1), positions=[state.shape[1] - 1],
        )[0]
        full = torch.softmax(logits, dim=-1)
        restricted = torch.softmax(logits[ids.to(logits.device)], dim=-1)

        p_answer.append(float(restricted[expected_index]))
        p_full.append(float(full[vocab_ids[sample.answer]]))
        correct.append(int(restricted.argmax()) == expected_index)

    return {
        "task": sample.task,
        "level": sample.level,
        "answer": sample.answer,
        "n_layers": n_layers,
        "p_answer": p_answer,
        "p_answer_full": p_full,
        "correct_by_layer": correct,
        "emergence_layer": emergence_layer(correct),
        "final_correct": correct[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = resolve(config["paths"]["results_dir"])
    dataset_path = Path(args.dataset) if args.dataset else out_dir / "synthetic_dataset.json"
    if not dataset_path.exists():
        raise SystemExit(f"no dataset at {dataset_path}; run build_dataset.py first")

    samples = [
        S.Sample.from_dict(r)
        for r in json.loads(dataset_path.read_text(encoding="utf-8"))["samples"]
    ]
    if args.tasks:
        samples = [s for s in samples if s.task in args.tasks]
    if args.limit:
        samples = samples[: args.limit]

    model = build_model(args, config)
    vocab = {name: T.TASKS[name].vocabulary_ids(model.processor) for name in T.TASKS}

    store = ResultsStore(
        out_dir / "answer_lens.json",
        meta={"model_id": args.model_id or config["model_id"], "stage": "answer_lens"},
    )

    for sample in tqdm(samples, desc="answer lens"):
        if sample.sample_id in store:
            continue
        store.set(
            sample.sample_id,
            analyse(model, sample, T.TASKS[sample.task], vocab[sample.task]),
        )
        store.save()

    store.save()
    report(store)
    print(f"\nwritten to {out_dir / 'answer_lens.json'}")


def report(store) -> None:
    records = list(store.images.values())
    if not records:
        print("no results")
        return

    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)

    n_layers = records[0]["n_layers"]
    print(f"\n{'=' * 70}")
    print(f"ANSWER-EMERGENCE LENS  ({len(records)} samples)")
    print("P(correct answer) within the answer vocabulary, decoded at the last token.")
    print("=" * 70)

    for task, task_records in sorted(by_task.items()):
        correct = [r for r in task_records if r["final_correct"]]
        print(f"\n--- {task}  ({len(task_records)} samples, {len(correct)} correct) ---")
        if not correct:
            print("  nothing answered correctly; emergence is undefined")
            continue

        layers = [r["emergence_layer"] for r in correct if r["emergence_layer"] is not None]
        if layers:
            print(f"  emergence layer: mean {statistics.mean(layers):.1f}, "
                  f"median {statistics.median(layers):.0f} of {n_layers - 1}")

        print("  mean P(answer) by layer:")
        for layer in range(n_layers):
            value = statistics.mean(r["p_answer"][layer] for r in correct)
            name = "embed" if layer == 0 else f"L{layer}"
            print(f"    {name:>6}: {value * 100:5.1f}%  {'#' * int(value * 50)}")


if __name__ == "__main__":
    main()
