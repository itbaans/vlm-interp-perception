"""Attention knockout on the perception tasks (secondary to patching).

Blocks attention from a token group to the last position over layer windows, exactly as in
the COCO replication, and measures how much of the correct answer survives.

**Read this alongside the patching results, not instead of them.** Only 8 of the 30 LM
layers have attention, so knockout cannot see what the other 22 short-conv layers do. It is
included because where the two methods agree the conclusion is much stronger, and where
they disagree that gap is itself the finding.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import patching as PA  # noqa: E402
import synthetic as S  # noqa: E402
import tasks as T  # noqa: E402
from experiment import (  # noqa: E402
    ResultsStore,
    add_common_args,
    build_model,
    load_config,
    resolve,
)
from hooked_lfm2vl import layer_windows  # noqa: E402


def evaluate(model, sample, task, prompt, vocab_ids, block_dict=None) -> dict:
    image = sample.clean.render()
    if block_dict:
        with model.block_attention(block_dict):
            probs = model.next_token_distribution(image, prompt)
    else:
        probs = model.next_token_distribution(image, prompt)

    answers = list(vocab_ids)
    ids = torch.tensor(list(vocab_ids.values()), device=probs.device)
    restricted = probs[ids]
    predicted = answers[int(restricted.argmax())]
    return {
        "predicted": predicted,
        "correct": predicted == sample.answer,
        "p_answer": float(probs[vocab_ids[sample.answer]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if (args.attn_implementation or config["attn_implementation"]) != "eager":
        print("WARNING: attention knockout needs an additive float mask. "
              "Run with --attn-implementation eager.")

    out_dir = resolve(config["paths"]["results_dir"])
    dataset_path = Path(args.dataset) if args.dataset else out_dir / "synthetic_dataset.json"
    baseline_path = Path(args.baseline) if args.baseline else out_dir / "baseline.json"
    if not dataset_path.exists():
        raise SystemExit(f"no dataset at {dataset_path}; run build_dataset.py first")

    samples = [
        S.Sample.from_dict(r)
        for r in json.loads(dataset_path.read_text(encoding="utf-8"))["samples"]
    ]
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["images"]
        correct = [s for s in samples if baseline.get(s.sample_id, {})
                   .get("clean", {}).get("correct")]
        print(f"{len(correct)} of {len(samples)} answered correctly unablated")
        samples = correct
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("nothing to knock out; check the baseline table first")

    model = build_model(args, config)
    windows = layer_windows(model.attention_layers, model.num_layers)
    vocab = {name: T.TASKS[name].vocabulary_ids(model.processor) for name in T.TASKS}
    print(f"attention layers {model.attention_layers} of {model.num_layers}")

    store = ResultsStore(
        out_dir / "knockout.json",
        meta={
            "model_id": args.model_id or config["model_id"],
            "attention_layers": model.attention_layers,
            "num_layers": model.num_layers,
            "windows": windows,
            "stage": "knockout",
        },
    )

    for sample in tqdm(samples, desc="knockout"):
        if sample.sample_id in store:
            continue
        task = T.TASKS[sample.task]
        prompt = T.build_prompt(model.processor, task, sample)
        inputs = model.prepare(sample.clean.render(), prompt)
        groups = PA.token_position_groups(sample, inputs, model)
        last = inputs["input_ids"].shape[1] - 1

        record = {
            "task": sample.task,
            "level": sample.level,
            "baseline": evaluate(model, sample, task, prompt, vocab[sample.task]),
            "knockout": {},
        }

        for group in ("changed", "targets", "distractors", "background"):
            positions = groups.get(group, [])
            if not positions:
                continue
            pairs = [(last, p) for p in positions if last >= p]
            for window_name, layers in windows.items():
                if not layers:
                    continue
                record["knockout"][f"{group}@{window_name}"] = evaluate(
                    model, sample, task, prompt, vocab[sample.task],
                    {layer: pairs for layer in layers},
                )

        store.set(sample.sample_id, record)
        store.save()

    store.save()
    report(store)
    print(f"\nwritten to {out_dir / 'knockout.json'}")


def report(store) -> None:
    records = [r for r in store.images.values() if r.get("baseline", {}).get("correct")]
    if not records:
        print("no correctly-answered samples to report")
        return

    windows = list(store.meta.get("windows", {}))
    groups = ["changed", "targets", "distractors", "background"]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)

    print(f"\n{'=' * 76}")
    print(f"ATTENTION KNOCKOUT  ({len(records)} samples)")
    print("Relative accuracy: 1.00 = no impact, 0.00 = every answer wrong.")
    print(f"Only {len(store.meta.get('attention_layers', []))} of "
          f"{store.meta.get('num_layers')} layers can be blocked at all.")
    print("=" * 76)

    for task, task_records in sorted(by_task.items()):
        print(f"\n--- {task}  ({len(task_records)} samples) ---")
        header = f"{'from':<14}" + "".join(f"{w.replace('_','-'):>12}" for w in windows)
        print(header)
        print("-" * len(header))
        for group in groups:
            line = f"{group:<14}"
            for window in windows:
                key = f"{group}@{window}"
                cell = [r["knockout"][key]["correct"] for r in task_records
                        if key in r["knockout"]]
                line += f"{sum(cell) / len(cell):>12.2f}" if cell else f"{'-':>12}"
            print(line)


if __name__ == "__main__":
    main()
