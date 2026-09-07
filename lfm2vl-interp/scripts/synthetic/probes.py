"""Linear probes on the last-token residual stream, per layer, per task.

Collects the residual stream at the final position for every sample, then fits one probe
per layer to predict the correct answer. Reported against the majority-class baseline,
because a task with 2 answers ("above"/"below") gets 50% for free and a raw accuracy
number would flatter it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import probes as PR  # noqa: E402
import synthetic as S  # noqa: E402
import tasks as T  # noqa: E402
from experiment import add_common_args, build_model, load_config, resolve  # noqa: E402


def collect(model, samples) -> dict[str, tuple[list[torch.Tensor], torch.Tensor]]:
    """Per task: ``(activations_by_layer, labels)`` at the last token position."""
    per_task: dict[str, list[list[torch.Tensor]]] = defaultdict(list)
    per_task_labels: dict[str, list[int]] = defaultdict(list)

    for sample in tqdm(samples, desc="collecting"):
        task = T.TASKS[sample.task]
        prompt = T.build_prompt(model.processor, task, sample)
        outputs = model.forward(sample.clean.render(), prompt, output_hidden_states=True)
        per_task[sample.task].append(
            [state[0, -1, :].detach().float().cpu() for state in outputs.hidden_states]
        )
        per_task_labels[sample.task].append(task.vocabulary.index(sample.answer))

    collected = {}
    for name, rows in per_task.items():
        n_layers = len(rows[0])
        activations = [torch.stack([row[layer] for row in rows]) for layer in range(n_layers)]
        collected[name] = (activations, torch.tensor(per_task_labels[name]))
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--min-samples", type=int, default=20,
                        help="Skip tasks with fewer samples than this; a probe on a handful "
                             "of points reports noise")
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
    if args.limit:
        samples = samples[: args.limit]

    model = build_model(args, config)
    collected = collect(model, samples)

    results = {}
    for task_name, (activations, labels) in collected.items():
        if len(labels) < args.min_samples:
            print(f"skipping {task_name}: only {len(labels)} samples "
                  f"(need {args.min_samples})")
            continue
        probe_results = PR.probe_all_layers(activations, labels)
        results[task_name] = [vars(r) for r in probe_results]

        print(f"\n--- {task_name} ({len(labels)} samples, "
              f"{probe_results[0].n_classes} classes, "
              f"majority baseline {probe_results[0].majority_baseline * 100:.0f}%) ---")
        for result in probe_results:
            name = "embed" if result.layer == 0 else f"L{result.layer}"
            bar = "#" * max(0, int(result.above_baseline * 60))
            print(f"  {name:>6}: test {result.test_accuracy * 100:5.1f}%  "
                  f"({result.above_baseline * 100:+5.1f} pts over baseline)  {bar}")

        best = max(probe_results, key=lambda r: r.above_baseline)
        print(f"  best: layer {best.layer} at {best.test_accuracy * 100:.1f}% "
              f"({best.above_baseline * 100:+.1f} pts)")

    output = out_dir / "probes.json"
    output.write_text(json.dumps({"meta": {"stage": "probes"}, "tasks": results}),
                      encoding="utf-8")
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()
