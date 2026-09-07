"""Activation patching over minimal pairs: where is each ability computed?

For every separable sample -- one the model answers correctly on *both* the clean and the
counterfactual scene -- splice the clean residual stream into the counterfactual run at
each (layer, token group) and measure how much of the clean answer returns.

Two passes:

* **coarse** -- every layer x every token group. ~8 groups x 30 layers per sample.
* **fine** -- at the best few layers from the coarse pass, patch each visual token on its
  own, producing a spatial recovery map that can be laid over the image.

Only separable samples are used: if the model already gets the counterfactual wrong there
is nothing for a patch to restore, and including those would dilute the signal with noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

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

#: Exit code meaning "the stage ran fine, but its gate left nothing to patch".
#: Distinct from a crash so an orchestrated run can skip ahead instead of aborting.
GATED_EXIT_CODE = 3


def fine_pass(model, sample, task, processor, run, grid, layers) -> dict[int, list[float]]:
    """Recovery from patching each visual token individually, per layer."""
    out: dict[int, list[float]] = {}
    for layer in layers:
        out[layer] = [
            PA.patch_and_score(model, sample, task, processor, run, layer, [index])
            for index in grid.indices
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Restrict the coarse sweep (default: every layer)")
    parser.add_argument("--no-fine-pass", action="store_true")
    parser.add_argument(
        "--fine-samples", type=int, default=None,
        help="How many samples get the per-token fine pass. It costs roughly 3x the "
        "coarse sweep per sample (fine_pass_layers x 256 forwards versus 8 groups x 30), "
        "so it is capped by default: the layer x group result comes from every sample, "
        "the spatial maps from a subset.",
    )
    parser.add_argument(
        "--ignore-gate",
        action="store_true",
        help="Patch samples the model answers wrongly too. Recovery is not interpretable "
        "for those -- there is no correct answer to restore -- so this is for validating "
        "the pipeline, not for producing results.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = resolve(config["paths"]["results_dir"])

    dataset_path = Path(args.dataset) if args.dataset else out_dir / "synthetic_dataset.json"
    baseline_path = Path(args.baseline) if args.baseline else out_dir / "baseline.json"
    for path in (dataset_path, baseline_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; run build_dataset.py and baseline.py first")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["images"]
    samples = [S.Sample.from_dict(r) for r in dataset["samples"]]

    if args.tasks:
        samples = [s for s in samples if s.task in args.tasks]

    separable = [s for s in samples if baseline.get(s.sample_id, {}).get("separable")]
    print(f"{len(separable)} separable of {len(samples)} samples "
          f"(both clean and counterfactual answered correctly)")
    if args.ignore_gate and not separable:
        print("WARNING: --ignore-gate, patching samples the model answers wrongly. "
              "Recovery values below are NOT interpretable as results.")
        separable = samples
    elif not separable:
        print(
            "no separable samples. The model cannot answer these tasks reliably enough to "
            "patch -- read the baseline table and pick easier (task, level) cells."
        )
        # Distinct from a crash: the stage ran correctly and found nothing to do, so an
        # orchestrated run can skip ahead instead of aborting the remaining stages.
        raise SystemExit(GATED_EXIT_CODE)

    limit = args.limit or config["patching"]["max_samples"]
    separable = separable[:limit]

    model = build_model(args, config)
    layers = args.layers if args.layers is not None else list(range(model.num_layers))
    n_fine = 0 if args.no_fine_pass else config["patching"]["fine_pass_layers"]
    fine_budget = (
        0 if args.no_fine_pass
        else (args.fine_samples if args.fine_samples is not None
              else config["patching"].get("fine_samples", 12))
    )
    if n_fine:
        print(f"coarse sweep on all {len(separable)} samples; "
              f"per-token fine pass on the first {fine_budget}")
    fine_done = 0

    store = ResultsStore(
        out_dir / "patching.json",
        meta={
            "model_id": args.model_id or config["model_id"],
            "layers": layers,
            "num_layers": model.num_layers,
            "attention_layers": model.attention_layers,
            "stage": "patching",
            "fine_pass_layers": n_fine,
        },
    )

    for sample in tqdm(separable, desc="patching"):
        if sample.sample_id in store:
            continue
        task = T.TASKS[sample.task]
        prompt = T.build_prompt(model.processor, task, sample)

        run = PA.prepare_run(model, sample, task, model.processor)
        if abs(run.separation) < 1e-4:
            # The two runs give the same logit difference; nothing to recover.
            store.set(sample.sample_id, {"task": sample.task, "level": sample.level,
                                         "skipped": "pair not separated"})
            store.save()
            continue

        inputs = model.prepare(sample.clean.render(), prompt)
        groups = PA.token_position_groups(sample, inputs, model)
        coarse = PA.sweep(model, sample, task, model.processor, run, groups, layers=layers)

        record = {
            "task": sample.task,
            "level": sample.level,
            "clean_diff": run.clean_diff,
            "cf_diff": run.cf_diff,
            "separation": run.separation,
            "group_sizes": {k: len(v) for k, v in groups.items()},
            "coarse": {g: {str(l): v for l, v in by_layer.items()}
                       for g, by_layer in coarse.items()},
        }

        if n_fine and fine_done < fine_budget:
            fine_done += 1
            # Pick the layers where patching the changed object mattered most.
            changed = coarse.get("changed", {})
            best = sorted(changed, key=lambda l: -abs(changed[l]))[:n_fine]
            grid = model.single_grid(inputs, (sample.clean.height, sample.clean.width))
            record["fine_layers"] = sorted(best)
            record["fine"] = {
                str(l): v
                for l, v in fine_pass(
                    model, sample, task, model.processor, run, grid, sorted(best)
                ).items()
            }

        store.set(sample.sample_id, record)
        store.save()

    store.save()
    report(store)
    print(f"\nwritten to {out_dir / 'patching.json'}")


def report(store) -> None:
    records = [r for r in store.images.values() if "coarse" in r]
    if not records:
        print("no patching results")
        return

    meta = store.meta
    layers = meta["layers"]
    groups = list(records[0]["coarse"])
    attention = set(meta.get("attention_layers", []))

    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)

    print(f"\n{'=' * 78}")
    print(f"ACTIVATION PATCHING  ({len(records)} samples)")
    print("Mean recovery: 0 = the patch changed nothing, 1 = the clean answer fully returned.")
    print(f"Layers marked * have attention; the rest are short-conv "
          f"({len(attention)} of {meta['num_layers']}).")
    print("=" * 78)

    for task, task_records in sorted(by_task.items()):
        print(f"\n--- {task}  ({len(task_records)} samples) ---")
        header = f"{'layer':<8}" + "".join(f"{g[:11]:>12}" for g in groups)

        # Group size confounds recovery: patching 195 background tokens moves the answer
        # more than patching 10 object tokens almost regardless of what they encode. The
        # comparison that means something is per-token, so print the sizes next to it.
        sizes = {
            g: statistics.mean([r["group_sizes"].get(g, 0) for r in task_records])
            for g in groups
        }
        print(f"{'tokens':<8}" + "".join(f"{sizes[g]:>12.0f}" for g in groups))
        print(header)
        print("-" * len(header))
        for layer in layers:
            values = []
            for group in groups:
                cell = [
                    r["coarse"][group][str(layer)] for r in task_records
                    if str(layer) in r["coarse"].get(group, {})
                ]
                cell = [v for v in cell if v == v]  # drop NaN
                values.append(statistics.mean(cell) if cell else float("nan"))
            marker = "*" if layer in attention else " "
            line = f"L{layer}{marker:<6}" + "".join(
                f"{v:>12.2f}" if v == v else f"{'-':>12}" for v in values
            )
            print(line)

        # The headline comparison: does patching the changed object beat the background?
        for group in ("changed", "background"):
            peak = max(
                (
                    statistics.mean([
                        r["coarse"][group][str(l)] for r in task_records
                        if str(l) in r["coarse"].get(group, {})
                        and r["coarse"][group][str(l)] == r["coarse"][group][str(l)]
                    ] or [float("nan")]),
                    l,
                )
                for l in layers
            )
            if peak[0] == peak[0]:
                size = sizes.get(group, 0) or 1
                print(f"  peak {group:<12} {peak[0]:.2f} at layer {peak[1]}"
                      f"   ({peak[0] / size:.4f} per token over {size:.0f} tokens)")
        print("  Compare per-token: a larger group recovers more almost by construction.")


if __name__ == "__main__":
    main()
