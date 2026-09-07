"""Combined report across every perception stage that has run.

Prints each stage's own table, then the cross-stage comparison that is the actual point of
the experiment: for each ability, where does patching say the information is *used*, where
does the lens say it *emerges*, and where do probes say it is *decodable*? Those three can
disagree, and the disagreement is informative -- information is usually linearly decodable
well before the model relies on it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from experiment import load_config, resolve  # noqa: E402

STAGES = ("baseline", "answer_lens", "attribute_lens", "patching", "knockout", "probes")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _peak_patching_layer(records: list[dict], group: str = "changed") -> tuple[int, float] | None:
    """Layer where patching ``group`` recovered most, averaged over samples."""
    by_layer: dict[int, list[float]] = defaultdict(list)
    for record in records:
        for layer, value in record.get("coarse", {}).get(group, {}).items():
            if value == value:  # not NaN
                by_layer[int(layer)].append(value)
    if not by_layer:
        return None
    means = {layer: statistics.mean(v) for layer, v in by_layer.items()}
    best = max(means, key=means.get)
    return best, means[best]


def summary(results: dict) -> None:
    """The cross-stage table: three different notions of 'where', side by side."""
    patching = results.get("patching")
    lens = results.get("answer_lens")
    probe = results.get("probes")
    baseline = results.get("baseline")

    tasks: set[str] = set()
    for blob, key in ((patching, "images"), (lens, "images"), (baseline, "images")):
        if blob:
            tasks |= {r["task"] for r in blob[key].values() if "task" in r}
    if probe:
        tasks |= set(probe.get("tasks", {}))
    if not tasks:
        return

    header = (f"{'Task':<14}{'accuracy':>10}{'emerges':>10}{'decodable':>11}"
              f"{'used (patch)':>14}{'recovery':>10}")
    print(f"\n{'=' * len(header)}")
    print("CROSS-STAGE SUMMARY - three different senses of 'where'")
    print("  emerges   : first layer the correct answer is argmax at the last token")
    print("  decodable : layer where a linear probe reads the answer best")
    print("  used      : layer where patching the changed object recovers most")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for task in sorted(tasks):
        accuracy = emerges = decodable = used = recovery = None

        if baseline:
            cell = [r for r in baseline["images"].values() if r.get("task") == task]
            if cell:
                accuracy = statistics.mean(r["clean"]["correct"] for r in cell)

        if lens:
            cell = [r for r in lens["images"].values()
                    if r.get("task") == task and r.get("final_correct")
                    and r.get("emergence_layer") is not None]
            if cell:
                emerges = statistics.mean(r["emergence_layer"] for r in cell)

        if probe and task in probe.get("tasks", {}):
            rows = probe["tasks"][task]
            best = max(rows, key=lambda r: r["test_accuracy"] - r["majority_baseline"])
            decodable = best["layer"]

        if patching:
            cell = [r for r in patching["images"].values()
                    if r.get("task") == task and "coarse" in r]
            peak = _peak_patching_layer(cell)
            if peak:
                used, recovery = peak

        def fmt(value, spec=".0f"):
            return format(value, spec) if value is not None else "-"

        print(f"{task:<14}"
              f"{(f'{accuracy * 100:.0f}%' if accuracy is not None else '-'):>10}"
              f"{fmt(emerges, '.1f'):>10}{fmt(decodable):>11}"
              f"{fmt(used):>14}{fmt(recovery, '.2f'):>10}")

    print("\nIf 'decodable' is much earlier than 'used', the information is present long")
    print("before the model relies on it - a real dissociation, not a measurement error.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.results_dir) if args.results_dir else resolve(
        config["paths"]["results_dir"]
    )

    results = {stage: load(out_dir / f"{stage}.json") for stage in STAGES}
    present = [s for s, v in results.items() if v]
    if not present:
        raise SystemExit(
            f"no results in {out_dir}. Expected any of: "
            + ", ".join(f"{s}.json" for s in STAGES)
        )
    print(f"stages present: {', '.join(present)}")

    # Each stage prints its own table via its own reporter, so the formats stay identical
    # whether you run the stage directly or the combined analysis.
    if results["baseline"]:
        import importlib.util

        for stage, module_name in (
            ("baseline", "baseline"), ("answer_lens", "answer_lens"),
            ("attribute_lens", "attribute_lens"), ("patching", "patching"),
            ("knockout", "knockout"),
        ):
            if not results[stage]:
                continue
            spec = importlib.util.spec_from_file_location(
                f"synthetic_{module_name}", Path(__file__).parent / f"{module_name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            class _Store:
                images = results[stage]["images"]
                meta = results[stage].get("meta", {})

            if stage == "baseline":
                module.report(_Store(), [])
            else:
                module.report(_Store())

    if results["probes"]:
        print(f"\n{'=' * 60}")
        print("LINEAR PROBES (test accuracy over majority baseline)")
        print("=" * 60)
        for task, rows in results["probes"]["tasks"].items():
            best = max(rows, key=lambda r: r["test_accuracy"] - r["majority_baseline"])
            print(f"  {task:<14} best layer {best['layer']:>3}  "
                  f"{best['test_accuracy'] * 100:5.1f}%  "
                  f"({(best['test_accuracy'] - best['majority_baseline']) * 100:+.1f} pts)")

    summary(results)


if __name__ == "__main__":
    main()
