"""Rebuild the sample viewer from whatever results currently exist.

Run it any time: with no results it is dataset mode, and each stage that has run enriches
the cards. Kept separate from the stage scripts so the viewer can be refreshed without
re-running a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import dataviewer  # noqa: E402
import geometry as G  # noqa: E402
import synthetic as S  # noqa: E402
from experiment import load_config, resolve  # noqa: E402


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("images", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of cards; the dataset is stored interleaved "
                             "so a limit stays balanced across tasks and levels")
    parser.add_argument("--no-processor", action="store_true",
                        help="Skip the tokenizer, so answer token ids are not shown")
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

    baseline = load(out_dir / "baseline.json")
    answer_lens = load(out_dir / "answer_lens.json")
    patching = load(out_dir / "patching.json")

    results: dict[str, dict] = {}
    for sample in samples:
        entry: dict = {}
        if sample.sample_id in baseline:
            entry.update(baseline[sample.sample_id].get("clean", {}))
            entry["separable"] = baseline[sample.sample_id].get("separable")
        if sample.sample_id in answer_lens:
            entry["answer_by_layer"] = answer_lens[sample.sample_id].get("p_answer")
            entry["emergence_layer"] = answer_lens[sample.sample_id].get("emergence_layer")
        if sample.sample_id in patching:
            entry["patching"] = patching[sample.sample_id].get("coarse")
        if entry:
            results[sample.sample_id] = entry

    processor = None
    if not args.no_processor:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(config["model_id"])

    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)
    output = Path(args.output) if args.output else resolve(config["paths"]["viewer"])
    stages = [n for n, d in (("baseline", baseline), ("answer lens", answer_lens),
                             ("patching", patching)) if d]

    dataviewer.build(
        samples, output, grid, processor=processor, results=results,
        title="Perception dataset",
        subtitle=f"{len(samples)} samples · {grid.n_rows}x{grid.n_cols} token grid · "
                 + (f"stages: {', '.join(stages)}" if stages else "dataset mode (no model run yet)"),
    )
    print(f"wrote {output}  ({len(results)} of {len(samples)} samples scored)")


if __name__ == "__main__":
    main()
