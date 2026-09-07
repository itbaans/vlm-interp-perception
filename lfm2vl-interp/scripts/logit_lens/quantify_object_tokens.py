"""Quantify the logit lens (paper Section 4.1, the 23.7% result).

The paper's headline number for this experiment:

    "We analyzed 170 COCO validation images with objects of sizes between 20,000 and
    30,000 square pixels. We found that in the best-performing layer for each image, an
    average of 23.7% of the object patch token positions correspond to the correct
    object class token. The best-performing layer occurs on average at layer 25.7
    (out of 33)."

and for Qwen2VL-2B, 6.5% at layer 25.1 of 29. The original repo ships the interactive
HTML but never this measurement, so it is implemented here from the description.

Two criteria are reported, because the paper does not say which it used:

* **strict** -- the top-1 decoded token id is the first token of the class name
* **lenient** -- or the decoded string is a >=3-character prefix of it, which is what
  the paper's own Figure 3 labelling does ("swe"(ater), "diam"(ond))

Note the larger object-area window: this uses 20,000-30,000 px^2, not the
1,000-2,000 px^2 window the ablation experiments use.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import prompts as P  # noqa: E402
from coco import CocoInstances, iter_filtered_images  # noqa: E402
from experiment import (  # noqa: E402
    ResultsStore,
    add_common_args,
    build_model,
    coco_paths,
    load_config,
    resolve,
)
from logit_lens import object_token_hit_rate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--area-window", type=float, nargs=2, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--question", default=P.DESCRIBE_QUESTION)
    args = parser.parse_args()

    config = load_config(args.config)
    image_dir, ann_file, split = coco_paths(args, config)
    area_window = tuple(args.area_window or config["filters"]["logit_lens_area_window"])
    num_images = args.num_images or config["filters"]["logit_lens_num_images"]

    coco = CocoInstances(ann_file)
    candidates = list(
        iter_filtered_images(
            coco,
            area_window=area_window,
            max_annotations=config["filters"]["max_annotations"],
        )
    )
    print(f"{len(candidates)} images with objects of area {area_window} in {split}")
    candidates = candidates[: args.limit or num_images]

    model = build_model(args, config)
    prompt = P.build_prompt(model.processor, args.question)

    results_path = resolve(
        args.results or Path(config["paths"]["results_dir"]) / f"logit_lens_{split}.json"
    )
    store = ResultsStore(
        results_path,
        meta={
            "split": split,
            "model_id": args.model_id or config["model_id"],
            "area_window": list(area_window),
            "num_layers": model.num_layers,
            "question": args.question,
        },
    )

    for item in tqdm(candidates, desc="logit lens"):
        if item.img_id in store:
            continue
        image_path = image_dir / item.file_name
        if not image_path.exists():
            continue

        image = item.open(image_dir)
        inputs = model.prepare(image, prompt)
        grid = model.single_grid(inputs, (image.height, image.width))
        outputs = model.forward(image, prompt, output_hidden_states=True)

        result = object_token_hit_rate(
            outputs.hidden_states,
            model.final_norm,
            model.lm_head,
            model.processor.tokenizer,
            grid,
            item.annotation,
            item.class_name,
        )
        result["class_name"] = item.class_name
        result["grid"] = [grid.n_rows, grid.n_cols]
        store.set(item.img_id, result)
        store.save()

    store.save()
    report(store, model.num_layers)
    print(f"\nWritten to {results_path}")


def report(store, num_layers: int) -> None:
    usable = [r for r in store.images.values() if r.get("n_object_tokens", 0) > 0]
    if not usable:
        print("no images with object tokens")
        return

    print(f"\n{'=' * 66}")
    print(f"Logit lens: object-token -> class-token correspondence ({len(usable)} images)")
    print("=" * 66)

    for criterion in ("strict", "lenient"):
        best_rates = [r[f"best_rate_{criterion}"] for r in usable]
        best_layers = [r[f"best_layer_{criterion}"] for r in usable]
        print(f"\n{criterion}:")
        print(f"  mean best-layer hit rate : {statistics.mean(best_rates) * 100:.1f}%")
        print(f"  mean best layer          : {statistics.mean(best_layers):.1f} / {num_layers}")
        print(f"  median best layer        : {statistics.median(best_layers):.1f}")

    # Per-layer average, to see the mid-to-late rise the paper describes.
    n_entries = max(len(r["strict"]) for r in usable)
    print("\nper-layer mean hit rate (strict / lenient):")
    for layer in range(n_entries):
        strict = [r["strict"][layer] for r in usable if layer < len(r["strict"])]
        lenient = [r["lenient"][layer] for r in usable if layer < len(r["lenient"])]
        label = "embed" if layer == 0 else f"L{layer}"
        bar = "#" * int(statistics.mean(lenient) * 100)
        print(
            f"  {label:>6}: {statistics.mean(strict) * 100:5.1f}% / "
            f"{statistics.mean(lenient) * 100:5.1f}%  {bar}"
        )

    print("\nPaper reference: LLaVA-1.5 23.7% at layer 25.7/33; Qwen2VL-2B 6.5% at 25.1/29.")


if __name__ == "__main__":
    main()
