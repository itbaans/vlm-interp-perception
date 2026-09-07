"""Build the filtered, hallucination-controlled image set (paper Section 3, "Dataset").

Two filtering steps, following the paper:

1. **Simpler images.** At most 3 annotations, and some category with exactly one
   instance whose area is in the configured window.
2. **Hallucination control.** Keep an image only if the model names the object when
   shown the original *and* fails to name it when the object's pixels are destroyed
   with noise. Without this, ablating object tokens could look ineffective simply
   because the model was guessing the object from context all along.

For LLaVA the paper ended with 4,318 images. LFM2.5-VL will land somewhere else, and
that count is itself a result worth reporting.

Writes a manifest JSON that every downstream experiment reads, so the expensive
filtering happens once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import geometry as G  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--skip-hallucination-control",
        action="store_true",
        help="Apply only the COCO filter. Fast, but keeps images whose object the "
        "model can infer from context.",
    )
    parser.add_argument("--area-window", type=float, nargs=2, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    image_dir, ann_file, split = coco_paths(args, config)
    area_window = tuple(args.area_window or config["filters"]["area_window"])

    results_path = resolve(
        args.results or Path(config["paths"]["results_dir"]) / f"dataset_{split}.json"
    )
    store = ResultsStore(
        results_path,
        meta={
            "split": split,
            "area_window": list(area_window),
            "max_annotations": config["filters"]["max_annotations"],
            "hallucination_control": not args.skip_hallucination_control,
            "model_id": args.model_id or config["model_id"],
        },
    )

    print(f"Loading annotations from {ann_file}")
    coco = CocoInstances(ann_file)
    candidates = list(
        iter_filtered_images(
            coco,
            area_window=area_window,
            max_annotations=config["filters"]["max_annotations"],
        )
    )
    print(f"{len(candidates)} images passed the COCO filter (of {len(coco)})")

    if args.skip_hallucination_control:
        for item in candidates:
            store.set(
                item.img_id,
                {
                    "file_name": item.file_name,
                    "width": item.width,
                    "height": item.height,
                    "class_name": item.class_name,
                    "category_id": item.annotation["category_id"],
                    "annotation_id": item.annotation["id"],
                    "kept": True,
                },
            )
        store.save()
        print(f"Wrote {len(store)} images to {results_path}")
        return

    model = build_model(args, config)
    describe = P.describe_prompt(model.processor)

    kept = 0
    for item in tqdm(candidates, desc="hallucination control"):
        if args.limit is not None and kept >= args.limit:
            break
        if item.img_id in store:
            kept += int(store[item.img_id]["kept"])
            continue

        try:
            image = item.open(image_dir)
        except FileNotFoundError:
            continue

        grid = G.token_grid_for_image(orig_h=image.height, orig_w=image.width)
        cells = G.annotation_to_cells(item.annotation, grid)
        if not cells:
            continue

        original_answer = model.generate(image, describe, max_new_tokens=200)
        names_original = P.mentions_class(original_answer, item.class_name)

        record = {
            "file_name": item.file_name,
            "width": item.width,
            "height": item.height,
            "class_name": item.class_name,
            "category_id": item.annotation["category_id"],
            "annotation_id": item.annotation["id"],
            "n_object_cells": len(cells),
            "n_visual_tokens": grid.n_tokens,
            "grid": [grid.n_rows, grid.n_cols],
            "names_original": names_original,
        }

        if names_original:
            masked = G.mask_image_regions(image, cells, grid, patch_type="noise")
            masked_answer = model.generate(masked, describe, max_new_tokens=200)
            record["names_masked"] = P.mentions_class(masked_answer, item.class_name)
            record["description"] = original_answer
        else:
            record["names_masked"] = None

        # Keep only: identified in the original, not identified once masked.
        record["kept"] = bool(names_original and record["names_masked"] is False)
        kept += int(record["kept"])

        store.set(item.img_id, record)
        store.save()

    total = len(store)
    kept_ids = [i for i, r in store.images.items() if r["kept"]]
    print(f"\nEvaluated {total} images; {len(kept_ids)} passed both filters.")
    print(f"(Paper's LLaVA-1.5 dataset after the same two steps: 4,318 images.)")
    print(f"Manifest written to {results_path}")


if __name__ == "__main__":
    main()
