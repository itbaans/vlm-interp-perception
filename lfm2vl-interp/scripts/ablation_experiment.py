"""Token ablation, generative and polling settings (paper Section 3 / Table 1).

For each image in the manifest, replace selected visual-token embeddings with the mean
visual token and check whether the model still identifies the object:

* **Generative** -- "Describe this image.", does the class name appear in the output?
* **Polling** -- "Is there a [o] in this image?", does the answer still start with yes?

Ablation sets, following the paper:

* ``object``, ``object+1``, ``object+2`` -- the object's visual tokens, dilated by 0/1/2
* ``register`` -- tokens whose norm exceeds mean + 2 sigma
* ``random_n`` -- n random visual tokens (the weak baseline)
* ``gradient_n`` -- the n highest integrated-gradients tokens (the strong baseline)

The comparison that matters is object+1 against random/gradient at a *similar token
count*, so the effective count of every condition is recorded per image.

One thing to watch on this model: a typical COCO image yields ~234 visual tokens, not
LLaVA's 576. The paper's ``n=250`` baselines therefore cover essentially the whole
image here, and are capped at the grid size; ``analyze_results.py`` reports the
realised counts so the comparison stays honest.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import geometry as G  # noqa: E402
import prompts as P  # noqa: E402
from attribution import integrated_gradients, yes_token_id  # noqa: E402
from experiment import (  # noqa: E402
    ResultsStore,
    add_common_args,
    build_model,
    coco_paths,
    load_config,
    resolve,
)


def check_identification(model, image, class_name, indices, replacement) -> dict:
    """Run both settings under one ablation set."""
    describe = P.describe_prompt(model.processor)
    polling = P.polling_prompt(model.processor, class_name)

    if indices:
        with model.ablate_inputs(indices, replacement):
            description = model.generate(image, describe, max_new_tokens=200)
        with model.ablate_inputs(indices, replacement):
            answer = model.generate(image, polling, max_new_tokens=10)
    else:
        description = model.generate(image, describe, max_new_tokens=200)
        answer = model.generate(image, polling, max_new_tokens=10)

    return {
        "generative": P.mentions_class(description, class_name),
        "polling": P.says_yes(answer),
        "n_ablated": len(indices),
        "description": description,
        "answer": answer,
    }


def build_ablation_sets(model, image, item, grid, config, mean_vector, rng) -> dict[str, list[int]]:
    """All the index sets for one image, keyed by condition name."""
    annotation = item["annotation"]
    sets: dict[str, list[int]] = {}

    for buffer in config["ablation"]["buffers"]:
        sets[f"object_{buffer}"] = G.object_token_indices(annotation, grid, buffer=buffer)

    embeds = model.get_inputs_embeds(image, P.describe_prompt(model.processor))
    sets["register"] = G.register_indices(embeds[:, grid.start : grid.end, :], grid)

    for count in config["ablation"]["baseline_counts"]:
        sets[f"random_{count}"] = G.random_indices(grid, count, rng=rng)

    inputs = model.prepare(image, P.polling_prompt(model.processor, item["class_name"]))
    ig_grid = model.single_grid(inputs, (item["height"], item["width"]))
    attribution = integrated_gradients(
        model,
        inputs,
        ig_grid,
        mean_vector,
        yes_token_id(model.processor),
        steps=config["ablation"]["integrated_gradient_steps"],
    )
    for count in config["ablation"]["baseline_counts"]:
        # Attribution was computed on the polling prompt, whose visual span may sit at a
        # different offset; re-anchor the picked cells onto the describe-prompt grid.
        picked = G.top_gradient_indices(attribution, ig_grid, count)
        sets[f"gradient_{count}"] = [i - ig_grid.start + grid.start for i in picked]

    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--manifest", default=None, help="Output of build_dataset.py")
    parser.add_argument("--mean-vector", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    image_dir, ann_file, split = coco_paths(args, config)

    manifest_path = resolve(
        args.manifest or Path(config["paths"]["results_dir"]) / f"dataset_{split}.json"
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    kept = {k: v for k, v in manifest["images"].items() if v.get("kept")}
    print(f"{len(kept)} images in {manifest_path}")

    from coco import CocoInstances

    coco = CocoInstances(ann_file)

    mean_path = resolve(args.mean_vector or config["ablation"]["mean_vector"])
    if not mean_path.exists():
        raise SystemExit(
            f"mean vector not found at {mean_path}; run scripts/compute_mean_visual_token.py first"
        )
    mean_vector = torch.load(mean_path, map_location="cpu", weights_only=True)

    model = build_model(args, config)
    rng = random.Random(args.seed)

    results_path = resolve(
        args.results or Path(config["paths"]["results_dir"]) / f"ablation_{split}.json"
    )
    store = ResultsStore(
        results_path,
        meta={
            "split": split,
            "model_id": args.model_id or config["model_id"],
            "manifest": str(manifest_path),
            "mean_vector": str(mean_path),
            "seed": args.seed,
            "setting": "generative+polling",
        },
    )

    limit = args.limit or config["ablation"]["max_images"]
    processed = 0

    for img_id, item in tqdm(kept.items(), desc="ablation"):
        if processed >= limit:
            break
        if img_id in store:
            processed += 1
            continue

        annotations = coco.anns_by_image[int(img_id)]
        annotation = next(
            (a for a in annotations if a["id"] == item["annotation_id"]), None
        )
        if annotation is None:
            continue
        item = {**item, "annotation": annotation}

        image_path = image_dir / item["file_name"]
        if not image_path.exists():
            continue
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        inputs = model.prepare(image, P.describe_prompt(model.processor))
        grid = model.single_grid(inputs, (image.height, image.width))

        record = {
            "class_name": item["class_name"],
            "n_visual_tokens": grid.n_tokens,
            "grid": [grid.n_rows, grid.n_cols],
            "conditions": {},
            "indices": {},
        }

        record["conditions"]["no_ablation"] = check_identification(
            model, image, item["class_name"], [], mean_vector
        )

        sets = build_ablation_sets(model, image, item, grid, config, mean_vector, rng)
        for name, indices in sets.items():
            record["conditions"][name] = check_identification(
                model, image, item["class_name"], indices, mean_vector
            )
            record["indices"][name] = indices

        store.set(img_id, record)
        store.save()
        processed += 1

    # Save even when nothing matched: a missing file is indistinguishable from a crash,
    # an empty one says the stage ran and found nothing.
    store.save()
    print(f"\n{len(store)} images written to {results_path}")
    print("Run scripts/analyze_results.py to render the Table 1 comparison.")


if __name__ == "__main__":
    main()
