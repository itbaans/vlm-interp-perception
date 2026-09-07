"""Token ablation, VQA setting (paper Section 3, method 3 / Table 1 "VQA" column).

The paper manually curated 100 COCO images with a specific question about the object
("What is on the bed?"), choosing questions with unambiguous answers and avoiding
objects implied by the question. The assistant turn is prefilled with "It is a" so the
next generated token *is* the answer, which makes the before/after comparison a single
token rather than a string search.

Those 100 questions ship in ``data/clean_questions.json`` and are reused verbatim here,
so this experiment is directly comparable to the paper's.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import geometry as G  # noqa: E402
import prompts as P  # noqa: E402
from attribution import integrated_gradients, yes_token_id  # noqa: E402
from coco import CocoInstances, load_clean_questions, select_annotation  # noqa: E402
from experiment import (  # noqa: E402
    ResultsStore,
    add_common_args,
    build_model,
    coco_paths,
    load_config,
    resolve,
)


def answer_under_ablation(model, image, prompt, class_name, indices, replacement) -> dict:
    """Next-token answer with an optional ablation applied."""
    if indices:
        with model.ablate_inputs(indices, replacement):
            probs = model.next_token_distribution(image, prompt)
    else:
        probs = model.next_token_distribution(image, prompt)

    top_id = int(probs.argmax())
    correct_id = P.first_answer_token_id(model.processor, class_name)

    return {
        "is_correct": P.answer_matches_class(model.processor, top_id, class_name),
        "generated_token": model.processor.tokenizer.decode([top_id]),
        "generated_token_prob": float(probs[top_id]),
        "correct_token": model.processor.tokenizer.decode([correct_id]),
        "correct_token_prob": float(probs[correct_id]),
        "n_ablated": len(indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--clean-questions", default=None)
    parser.add_argument("--mean-vector", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    image_dir, ann_file, split = coco_paths(args, config)

    questions_path = resolve(args.clean_questions or config["data"]["clean_questions"])
    questions = load_clean_questions(questions_path)
    print(f"{len(questions)} curated questions from {questions_path}")

    coco = CocoInstances(ann_file)
    available = [q for q in questions if q in coco.images]
    print(f"{len(available)} of them are in split {split}")
    if not available:
        raise SystemExit(
            f"none of the curated image ids are in {split}. "
            "The curated set was built on the paper's filtered train2017 images -- "
            "try --split train2017."
        )

    mean_path = resolve(args.mean_vector or config["ablation"]["mean_vector"])
    if not mean_path.exists():
        raise SystemExit(f"mean vector not found at {mean_path}")
    mean_vector = torch.load(mean_path, map_location="cpu", weights_only=True)

    model = build_model(args, config)
    rng = random.Random(args.seed)

    results_path = resolve(
        args.results or Path(config["paths"]["results_dir"]) / f"ablation_vqa_{split}.json"
    )
    store = ResultsStore(
        results_path,
        meta={
            "split": split,
            "model_id": args.model_id or config["model_id"],
            "mean_vector": str(mean_path),
            "setting": "vqa",
            "n_questions": len(available),
        },
    )

    processed = 0
    for img_id in tqdm(available, desc="vqa ablation"):
        if args.limit is not None and processed >= args.limit:
            break
        if img_id in store:
            processed += 1
            continue

        annotation = select_annotation(
            coco.anns_by_image.get(img_id, []),
            area_window=tuple(config["filters"]["area_window"]),
            max_annotations=config["filters"]["max_annotations"],
        )
        if annotation is None:
            continue

        info = coco.images[img_id]
        image_path = image_dir / info["file_name"]
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        class_name = coco.class_name(annotation["category_id"]).lower()

        prompt = P.vqa_prompt(model.processor, questions[img_id])
        inputs = model.prepare(image, prompt)
        grid = model.single_grid(inputs, (image.height, image.width))

        baseline = answer_under_ablation(model, image, prompt, class_name, [], mean_vector)
        record = {
            "question": questions[img_id],
            "class_name": class_name,
            "ambiguous_class": P.first_token_is_ambiguous(model.processor, class_name),
            "n_visual_tokens": grid.n_tokens,
            "grid": [grid.n_rows, grid.n_cols],
            "conditions": {"no_ablation": baseline},
            "indices": {},
        }

        # Only images the model answers correctly unablated can show a degradation.
        if not baseline["is_correct"]:
            store.set(img_id, record)
            store.save()
            processed += 1
            continue

        sets: dict[str, list[int]] = {}
        for buffer in config["ablation"]["buffers"]:
            sets[f"object_{buffer}"] = G.object_token_indices(annotation, grid, buffer=buffer)

        embeds = model.get_inputs_embeds(image, prompt)
        sets["register"] = G.register_indices(embeds[:, grid.start : grid.end, :], grid)

        for count in config["ablation"]["baseline_counts"]:
            sets[f"random_{count}"] = G.random_indices(grid, count, rng=rng)

        attribution = integrated_gradients(
            model,
            inputs,
            grid,
            mean_vector,
            yes_token_id(model.processor),
            steps=config["ablation"]["integrated_gradient_steps"],
        )
        for count in config["ablation"]["baseline_counts"]:
            sets[f"gradient_{count}"] = G.top_gradient_indices(attribution, grid, count)

        for name, indices in sets.items():
            record["conditions"][name] = answer_under_ablation(
                model, image, prompt, class_name, indices, mean_vector
            )
            record["indices"][name] = indices

        store.set(img_id, record)
        store.save()
        processed += 1

    # Save even when nothing matched, so downstream stages see an empty result rather
    # than a missing file.
    store.save()
    print(f"\n{len(store)} images written to {results_path}")


if __name__ == "__main__":
    main()
