"""Attention knockout (paper Section 4.2 / Table 2).

Block attention between chosen token groups over a window of layers and measure how much
the correct answer degrades. The paper's finding on LLaVA: blocking object tokens ->
last token position hurts most in the middle-to-late layers, while blocking all visual
tokens -> the last row of visual tokens does essentially nothing (contradicting the
"summarise into the last row" hypothesis).

**This model is a hybrid, and that changes what knockout can show.** Of LFM2.5-VL's 30
LM layers only 8 have attention (indices 2, 5, 9, 13, 17, 21, 24, 27); the other 22 are
``Lfm2ShortConv`` with ``conv_L_cache = 3``, which mix ~2 positions backwards each,
ignore token-pair masking entirely, and cannot be blocked. So:

* Layer windows are defined over the 8 attention layers. The paper's five windows are
  reproduced by mapping each to the attention layers it contains, and -- since 8 layers
  is cheap to sweep exhaustively -- each layer is also blocked alone, plus sliding
  windows of 3.
* A **distance control** runs the strongest condition again with the question padded so
  the object tokens sit far enough from the last position that no chain of short-conv
  layers could bridge the gap. If the effect survives the padding it is genuine
  attention routing; if it vanishes, part of what we measured was conv leakage.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import geometry as G  # noqa: E402
import prompts as P  # noqa: E402
from coco import CocoInstances, load_clean_questions, select_annotation  # noqa: E402
from experiment import (  # noqa: E402
    ResultsStore,
    add_common_args,
    build_model,
    coco_paths,
    load_config,
    resolve,
)
from hooked_lfm2vl import layer_windows  # noqa: E402

# Padding that pushes the last token far from the visual span without changing what is
# being asked. It must exceed the short-conv layers' reach -- 2 positions per conv layer,
# so 44 tokens on this model -- or the control proves nothing, because a chain of conv
# layers could still bridge the gap on its own. Sized here for a comfortable margin.
DISTANCE_PADDING = (
    " Please look at the picture carefully and consider it in full detail before you "
    "answer, taking into account everything that is visible in the scene, including the "
    "background, the foreground, the lighting, the colours, the textures, the spatial "
    "arrangement of every element, and any objects that may be partially hidden or only "
    "just visible at the edges of the frame. Think it through completely, weigh up what "
    "you can see against what you would expect to see, and only then give your answer."
)


def block_pairs(from_indices, to_indices) -> list[tuple[int, int]]:
    """``(query, key)`` pairs blocking information flow from ``from_indices`` to ``to_indices``."""
    return [(to, frm) for to, frm in itertools.product(to_indices, from_indices) if to >= frm]


def evaluate(model, image, prompt, class_name, block_dict=None) -> dict:
    """Next-token result, optionally under attention blocking."""
    if block_dict:
        with model.block_attention(block_dict):
            probs = model.next_token_distribution(image, prompt)
    else:
        probs = model.next_token_distribution(image, prompt)

    top_id = int(probs.argmax())
    correct_id = P.first_answer_token_id(model.processor, class_name)
    return {
        "is_correct": P.answer_matches_class(model.processor, top_id, class_name),
        "generated_token": model.processor.tokenizer.decode([top_id]),
        "generated_token_prob": float(probs[top_id]),
        "correct_token_prob": float(probs[correct_id]),
    }


def build_source_groups(annotation, grid, last_position) -> dict[str, list[int]]:
    """The paper's "From" groups, as sequence indices."""
    object_0 = G.object_token_indices(annotation, grid, buffer=0)
    object_1 = G.object_token_indices(annotation, grid, buffer=1)
    object_2 = G.object_token_indices(annotation, grid, buffer=2)
    last_row = list(range(grid.end - grid.n_cols, grid.end))

    return {
        "O": object_0,
        "O+1": object_1,
        "O+2": object_2,
        "I-(O+1)": [i for i in grid.indices if i not in set(object_1)],
        "I-LVR": [i for i in grid.indices if i not in set(last_row)],
        "_last_row": last_row,
    }


def build_experiments(groups, grid, last_position, windows, config, attention_layers) -> dict:
    """Every (from, to, layer-window) combination to run."""
    last_row = groups["_last_row"]
    experiments: dict[str, dict] = {}

    # To the last token position: the paper's main result.
    for source in ("O", "O+1", "O+2", "I-(O+1)"):
        for window_name, layers in windows.items():
            experiments[f"{source}->LTP@{window_name}"] = {
                "from": groups[source],
                "to": [last_position],
                "layers": layers,
            }

    # To the last visual row: the paper's control against the "summarisation" hypothesis.
    for source in ("O+1", "I-LVR"):
        for window_name, layers in windows.items():
            experiments[f"{source}->LVR@{window_name}"] = {
                "from": groups[source],
                "to": last_row,
                "layers": layers,
            }

    if config["attention_knockout"]["per_layer_sweep"]:
        for layer in attention_layers:
            experiments[f"O+1->LTP@L{layer}"] = {
                "from": groups["O+1"],
                "to": [last_position],
                "layers": [layer],
            }

    width = config["attention_knockout"]["sliding_window"]
    if width:
        for start in range(len(attention_layers) - width + 1):
            layers = attention_layers[start : start + width]
            experiments[f"O+1->LTP@W{layers[0]}-{layers[-1]}"] = {
                "from": groups["O+1"],
                "to": [last_position],
                "layers": layers,
            }

    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--clean-questions", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if (args.attn_implementation or config["attn_implementation"]) != "eager":
        print(
            "WARNING: attention knockout needs an additive float mask. "
            "Run with --attn-implementation eager."
        )

    image_dir, ann_file, split = coco_paths(args, config)
    questions = load_clean_questions(resolve(args.clean_questions or config["data"]["clean_questions"]))
    coco = CocoInstances(ann_file)
    available = [q for q in questions if q in coco.images]
    print(f"{len(available)} curated questions available in {split}")
    if not available:
        raise SystemExit(f"no curated image ids in {split}; try --split train2017")

    model = build_model(args, config)
    attention_layers = model.attention_layers
    windows = layer_windows(attention_layers, model.num_layers)
    print(f"attention layers: {attention_layers}")
    print(f"layer windows: { {k: v for k, v in windows.items()} }")

    results_path = resolve(
        args.results or Path(config["paths"]["results_dir"]) / f"attention_knockout_{split}.json"
    )
    store = ResultsStore(
        results_path,
        meta={
            "split": split,
            "model_id": args.model_id or config["model_id"],
            "attention_layers": attention_layers,
            "windows": windows,
            "num_layers": model.num_layers,
            "conv_layers": [i for i in range(model.num_layers) if i not in attention_layers],
        },
    )

    processed = 0
    for img_id in tqdm(available, desc="knockout"):
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
        last_position = inputs["input_ids"].shape[1] - 1

        baseline = evaluate(model, image, prompt, class_name)
        record = {
            "question": questions[img_id],
            "class_name": class_name,
            "ambiguous_class": P.first_token_is_ambiguous(model.processor, class_name),
            "n_visual_tokens": grid.n_tokens,
            "visual_span": [grid.start, grid.end],
            "last_position": last_position,
            "distance_to_last": last_position - grid.end,
            "baseline": baseline,
            "knockout": {},
        }

        if not baseline["is_correct"]:
            store.set(img_id, record)
            store.save()
            processed += 1
            continue

        groups = build_source_groups(annotation, grid, last_position)
        experiments = build_experiments(
            groups, grid, last_position, windows, config, attention_layers
        )

        for name, spec in experiments.items():
            if not spec["from"] or not spec["layers"]:
                continue
            pairs = block_pairs(spec["from"], spec["to"])
            record["knockout"][name] = evaluate(
                model, image, prompt, class_name, {layer: pairs for layer in spec["layers"]}
            )

        # Distance control: same blocking, but with the object tokens pushed further
        # from the last position than any chain of short-conv layers can reach.
        if config["attention_knockout"]["distance_control"]:
            padded_prompt = P.vqa_prompt(model.processor, questions[img_id] + DISTANCE_PADDING)
            padded_inputs = model.prepare(image, padded_prompt)
            padded_grid = model.single_grid(padded_inputs, (image.height, image.width))
            padded_last = padded_inputs["input_ids"].shape[1] - 1
            padded_object = G.object_token_indices(annotation, padded_grid, buffer=1)

            control = {
                "distance_to_last": padded_last - padded_grid.end,
                "conv_reach": 2 * len([i for i in range(model.num_layers) if i not in attention_layers]),
                "baseline": evaluate(model, image, padded_prompt, class_name),
            }
            pairs = block_pairs(padded_object, [padded_last])
            control["all_layers"] = evaluate(
                model,
                image,
                padded_prompt,
                class_name,
                {layer: pairs for layer in attention_layers},
            )
            record["distance_control"] = control

        store.set(img_id, record)
        store.save()
        processed += 1

    # Save even when nothing matched, so downstream stages see an empty result rather
    # than a missing file.
    store.save()
    print(f"\n{len(store)} images written to {results_path}")
    print("Run scripts/analyze_results.py to render the Table 2 comparison.")


if __name__ == "__main__":
    main()
