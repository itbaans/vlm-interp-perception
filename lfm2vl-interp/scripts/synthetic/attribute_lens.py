"""Attribute lens: do a shape's own visual tokens decode to its colour and kind?

The COCO replication asked whether an object's visual tokens decode to its *class*. Here
the ground truth is far richer -- we know each shape's colour, kind and exact token cells --
so the same machinery answers a sharper question: is "red" locally encoded at the red
circle's tokens, and is "circle" encoded there too, or only one of the two?

For every shape in every scene, the tokens covering that shape are decoded at every layer
and scored against three targets:

* its **colour** (` red`)
* its **kind** (` circle`)
* a **control**: the colour it is *not*, drawn from the same vocabulary, which measures how
  much of any apparent hit rate is just the layer's general bias toward colour words.

Without that control a lens that decodes every visual token to ' red' at some layer would
look like perfect colour encoding.
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

import geometry as G  # noqa: E402
import logit_lens as LL  # noqa: E402
import synthetic as S  # noqa: E402
import tasks as T  # noqa: E402
from experiment import (  # noqa: E402
    ResultsStore,
    add_common_args,
    build_model,
    load_config,
    resolve,
)


def hit_rates(top1_ids, target_ids: set[int]) -> list[float]:
    """Per-layer fraction of positions whose top-1 token is in ``target_ids``."""
    return [
        sum(1 for token in layer_ids.tolist() if token in target_ids) / len(layer_ids)
        for layer_ids in top1_ids
    ]


def analyse(model, sample, grid) -> dict:
    """Decode every shape's own tokens; score colour, kind and a control."""
    import prompts as P

    prompt = P.describe_prompt(model.processor)
    inputs = model.prepare(sample.clean.render(), prompt)
    real_grid = model.single_grid(inputs, (sample.clean.height, sample.clean.width))
    outputs = model.forward(sample.clean.render(), prompt, output_hidden_states=True)

    tokenizer = model.processor.tokenizer
    shapes = []
    for index, shape in enumerate(sample.clean.shapes):
        positions = S.cells_to_token_indices(S.shape_cells(shape, real_grid), real_grid)
        if not positions:
            continue

        top1 = LL.top1_token_ids(
            outputs.hidden_states, model.final_norm, model.lm_head, positions
        )
        other_colour = next(c for c in S.COLOURS if c != shape.colour)

        shapes.append({
            "index": index,
            "colour": shape.colour,
            "kind": shape.kind,
            "n_tokens": len(positions),
            "colour_hits": hit_rates(top1, LL.class_match_ids(tokenizer, shape.colour)),
            "kind_hits": hit_rates(top1, LL.class_match_ids(tokenizer, shape.kind)),
            "control_hits": hit_rates(top1, LL.class_match_ids(tokenizer, other_colour)),
        })

    return {
        "task": sample.task,
        "level": sample.level,
        "n_layers": len(outputs.hidden_states),
        "shapes": shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--dataset", default=None)
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
    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)

    store = ResultsStore(
        out_dir / "attribute_lens.json",
        meta={"model_id": args.model_id or config["model_id"], "stage": "attribute_lens"},
    )

    for sample in tqdm(samples, desc="attribute lens"):
        if sample.sample_id in store:
            continue
        store.set(sample.sample_id, analyse(model, sample, grid))
        store.save()

    store.save()
    report(store)
    print(f"\nwritten to {out_dir / 'attribute_lens.json'}")


def report(store) -> None:
    records = list(store.images.values())
    shapes = [s for r in records for s in r["shapes"]]
    if not shapes:
        print("no shapes analysed")
        return

    n_layers = records[0]["n_layers"]
    print(f"\n{'=' * 74}")
    print(f"ATTRIBUTE LENS  ({len(shapes)} shapes across {len(records)} scenes)")
    print("Fraction of a shape's own visual tokens decoding to its colour / kind.")
    print("'control' is a colour the shape is NOT -- the floor any real signal must clear.")
    print("=" * 74)

    header = f"{'layer':<8}{'colour':>12}{'kind':>12}{'control':>12}"
    print(header)
    print("-" * len(header))
    for layer in range(n_layers):
        colour = statistics.mean(s["colour_hits"][layer] for s in shapes)
        kind = statistics.mean(s["kind_hits"][layer] for s in shapes)
        control = statistics.mean(s["control_hits"][layer] for s in shapes)
        name = "embed" if layer == 0 else f"L{layer}"
        print(f"{name:<8}{colour * 100:>11.1f}%{kind * 100:>11.1f}%{control * 100:>11.1f}%")

    for label, key in (("colour", "colour_hits"), ("kind", "kind_hits")):
        peak_layer = max(
            range(n_layers), key=lambda l: statistics.mean(s[key][l] for s in shapes)
        )
        peak = statistics.mean(s[key][peak_layer] for s in shapes)
        control = statistics.mean(s["control_hits"][peak_layer] for s in shapes)
        print(f"\npeak {label}: {peak * 100:.1f}% at layer {peak_layer} "
              f"(control {control * 100:.1f}% -> margin {(peak - control) * 100:+.1f} pts)")

    # Per-colour breakdown: some colours may be far more legible than others.
    by_colour: dict[str, list[dict]] = defaultdict(list)
    for shape in shapes:
        by_colour[shape["colour"]].append(shape)
    print("\nbest-layer colour hit rate by colour:")
    for colour, group in sorted(by_colour.items()):
        best = max(
            statistics.mean(s["colour_hits"][l] for s in group) for l in range(n_layers)
        )
        print(f"  {colour:<8} {best * 100:5.1f}%  (n={len(group)})")


if __name__ == "__main__":
    main()
