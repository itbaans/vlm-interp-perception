"""Render the paper's Table 1 and Table 2 from the results JSONs.

**Table 1** reports, per ablation condition, the percentage of *initially correct*
object identifications that survive ablation -- 100 % means the ablation changed
nothing, 0 % means it destroyed every correct answer. Average ablated-token counts are
printed alongside, because the whole argument rests on comparing object ablation to the
baselines *at matched token counts* -- which matters more here than in the paper, since
this model has ~234 visual tokens rather than 576.

**Table 2** reports relative accuracy under attention blocking, 1.00 being no impact.

All arithmetic lives in ``src/aggregate.py``, shared with the figures and the report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import aggregate as A  # noqa: E402
from experiment import load_config, resolve  # noqa: E402


def _ambiguity_note(ambiguous: dict) -> None:
    if not ambiguous["n"]:
        return
    print(
        f"Note: {ambiguous['n']} of the scored images ({ambiguous['share']:.0f}%) have a class "
        f"whose first token is too short to identify it on its own: "
        f"{', '.join(ambiguous['classes'])}.\n"
        f"      Prefer the correct-token probability column for those."
    )


def table_one(results: dict, settings: tuple[str, ...]) -> None:
    rows = A.ablation_rows(results, settings)
    if not rows:
        print("no ablation conditions found")
        return

    header = f"{'Ablation Type':<22}{'Avg Tokens':>12}" + "".join(
        f"{s.capitalize() + ' dec.':>16}" for s in settings
    )
    print("\n" + "=" * len(header))
    print(f"TABLE 1  Degradation after token ablation "
          f"({len(results.get('images', {}))} images)")
    print("Percent of initially-correct identifications destroyed by the ablation, so")
    print("HIGHER = greater impact. These are the paper's Table 1 units.")
    print("Compare object rows against the baselines at a similar Avg Tokens.")
    stats = A.visual_token_stats(results)
    if stats:
        print(
            f"Visual tokens per image: mean {stats['mean']:.1f}, "
            f"min {stats['min']}, max {stats['max']}"
        )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for row in rows:
        line = f"{row.label:<22}{row.avg_tokens:>12.1f}"
        for setting in settings:
            value = row.decrease(setting)
            line += f"{value:>15.2f}%" if value is not None else f"{'-':>16}"
        print(line)
        if row.condition == "register":
            print("-" * len(header))

    print("\nPaper (LLaVA-1.5): Object 33.33 / 15.38 / 33.33, +1 Buffer 71.79 / 51.28 / 86.67")
    print("across generative / polling / VQA, with +1 Buffer averaging 33.4 of 576 tokens.")


def table_one_vqa(results: dict) -> None:
    rows, summary = A.vqa_rows(results)
    header = (
        f"{'Ablation Type':<22}{'Avg Tokens':>12}{'VQA dec.':>14}{'Correct-tok prob':>20}"
    )

    print("\n" + "=" * len(header))
    print(
        f"TABLE 1 (VQA)  {summary['n_correct_before']} of {summary['n_total']} images "
        f"correct unablated"
    )
    print("=" * len(header))

    if not rows:
        print("no images answered correctly without ablation")
        return

    print(header)
    print("-" * len(header))
    print(f"{'(no ablation)':<22}{0:>12.1f}{0.0:>13.2f}%{summary['baseline_prob']:>20.4f}")
    print("-" * len(header))
    _ambiguity_note(summary["ambiguous"])

    for row in rows:
        print(
            f"{row.label:<22}{row.avg_tokens:>12.1f}"
            f"{row.decrease('vqa'):>13.2f}%{row.retained['correct_token_prob']:>20.4f}"
        )


def table_two(results: dict) -> None:
    summary = A.knockout_summary(results)
    windows = summary["windows"]
    header = f"{'From':<10}{'To':<6}" + "".join(f"{w:>12}" for w in windows)

    print("\n" + "=" * len(header))
    print(f"TABLE 2  Relative accuracy under attention blocking ({summary['n_usable']} images)")
    print("1.00 = no impact, 0.00 = every answer wrong.")
    print(
        f"Attention layers: {summary['meta'].get('attention_layers')} "
        f"of {summary['meta'].get('num_layers')} total"
    )
    print(f"Windows: {summary['meta'].get('windows')}")
    print("=" * len(header))

    if not summary["n_usable"]:
        print("no images answered correctly without blocking")
        return

    print(header)
    print("-" * len(header))
    _ambiguity_note(summary["ambiguous"])

    for (source, target), values in summary["grid"].items():
        line = f"{source:<10}{target:<6}"
        for window in windows:
            line += f"{values[window]:>12.2f}" if window in values else f"{'-':>12}"
        print(line)

    if summary["sweep"]:
        print("\nPer-layer and sliding-window sweep (O+1 -> last token position):")
        for name, value in sorted(summary["sweep"].items()):
            print(f"  {name:>10}: {value:>5.2f}  {'#' * int((1.0 - value) * 40)}")

    control = summary["control"]
    if control:
        print("\nDistance control (padded question, object tokens pushed from the last position):")
        print(f"  mean distance to last token       : {control['mean_distance']:.1f}")
        print(f"  short-conv reach (2 x conv layers): {control['conv_reach']}")
        if control["retained"] is not None:
            print(f"  accuracy, all attention blocked   : {control['retained']:.2f} "
                  f"(n={control['n']})")
            print("  If this matches the unpadded 'all' column, the effect is genuine attention")
            print("  routing; if it is much higher, short-conv leakage was carrying information.")

    print("\nPaper (LLaVA-1.5): O+1 -> LTP falls to 0.82 at mid-late and 0.67 across all layers;")
    print("O+1 -> last visual row stays at 1.00 everywhere.")


def table_logit_lens(results: dict) -> None:
    summary = A.logit_lens_summary(results)
    print("\n" + "=" * 66)
    print(f"LOGIT LENS  object-token -> class-token correspondence "
          f"({summary['n_images']} images)")
    print("=" * 66)
    if not summary["n_images"]:
        print("no images with object tokens")
        return

    num_layers = summary["num_layers"] or summary["n_entries"] - 1
    print(f"Averaged curve peaks at layer {summary['peak_layer']} / {num_layers} "
          f"({summary['peak_rate'] * 100:.1f}%)")

    for criterion in ("strict", "lenient"):
        best = summary["best"][criterion]
        print(f"\n{criterion}:")
        print(f"  images showing the effect : {best['n_with_signal']}/{summary['n_images']} "
              f"({best['share_with_signal']:.0f}%)")
        print(f"  mean best-layer hit rate  : {best['rate'] * 100:.1f}%  "
              f"(over those images; {best['rate_all_images'] * 100:.1f}% over all)")
        if best["layer"] is not None:
            print(f"  mean best layer           : {best['layer']:.1f} / {num_layers}")
            print(f"  median best layer         : {best['median_layer']:.0f} / {num_layers}")

    print("\nper-layer mean hit rate (strict / lenient):")
    for index in range(summary["n_entries"]):
        strict = summary["per_layer"]["strict"][index]
        lenient = summary["per_layer"]["lenient"][index]
        name = "embed" if index == 0 else f"L{index}"
        print(f"  {name:>6}: {strict * 100:5.1f}% / {lenient * 100:5.1f}%  "
              f"{'#' * int(lenient * 100)}")

    print("\nPaper reference: LLaVA-1.5 23.7% at layer 25.7/33; Qwen2VL-2B 6.5% at 25.1/29.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation", default=None, help="ablation_*.json")
    parser.add_argument("--vqa", default=None, help="ablation_vqa_*.json")
    parser.add_argument("--knockout", default=None, help="attention_knockout_*.json")
    parser.add_argument("--logit-lens", default=None, help="logit_lens_*.json")
    parser.add_argument("--split", default=None)
    parser.add_argument("--lens-split", default=None, help="Split the logit lens ran on")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = resolve(config["paths"]["results_dir"])
    split = args.split or config["data"]["split"]
    lens_split = args.lens_split or split

    paths = {
        "ablation": Path(args.ablation) if args.ablation else results_dir / f"ablation_{split}.json",
        "vqa": Path(args.vqa) if args.vqa else results_dir / f"ablation_vqa_{split}.json",
        "knockout": (
            Path(args.knockout) if args.knockout
            else results_dir / f"attention_knockout_{split}.json"
        ),
        "logit_lens": (
            Path(args.logit_lens) if args.logit_lens
            else results_dir / f"logit_lens_{lens_split}.json"
        ),
    }

    found = False
    if paths["ablation"].exists():
        table_one(A.load(paths["ablation"]), settings=("generative", "polling"))
        found = True
    if paths["vqa"].exists():
        table_one_vqa(A.load(paths["vqa"]))
        found = True
    if paths["knockout"].exists():
        table_two(A.load(paths["knockout"]))
        found = True
    if paths["logit_lens"].exists():
        table_logit_lens(A.load(paths["logit_lens"]))
        found = True

    if not found:
        raise SystemExit(
            "no results found in {}. Expected any of:\n  {}".format(
                results_dir, "\n  ".join(p.name for p in paths.values())
            )
        )


if __name__ == "__main__":
    main()
