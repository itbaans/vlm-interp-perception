"""Turn raw results JSON into the summary numbers the tables, plots and report share.

``analyze_results.py`` prints these, ``plots.py`` draws them and ``report.py`` embeds
them. Keeping the arithmetic in one place is what stops the printed table and the
figure beside it from disagreeing.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# Order matters: object conditions first, then the two baselines by ascending count.
CONDITION_ORDER = [
    "object_0",
    "object_1",
    "object_2",
    "register",
    "gradient_5",
    "gradient_10",
    "gradient_20",
    "gradient_40",
    "gradient_60",
    "gradient_100",
    "gradient_250",
    "random_5",
    "random_10",
    "random_20",
    "random_40",
    "random_60",
    "random_100",
    "random_250",
]

PRETTY = {
    "object_0": "Object",
    "object_1": "+ 1 Buffer",
    "object_2": "+ 2 Buffer",
    "register": "Register tokens",
}

# Short labels for direct-labelling points in the figures.
SHORT = {"object_0": "O", "object_1": "O+1", "object_2": "O+2", "register": "Reg"}


def label(condition: str) -> str:
    if condition in PRETTY:
        return PRETTY[condition]
    kind, _, count = condition.partition("_")
    kind = {"gradient": "Int. Gradients", "random": "Random"}.get(kind, kind)
    return f"{kind} ({count})"


def family(condition: str) -> str:
    """Which group a condition belongs to: ``object``, ``gradient`` or ``random``."""
    if condition.startswith("object") or condition == "register":
        return "object"
    return condition.partition("_")[0]


def load(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class AblationRow:
    condition: str
    avg_tokens: float
    n: int
    retained: dict[str, float | None] = field(default_factory=dict)
    denominator: dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return label(self.condition)

    @property
    def family(self) -> str:
        return family(self.condition)

    def decrease(self, setting: str) -> float | None:
        """Share of initially-correct answers destroyed, i.e. ``100 - retained``.

        The paper's Table 1 is in these units (its "Decrease (%)" columns), so this is
        the number to compare against directly. ``retained`` is kept alongside because
        the paper's caption describes the opposite convention, and reporting both
        removes the ambiguity.
        """
        value = self.retained.get(setting)
        return None if value is None else 100.0 - value


def ablation_rows(results: dict, settings: tuple[str, ...]) -> list[AblationRow]:
    """Percentage of *initially correct* identifications that survive each ablation.

    100 % means the ablation changed nothing; 0 % means it destroyed every correct
    answer. Images the model got wrong before ablation are excluded from the
    denominator -- they cannot degrade.
    """
    images = results.get("images", {})
    rows: list[AblationRow] = []

    for condition in CONDITION_ORDER:
        present = [r for r in images.values() if condition in r.get("conditions", {})]
        if not present:
            continue

        row = AblationRow(
            condition=condition,
            avg_tokens=statistics.mean(
                r["conditions"][condition].get("n_ablated", 0) for r in present
            ),
            n=len(present),
        )
        for setting in settings:
            correct_before = [
                r for r in present if r["conditions"].get("no_ablation", {}).get(setting) is True
            ]
            row.denominator[setting] = len(correct_before)
            if correct_before:
                still = sum(
                    1 for r in correct_before if r["conditions"][condition].get(setting) is True
                )
                row.retained[setting] = 100.0 * still / len(correct_before)
            else:
                row.retained[setting] = None
        rows.append(row)

    return rows


def vqa_rows(results: dict) -> tuple[list[AblationRow], dict]:
    """VQA setting, which judges a single generated token via ``is_correct``."""
    images = results.get("images", {})
    correct_before = {
        k: r for k, r in images.items() if r["conditions"].get("no_ablation", {}).get("is_correct")
    }
    summary = {
        "n_total": len(images),
        "n_correct_before": len(correct_before),
        "baseline_prob": (
            statistics.mean(
                r["conditions"]["no_ablation"]["correct_token_prob"]
                for r in correct_before.values()
            )
            if correct_before
            else None
        ),
        "ambiguous": ambiguity(correct_before),
    }
    if not correct_before:
        return [], summary

    rows: list[AblationRow] = []
    for condition in CONDITION_ORDER:
        present = [r for r in correct_before.values() if condition in r.get("conditions", {})]
        if not present:
            continue
        row = AblationRow(
            condition=condition,
            avg_tokens=statistics.mean(r["conditions"][condition]["n_ablated"] for r in present),
            n=len(present),
        )
        row.retained["vqa"] = 100.0 * sum(
            1 for r in present if r["conditions"][condition]["is_correct"]
        ) / len(present)
        row.denominator["vqa"] = len(present)
        row.retained["correct_token_prob"] = statistics.mean(
            r["conditions"][condition]["correct_token_prob"] for r in present
        )
        rows.append(row)

    return rows, summary


def ambiguity(records: dict) -> dict:
    """How much of a single-token-judged result rests on ambiguous class names.

    "teddy bear" tokenizes to ' t', which "tie", "train" and "toilet" also produce, so
    those images can score correct for the wrong reason. They stay in the sample --
    dropping them would bias it -- but the share is reported.
    """
    flagged = [r for r in records.values() if r.get("ambiguous_class")]
    if not flagged or not records:
        return {"n": 0, "share": 0.0, "classes": []}
    return {
        "n": len(flagged),
        "share": 100.0 * len(flagged) / len(records),
        "classes": sorted({r.get("class_name", "?") for r in flagged}),
    }


def visual_token_stats(results: dict) -> dict | None:
    counts = [
        r["n_visual_tokens"] for r in results.get("images", {}).values() if r.get("n_visual_tokens")
    ]
    if not counts:
        return None
    return {"mean": statistics.mean(counts), "min": min(counts), "max": max(counts)}


# --------------------------------------------------------------------------------------
# Attention knockout
# --------------------------------------------------------------------------------------


def knockout_summary(results: dict) -> dict:
    """Relative accuracy per (from, to) group and layer window, plus the sweeps."""
    images = results.get("images", {})
    meta = results.get("meta", {})
    usable = {k: r for k, r in images.items() if r.get("baseline", {}).get("is_correct")}

    summary = {
        "meta": meta,
        "n_total": len(images),
        "n_usable": len(usable),
        "windows": list(meta.get("windows", {}).keys())
        or ["early", "early_mid", "mid", "mid_late", "late", "all"],
        "grid": {},
        "sweep": {},
        "ambiguous": ambiguity(usable),
        "control": None,
    }
    if not usable:
        return summary

    names: list[str] = []
    for record in usable.values():
        for name in record.get("knockout", {}):
            if name not in names:
                names.append(name)

    for name in names:
        scores = [r["knockout"][name]["is_correct"] for r in usable.values() if name in r["knockout"]]
        if not scores:
            continue
        value = sum(scores) / len(scores)
        head, _, window = name.partition("@")
        source, _, target = head.partition("->")
        if window.startswith(("L", "W")):
            summary["sweep"][window] = value
        else:
            summary["grid"].setdefault((source, target), {})[window] = value

    controls = [r["distance_control"] for r in usable.values() if "distance_control" in r]
    if controls:
        correct = [c for c in controls if c["baseline"]["is_correct"]]
        summary["control"] = {
            "n": len(correct),
            "mean_distance": statistics.mean(c["distance_to_last"] for c in controls),
            "conv_reach": controls[0]["conv_reach"],
            "retained": (
                sum(c["all_layers"]["is_correct"] for c in correct) / len(correct)
                if correct
                else None
            ),
        }
    return summary


# --------------------------------------------------------------------------------------
# Logit lens
# --------------------------------------------------------------------------------------


def logit_lens_summary(results: dict) -> dict:
    """Per-layer object-token hit rates, averaged over images."""
    images = results.get("images", {})
    usable = [r for r in images.values() if r.get("n_object_tokens", 0) > 0]
    summary = {
        "n_images": len(usable),
        "num_layers": results.get("meta", {}).get("num_layers"),
        "per_layer": {},
        "best": {},
    }
    if not usable:
        return summary

    n_entries = max(len(r["strict"]) for r in usable)
    for criterion in ("strict", "lenient"):
        summary["per_layer"][criterion] = [
            statistics.mean([r[criterion][i] for r in usable if i < len(r[criterion])])
            for i in range(n_entries)
        ]

        # Images whose object tokens never decode to the class at ANY layer have a
        # best rate of zero, so their argmax is an arbitrary tie -- in practice layer 0.
        # Averaging those in drags the "best layer" toward the start of the network and
        # reports a peak that does not exist. The best-layer statistics are therefore
        # computed over images that actually show the effect, with the count disclosed.
        with_signal = [r for r in usable if r[f"best_rate_{criterion}"] > 0]
        summary["best"][criterion] = {
            "n_with_signal": len(with_signal),
            "share_with_signal": 100.0 * len(with_signal) / len(usable),
            "rate": (
                statistics.mean(r[f"best_rate_{criterion}"] for r in with_signal)
                if with_signal else 0.0
            ),
            "rate_all_images": statistics.mean(r[f"best_rate_{criterion}"] for r in usable),
            "layer": (
                statistics.mean(r[f"best_layer_{criterion}"] for r in with_signal)
                if with_signal else None
            ),
            "median_layer": (
                statistics.median(r[f"best_layer_{criterion}"] for r in with_signal)
                if with_signal else None
            ),
        }

    # The peak of the averaged curve: a more robust depth estimate than per-image argmax.
    strict_curve = summary["per_layer"]["strict"]
    summary["peak_layer"] = int(max(range(len(strict_curve)), key=strict_curve.__getitem__))
    summary["peak_rate"] = max(strict_curve)
    summary["n_entries"] = n_entries
    return summary
