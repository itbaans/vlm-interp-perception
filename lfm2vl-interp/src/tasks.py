"""Perception tasks: prompts, single-token answers, and minimal-pair construction.

Every task produces a ``Sample`` whose clean and counterfactual scenes differ in exactly
one shape attribute, flipping the correct answer. All answers are a single token so the
model's next-token distribution *is* the answer distribution -- no generation, no parsing.

The single-token requirement has one subtlety worth stating plainly, because getting it
wrong silently ruins a run:

    Space-prefixed digits are NOT single tokens: ' 3' -> [229, 27].
    Bare digits ARE:                              '3'  -> [27], and 0-9 are contiguous
                                                  ids 24-33.

So counting prompts use a prefill that **ends with a space** (the space becomes its own
token, id 229) and the answer is a bare digit. Word answers do the opposite: no trailing
space, and the answer carries the leading space (' red' -> [2959]). ``answer_style``
records which convention a task uses, and ``verify_single_token`` asserts it against the
real tokenizer.

Because digits are contiguous, a softmax restricted to ids 24-33 gives a full
``P(count = k)`` distribution -- a much richer signal than a correct/incorrect bit, and
what makes "off by one" distinguishable from "guessing".
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from synthetic import (
    COLOURS,
    DIRECTIONS,
    Sample,
    Scene,
    build_scene,
    recolour,
    reorient,
    reshape,
    swap_positions,
)

DIGITS = [str(d) for d in range(10)]
COLOUR_NAMES = list(COLOURS)
DIRECTION_NAMES = list(DIRECTIONS)


@dataclass(frozen=True)
class Task:
    """One perceptual ability under test."""

    name: str
    #: "bare" -> prefill ends with a space, answer is a bare token (digits).
    #: "spaced" -> prefill has no trailing space, answer carries a leading space (words).
    answer_style: str
    vocabulary: list[str]
    build: Callable[[random.Random, str, str], Sample]
    question: Callable[[Sample], str]
    prefill: Callable[[Sample], str]
    description: str = ""

    def answer_text(self, answer: str) -> str:
        """The exact string the model should emit next."""
        return answer if self.answer_style == "bare" else f" {answer}"

    def answer_token_id(self, processor, answer: str) -> int:
        ids = processor.tokenizer.encode(self.answer_text(answer), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"task {self.name!r} answer {answer!r} is {len(ids)} tokens, not 1 "
                f"(encoded {self.answer_text(answer)!r} -> {ids})"
            )
        return ids[0]

    def vocabulary_ids(self, processor) -> dict[str, int]:
        """Answer string -> token id, for the whole answer space of this task."""
        return {a: self.answer_token_id(processor, a) for a in self.vocabulary}


# --------------------------------------------------------------------------------------
# Scene builders
# --------------------------------------------------------------------------------------


def _level_config(level: str) -> dict:
    from synthetic import LEVELS

    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {list(LEVELS)}")
    return LEVELS[level]


def _build_count(rng: random.Random, level: str, sample_id: str) -> Sample:
    """Counting, with a colour filter from L3 upward.

    The counterfactual changes exactly one target shape so that it stops matching the
    query, keeping the same number of objects in the same places. Removing a shape instead
    would confound "one fewer red circle" with "one fewer object".

    *Which* attribute changes depends on what the question filters on. When the query is
    colour-filtered ("how many red circles"), recolouring one target works. When it is not
    ("how many circles"), recolouring changes nothing -- the shape is still a circle -- so
    the counterfactual must change the target's **kind** instead. Getting this wrong
    produces pairs whose labels say 2 -> 1 while both scenes show 2.
    """
    config = _level_config(level)
    total = rng.randint(*config["n_range"])
    filtered = config["colours"] > 1

    target_colour, distractor_colour = rng.sample(COLOUR_NAMES, 2)
    if config["kinds"] > 1:
        target_kind, other_kind = rng.sample(("circle", "square", "triangle", "star"), 2)
    else:
        target_kind, other_kind = rng.sample(("circle", "square", "triangle"), 2)
        if filtered:
            # One kind everywhere; the colour filter does the discriminating.
            other_kind = target_kind
    # The kind the changed shape becomes when the query is not colour-filtered.
    swap_kind = next(k for k in ("circle", "square", "triangle", "star") if k != target_kind)

    # At least 2 targets so the counterfactual answer stays >= 1.
    n_target = rng.randint(2, max(2, total - 1)) if filtered else total
    n_other = total - n_target

    specs = [(target_kind, target_colour, "up")] * n_target
    specs += [(other_kind, distractor_colour if filtered else target_colour, "up")] * n_other
    rng.shuffle(specs)

    clean = build_scene(specs, rng, crowded=config["crowded"])
    target_indices = clean.indices_of(
        colour=target_colour if filtered else None, kind=target_kind
    )
    changed = rng.choice(target_indices)
    # Change whichever attribute the query actually filters on, so the count really drops.
    counterfactual = (
        recolour(clean, changed, distractor_colour) if filtered
        else reshape(clean, changed, swap_kind)
    )

    return Sample(
        sample_id=sample_id,
        task="count",
        level=level,
        clean=clean,
        counterfactual=counterfactual,
        changed_index=changed,
        answer=str(len(target_indices)),
        answer_cf=str(len(target_indices) - 1),
        query={
            "colour": target_colour if filtered else None,
            "kind": target_kind,
            "filtered": filtered,
        },
    )


def _unique_kind_scene(rng: random.Random, level: str) -> tuple[Scene, int, str]:
    """A scene where one shape kind appears exactly once. Returns (scene, index, kind)."""
    config = _level_config(level)
    total = rng.randint(*config["n_range"])
    kinds = list(rng.sample(("circle", "square", "triangle", "star"), 3))
    unique_kind, filler_a, filler_b = kinds

    specs = [(unique_kind, rng.choice(COLOUR_NAMES), "up")]
    specs += [
        (rng.choice((filler_a, filler_b)), rng.choice(COLOUR_NAMES), "up")
        for _ in range(total - 1)
    ]
    rng.shuffle(specs)
    scene = build_scene(specs, rng, crowded=config["crowded"])
    index = scene.indices_of(kind=unique_kind)[0]
    return scene, index, unique_kind


def _build_colour(rng: random.Random, level: str, sample_id: str) -> Sample:
    """Read the colour of a uniquely identifiable shape; the pair flips that colour."""
    scene, index, kind = _unique_kind_scene(rng, level)
    original = scene.shapes[index].colour
    new_colour = rng.choice([c for c in COLOUR_NAMES if c != original])
    return Sample(
        sample_id=sample_id,
        task="colour",
        level=level,
        clean=scene,
        counterfactual=recolour(scene, index, new_colour),
        changed_index=index,
        answer=original,
        answer_cf=new_colour,
        query={"kind": kind},
    )


def _build_shape(rng: random.Random, level: str, sample_id: str) -> Sample:
    """Read the kind of a uniquely coloured shape; the pair changes that kind."""
    config = _level_config(level)
    total = rng.randint(*config["n_range"])
    target_colour, filler_colour = rng.sample(COLOUR_NAMES, 2)
    target_kind, alt_kind, filler_kind = rng.sample(
        ("circle", "square", "triangle", "star"), 3
    )

    specs = [(target_kind, target_colour, "up")]
    specs += [(filler_kind, filler_colour, "up") for _ in range(total - 1)]
    rng.shuffle(specs)
    scene = build_scene(specs, rng, crowded=config["crowded"])
    index = scene.indices_of(colour=target_colour)[0]

    return Sample(
        sample_id=sample_id,
        task="shape",
        level=level,
        clean=scene,
        counterfactual=reshape(scene, index, alt_kind),
        changed_index=index,
        answer=target_kind,
        answer_cf=alt_kind,
        query={"colour": target_colour},
    )


def _build_orientation(rng: random.Random, level: str, sample_id: str) -> Sample:
    """Read an arrow's direction; the pair rotates it."""
    config = _level_config(level)
    total = rng.randint(*config["n_range"])
    arrow_colour, filler_colour = rng.sample(COLOUR_NAMES, 2)
    direction = rng.choice(DIRECTION_NAMES)
    other = rng.choice([d for d in DIRECTION_NAMES if d != direction])
    filler_kind = rng.choice(("circle", "square", "star"))

    specs = [("arrow", arrow_colour, direction)]
    specs += [(filler_kind, filler_colour, "up") for _ in range(total - 1)]
    rng.shuffle(specs)
    scene = build_scene(specs, rng, crowded=config["crowded"])
    index = scene.indices_of(kind="arrow")[0]

    return Sample(
        sample_id=sample_id,
        task="orientation",
        level=level,
        clean=scene,
        counterfactual=reorient(scene, index, other),
        changed_index=index,
        answer=direction,
        answer_cf=other,
        query={"colour": arrow_colour},
    )


def _build_relation(rng: random.Random, level: str, sample_id: str) -> Sample:
    """Above/below between two named shapes; the pair swaps their positions."""
    config = _level_config(level)
    total = max(2, rng.randint(*config["n_range"]))
    colour_a, colour_b = rng.sample(COLOUR_NAMES, 2)
    kind_a, kind_b = rng.sample(("circle", "square", "triangle", "star"), 2)

    specs = [(kind_a, colour_a, "up"), (kind_b, colour_b, "up")]
    specs += [
        (rng.choice(("circle", "square")), rng.choice(COLOUR_NAMES), "up")
        for _ in range(total - 2)
    ]
    rng.shuffle(specs)
    scene = build_scene(specs, rng, crowded=config["crowded"])

    first = scene.indices_of(colour=colour_a, kind=kind_a)[0]
    second = scene.indices_of(colour=colour_b, kind=kind_b)[0]
    # Guarantee an unambiguous vertical relation.
    if abs(scene.shapes[first].cy - scene.shapes[second].cy) < 40:
        return _build_relation(rng, level, sample_id)

    above = scene.shapes[first].cy < scene.shapes[second].cy
    return Sample(
        sample_id=sample_id,
        task="relation",
        level=level,
        clean=scene,
        counterfactual=swap_positions(scene, first, second),
        changed_index=first,
        answer="above" if above else "below",
        answer_cf="below" if above else "above",
        query={
            "subject": f"{colour_a} {kind_a}",
            "object": f"{colour_b} {kind_b}",
            "subject_index": first,
            "object_index": second,
        },
    )


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------


def _count_question(sample: Sample) -> str:
    kind, colour = sample.query["kind"], sample.query["colour"]
    what = f"{colour} {kind}s" if colour else f"{kind}s"
    return f"How many {what} are there?"


def _count_prefill(sample: Sample) -> str:
    kind, colour = sample.query["kind"], sample.query["colour"]
    what = f"{colour} {kind}s" if colour else f"{kind}s"
    # Trailing space is deliberate: it becomes its own token, so the answer is a bare digit.
    return f"The number of {what} is "


TASKS: dict[str, Task] = {
    "count": Task(
        name="count",
        answer_style="bare",
        vocabulary=DIGITS,
        build=_build_count,
        question=_count_question,
        prefill=_count_prefill,
        description="How many objects match a colour/kind filter",
    ),
    "colour": Task(
        name="colour",
        answer_style="spaced",
        vocabulary=COLOUR_NAMES,
        build=_build_colour,
        question=lambda s: f"What colour is the {s.query['kind']}?",
        prefill=lambda s: f"The {s.query['kind']} is",
        description="Read the colour of a uniquely shaped object (binding)",
    ),
    "shape": Task(
        name="shape",
        answer_style="spaced",
        vocabulary=["circle", "square", "triangle", "star"],
        build=_build_shape,
        question=lambda s: f"What shape is the {s.query['colour']} object?",
        prefill=lambda s: f"The {s.query['colour']} object is a",
        description="Read the kind of a uniquely coloured object (binding, reversed)",
    ),
    "orientation": Task(
        name="orientation",
        answer_style="spaced",
        vocabulary=DIRECTION_NAMES,
        build=_build_orientation,
        question=lambda s: f"Which way does the {s.query['colour']} arrow point?",
        prefill=lambda s: f"The {s.query['colour']} arrow points",
        description="Read an arrow's direction",
    ),
    "relation": Task(
        name="relation",
        answer_style="spaced",
        vocabulary=["above", "below"],
        build=_build_relation,
        question=lambda s: (
            f"Is the {s.query['subject']} above or below the {s.query['object']}?"
        ),
        prefill=lambda s: f"The {s.query['subject']} is",
        description="Vertical spatial relation between two named objects",
    ),
}


def build_prompt(processor, task: Task, sample: Sample) -> str:
    """Full chat-template prompt with the assistant turn prefilled."""
    import prompts as P

    return P.build_prompt(processor, task.question(sample), prefill=task.prefill(sample))


def generate(
    task_name: str,
    level: str,
    n: int,
    seed: int = 0,
    start_index: int = 0,
) -> list[Sample]:
    """Deterministically generate ``n`` samples for one (task, level) cell."""
    task = TASKS[task_name]
    rng = random.Random(f"{task_name}:{level}:{seed}")
    return [
        task.build(rng, level, f"{task_name}-{level}-{start_index + i:04d}")
        for i in range(n)
    ]


def interleave(samples: list[Sample]) -> list[Sample]:
    """Round-robin samples across (task, level) cells so any prefix is balanced.

    Generation is naturally grouped -- all of count-L1, then all of count-L2, and so on --
    which makes ``--limit 12`` silently mean "only counting at L1 and L2". Storing the
    dataset interleaved means every downstream script's ``--limit`` takes a stratified
    sample instead, without any of them needing to know about it.
    """
    cells: dict[tuple[str, str], list[Sample]] = {}
    for sample in samples:
        cells.setdefault((sample.task, sample.level), []).append(sample)

    ordered: list[Sample] = []
    for position in range(max((len(v) for v in cells.values()), default=0)):
        for key in cells:
            if position < len(cells[key]):
                ordered.append(cells[key][position])
    return ordered


def verify_single_token(processor, task_names: list[str] | None = None) -> dict[str, dict]:
    """Assert every answer in every task vocabulary is exactly one token.

    Called by the dataset builder before anything is written, so a tokenization surprise
    fails immediately rather than silently corrupting a run.
    """
    report: dict[str, dict] = {}
    for name in task_names or list(TASKS):
        task = TASKS[name]
        report[name] = task.vocabulary_ids(processor)  # raises if any answer is multi-token
    return report
