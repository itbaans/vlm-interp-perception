"""Synthetic scene generation and task construction.

Mostly offline: the scene tests need no model and no processor. Only the single-token
answer assertions touch the real tokenizer.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

import geometry as G
import synthetic as S
import tasks as T


# --------------------------------------------------------------------------------------
# Canvas and token-grid alignment -- the reason the canvas is 512x512
# --------------------------------------------------------------------------------------


def test_canvas_maps_to_a_clean_16x16_token_grid():
    """512x512 is the one size that survives smart_resize untouched."""
    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)
    assert (grid.resized_h, grid.resized_w) == (S.CANVAS, S.CANVAS)  # no resampling
    assert (grid.n_rows, grid.n_cols) == (16, 16)
    assert grid.n_tokens == 256
    assert S.CANVAS / grid.n_cols == S.TOKEN_CELL_PX == 32


def test_shape_on_cell_boundaries_maps_to_exactly_those_cells():
    """With 32 px cells and no resampling this is an equality, not an approximation."""
    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)
    # A square spanning x,y in [64,128) -> cells rows 2-3, cols 2-3.
    shape = S.Shape(kind="square", colour="red", cx=96, cy=96, size=64)
    assert S.shape_cells(shape, grid) == {(2, 2), (2, 3), (3, 2), (3, 3)}


def test_cells_convert_to_contiguous_token_indices():
    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS, start=5)
    shape = S.Shape(kind="square", colour="red", cx=96, cy=96, size=64)
    indices = S.cells_to_token_indices(S.shape_cells(shape, grid), grid)
    assert indices == sorted(indices)
    assert all(grid.start <= i < grid.end for i in indices)


# --------------------------------------------------------------------------------------
# Shapes: the polygon that is drawn is the polygon that is annotated
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", S.SHAPE_KINDS)
def test_annotation_matches_rendered_pixels(kind):
    """IoU between the annotation mask and the non-white pixels must be near 1.

    This is the invariant that lets every helper in geometry.py operate on synthetic
    scenes: if the annotation drifted from the render, every token-index mapping
    downstream would be quietly wrong.
    """
    shape = S.Shape(kind=kind, colour="red", cx=256, cy=256, size=160)
    rendered = np.array(S.Scene([shape]).render())
    drawn = (rendered.sum(axis=-1) < 720)  # anything not white

    annotated = G.rasterize_annotation(shape.annotation(), S.CANVAS, S.CANVAS)

    intersection = (drawn & annotated).sum()
    union = (drawn | annotated).sum()
    assert union > 0
    assert intersection / union > 0.95, f"{kind}: IoU {intersection / union:.3f}"


@pytest.mark.parametrize("kind", S.SHAPE_KINDS)
def test_shape_stays_within_its_bounding_box(kind):
    shape = S.Shape(kind=kind, colour="blue", cx=200, cy=180, size=100)
    x0, y0, x1, y1 = shape.bbox
    for x, y in shape.points():
        assert x0 - 1 <= x <= x1 + 1
        assert y0 - 1 <= y <= y1 + 1


def test_rotation_changes_arrow_geometry_but_not_bbox_centre():
    up = S.Shape(kind="arrow", colour="red", cx=256, cy=256, size=120, direction="up")
    down = S.Shape(kind="arrow", colour="red", cx=256, cy=256, size=120, direction="down")
    assert up.points() != down.points()
    assert up.bbox == down.bbox  # same footprint, different orientation


def test_rotated_arrows_render_differently():
    left = S.Scene([S.Shape("arrow", "red", 256, 256, 160, "left")]).render()
    right = S.Scene([S.Shape("arrow", "red", 256, 256, 160, "right")]).render()
    assert not np.array_equal(np.array(left), np.array(right))


def test_invalid_shape_attributes_are_rejected():
    with pytest.raises(ValueError):
        S.Shape(kind="hexagon", colour="red", cx=10, cy=10, size=10)
    with pytest.raises(ValueError):
        S.Shape(kind="circle", colour="mauve", cx=10, cy=10, size=10)
    with pytest.raises(ValueError):
        S.Shape(kind="arrow", colour="red", cx=10, cy=10, size=10, direction="sideways")


# --------------------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["L1", "L2", "L3", "L4"])
def test_shapes_do_not_overlap_below_the_crowded_level(level):
    rng = random.Random(level)
    for _ in range(20):
        config = S.LEVELS[level]
        count = rng.randint(*config["n_range"])
        specs = [("circle", "red", "up")] * count
        scene = S.build_scene(specs, rng, crowded=config["crowded"])
        for i, a in enumerate(scene.shapes):
            for b in scene.shapes[i + 1:]:
                assert not a.overlaps(b), f"{level}: shapes overlap"


def test_all_shapes_stay_on_canvas():
    rng = random.Random(7)
    for level in S.LEVELS:
        config = S.LEVELS[level]
        count = config["n_range"][1]
        scene = S.build_scene([("star", "green", "up")] * count, rng,
                              crowded=config["crowded"])
        for shape in scene.shapes:
            x0, y0, x1, y1 = shape.bbox
            assert 0 <= x0 and 0 <= y0 and x1 <= S.CANVAS and y1 <= S.CANVAS


def test_requested_shape_count_is_what_gets_rendered():
    rng = random.Random(3)
    for count in range(2, 8):
        scene = S.build_scene([("circle", "blue", "up")] * count, rng)
        assert len(scene.shapes) == count


# --------------------------------------------------------------------------------------
# Scene queries
# --------------------------------------------------------------------------------------


def test_scene_select_and_count_filter_correctly():
    shapes = [
        S.Shape("circle", "red", 60, 60, 40),
        S.Shape("circle", "red", 160, 60, 40),
        S.Shape("circle", "blue", 260, 60, 40),
        S.Shape("square", "red", 360, 60, 40),
    ]
    scene = S.Scene(shapes)
    assert scene.count(colour="red") == 3
    assert scene.count(kind="circle") == 3
    assert scene.count(colour="red", kind="circle") == 2
    assert scene.indices_of(colour="blue") == [2]


def test_background_cells_exclude_every_shape():
    grid = G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)
    scene = S.Scene([S.Shape("square", "red", 96, 96, 64)])
    background = S.background_cells(scene, grid)
    assert (2, 2) not in background
    assert (0, 0) in background
    assert len(background) == 256 - 4


# --------------------------------------------------------------------------------------
# Minimal pairs -- the core invariant of the whole experiment
# --------------------------------------------------------------------------------------

ALL_TASKS = list(T.TASKS)


@pytest.mark.parametrize("task_name", ALL_TASKS)
@pytest.mark.parametrize("level", ["L1", "L3", "L5"])
def test_minimal_pair_changes_exactly_one_shape(task_name, level):
    """Clean and counterfactual must differ in exactly one shape, and the answer must flip.

    If more than one shape moved, patching would attribute the effect to the wrong tokens.
    """
    for sample in T.generate(task_name, level, n=6, seed=1):
        clean, cf = sample.clean.shapes, sample.counterfactual.shapes
        assert len(clean) == len(cf)

        differing = [i for i, (a, b) in enumerate(zip(clean, cf)) if a != b]
        if sample.task == "relation":
            # A position swap necessarily moves two shapes; that is the intended edit.
            assert len(differing) == 2
        else:
            assert differing == [sample.changed_index], (
                f"{task_name}/{level}: {len(differing)} shapes differ"
            )
        assert sample.answer != sample.answer_cf


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_pair_images_have_identical_dimensions(task_name):
    """Patching requires aligned token positions, which requires identical image sizes."""
    for sample in T.generate(task_name, "L3", n=3, seed=2):
        assert sample.clean.render().size == sample.counterfactual.render().size
        assert sample.clean.render().size == (S.CANVAS, S.CANVAS)


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_pair_renders_actually_differ(task_name):
    for sample in T.generate(task_name, "L3", n=3, seed=3):
        a = np.array(sample.clean.render())
        b = np.array(sample.counterfactual.render())
        assert not np.array_equal(a, b)


def test_count_counterfactual_is_one_fewer():
    for sample in T.generate("count", "L3", n=8, seed=4):
        assert int(sample.answer_cf) == int(sample.answer) - 1
        assert int(sample.answer) >= 2  # so the counterfactual stays >= 1


@pytest.mark.parametrize("level", ["L1", "L2", "L3", "L4", "L5"])
def test_count_answers_match_the_actual_scenes(level):
    """Both labels must be recomputable from the scenes, at every level.

    This is the check that catches a counterfactual editing the wrong attribute: at the
    unfiltered levels the question is "how many circles", so recolouring a circle leaves
    the count unchanged and the pair would claim 2 -> 1 while both scenes show 2.
    """
    for sample in T.generate("count", level, n=8, seed=5):
        colour, kind = sample.query["colour"], sample.query["kind"]
        assert int(sample.answer) == sample.clean.count(colour=colour, kind=kind)
        assert int(sample.answer_cf) == sample.counterfactual.count(colour=colour, kind=kind)


@pytest.mark.parametrize("level", ["L1", "L2", "L3", "L4", "L5"])
def test_count_pair_keeps_the_same_number_of_objects(level):
    """Only the *matching* count changes; the object count must not."""
    for sample in T.generate("count", level, n=6, seed=19):
        assert len(sample.clean.shapes) == len(sample.counterfactual.shapes)


def test_count_recolour_keeps_object_positions_identical():
    """The pair must not change *how many objects* there are, only how many match."""
    for sample in T.generate("count", "L3", n=5, seed=6):
        for a, b in zip(sample.clean.shapes, sample.counterfactual.shapes):
            assert (a.cx, a.cy, a.size, a.kind) == (b.cx, b.cy, b.size, b.kind)


def test_colour_answer_matches_the_queried_shape():
    for sample in T.generate("colour", "L3", n=6, seed=7):
        queried = sample.clean.select(kind=sample.query["kind"])
        assert len(queried) == 1, "the queried kind must be unique in the scene"
        assert queried[0].colour == sample.answer


def test_shape_answer_matches_the_queried_colour():
    for sample in T.generate("shape", "L3", n=6, seed=8):
        queried = sample.clean.select(colour=sample.query["colour"])
        assert len(queried) == 1
        assert queried[0].kind == sample.answer


def test_orientation_answer_matches_the_arrow():
    for sample in T.generate("orientation", "L3", n=6, seed=9):
        arrows = sample.clean.select(kind="arrow")
        assert len(arrows) == 1
        assert arrows[0].direction == sample.answer


def test_relation_answer_matches_the_geometry():
    for sample in T.generate("relation", "L3", n=6, seed=10):
        subject = sample.clean.shapes[sample.query["subject_index"]]
        obj = sample.clean.shapes[sample.query["object_index"]]
        expected = "above" if subject.cy < obj.cy else "below"
        assert sample.answer == expected


# --------------------------------------------------------------------------------------
# Determinism and serialisation
# --------------------------------------------------------------------------------------


def test_generation_is_deterministic_for_a_seed():
    a = T.generate("count", "L3", n=4, seed=11)
    b = T.generate("count", "L3", n=4, seed=11)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_different_seeds_give_different_scenes():
    a = T.generate("count", "L3", n=4, seed=12)
    b = T.generate("count", "L3", n=4, seed=13)
    assert [s.to_dict() for s in a] != [s.to_dict() for s in b]


def test_sample_ids_are_unique_and_stable():
    samples = T.generate("colour", "L2", n=5, seed=14)
    ids = [s.sample_id for s in samples]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "colour-L2-0000"


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_sample_round_trips_through_json(task_name):
    for sample in T.generate(task_name, "L3", n=2, seed=15):
        restored = S.Sample.from_dict(sample.to_dict())
        assert restored.to_dict() == sample.to_dict()
        assert restored.clean.shapes == sample.clean.shapes
        assert restored.answer == sample.answer


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="unknown level"):
        T.generate("count", "L9", n=1)


# --------------------------------------------------------------------------------------
# Single-token answers -- needs the real tokenizer
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_every_answer_is_exactly_one_token(processor, task_name):
    """The finding this guards: ' 3' is two tokens, '3' is one."""
    ids = T.TASKS[task_name].vocabulary_ids(processor)
    assert len(ids) == len(T.TASKS[task_name].vocabulary)
    assert all(isinstance(i, int) for i in ids.values())


def test_digit_ids_are_contiguous(processor):
    """Contiguity is what makes a full P(count=k) distribution cheap to read off."""
    ids = [T.TASKS["count"].answer_token_id(processor, str(d)) for d in range(10)]
    assert ids == list(range(ids[0], ids[0] + 10))


def test_count_prefill_ends_with_a_standalone_space_token(processor):
    """The trailing space must survive tokenization, or digits become two tokens."""
    sample = T.generate("count", "L1", n=1, seed=16)[0]
    prefill = T.TASKS["count"].prefill(sample)
    assert prefill.endswith(" ")
    ids = processor.tokenizer.encode(prefill, add_special_tokens=False)
    assert processor.tokenizer.decode([ids[-1]]) == " "


def test_word_task_prefills_have_no_trailing_space(processor):
    for name in ("colour", "shape", "orientation", "relation"):
        sample = T.generate(name, "L3", n=1, seed=17)[0]
        assert not T.TASKS[name].prefill(sample).endswith(" ")


def test_verify_single_token_reports_every_task(processor):
    report = T.verify_single_token(processor)
    assert set(report) == set(T.TASKS)
    assert report["count"]["3"] == T.TASKS["count"].answer_token_id(processor, "3")


def test_prompt_contains_the_question_and_prefill(processor):
    sample = T.generate("colour", "L3", n=1, seed=18)[0]
    task = T.TASKS["colour"]
    prompt = T.build_prompt(processor, task, sample)
    assert task.question(sample) in prompt
    assert prompt.endswith(task.prefill(sample))
    assert "<image>" in prompt  # inserted by the chat template, not by us
