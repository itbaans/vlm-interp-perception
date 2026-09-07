"""Visual-token span detection, checked against the real processor.

These run without model weights: the processor is real, the model is the tiny
random-weight miniature, and none of the assertions depend on what the weights are.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

import geometry as G
import prompts as P


@pytest.fixture
def prompt(tiny):
    return P.describe_prompt(tiny.processor)


# --------------------------------------------------------------------------------------
# The core invariant
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (500, 375), (375, 500), (1024, 768)])
def test_grid_accounts_for_every_image_token(tiny, prompt, size):
    """sum(n_rows * n_cols) over tiles == number of image-token positions.

    This is the assertion that catches any geometry drift; ``image_token_spans`` raises
    if it fails, so simply not raising is the test.
    """
    width, height = size
    inputs = tiny.prepare(Image.new("RGB", (width, height)), prompt)
    grids = tiny.image_token_spans(inputs, (height, width))

    expected = int((inputs["input_ids"][0] == tiny.image_token_id).sum())
    assert sum(g.n_tokens for g in grids) == expected


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (500, 375), (1024, 768), (3000, 2000)])
def test_offline_geometry_matches_the_processor(tiny, prompt, size):
    """``token_grid_for_image`` must agree with what the processor actually produced.

    With splitting disabled the offline arithmetic is authoritative for every size,
    which is what lets scripts compute grids without running the vision tower.
    """
    width, height = size
    inputs = tiny.prepare(Image.new("RGB", (width, height)), prompt)
    from_processor = tiny.single_grid(inputs, (height, width))
    offline = G.token_grid_for_image(orig_h=height, orig_w=width, start=from_processor.start)

    assert (offline.n_rows, offline.n_cols) == (from_processor.n_rows, from_processor.n_cols)
    assert offline.resized_h == from_processor.resized_h
    assert offline.resized_w == from_processor.resized_w


def test_common_coco_image_has_the_expected_grid(tiny, prompt, image):
    inputs = tiny.prepare(image, prompt)
    grid = tiny.single_grid(inputs, (480, 640))
    assert (grid.n_rows, grid.n_cols, grid.n_tokens) == (13, 18, 234)


# --------------------------------------------------------------------------------------
# Span placement
# --------------------------------------------------------------------------------------


def test_span_covers_exactly_the_image_token_positions(tiny, prompt, image):
    inputs = tiny.prepare(image, prompt)
    grid = tiny.single_grid(inputs, (480, 640))

    input_ids = inputs["input_ids"][0]
    positions = torch.nonzero(input_ids == tiny.image_token_id).flatten().tolist()
    assert positions == list(range(grid.start, grid.end))


def test_span_is_bracketed_by_the_image_delimiters(tiny, prompt, image):
    """``<|image_start|>`` and ``<|image_end|>`` must sit outside the span, not in it."""
    inputs = tiny.prepare(image, prompt)
    grid = tiny.single_grid(inputs, (480, 640))
    tokenizer = tiny.processor.tokenizer
    input_ids = inputs["input_ids"][0].tolist()

    assert tokenizer.decode([input_ids[grid.start - 1]]) == "<|image_start|>"
    assert tokenizer.decode([input_ids[grid.end]]) == "<|image_end|>"


def test_span_does_not_start_at_zero(tiny, prompt, image):
    """The visual tokens follow the chat header, so ``start`` must be an honest offset."""
    inputs = tiny.prepare(image, prompt)
    assert tiny.single_grid(inputs, (480, 640)).start > 0


def test_last_visual_row_indices_are_the_final_n_cols_tokens(tiny, prompt, image):
    """The 'last row of visual tokens' condition from the paper's Table 2."""
    inputs = tiny.prepare(image, prompt)
    grid = tiny.single_grid(inputs, (480, 640))
    last_row = [grid.cell_to_index(grid.n_rows - 1, c) for c in range(grid.n_cols)]
    assert last_row == list(range(grid.end - grid.n_cols, grid.end))


# --------------------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------------------


def test_single_grid_rejects_a_tiled_image(tiny, prompt):
    """With splitting enabled a large image yields several tiles; single_grid must refuse."""
    tiny.do_image_splitting = True
    try:
        inputs = tiny.prepare(Image.new("RGB", (1024, 768)), prompt)
        grids = tiny.image_token_spans(inputs, (768, 1024))
        assert len(grids) > 1
        with pytest.raises(ValueError, match="single tile"):
            tiny.single_grid(inputs, (768, 1024))
    finally:
        tiny.do_image_splitting = False


def test_tiled_spans_still_account_for_every_image_token(tiny, prompt):
    tiny.do_image_splitting = True
    try:
        inputs = tiny.prepare(Image.new("RGB", (1024, 768)), prompt)
        grids = tiny.image_token_spans(inputs, (768, 1024))
        expected = int((inputs["input_ids"][0] == tiny.image_token_id).sum())
        assert sum(g.n_tokens for g in grids) == expected
        # Runs are disjoint and ordered.
        for earlier, later in zip(grids, grids[1:]):
            assert earlier.end <= later.start
    finally:
        tiny.do_image_splitting = False


def test_text_only_input_has_no_image_tokens(tiny):
    prompt = P.build_prompt(tiny.processor, "Hello.", with_image=False)
    inputs = tiny.processor(text=prompt, return_tensors="pt")
    assert int((inputs["input_ids"][0] == tiny.image_token_id).sum()) == 0


# --------------------------------------------------------------------------------------
# Object indices on a real span
# --------------------------------------------------------------------------------------


def test_object_indices_land_inside_the_span(tiny, prompt, image):
    inputs = tiny.prepare(image, prompt)
    grid = tiny.single_grid(inputs, (480, 640))
    ann = {
        "bbox": [300, 200, 40, 40],
        "area": 1600,
        "iscrowd": 0,
        "segmentation": [[300, 200, 340, 200, 340, 240, 300, 240]],
    }

    plain = G.object_token_indices(ann, grid, buffer=0)
    buffered = G.object_token_indices(ann, grid, buffer=1)

    assert plain, "a 40x40 object should cover at least one cell"
    assert set(plain).issubset(buffered)
    assert len(buffered) > len(plain)
    for index in buffered:
        assert grid.start <= index < grid.end
        # And every index must really be an image-token position.
        assert inputs["input_ids"][0, index].item() == tiny.image_token_id
