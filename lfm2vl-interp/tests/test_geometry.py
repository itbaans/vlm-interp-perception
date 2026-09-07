"""Offline geometry tests: no model, no processor, no network."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from geometry import (  # noqa: E402
    TokenGrid,
    add_buffer,
    annotation_to_cells,
    cells_to_indices,
    grid_from_spatial_shape,
    is_image_too_large,
    mask_image_regions,
    object_token_indices,
    random_indices,
    smart_resize,
    token_grid_for_image,
)


# --------------------------------------------------------------------------------------
# Resize arithmetic
# --------------------------------------------------------------------------------------


def test_landscape_coco_image_is_not_split():
    # 640x480 -> 307,200 px, under the 256 * 16^2 * 2^2 * 2.0 = 524,288 budget.
    assert is_image_too_large(480, 640) is False


def test_portrait_coco_image_is_not_split():
    assert is_image_too_large(640, 480) is False


def test_very_large_image_is_split():
    assert is_image_too_large(2000, 3000) is True


def test_landscape_coco_grid_matches_hand_computation():
    """640x480 -> 416x576 px -> 26x36 patches -> 13x18 = 234 visual tokens."""
    grid = token_grid_for_image(orig_h=480, orig_w=640)
    assert (grid.resized_h, grid.resized_w) == (416, 576)
    assert (grid.n_rows, grid.n_cols) == (13, 18)
    assert grid.n_tokens == 234


def test_portrait_coco_grid_is_the_transpose():
    grid = token_grid_for_image(orig_h=640, orig_w=480)
    assert (grid.resized_h, grid.resized_w) == (576, 416)
    assert (grid.n_rows, grid.n_cols) == (18, 13)
    assert grid.n_tokens == 234


@pytest.mark.parametrize(
    "orig_h,orig_w",
    [(480, 640), (640, 480), (500, 375), (333, 500), (128, 128), (1024, 768), (200, 900)],
)
def test_resize_invariants(orig_h, orig_w):
    """Every resize is divisible by 32 and lands inside the token budget."""
    grid = token_grid_for_image(orig_h=orig_h, orig_w=orig_w)
    assert grid.resized_h % 32 == 0
    assert grid.resized_w % 32 == 0
    assert 64 <= grid.n_tokens <= 256
    # Divisibility by 32 means the 2x2 unshuffle never pads, so ceil == exact division.
    assert grid.n_rows * 2 == grid.resized_h // 16
    assert grid.n_cols * 2 == grid.resized_w // 16


def test_tiny_image_is_scaled_up_to_the_minimum():
    grid = token_grid_for_image(orig_h=40, orig_w=40)
    assert grid.n_tokens >= 64


def test_smart_resize_returns_height_width_order():
    # Guards against the (w_bar, h_bar) ordering the HF implementation uses.
    height, width = smart_resize(480, 640)
    assert width > height


def test_grid_from_spatial_shape_agrees_with_offline_computation():
    """The processor's spatial_shapes and our offline arithmetic must not diverge."""
    offline = token_grid_for_image(orig_h=480, orig_w=640, start=7)
    from_processor = grid_from_spatial_shape([26, 36], orig_h=480, orig_w=640, start=7)
    assert (from_processor.n_rows, from_processor.n_cols) == (offline.n_rows, offline.n_cols)
    assert from_processor.resized_h == offline.resized_h
    assert from_processor.resized_w == offline.resized_w


# --------------------------------------------------------------------------------------
# Index bookkeeping
# --------------------------------------------------------------------------------------


def _grid(start: int = 0) -> TokenGrid:
    return TokenGrid(
        n_rows=13, n_cols=18, start=start, orig_h=480, orig_w=640, resized_h=416, resized_w=576
    )


def test_cell_index_roundtrip():
    grid = _grid(start=5)
    for row in range(grid.n_rows):
        for col in range(grid.n_cols):
            assert grid.index_to_cell(grid.cell_to_index(row, col)) == (row, col)


def test_indices_are_row_major_and_offset_by_start():
    grid = _grid(start=100)
    assert grid.cell_to_index(0, 0) == 100
    assert grid.cell_to_index(0, 17) == 117
    assert grid.cell_to_index(1, 0) == 118  # next row starts right after
    assert grid.end == 100 + 234


def test_out_of_range_cells_raise():
    grid = _grid()
    with pytest.raises(IndexError):
        grid.cell_to_index(13, 0)
    with pytest.raises(IndexError):
        grid.cell_to_index(0, 18)
    with pytest.raises(IndexError):
        grid.index_to_cell(grid.end)


# --------------------------------------------------------------------------------------
# Buffer dilation
# --------------------------------------------------------------------------------------


def test_buffer_one_grows_interior_cell_to_3x3():
    assert add_buffer([(5, 5)], 13, 18, 1) == {(r, c) for r in (4, 5, 6) for c in (4, 5, 6)}


def test_buffer_two_grows_interior_cell_to_5x5():
    grown = add_buffer([(5, 5)], 13, 18, 2)
    assert grown == {(r, c) for r in range(3, 8) for c in range(3, 8)}
    assert len(grown) == 25


def test_buffer_does_not_wrap_across_rows():
    """The original code's flat-index dilation could wrap column 0 into the previous row."""
    grown = add_buffer([(5, 0)], 13, 18, 1)
    assert grown == {(4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1)}
    assert all(col >= 0 for _, col in grown)
    assert (4, 17) not in grown


def test_buffer_clips_at_corners():
    assert add_buffer([(0, 0)], 13, 18, 1) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert add_buffer([(12, 17)], 13, 18, 1) == {(11, 16), (11, 17), (12, 16), (12, 17)}


def test_buffer_zero_is_identity():
    cells = {(3, 4), (3, 5)}
    assert add_buffer(cells, 13, 18, 0) == cells


# --------------------------------------------------------------------------------------
# Annotation -> cells
# --------------------------------------------------------------------------------------


def _rect_annotation(x: float, y: float, w: float, h: float) -> dict:
    return {
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0,
        "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
    }


def test_rectangle_maps_to_the_expected_cells():
    """A rect on exact cell boundaries must hit exactly those cells."""
    grid = _grid()
    cell_w = grid.orig_w / grid.n_cols  # 640/18
    cell_h = grid.orig_h / grid.n_rows  # 480/13
    # Cover columns 2-3 and rows 4-5 exactly, inset slightly to dodge edge rounding.
    ann = _rect_annotation(
        x=2 * cell_w + 1, y=4 * cell_h + 1, w=2 * cell_w - 2, h=2 * cell_h - 2
    )
    assert annotation_to_cells(ann, grid) == {(4, 2), (4, 3), (5, 2), (5, 3)}


def test_single_small_object_lands_in_one_cell():
    grid = _grid()
    ann = _rect_annotation(x=100, y=100, w=4, h=4)
    cells = annotation_to_cells(ann, grid)
    assert len(cells) == 1
    row, col = next(iter(cells))
    left, top, right, bottom = grid.cell_box(row, col)
    assert left <= 100 < right and top <= 100 < bottom


def test_threshold_shrinks_the_cell_set():
    grid = _grid()
    # A thin sliver straddling a cell boundary: any-overlap catches both cells,
    # a coverage requirement catches neither.
    cell_w = grid.orig_w / grid.n_cols
    ann = _rect_annotation(x=3 * cell_w - 2, y=200, w=4, h=4)
    assert len(annotation_to_cells(ann, grid, threshold=0.0)) == 2
    assert annotation_to_cells(ann, grid, threshold=0.5) == set()


def test_full_image_annotation_covers_every_cell():
    grid = _grid()
    ann = _rect_annotation(x=0, y=0, w=grid.orig_w, h=grid.orig_h)
    assert len(annotation_to_cells(ann, grid)) == grid.n_tokens


def test_bbox_fallback_when_segmentation_missing():
    grid = _grid()
    ann = {"bbox": [100, 100, 4, 4], "area": 16, "iscrowd": 0}
    assert len(annotation_to_cells(ann, grid)) == 1


def test_multi_polygon_segmentation_unions_the_parts():
    grid = _grid()
    ann = {
        "bbox": [0, 0, 640, 480],
        "area": 32,
        "iscrowd": 0,
        # Both squares sit well inside a single cell, so each contributes exactly one.
        "segmentation": [
            [10, 10, 14, 10, 14, 14, 10, 14],  # inside cell (0, 0)
            [620, 460, 624, 460, 624, 464, 620, 464],  # inside cell (12, 17)
        ],
    }
    assert annotation_to_cells(ann, grid) == {(0, 0), (12, 17)}


def test_object_token_indices_respects_span_offset():
    grid = _grid(start=50)
    ann = _rect_annotation(x=100, y=100, w=4, h=4)
    plain = object_token_indices(ann, grid, buffer=0)
    buffered = object_token_indices(ann, grid, buffer=1)
    assert len(plain) == 1
    assert len(buffered) == 9  # interior cell -> 3x3
    assert set(plain).issubset(buffered)
    assert all(grid.start <= i < grid.end for i in buffered)


def test_cells_to_indices_is_sorted():
    grid = _grid(start=3)
    indices = cells_to_indices({(5, 5), (0, 0), (2, 9)}, grid)
    assert indices == sorted(indices)


# --------------------------------------------------------------------------------------
# Baselines and image masking
# --------------------------------------------------------------------------------------


def test_random_indices_are_capped_at_the_grid_size():
    grid = _grid(start=10)
    # The paper's random_250 condition exceeds this model's 234-token budget.
    indices = random_indices(grid, 250)
    assert len(indices) == grid.n_tokens
    assert set(indices) == set(grid.indices)


def test_random_indices_stay_inside_the_visual_span():
    grid = _grid(start=10)
    indices = random_indices(grid, 40)
    assert len(indices) == 40
    assert len(set(indices)) == 40
    assert all(grid.start <= i < grid.end for i in indices)


def test_top_gradient_indices_are_confined_to_the_visual_span():
    torch = pytest.importorskip("torch")
    grid = _grid(start=10)
    from geometry import top_gradient_indices

    # Huge scores on text positions must not be selected.
    scores = torch.zeros(400)
    scores[:10] = 1e6
    scores[grid.start + 5] = 10.0
    scores[grid.start + 7] = 9.0
    picked = top_gradient_indices(scores, grid, count=2)
    assert picked == [grid.start + 5, grid.start + 7]


def test_mask_image_regions_only_touches_targeted_cells():
    grid = _grid()
    image = Image.new("RGB", (grid.orig_w, grid.orig_h), (10, 20, 30))
    masked = mask_image_regions(image, {(4, 2)}, grid, patch_type="gray")

    before, after = np.array(image), np.array(masked)
    left, top, right, bottom = grid.cell_box(4, 2)
    assert np.all(after[top:bottom, left:right] == 128)

    untouched = np.ones(before.shape[:2], dtype=bool)
    untouched[top:bottom, left:right] = False
    assert np.array_equal(before[untouched], after[untouched])


def test_mask_image_regions_covers_the_whole_image_when_given_all_cells():
    grid = _grid()
    image = Image.new("RGB", (grid.orig_w, grid.orig_h), (10, 20, 30))
    all_cells = {(r, c) for r in range(grid.n_rows) for c in range(grid.n_cols)}
    masked = np.array(mask_image_regions(image, all_cells, grid, patch_type="gray"))
    assert np.all(masked == 128)
