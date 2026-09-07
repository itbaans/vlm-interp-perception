"""Geometry: COCO annotations <-> LFM2.5-VL visual-token indices.

This replaces the LLaVA-specific half of ``llava-interp-main/src/utils.py``.

LLaVA center-crops every image to a square, resizes to 336x336 and emits a fixed
24x24 = 576 grid of visual tokens, so the original code could hard-code ``width = 24``
and iterate over ``range(576)``. LFM2.5-VL does none of that:

* ``smart_resize`` scales the **whole** image (no crop), preserving aspect ratio,
  to dimensions divisible by ``encoder_patch_size * downsample_factor`` (= 32),
  such that the resulting token count lands in ``[min_image_tokens, max_image_tokens]``.
* The SigLIP2 NaFlex tower emits ``(H/16) x (W/16)`` patches.
* The projector applies a 2x2 pixel-unshuffle, so the LM sees a ``(H/32) x (W/32)``
  grid of visual tokens, flattened row-major.

Because the resize maps the *entire* original image onto the *entire* resized image,
token-grid cell ``(i, j)`` corresponds exactly to the original-image rectangle::

    x in [j * orig_w / n_cols, (j+1) * orig_w / n_cols)
    y in [i * orig_h / n_rows, (i+1) * orig_h / n_rows)

which is what lets us map COCO annotations to token indices without ever touching
the resized pixels.

The functions here mirror ``transformers.models.lfm2_vl.image_processing_lfm2_vl_fast``
so that geometry can be computed offline, with no model and no processor. At runtime
the authoritative grid comes from the processor's ``spatial_shapes`` output; see
``HookedLFM2VL.image_token_spans``. ``test_geometry.py`` checks the two against
each other.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

# LFM2.5-VL-3B defaults, from the checkpoint's config.json / processor_config.json.
ENCODER_PATCH_SIZE = 16
DOWNSAMPLE_FACTOR = 2
MIN_IMAGE_TOKENS = 64
MAX_IMAGE_TOKENS = 256
MAX_PIXELS_TOLERANCE = 2.0
TILE_SIZE = 512


# --------------------------------------------------------------------------------------
# Resize / grid arithmetic (mirrors Lfm2VlImageProcessorFast)
# --------------------------------------------------------------------------------------


def round_by_factor(number: float, factor: int) -> int:
    """Closest integer to ``number`` divisible by ``factor``."""
    return round(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    downsample_factor: int = DOWNSAMPLE_FACTOR,
    min_image_tokens: int = MIN_IMAGE_TOKENS,
    max_image_tokens: int = MAX_IMAGE_TOKENS,
    encoder_patch_size: int = ENCODER_PATCH_SIZE,
) -> tuple[int, int]:
    """Return ``(new_height, new_width)`` for the single-tile path.

    Mirrors ``Lfm2VlImageProcessorFast.smart_resize``, except that method returns
    ``(w_bar, h_bar)`` while this one returns ``(height, width)`` to match the order
    the rest of this codebase uses.
    """
    total_factor = encoder_patch_size * downsample_factor
    min_pixels = min_image_tokens * encoder_patch_size**2 * downsample_factor**2
    max_pixels = max_image_tokens * encoder_patch_size**2 * downsample_factor**2

    h_bar = max(total_factor, round_by_factor(height, total_factor))
    w_bar = max(total_factor, round_by_factor(width, total_factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(total_factor, math.floor(height / beta / total_factor) * total_factor)
        w_bar = max(total_factor, math.floor(width / beta / total_factor) * total_factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / total_factor) * total_factor
        w_bar = math.ceil(width * beta / total_factor) * total_factor

    return h_bar, w_bar


def is_image_too_large(
    height: int,
    width: int,
    max_image_tokens: int = MAX_IMAGE_TOKENS,
    encoder_patch_size: int = ENCODER_PATCH_SIZE,
    downsample_factor: int = DOWNSAMPLE_FACTOR,
    max_pixels_tolerance: float = MAX_PIXELS_TOLERANCE,
) -> bool:
    """Whether the processor would split this image into tiles rather than resize it.

    Mirrors ``Lfm2VlImageProcessorFast._is_image_too_large``. Typical COCO images
    (640x480) come out False, so they take the clean single-grid path.
    """
    total_factor = encoder_patch_size * downsample_factor
    h_bar = max(encoder_patch_size, round_by_factor(height, total_factor))
    w_bar = max(encoder_patch_size, round_by_factor(width, total_factor))
    budget = max_image_tokens * encoder_patch_size**2 * downsample_factor**2
    return h_bar * w_bar > budget * max_pixels_tolerance


@dataclass(frozen=True)
class TokenGrid:
    """The visual-token grid for one image (single-tile path).

    Attributes:
        n_rows, n_cols: shape of the token grid; ``n_rows * n_cols`` tokens total.
        start: absolute index of the first visual token in the LM sequence.
        orig_h, orig_w: original image size, the frame COCO annotations live in.
        resized_h, resized_w: what the vision tower actually sees.
    """

    n_rows: int
    n_cols: int
    start: int
    orig_h: int
    orig_w: int
    resized_h: int
    resized_w: int

    @property
    def n_tokens(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def end(self) -> int:
        """Exclusive end index of the visual-token run."""
        return self.start + self.n_tokens

    @property
    def indices(self) -> list[int]:
        return list(range(self.start, self.end))

    def cell_to_index(self, row: int, col: int) -> int:
        if not (0 <= row < self.n_rows and 0 <= col < self.n_cols):
            raise IndexError(f"cell ({row}, {col}) outside {self.n_rows}x{self.n_cols} grid")
        return self.start + row * self.n_cols + col

    def index_to_cell(self, index: int) -> tuple[int, int]:
        if not (self.start <= index < self.end):
            raise IndexError(f"index {index} outside visual span [{self.start}, {self.end})")
        return divmod(index - self.start, self.n_cols)

    def cell_box(self, row: int, col: int) -> tuple[int, int, int, int]:
        """Original-image pixel box ``(left, top, right, bottom)`` covered by a cell."""
        left = int(round(col * self.orig_w / self.n_cols))
        right = int(round((col + 1) * self.orig_w / self.n_cols))
        top = int(round(row * self.orig_h / self.n_rows))
        bottom = int(round((row + 1) * self.orig_h / self.n_rows))
        return left, top, right, bottom


def token_grid_for_image(
    orig_h: int,
    orig_w: int,
    start: int = 0,
    downsample_factor: int = DOWNSAMPLE_FACTOR,
    min_image_tokens: int = MIN_IMAGE_TOKENS,
    max_image_tokens: int = MAX_IMAGE_TOKENS,
    encoder_patch_size: int = ENCODER_PATCH_SIZE,
) -> TokenGrid:
    """Compute the token grid offline, without running the processor.

    Only valid for the single-tile path; callers that allow tiling should build the
    grid from the processor's ``spatial_shapes`` instead (see ``grid_from_spatial_shape``).
    """
    resized_h, resized_w = smart_resize(
        orig_h,
        orig_w,
        downsample_factor=downsample_factor,
        min_image_tokens=min_image_tokens,
        max_image_tokens=max_image_tokens,
        encoder_patch_size=encoder_patch_size,
    )
    n_patches_h = resized_h // encoder_patch_size
    n_patches_w = resized_w // encoder_patch_size
    return TokenGrid(
        n_rows=math.ceil(n_patches_h / downsample_factor),
        n_cols=math.ceil(n_patches_w / downsample_factor),
        start=start,
        orig_h=orig_h,
        orig_w=orig_w,
        resized_h=resized_h,
        resized_w=resized_w,
    )


def grid_from_spatial_shape(
    spatial_shape: Sequence[int],
    orig_h: int,
    orig_w: int,
    start: int,
    downsample_factor: int = DOWNSAMPLE_FACTOR,
    encoder_patch_size: int = ENCODER_PATCH_SIZE,
) -> TokenGrid:
    """Build a ``TokenGrid`` from the processor's authoritative ``spatial_shapes`` row.

    ``spatial_shape`` is ``[n_patches_h, n_patches_w]`` for one tile. The token grid is
    that divided by ``downsample_factor``, matching the 2x2 pixel-unshuffle in
    ``Lfm2VlMultiModalProjector``.
    """
    n_patches_h, n_patches_w = int(spatial_shape[0]), int(spatial_shape[1])
    return TokenGrid(
        n_rows=math.ceil(n_patches_h / downsample_factor),
        n_cols=math.ceil(n_patches_w / downsample_factor),
        start=start,
        orig_h=orig_h,
        orig_w=orig_w,
        resized_h=n_patches_h * encoder_patch_size,
        resized_w=n_patches_w * encoder_patch_size,
    )


# --------------------------------------------------------------------------------------
# COCO annotation -> token cells
# --------------------------------------------------------------------------------------


def rasterize_annotation(ann: dict, orig_h: int, orig_w: int) -> np.ndarray:
    """Rasterize one COCO annotation to a boolean mask at original resolution.

    Polygon segmentations are drawn with PIL so that the common case needs no
    ``pycocotools`` (which has no reliable Windows wheel and is only needed for the
    RLE-encoded ``iscrowd=1`` annotations, which the paper's filter excludes anyway).
    """
    segmentation = ann.get("segmentation")

    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], (list, tuple)):
        canvas = Image.new("1", (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon in segmentation:
            if len(polygon) < 6:  # need at least 3 points
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
            draw.polygon(points, fill=1)
        return np.array(canvas, dtype=bool)

    if segmentation is not None:
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "This annotation uses RLE segmentation, which needs pycocotools. "
                "Install it, or filter out iscrowd=1 annotations."
            ) from exc
        rle = segmentation
        if isinstance(rle, dict) and isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, orig_h, orig_w)
        return mask_utils.decode(rle).astype(bool)

    # No segmentation at all: fall back to the bounding box.
    mask = np.zeros((orig_h, orig_w), dtype=bool)
    x, y, w, h = ann["bbox"]
    x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    x1, y1 = min(orig_w, int(math.ceil(x + w))), min(orig_h, int(math.ceil(y + h)))
    mask[y0:y1, x0:x1] = True
    return mask


def _cell_coverage(mask: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Fraction of each token cell covered by ``mask``, as an ``(n_rows, n_cols)`` array."""
    orig_h, orig_w = mask.shape
    # Integral image so each cell is an O(1) box sum.
    integral = np.zeros((orig_h + 1, orig_w + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)

    row_edges = np.round(np.linspace(0, orig_h, n_rows + 1)).astype(int)
    col_edges = np.round(np.linspace(0, orig_w, n_cols + 1)).astype(int)

    top, bottom = row_edges[:-1, None], row_edges[1:, None]
    left, right = col_edges[None, :-1], col_edges[None, 1:]

    counts = (
        integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
    )
    areas = np.maximum((bottom - top) * (right - left), 1)
    return counts / areas


def annotation_to_cells(
    ann: dict,
    grid: TokenGrid,
    threshold: float = 0.0,
) -> set[tuple[int, int]]:
    """Token-grid cells overlapping an annotation.

    Args:
        threshold: minimum fraction of the cell that must be covered. The paper's
            ``find_overlapping_patches`` accepted *any* overlap, which is
            ``threshold=0.0`` here (the default, for comparability). Raise it to
            require more coverage.
    """
    mask = rasterize_annotation(ann, grid.orig_h, grid.orig_w)
    coverage = _cell_coverage(mask, grid.n_rows, grid.n_cols)
    hit = coverage > threshold if threshold > 0 else coverage > 0
    return {(int(r), int(c)) for r, c in zip(*np.nonzero(hit))}


def add_buffer(
    cells: Iterable[tuple[int, int]],
    n_rows: int,
    n_cols: int,
    amount: int = 1,
) -> set[tuple[int, int]]:
    """Dilate a cell set by ``amount`` steps of 8-connectivity.

    Generalises the original ``_add_buffer``, which hard-coded ``width = height = 24``.
    Neighbours are clipped at the grid edges, so a cell in column 0 never wraps into
    the previous row.
    """
    current = {(int(r), int(c)) for r, c in cells}
    for _ in range(max(0, amount)):
        grown = set(current)
        for row, col in current:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols:
                        grown.add((nr, nc))
        current = grown
    return current


def cells_to_indices(cells: Iterable[tuple[int, int]], grid: TokenGrid) -> list[int]:
    """Absolute LM sequence indices for a set of grid cells, sorted."""
    return sorted(grid.cell_to_index(row, col) for row, col in cells)


def object_token_indices(
    ann: dict,
    grid: TokenGrid,
    buffer: int = 0,
    threshold: float = 0.0,
) -> list[int]:
    """Sequence indices of the visual tokens covering an object, optionally dilated.

    This is the replacement for ``utils.get_object_patch_indices``, which combined a
    center-crop correction, a 576-iteration per-patch IoU loop and a 24-wide dilation.
    """
    cells = annotation_to_cells(ann, grid, threshold=threshold)
    if buffer > 0:
        cells = add_buffer(cells, grid.n_rows, grid.n_cols, buffer)
    return cells_to_indices(cells, grid)


# --------------------------------------------------------------------------------------
# Baselines and image-space ablation
# --------------------------------------------------------------------------------------


def random_indices(grid: TokenGrid, count: int, rng: random.Random | None = None) -> list[int]:
    """``count`` random visual-token indices, capped at the grid size.

    The cap matters here in a way it did not for LLaVA: a typical COCO image gives
    LFM2.5-VL ~234 visual tokens, so the paper's ``random_250`` condition would
    otherwise ask for more tokens than exist.
    """
    rng = rng or random
    count = min(count, grid.n_tokens)
    return sorted(rng.sample(grid.indices, count))


def register_indices(visual_embeds, grid: TokenGrid, num_std: float = 2.0) -> list[int]:
    """Visual tokens whose norm exceeds mean + ``num_std`` * std ("register tokens").

    Ports ``utils.get_register_indices`` unchanged apart from taking a ``TokenGrid``
    instead of a list of start/end tuples.
    """
    import torch

    norms = torch.norm(visual_embeds.detach().float().squeeze(), dim=-1)
    cutoff = norms.mean() + num_std * norms.std()
    offsets = torch.nonzero(norms > cutoff).flatten().tolist()
    return [grid.start + offset for offset in offsets]


def top_gradient_indices(
    attributions,
    grid: TokenGrid,
    count: int,
) -> list[int]:
    """Top-``count`` visual tokens by attribution score, restricted to the visual span.

    ``attributions`` is a 1-D tensor over the whole sequence. The original code sorted
    the entire sequence and sliced the top-n, which could return text positions; here
    the selection is confined to the visual tokens so the token counts stay comparable
    to the object-ablation conditions.
    """
    import torch

    scores = attributions.detach().float().flatten()[grid.start : grid.end]
    count = min(count, grid.n_tokens)
    order = torch.argsort(scores, descending=True)[:count]
    return sorted(grid.start + int(i) for i in order)


def mask_image_regions(
    image: Image.Image,
    cells: Iterable[tuple[int, int]],
    grid: TokenGrid,
    patch_type: str = "noise",
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Overwrite the original-image regions behind given token cells.

    Used for the paper's hallucination control: an image is kept only if the model
    names the object in the original but not once the object's pixels are destroyed.
    Unlike ``utils.replace_image_regions_with_patches`` this works in original-image
    coordinates on a non-square image, since LFM2.5-VL never center-crops.
    """
    rng = rng or np.random.default_rng()
    array = np.array(image.convert("RGB"))

    for row, col in cells:
        left, top, right, bottom = grid.cell_box(row, col)
        if right <= left or bottom <= top:
            continue
        shape = (bottom - top, right - left, 3)
        if patch_type == "noise":
            patch = rng.integers(0, 256, size=shape, dtype=np.uint8)
        elif patch_type == "gray":
            patch = np.full(shape, 128, dtype=np.uint8)
        else:
            raise ValueError(f"unsupported patch_type {patch_type!r}; use 'noise' or 'gray'")
        array[top:bottom, left:right, :] = patch

    return Image.fromarray(array, mode="RGB")
