"""Synthetic scene generation for the perception-probing experiments.

Why synthetic: COCO cannot answer "where is counting computed" because there is no way to
change exactly one property of an image and watch what moves. Here we draw the shapes, so
we know every object's polygon exactly and can build **minimal pairs** -- two scenes
identical in every pixel except one shape's colour, where the correct answer changes from
3 to 2. That contrast is what activation patching needs.

Two design constraints drive everything in this module:

* **512x512 canvas.** It is the one size that passes through the processor's
  ``smart_resize`` untouched, giving a clean 16x16 = 256 visual-token grid at exactly
  32 px per token cell. A shape drawn at (64,64)-(128,128) therefore covers token cells
  rows 2-3, cols 2-3 with zero interpolation error.
* **Shapes are drawn from the same polygon that is emitted as the annotation.** Circles
  are 64-gons rather than ``draw.ellipse`` calls, so the COCO-style annotation this module
  emits describes the rendered pixels exactly. That means every existing helper in
  ``geometry.py`` -- ``annotation_to_cells``, ``object_token_indices`` -- works on
  synthetic scenes with no new code.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

CANVAS = 512
TOKEN_CELL_PX = 32  # 512 / 16, the visual-token grid pitch
BACKGROUND = (255, 255, 255)

# Rendering is supersampled then downscaled, so edges are smooth rather than jagged.
# The emitted polygon still describes the shape exactly; only the boundary pixels are
# blended, which is why the annotation/pixel agreement test uses IoU rather than equality.
SUPERSAMPLE = 4

# Well-separated in RGB, and every name is a single token space-prefixed.
COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (214, 45, 40),
    "blue": (38, 92, 208),
    "green": (26, 148, 72),
    "yellow": (242, 198, 28),
}

SHAPE_KINDS = ("circle", "square", "triangle", "star", "arrow")

# Arrow/triangle orientations, in degrees clockwise from pointing up.
DIRECTIONS: dict[str, int] = {"up": 0, "right": 90, "down": 180, "left": 270}


# --------------------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------------------


def _rotate(points: list[tuple[float, float]], degrees: float, cx: float, cy: float):
    if degrees % 360 == 0:
        return points
    theta = math.radians(degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    rotated = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        # Screen coordinates have y pointing down, so a clockwise visual rotation is the
        # standard counter-clockwise matrix with the y axis flipped.
        rotated.append((cx + dx * cos - dy * sin, cy + dx * sin + dy * cos))
    return rotated


@dataclass(frozen=True)
class Shape:
    """One rendered object. ``size`` is the side of its bounding box in pixels."""

    kind: str
    colour: str
    cx: int
    cy: int
    size: int
    direction: str = "up"  # only meaningful for arrow and triangle

    def __post_init__(self):
        if self.kind not in SHAPE_KINDS:
            raise ValueError(f"unknown shape kind {self.kind!r}")
        if self.colour not in COLOURS:
            raise ValueError(f"unknown colour {self.colour!r}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {self.direction!r}")

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        half = self.size / 2
        return (
            int(round(self.cx - half)), int(round(self.cy - half)),
            int(round(self.cx + half)), int(round(self.cy + half)),
        )

    @property
    def rgb(self) -> tuple[int, int, int]:
        return COLOURS[self.colour]

    def points(self) -> list[tuple[float, float]]:
        """Outline as a list of points. This is both what is drawn and what is annotated."""
        half = self.size / 2
        cx, cy = float(self.cx), float(self.cy)

        if self.kind == "circle":
            # A 64-gon: visually a circle, but exactly representable as a polygon so the
            # annotation matches the rendered pixels.
            pts = [
                (cx + half * math.cos(2 * math.pi * i / 64),
                 cy + half * math.sin(2 * math.pi * i / 64))
                for i in range(64)
            ]
        elif self.kind == "square":
            pts = [(cx - half, cy - half), (cx + half, cy - half),
                   (cx + half, cy + half), (cx - half, cy + half)]
        elif self.kind == "triangle":
            pts = [(cx, cy - half), (cx + half, cy + half), (cx - half, cy + half)]
        elif self.kind == "star":
            pts = []
            for i in range(10):
                radius = half if i % 2 == 0 else half * 0.42
                angle = math.pi / 2 * 3 + i * math.pi / 5
                pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        elif self.kind == "arrow":
            # Shaft plus head, pointing up before rotation.
            shaft, head = half * 0.34, half * 0.55
            pts = [
                (cx, cy - half),
                (cx + head, cy - half + head * 1.4),
                (cx + shaft, cy - half + head * 1.4),
                (cx + shaft, cy + half),
                (cx - shaft, cy + half),
                (cx - shaft, cy - half + head * 1.4),
                (cx - head, cy - half + head * 1.4),
            ]
        else:  # pragma: no cover - guarded in __post_init__
            raise ValueError(self.kind)

        if self.kind in ("arrow", "triangle"):
            pts = _rotate(pts, DIRECTIONS[self.direction], cx, cy)
        return pts

    def polygon(self) -> list[float]:
        """Flat ``[x0, y0, x1, y1, ...]`` polygon, the COCO segmentation format."""
        return [coordinate for point in self.points() for coordinate in point]

    def annotation(self, ann_id: int = 1, category_id: int = 1) -> dict:
        """COCO-style annotation, so every helper in ``geometry.py`` accepts this shape."""
        x0, y0, x1, y1 = self.bbox
        return {
            "id": ann_id,
            "category_id": category_id,
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "area": float(self.size * self.size),
            "iscrowd": 0,
            "segmentation": [self.polygon()],
        }

    def overlaps(self, other: "Shape", margin: int = 0) -> bool:
        ax0, ay0, ax1, ay1 = self.bbox
        bx0, by0, bx1, by1 = other.bbox
        return not (
            ax1 + margin <= bx0 or bx1 + margin <= ax0
            or ay1 + margin <= by0 or by1 + margin <= ay0
        )

    def describe(self) -> str:
        text = f"{self.colour} {self.kind}"
        if self.kind in ("arrow", "triangle") and self.direction != "up":
            text += f" pointing {self.direction}"
        return text


# --------------------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------------------


@dataclass
class Scene:
    """A set of shapes on a plain canvas."""

    shapes: list[Shape]
    width: int = CANVAS
    height: int = CANVAS

    def render(self) -> Image.Image:
        scale = SUPERSAMPLE
        canvas = Image.new("RGB", (self.width * scale, self.height * scale), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        for shape in self.shapes:
            scaled = [(x * scale, y * scale) for x, y in shape.points()]
            draw.polygon(scaled, fill=shape.rgb)
        return canvas.resize((self.width, self.height), Image.LANCZOS)

    def annotations(self) -> list[dict]:
        return [s.annotation(ann_id=i + 1) for i, s in enumerate(self.shapes)]

    def count(self, colour: str | None = None, kind: str | None = None) -> int:
        return len(list(self.select(colour=colour, kind=kind)))

    def select(self, colour: str | None = None, kind: str | None = None) -> list[Shape]:
        return [
            s for s in self.shapes
            if (colour is None or s.colour == colour) and (kind is None or s.kind == kind)
        ]

    def indices_of(self, colour: str | None = None, kind: str | None = None) -> list[int]:
        return [
            i for i, s in enumerate(self.shapes)
            if (colour is None or s.colour == colour) and (kind is None or s.kind == kind)
        ]


# --------------------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------------------

# The difficulty ladder. `crowded` allows bounding boxes to touch and overlap.
LEVELS: dict[str, dict] = {
    "L1": {"n_range": (2, 3), "colours": 1, "kinds": 1, "crowded": False},
    "L2": {"n_range": (4, 6), "colours": 1, "kinds": 1, "crowded": False},
    "L3": {"n_range": (4, 6), "colours": 2, "kinds": 1, "crowded": False},
    "L4": {"n_range": (4, 6), "colours": 2, "kinds": 2, "crowded": False},
    "L5": {"n_range": (5, 7), "colours": 2, "kinds": 2, "crowded": True},
}


def place(
    count: int,
    rng: random.Random,
    crowded: bool = False,
    size_range: tuple[int, int] = (56, 88),
    margin: int = 24,
) -> list[tuple[int, int, int]]:
    """Choose ``count`` non-overlapping ``(cx, cy, size)`` slots by rejection sampling.

    Positions come from a jittered grid rather than uniform sampling so that shapes spread
    out instead of clumping in one corner, which would confound "counting" with "finding".
    """
    slots_per_side = max(2, math.ceil(math.sqrt(count)) + (0 if crowded else 1))
    cell = CANVAS / slots_per_side
    max_size = min(size_range[1], int(cell) - (4 if crowded else margin))
    low = min(size_range[0], max_size)

    cells = [(r, c) for r in range(slots_per_side) for c in range(slots_per_side)]
    rng.shuffle(cells)

    placed: list[tuple[int, int, int]] = []
    for row, col in cells:
        if len(placed) == count:
            break
        size = rng.randint(low, max_size)
        slack = max(0, int(cell) - size - (0 if crowded else margin // 2))
        jitter_x = rng.randint(-slack // 2, slack // 2) if slack else 0
        jitter_y = rng.randint(-slack // 2, slack // 2) if slack else 0
        cx = int(col * cell + cell / 2 + jitter_x)
        cy = int(row * cell + cell / 2 + jitter_y)
        # Keep the whole shape on canvas.
        cx = min(max(cx, size // 2 + 2), CANVAS - size // 2 - 2)
        cy = min(max(cy, size // 2 + 2), CANVAS - size // 2 - 2)
        placed.append((cx, cy, size))

    if len(placed) < count:
        raise RuntimeError(f"could not place {count} shapes (got {len(placed)})")
    return placed


def build_scene(
    specs: Sequence[tuple[str, str, str]],
    rng: random.Random,
    crowded: bool = False,
) -> Scene:
    """Build a scene from ``(kind, colour, direction)`` specs, placing them automatically."""
    slots = place(len(specs), rng, crowded=crowded)
    shapes = [
        Shape(kind=kind, colour=colour, cx=cx, cy=cy, size=size, direction=direction)
        for (kind, colour, direction), (cx, cy, size) in zip(specs, slots)
    ]
    return Scene(shapes=shapes)


# --------------------------------------------------------------------------------------
# Samples: a clean scene, its counterfactual, and what changed
# --------------------------------------------------------------------------------------


@dataclass
class Sample:
    """A minimal pair plus everything the downstream stages need.

    ``clean`` and ``counterfactual`` differ in exactly one shape -- ``changed_index`` --
    and that single difference flips the correct answer from ``answer`` to ``answer_cf``.
    Both scenes render at the same size with the same prompt, so their token positions
    align, which is what makes activation patching valid.
    """

    sample_id: str
    task: str
    level: str
    clean: Scene
    counterfactual: Scene
    changed_index: int
    answer: str
    answer_cf: str
    query: dict = field(default_factory=dict)

    @property
    def changed_shape(self) -> Shape:
        return self.clean.shapes[self.changed_index]

    def scene(self, variant: str = "clean") -> Scene:
        return self.clean if variant == "clean" else self.counterfactual

    def to_dict(self) -> dict:
        """JSON-serialisable record. Scenes are stored as shape lists, not pixels."""
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "level": self.level,
            "changed_index": self.changed_index,
            "answer": self.answer,
            "answer_cf": self.answer_cf,
            "query": self.query,
            "clean": [vars(s) for s in self.clean.shapes],
            "counterfactual": [vars(s) for s in self.counterfactual.shapes],
        }

    @classmethod
    def from_dict(cls, record: dict) -> "Sample":
        return cls(
            sample_id=record["sample_id"],
            task=record["task"],
            level=record["level"],
            clean=Scene([Shape(**s) for s in record["clean"]]),
            counterfactual=Scene([Shape(**s) for s in record["counterfactual"]]),
            changed_index=record["changed_index"],
            answer=record["answer"],
            answer_cf=record["answer_cf"],
            query=record.get("query", {}),
        )


def recolour(scene: Scene, index: int, colour: str) -> Scene:
    """Copy of ``scene`` with one shape's colour changed and everything else identical."""
    shapes = list(scene.shapes)
    shapes[index] = replace(shapes[index], colour=colour)
    return Scene(shapes=shapes, width=scene.width, height=scene.height)


def reshape(scene: Scene, index: int, kind: str) -> Scene:
    """Copy of ``scene`` with one shape's kind changed, keeping position and size."""
    shapes = list(scene.shapes)
    shapes[index] = replace(shapes[index], kind=kind)
    return Scene(shapes=shapes, width=scene.width, height=scene.height)


def reorient(scene: Scene, index: int, direction: str) -> Scene:
    """Copy of ``scene`` with one shape rotated, keeping position, size and colour."""
    shapes = list(scene.shapes)
    shapes[index] = replace(shapes[index], direction=direction)
    return Scene(shapes=shapes, width=scene.width, height=scene.height)


def swap_positions(scene: Scene, first: int, second: int) -> Scene:
    """Copy of ``scene`` with two shapes' centres exchanged (sizes stay with the shape)."""
    shapes = list(scene.shapes)
    a, b = shapes[first], shapes[second]
    shapes[first] = replace(a, cx=b.cx, cy=b.cy)
    shapes[second] = replace(b, cx=a.cx, cy=a.cy)
    return Scene(shapes=shapes, width=scene.width, height=scene.height)


# --------------------------------------------------------------------------------------
# Token-cell helpers -- thin wrappers over geometry.py, kept here so callers do not need
# to know that a Shape is annotation-shaped.
# --------------------------------------------------------------------------------------


#: Minimum fraction of a token cell a shape must cover to count as occupying it.
#: Polygon fills are boundary-inclusive, so a shape spanning x in [64, 128] puts a single
#: pixel column into the next cell -- 32 of 1024 px, ~3%. Counting that would inflate every
#: object's token set by a whole ring of cells and blur exactly the localization this
#: experiment is trying to measure. 5% excludes the spill while keeping any real overlap.
#: (The COCO code keeps threshold=0.0, which is faithful to the paper's any-overlap rule.)
CELL_COVERAGE_THRESHOLD = 0.05


def shape_cells(
    shape: Shape, grid, threshold: float = CELL_COVERAGE_THRESHOLD
) -> set[tuple[int, int]]:
    """Token-grid cells a shape covers."""
    from geometry import annotation_to_cells

    return annotation_to_cells(shape.annotation(), grid, threshold=threshold)


def scene_cells(
    scene: Scene, grid, threshold: float = CELL_COVERAGE_THRESHOLD
) -> list[set[tuple[int, int]]]:
    """Per-shape token cells, in scene order."""
    return [shape_cells(s, grid, threshold=threshold) for s in scene.shapes]


def background_cells(
    scene: Scene, grid, threshold: float = CELL_COVERAGE_THRESHOLD
) -> set[tuple[int, int]]:
    """Cells no shape covers."""
    covered: set[tuple[int, int]] = set()
    for cells in scene_cells(scene, grid, threshold=threshold):
        covered |= cells
    everything = {(r, c) for r in range(grid.n_rows) for c in range(grid.n_cols)}
    return everything - covered


def cells_to_token_indices(cells: Iterable[tuple[int, int]], grid) -> list[int]:
    from geometry import cells_to_indices

    return cells_to_indices(cells, grid)
