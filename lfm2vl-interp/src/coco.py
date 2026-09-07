"""COCO loading and the paper's image filter.

Ports ``llava-interp-main/src/ImageDatasets.py`` plus the ``filter_fn`` that is
copy-pasted into each of the paper's four ablation scripts. Two changes:

* ``pycocotools`` is optional -- a small pure-Python reader covers
  ``instances_*.json``, which is all these experiments need. That keeps local
  development working on Windows, where pycocotools has no reliable wheel.
* The area window is a parameter rather than a literal, since the paper's
  ``(1000, 2000)`` was chosen for LLaVA's 336x336 center crop while LFM2.5-VL sees
  roughly 416x576. The default stays at the paper's values for comparability.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image

# Paper Section 3: "an object whose size is between 1,000-2,000 square pixels".
PAPER_AREA_WINDOW = (1000, 2000)
# Paper Section 4.1 uses a separate, larger window for the logit-lens measurement:
# "objects of sizes between 20,000 and 30,000 square pixels".
PAPER_LOGIT_LENS_AREA_WINDOW = (20000, 30000)
PAPER_MAX_ANNOTATIONS = 3


@dataclass
class CocoImage:
    """One image plus the single annotation the filter selected for it."""

    img_id: int
    file_name: str
    width: int
    height: int
    annotation: dict
    class_name: str

    def open(self, image_dir: str | Path) -> Image.Image:
        return Image.open(Path(image_dir) / self.file_name).convert("RGB")


class CocoInstances:
    """Minimal reader for a COCO ``instances_*.json`` file."""

    def __init__(self, ann_file: str | Path):
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.categories = {cat["id"]: cat for cat in data["categories"]}
        self.anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def class_name(self, category_id: int) -> str:
        return self.categories[category_id]["name"]

    def image_ids(self) -> list[int]:
        return sorted(self.images)


def select_annotation(
    anns: list[dict],
    area_window: tuple[float, float] = PAPER_AREA_WINDOW,
    max_annotations: int = PAPER_MAX_ANNOTATIONS,
) -> dict | None:
    """The paper's filter, returning the chosen annotation or None.

    Keeps images that have at most ``max_annotations`` objects and contain some category
    with exactly one instance whose area falls inside ``area_window``. Crowd regions are
    rejected outright -- they are RLE-encoded, and the paper's "only one instance of that
    object" condition excludes them in spirit anyway.
    """
    if not anns or len(anns) > max_annotations:
        return None

    by_category: dict[int, list[dict]] = defaultdict(list)
    for ann in anns:
        if ann.get("iscrowd", 0):
            return None
        by_category[ann["category_id"]].append(ann)

    low, high = area_window
    for group in by_category.values():
        if len(group) == 1 and low < group[0]["area"] < high:
            return group[0]
    return None


def iter_filtered_images(
    coco: CocoInstances,
    area_window: tuple[float, float] = PAPER_AREA_WINDOW,
    max_annotations: int = PAPER_MAX_ANNOTATIONS,
    keep: Callable[[int], bool] | None = None,
) -> Iterator[CocoImage]:
    """Yield the images that pass the filter, in stable image-id order.

    Args:
        keep: optional extra predicate on image id, used to restrict a run to the
            curated VQA subset.
    """
    for img_id in coco.image_ids():
        if keep is not None and not keep(img_id):
            continue
        ann = select_annotation(
            coco.anns_by_image.get(img_id, []),
            area_window=area_window,
            max_annotations=max_annotations,
        )
        if ann is None:
            continue
        info = coco.images[img_id]
        yield CocoImage(
            img_id=img_id,
            file_name=info["file_name"],
            width=info["width"],
            height=info["height"],
            annotation=ann,
            class_name=coco.class_name(ann["category_id"]),
        )


def load_clean_questions(path: str | Path) -> dict[int, str]:
    """Load the paper's 100 curated VQA questions, dropping the empty entries.

    ``data/clean_questions.json`` has one key per filtered image (4,318 of them) but
    only 100 carry a question; the rest are empty lists.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(img_id): questions[0] for img_id, questions in raw.items() if questions}
