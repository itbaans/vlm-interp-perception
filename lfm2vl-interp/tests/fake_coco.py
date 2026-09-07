"""Build a miniature COCO-shaped dataset so the scripts can be smoke-run offline.

Real COCO is 19 GB. Every script only needs ``instances_*.json`` plus the image files
it names, so this writes a handful of generated images with polygon annotations whose
areas land inside the paper's filter window.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

CLASSES = [(1, "dog"), (2, "cat"), (3, "bench")]


def _annotation(ann_id: int, image_id: int, category_id: int, x: int, y: int, size: int) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, size, size],
        "area": size * size,
        "iscrowd": 0,
        "segmentation": [[x, y, x + size, y, x + size, y + size, x, y + size]],
    }


def build_fake_coco(root: Path, split: str = "val2017", n_images: int = 4) -> dict[str, Path]:
    """Write images + annotations under ``root``. Returns the paths created."""
    image_dir = root / split
    ann_dir = root / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    for index in range(n_images):
        image_id = 1000 + index
        width, height = (640, 480) if index % 2 == 0 else (480, 640)
        file_name = f"{image_id:012d}.jpg"

        # Object area 1600 px^2, inside the paper's (1000, 2000) window.
        size = 40
        x, y = 100 + 20 * index, 90 + 20 * index
        category_id, _ = CLASSES[index % len(CLASSES)]

        canvas = Image.new("RGB", (width, height), (140, 150, 160))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([x, y, x + size, y + size], fill=(220, 60, 40))
        canvas.save(image_dir / file_name, quality=90)

        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        annotations.append(_annotation(index + 1, image_id, category_id, x, y, size))

    ann_file = ann_dir / f"instances_{split}.json"
    ann_file.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": cid, "name": name} for cid, name in CLASSES],
            }
        ),
        encoding="utf-8",
    )

    questions_file = root / "clean_questions.json"
    questions_file.write_text(
        json.dumps({str(img["id"]): ["What is in the image?"] for img in images}),
        encoding="utf-8",
    )

    return {"root": root, "image_dir": image_dir, "ann_file": ann_file, "questions": questions_file}


def build_fake_config(root: Path, split: str = "val2017") -> Path:
    """A config pointing at the fake dataset, with the expensive knobs turned down."""
    config = {
        "model_id": "LiquidAI/LFM2.5-VL-3B",
        "device": "cpu",
        "dtype": "float32",
        "attn_implementation": "eager",
        "do_image_splitting": False,
        "min_image_tokens": 64,
        "max_image_tokens": 256,
        "attention_layers": [2, 5],
        "num_layers": 6,
        "data": {
            "coco_root": str(root),
            "split": split,
            "clean_questions": str(root / "clean_questions.json"),
        },
        "filters": {
            "area_window": [1000, 2000],
            "max_annotations": 3,
            "logit_lens_area_window": [1000, 2000],
            "logit_lens_num_images": 4,
        },
        "ablation": {
            "mean_vector": str(root / "mean_visual_token.pt"),
            "baseline_counts": [5, 10],
            "buffers": [0, 1],
            "integrated_gradient_steps": 2,
            "max_images": 4,
        },
        "attention_knockout": {
            "per_layer_sweep": True,
            "sliding_window": 2,
            "distance_control": True,
            "distance_control_padding": 60,
        },
        "paths": {
            "results_dir": str(root / "results"),
            "logit_lens_dir": str(root / "results" / "logit_lens"),
        },
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
