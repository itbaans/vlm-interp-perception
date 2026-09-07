"""Shared plumbing for the experiment scripts: config, CLI, model building, results.

The paper's scripts each re-declare the COCO filter, re-parse the same arguments and
re-implement resumable JSON saving. That is centralised here so the experiment scripts
contain only their experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "lfm2_5_vl_3b.yaml"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root unless already absolute."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="Path to the YAML config")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda:0")
    parser.add_argument("--model-id", default=None, help="Override the HF model id")
    parser.add_argument("--coco-root", default=None, help="Override the COCO root directory")
    parser.add_argument("--split", default=None, help="COCO split, e.g. train2017 / val2017")
    parser.add_argument("--results", default=None, help="Results JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N images")
    parser.add_argument(
        "--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"]
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a random-weight miniature model. Exercises every code path on CPU "
        "without downloading the 3B checkpoint; results are meaningless.",
    )
    return parser


def build_model(args, config: dict):
    """Construct the hooked model described by config + CLI overrides."""
    from hooked_lfm2vl import HookedLFM2VL

    attn = args.attn_implementation or config.get("attn_implementation", "eager")

    if args.tiny:
        from tiny_model import build_tiny_hooked_model

        model = build_tiny_hooked_model(
            model_id=args.model_id or config["model_id"], attn_implementation=attn
        )
        model.do_image_splitting = config.get("do_image_splitting", False)
        return model

    return HookedLFM2VL(
        model_id=args.model_id or config["model_id"],
        device=args.device or config.get("device", "cuda:0"),
        dtype=config.get("dtype", "bfloat16"),
        attn_implementation=attn,
        do_image_splitting=config.get("do_image_splitting", False),
    )


def coco_paths(args, config: dict) -> tuple[Path, Path, str]:
    """Return ``(image_dir, annotation_file, split)``."""
    root = resolve(args.coco_root or config["data"]["coco_root"])
    split = args.split or config["data"]["split"]
    return root / split, root / "annotations" / f"instances_{split}.json", split


class ResultsStore:
    """Resumable per-image results, written atomically after each image.

    The paper's scripts rewrite the whole JSON in place on every iteration, so a crash
    mid-write truncates the file. Here the write goes to a temp file and is renamed.
    """

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"meta": meta or {}, "images": {}}
        if meta:
            self.data.setdefault("meta", {}).update(meta)
        self.data.setdefault("images", {})

    def __contains__(self, img_id) -> bool:
        return str(img_id) in self.data["images"]

    def __len__(self) -> int:
        return len(self.data["images"])

    def __getitem__(self, img_id) -> Any:
        return self.data["images"][str(img_id)]

    def set(self, img_id, value: dict) -> None:
        self.data["images"][str(img_id)] = value

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)

    @property
    def images(self) -> dict:
        return self.data["images"]

    @property
    def meta(self) -> dict:
        return self.data["meta"]
