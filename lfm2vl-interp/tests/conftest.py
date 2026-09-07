"""Shared fixtures. Puts ``src/`` on the path and builds the tiny model once per session."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


@pytest.fixture(scope="session")
def processor():
    """The real LFM2.5-VL processor (tokenizer + image processor, no model weights)."""
    from transformers import AutoProcessor

    from hooked_lfm2vl import DEFAULT_MODEL_ID

    try:
        return AutoProcessor.from_pretrained(DEFAULT_MODEL_ID)
    except Exception as exc:  # pragma: no cover - offline environments
        pytest.skip(f"processor unavailable: {exc}")


@pytest.fixture(scope="session")
def tiny(processor):
    """A random-weight LFM2-VL on CPU, structurally faithful to the real model."""
    from tiny_model import build_tiny_hooked_model

    return build_tiny_hooked_model()


@pytest.fixture
def image():
    """A 640x480 image: the common COCO shape, 234 visual tokens on a 13x18 grid."""
    from PIL import Image

    return Image.new("RGB", (640, 480), (100, 110, 120))
