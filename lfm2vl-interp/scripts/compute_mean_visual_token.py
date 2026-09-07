"""Compute the mean visual token used as the ablation replacement (paper Section 3).

The paper replaces ablated visual tokens with
``e_bar = (1/N) * sum e_i`` over all visual tokens from 50,000 ImageNet validation
images, arguing that mean ablation "preserve[s] the norm of the image token and keep[s]
them in-distribution, as their norms are typically much higher than the norm of text
tokens".

Differences here:

* Images come from any folder -- COCO train2017 by default, so no extra dataset
  download is needed on the cloud box. The vector serves the same purpose.
* The paper's pipeline is three scripts (``save_post_adapter_acts`` writes multi-GB
  ``.pt`` shards, ``estimate_acts_size`` measures them, ``calculate_mean_vector``
  averages them). This does it in one streaming pass with no intermediate files, which
  matters more here because LFM2.5-VL's token count varies per image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiment import add_common_args, build_model, load_config, resolve  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_paths(folder: Path, limit: int | None) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit else paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Folder of images to average over. Defaults to the configured COCO split.",
    )
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--output", default=None, help="Where to write the .pt vector")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.image_dir:
        image_dir = Path(args.image_dir)
    else:
        root = resolve(args.coco_root or config["data"]["coco_root"])
        image_dir = root / (args.split or config["data"]["split"])

    output = resolve(args.output or config["ablation"]["mean_vector"])
    paths = iter_image_paths(image_dir, args.num_images)
    if not paths:
        raise SystemExit(f"no images found in {image_dir}")
    print(f"Averaging visual tokens over {len(paths)} images from {image_dir}")

    model = build_model(args, config)

    # Accumulate in float64 on CPU: bf16 would lose precision over millions of tokens.
    total = None
    count = 0
    skipped = 0

    for path in tqdm(paths, desc="visual tokens"):
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, ValueError):
            skipped += 1
            continue

        features = model.visual_token_features(image).to(torch.float64).cpu()
        if total is None:
            total = features.sum(dim=0)
        else:
            total += features.sum(dim=0)
        count += features.shape[0]

    if not count:
        raise SystemExit("no visual tokens accumulated")

    mean_vector = (total / count).to(torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_vector, output)

    print(f"\n{count} visual tokens over {len(paths) - skipped} images ({skipped} skipped)")
    print(f"mean vector: dim={mean_vector.numel()} norm={mean_vector.norm():.4f}")
    print(f"written to {output}")


if __name__ == "__main__":
    main()
