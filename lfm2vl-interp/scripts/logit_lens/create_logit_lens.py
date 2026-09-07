"""Generate interactive logit-lens HTMLs (paper Section 4.1, Figure 3).

One self-contained HTML per image: hover any image region to see what the residual
stream at that visual-token position decodes to, layer by layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import prompts as P  # noqa: E402
from experiment import add_common_args, build_model, load_config, resolve  # noqa: E402
from logit_lens import create_interactive_logit_lens  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--save-folder", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", default=P.DESCRIBE_QUESTION)
    args = parser.parse_args()

    config = load_config(args.config)
    save_folder = resolve(args.save_folder or config["paths"]["logit_lens_dir"])
    save_folder.mkdir(parents=True, exist_ok=True)

    image_folder = Path(args.image_folder)
    paths = sorted(p for p in image_folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images found in {image_folder}")

    model = build_model(args, config)
    prompt = P.build_prompt(model.processor, args.question)

    for path in tqdm(paths, desc="logit lens"):
        image = Image.open(path).convert("RGB")
        inputs = model.prepare(image, prompt)
        grid = model.single_grid(inputs, (image.height, image.width))
        outputs = model.forward(image, prompt, output_hidden_states=True)

        create_interactive_logit_lens(
            outputs.hidden_states,
            model.final_norm,
            model.lm_head,
            model.processor.tokenizer,
            inputs["input_ids"][0].tolist(),
            image,
            grid,
            prompt,
            save_folder / f"{path.stem}_logit_lens.html",
            top_k=args.top_k,
            misc_text=f"{path.name} ({image.width}x{image.height})",
        )

    print(f"\nWrote {len(paths)} files to {save_folder}")
    print("Run scripts/logit_lens/generate_overview.py to build an index.")


if __name__ == "__main__":
    main()
