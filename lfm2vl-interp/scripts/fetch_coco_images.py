"""Download only the COCO images these experiments actually use.

The full ``train2017`` zip is 19 GB, and unpacking it needs 19 GB more — which does not
fit on a typical GPU box's system disk. But these experiments only ever touch a filtered
subset: images with a single small annotated object, the 100 curated VQA images, and a
few hundred larger-object images for the logit lens. COCO serves images individually at
``http://images.cocodataset.org/<split>/<file_name>``, so fetching just that subset costs
a few hundred MB instead of 38 GB, and is faster.

The union that gets downloaded:

* ablation candidates — the paper's COCO filter on the main split, capped by ``--budget``
* the curated VQA image ids from ``clean_questions.json``
* logit-lens candidates — the larger-object filter on the lens split
* whatever the mean-visual-token pass needs, drawn from the ablation candidates

Existing files are skipped, so re-running only fetches what is missing.
"""

from __future__ import annotations

import argparse
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coco import CocoInstances, iter_filtered_images, load_clean_questions  # noqa: E402
from experiment import load_config, resolve  # noqa: E402

BASE_URL = "http://images.cocodataset.org"
_print_lock = threading.Lock()


def fetch_one(split: str, file_name: str, target_dir: Path, retries: int = 3) -> int:
    """Download one image unless it is already there. Returns bytes written."""
    target = target_dir / file_name
    if target.exists() and target.stat().st_size > 0:
        return 0

    url = f"{BASE_URL}/{split}/{file_name}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(target)  # atomic, so an interrupted run leaves no half file
            return len(data)
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt == retries - 1:
                with _print_lock:
                    print(f"  failed: {file_name}", flush=True)
                return 0
    return 0


def download_many(split: str, file_names: list[str], target_dir: Path, workers: int = 16) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    missing = [f for f in file_names if not (target_dir / f).exists()]
    print(f"{split}: {len(file_names)} needed, {len(file_names) - len(missing)} present, "
          f"{len(missing)} to download")
    if not missing:
        return

    total = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, split, name, target_dir): name for name in missing}
        for future in as_completed(futures):
            total += future.result()
            done += 1
            if done % 100 == 0 or done == len(missing):
                print(f"  {done}/{len(missing)}  ({total / 1e6:.0f} MB)", flush=True)
    print(f"{split}: {total / 1e6:.0f} MB downloaded")


def collect(config: dict, args) -> dict[str, list[str]]:
    """Work out the file names needed per split."""
    coco_root = resolve(args.coco_root or config["data"]["coco_root"])
    main_split = args.split or config["data"]["split"]
    lens_split = args.lens_split or main_split
    needed: dict[str, set[str]] = {main_split: set(), lens_split: set()}

    ann = coco_root / "annotations" / f"instances_{main_split}.json"
    print(f"reading {ann}")
    coco = CocoInstances(ann)

    candidates = list(
        iter_filtered_images(
            coco,
            area_window=tuple(config["filters"]["area_window"]),
            max_annotations=config["filters"]["max_annotations"],
        )
    )
    print(f"{main_split}: {len(candidates)} images pass the ablation filter")
    for item in candidates[: args.budget]:
        needed[main_split].add(item.file_name)

    questions = load_clean_questions(resolve(config["data"]["clean_questions"]))
    curated = [i for i in questions if i in coco.images]
    print(f"{main_split}: {len(curated)} curated VQA images")
    for img_id in curated:
        needed[main_split].add(coco.images[img_id]["file_name"])

    lens_coco = coco
    if lens_split != main_split:
        lens_ann = coco_root / "annotations" / f"instances_{lens_split}.json"
        print(f"reading {lens_ann}")
        lens_coco = CocoInstances(lens_ann)

    lens_items = list(
        iter_filtered_images(
            lens_coco,
            area_window=tuple(config["filters"]["logit_lens_area_window"]),
            max_annotations=config["filters"]["max_annotations"],
        )
    )
    lens_budget = max(config["filters"]["logit_lens_num_images"], args.lens_budget)
    print(f"{lens_split}: {len(lens_items)} images pass the logit-lens filter, "
          f"taking {min(lens_budget, len(lens_items))}")
    for item in lens_items[:lens_budget]:
        needed[lens_split].add(item.file_name)

    return {split: sorted(names) for split, names in needed.items() if names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--coco-root", default=None)
    parser.add_argument("--split", default=None, help="Main experiment split")
    parser.add_argument("--lens-split", default=None, help="Logit-lens split")
    parser.add_argument("--budget", type=int, default=6000,
                        help="Max ablation-candidate images to fetch")
    parser.add_argument("--lens-budget", type=int, default=400,
                        help="Max logit-lens images to fetch")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)
    coco_root = resolve(args.coco_root or config["data"]["coco_root"])

    needed = collect(config, args)
    print()
    for split, names in needed.items():
        download_many(split, names, coco_root / split, workers=args.workers)

    print("\ndone")
    for split in needed:
        folder = coco_root / split
        count = sum(1 for _ in folder.glob("*.jpg"))
        size = sum(p.stat().st_size for p in folder.glob("*.jpg"))
        print(f"  {folder}: {count} images, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
