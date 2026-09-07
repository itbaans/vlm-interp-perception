#!/usr/bin/env python3
"""Run the whole replication end to end and bundle the results for download.

    python3 run_all.py                  # standard preset, ~3-5 h on one A100
    python3 run_all.py --preset quick   # ~20 min, proves the pipeline works
    python3 run_all.py --serve 8000     # …then download the bundle from a browser

One command handles everything: it builds the virtualenv and installs dependencies,
downloads only the COCO splits actually needed, runs all three experiments, renders the
figures and tables, writes a self-contained HTML report and zips the lot.

The script is deliberately stdlib-only until the virtualenv exists, so it can be run by
whatever ``python3`` the box happens to have. After bootstrapping it re-executes itself
inside the venv.

Every stage is resumable: completed stages are recorded in ``outputs/state.json`` and
the experiment scripts themselves checkpoint per image, so an interrupted run continues
where it stopped. ``--force`` reruns everything.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
OUTPUTS = ROOT / "outputs"
IN_VENV_FLAG = "--_in-venv"

COCO_URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
}
COCO_SIZES = {"annotations": "241 MB", "val2017": "1 GB", "train2017": "19 GB"}

PRESETS = {
    "quick": {
        "ablation_images": 50, "vqa_limit": 25, "knockout_limit": 25,
        "lens_images": 30, "lens_pages": 6, "mean_images": 300,
        "ig_steps": 10, "baseline_counts": [5, 20, 60], "image_budget": 400,
        "perception_samples": 25, "perception_per_cell": 4,
        "estimate": "~20 min", "purpose": "proves the pipeline end to end",
    },
    "standard": {
        "ablation_images": 500, "vqa_limit": None, "knockout_limit": None,
        "lens_images": 170, "lens_pages": 20, "mean_images": 5000,
        "ig_steps": 50, "baseline_counts": [5, 10, 20, 40, 60, 100, 250],
        "image_budget": 4000, "perception_samples": 150, "perception_per_cell": 20,
        "estimate": "~3-5 h", "purpose": "enough images for real numbers",
    },
    "full": {
        "ablation_images": 1000, "vqa_limit": None, "knockout_limit": None,
        "lens_images": 300, "lens_pages": 40, "mean_images": 10000,
        "ig_steps": 50, "baseline_counts": [5, 10, 20, 40, 60, 100, 250],
        "image_budget": 8000, "perception_samples": 400, "perception_per_cell": 40,
        "estimate": "~10 h+", "purpose": "paper scale",
    },
}

# Two experiment suites share this runner: the COCO replication of the paper, and the
# synthetic perception probe. They differ only in their stage list and dispatch; the venv
# bootstrap, GPU check, resumability, bundling and retrieval are common.
SUITES = {
    "replication": [
        "data", "verify", "dataset", "meanvec", "ablation", "vqa",
        "knockout", "lens_quant", "lens_pages", "analyze", "figures", "report", "bundle",
    ],
    "perception": [
        "verify", "generate", "baseline", "answer_lens", "attribute_lens",
        "patch", "attn_knockout", "probe", "viewer", "analyze", "bundle",
    ],
}
STAGES = SUITES["replication"]

LOG_PATH: Path | None = None


# --------------------------------------------------------------------------------------
# Logging and process helpers
# --------------------------------------------------------------------------------------


def use_utf8_console() -> None:
    """Make our own stdout/stderr UTF-8 tolerant.

    Child stages emit progress bars and box-drawing characters. We read their output as
    UTF-8 and re-print it, so on a Windows console defaulting to cp1252 the *parent* is
    what crashes -- on a character it merely relayed.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # not a reconfigurable stream
            pass


def log(message: str = "", prefix: str = "") -> None:
    line = f"{prefix}{message}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def banner(title: str) -> None:
    log()
    log("=" * 74)
    log(f"  {title}")
    log("=" * 74)


def run(command: list[str], capture: Path | None = None, check: bool = True) -> int:
    """Run a command, streaming output to the console, the log and optionally a file."""
    log(f"$ {' '.join(str(c) for c in command)}", prefix="")
    process = subprocess.Popen(
        [str(c) for c in command],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            # Without this a child inherits a cp1252 stdout on Windows and dies the first
            # time it prints a non-ASCII character.
            "PYTHONIOENCODING": "utf-8",
        },
    )
    captured: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        captured.append(line)
        if LOG_PATH:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    code = process.wait()

    if capture:
        capture.parent.mkdir(parents=True, exist_ok=True)
        capture.write_text("\n".join(captured), encoding="utf-8")
    if check and code != 0:
        # str() each item: commands carry Path objects, and a TypeError here would
        # mask the actual failure.
        pretty = " ".join(str(c) for c in command)
        raise SystemExit(f"\ncommand failed with exit code {code}:\n  {pretty}")
    return code


def human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
    return f"{seconds // 3600:.0f}h {(seconds % 3600) / 60:.0f}m"


# --------------------------------------------------------------------------------------
# Stage 0 -- environment
# --------------------------------------------------------------------------------------


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def bootstrap(argv: list[str]) -> None:
    """Create the venv, install requirements, then re-exec this script inside it.

    The venv is created with ``--system-site-packages`` so a CUDA build of torch that
    the cloud image already provides is reused rather than replaced with a CPU wheel.
    """
    banner("Stage 0/13 - environment")
    python = venv_python(VENV)

    if not python.exists():
        log(f"creating virtualenv at {VENV} (with --system-site-packages)")
        subprocess.check_call(
            [sys.executable, "-m", "venv", "--system-site-packages", str(VENV)]
        )
    else:
        log(f"reusing virtualenv at {VENV}")

    log("installing requirements (torch is skipped if the image already provides it)")
    subprocess.check_call([str(python), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")]
    )

    log("re-executing inside the virtualenv")
    code = subprocess.call([str(python), str(ROOT / "run_all.py"), *argv, IN_VENV_FLAG])
    sys.exit(code)


def check_gpu(allow_cpu: bool, tiny: bool) -> str:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"GPU: {name} ({memory:.0f} GB) - torch {torch.__version__}")
        return name

    message = f"no CUDA device visible (torch {torch.__version__})"
    if tiny or allow_cpu:
        log(f"WARNING: {message} - continuing on CPU")
        return "cpu"
    raise SystemExit(
        f"\n{message}.\n"
        "The 3B model needs a GPU. Either fix the CUDA install, or pass --allow-cpu "
        "to proceed anyway (very slow), or --tiny to exercise the pipeline with a "
        "random-weight miniature."
    )


# --------------------------------------------------------------------------------------
# Stage 1 -- data
# --------------------------------------------------------------------------------------


def download(url: str, target: Path) -> None:
    """Fetch a file, preferring curl/wget so an interrupted download can resume."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        run(["curl", "-L", "-C", "-", "--retry", "3", "-o", str(target), url])
        return
    if shutil.which("wget"):
        run(["wget", "-c", "-O", str(target), url])
        return

    log(f"downloading {url} with urllib (no curl/wget; not resumable)")
    import urllib.request

    def progress(count, block, total):
        if total > 0 and count % 500 == 0:
            done = count * block
            sys.stdout.write(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, target, reporthook=progress)
    sys.stdout.write("\n")


def ensure_coco_part(coco_root: Path, part: str, marker: Path) -> None:
    if marker.exists():
        log(f"{part}: already present at {marker}")
        return

    archive = coco_root / f"{part}.zip"
    log(f"{part}: downloading ({COCO_SIZES[part]})")
    if not archive.exists():
        download(COCO_URLS[part], archive)

    log(f"{part}: extracting")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(coco_root)
    archive.unlink(missing_ok=True)

    if not marker.exists():
        raise SystemExit(f"{part} extracted but {marker} is missing")


def detect_curated_split(coco_root: Path, questions_path: Path) -> str:
    """Find which COCO split holds the paper's 100 curated VQA image ids.

    The reference repo is inconsistent about this -- its ablation script uses
    train2017 while the attention script defaults to val2017 -- so rather than guess,
    check the annotation files and use whichever actually contains the ids.
    """
    with open(questions_path, "r", encoding="utf-8") as f:
        wanted = {int(k) for k, v in json.load(f).items() if v}

    best, best_hits = "train2017", -1
    for split in ("val2017", "train2017"):
        ann_file = coco_root / "annotations" / f"instances_{split}.json"
        if not ann_file.exists():
            continue
        with open(ann_file, "r", encoding="utf-8") as f:
            ids = {img["id"] for img in json.load(f)["images"]}
        hits = len(wanted & ids)
        log(f"  {split}: {hits} of {len(wanted)} curated ids present")
        if hits > best_hits:
            best, best_hits = split, hits

    if best_hits <= 0:
        raise SystemExit("none of the curated VQA image ids were found in either split")
    return best


def stage_data(args, coco_root: Path, questions_path: Path, fetch: bool = True) -> dict:
    """Download only what is needed: annotations, then the split(s) actually used."""
    if fetch:
        ensure_coco_part(coco_root, "annotations", coco_root / "annotations")
    elif not (coco_root / "annotations").exists():
        raise SystemExit(f"no annotations at {coco_root / 'annotations'} and downloading is off")

    log("\ndetecting which split holds the curated VQA questions")
    main_split = detect_curated_split(coco_root, questions_path)
    lens_split = args.lens_split or "val2017"
    if not (coco_root / lens_split).exists() and not fetch:
        lens_split = main_split
    log(f"  -> experiments run on {main_split}; logit lens on {lens_split}")

    if fetch and args.full_zips:
        for split in dict.fromkeys([main_split, lens_split]):
            ensure_coco_part(coco_root, split, coco_root / split)

    return {"main_split": main_split, "lens_split": lens_split}


# --------------------------------------------------------------------------------------
# Derived config
# --------------------------------------------------------------------------------------


def write_run_config(
    args, preset: dict, splits: dict, coco_root: Path, questions_path: Path, outputs: Path
) -> Path:
    """Materialise a config with the preset's overrides applied.

    The experiment scripts all read a YAML config, so scaling a run is a matter of
    writing one derived config rather than threading a dozen flags through each script.
    """
    import yaml

    with open(ROOT / "configs" / "lfm2_5_vl_3b.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["device"] = args.device
    config["data"]["coco_root"] = str(coco_root)
    config["data"]["split"] = splits["main_split"]
    config["data"]["clean_questions"] = str(questions_path)
    config["ablation"]["max_images"] = preset["ablation_images"]
    config["ablation"]["integrated_gradient_steps"] = preset["ig_steps"]
    config["ablation"]["baseline_counts"] = preset["baseline_counts"]
    config["filters"]["logit_lens_num_images"] = preset["lens_images"]
    if args.tiny:
        # The toy fixtures have small objects, and a random-weight model never names a
        # class. Widen the logit-lens area window so the stage has candidates at all;
        # the hallucination control is skipped for the same reason (see the dataset stage).
        config["filters"]["logit_lens_area_window"] = config["filters"]["area_window"]
    config["paths"]["results_dir"] = str(outputs / "raw")
    config["paths"]["logit_lens_dir"] = str(outputs / "logit_lens")
    # Keep every produced artifact inside the run's own outputs directory. Left at the
    # config default this writes into the source tree, so a --tiny smoke run would
    # overwrite a real 2048-dim mean vector with a 64-dim one from the miniature model.
    config["ablation"]["mean_vector"] = str(outputs / "mean_visual_token.pt")

    path = outputs / "run_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


# --------------------------------------------------------------------------------------
# Bundling
# --------------------------------------------------------------------------------------


def bundle(outputs: Path, manifest: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    archive = outputs / f"lfm2vl-results-{stamp}.zip"

    include = ["report.html", "figures", "tables", "logit_lens", "raw", "run.log",
               "manifest.json", "run_config.yaml",
               # perception suite
               "viewer.html", "synthetic"]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in include:
            path = outputs / name
            if not path.exists():
                continue
            if path.is_file():
                zf.write(path, name)
            else:
                for item in sorted(path.rglob("*")):
                    if item.is_file():
                        zf.write(item, str(item.relative_to(outputs)))
    return archive


def retrieval_instructions(archive: Path) -> None:
    import getpass
    import socket

    user = getpass.getuser()
    host = socket.gethostname()
    size = archive.stat().st_size / 1e6

    banner("Results ready")
    log(f"{archive}  ({size:.1f} MB)")
    log("")
    log("Pull it to your local PC with either of these, run FROM your local machine:")
    log("")
    log(f"  scp {user}@{host}:{archive} .")
    log(f"  rsync -avP {user}@{host}:{archive} .")
    log("")
    log("Replace the host with your provider's SSH address and port if it differs, e.g.")
    log(f"  scp -P 22022 root@1.2.3.4:{archive} .")
    log("")
    log("Or serve it over HTTP instead (handy when the provider forwards a port):")
    log(f"  python3 run_all.py --serve 8000    # then open http://<host>:8000/")
    log("")
    log("Unzip locally and open report.html - it is self-contained, figures included.")


def serve(outputs: Path, port: int) -> None:
    import http.server
    import socketserver

    os.chdir(outputs)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("0.0.0.0", port), handler) as server:
        log(f"serving {outputs} at http://0.0.0.0:{port}/  (ctrl-c to stop)")
        log("If your provider needs a forwarded port, expose this one in its dashboard.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log("\nstopped")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", choices=list(SUITES), default="replication",
                        help="replication = the COCO paper replication; "
                             "perception = synthetic counting/colour/shape probing")
    parser.add_argument("--preset", choices=list(PRESETS), default="standard")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coco-root", default=None, help="Default: data/coco")
    parser.add_argument("--lens-split", default=None, help="Split for the logit lens (val2017)")
    parser.add_argument("--clean-questions", default=None,
                        help="Curated VQA questions JSON. Default: data/clean_questions.json")
    parser.add_argument("--no-download", action="store_true",
                        help="Never fetch COCO; use whatever is already at --coco-root")
    parser.add_argument("--full-zips", action="store_true",
                        help="Download whole COCO split zips instead of only the images the "
                             "filters select. train2017 needs 38 GB transiently; the default "
                             "selective fetch needs a few hundred MB.")
    parser.add_argument("--outputs", default=None, help="Default: outputs/")
    parser.add_argument("--config", default=None,
                        help="Override the suite's config. Default: "
                             "configs/lfm2_5_vl_3b.yaml (replication) or "
                             "configs/synthetic.yaml (perception).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per stage. Overrides the preset. The perception "
                             "dataset is stored interleaved, so a cap stays balanced "
                             "across tasks and levels rather than truncating to one task.")

    parser.add_argument("--only", nargs="+", metavar="STAGE",
                        help=f"Run only these stages. Choices: {' '.join(STAGES)}")
    parser.add_argument("--skip", nargs="+", metavar="STAGE", default=[],
                        help="Skip these stages")
    parser.add_argument("--from", dest="from_stage", metavar="STAGE",
                        help="Start at this stage, skipping earlier ones")
    parser.add_argument("--force", action="store_true",
                        help="Rerun stages already recorded as complete")

    parser.add_argument("--serve", type=int, metavar="PORT",
                        help="Serve the outputs directory over HTTP and exit")
    parser.add_argument("--no-venv", action="store_true",
                        help="Use the current interpreter instead of building a venv")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Proceed without a GPU (very slow)")
    parser.add_argument("--tiny", action="store_true",
                        help="Random-weight miniature model: exercises every stage in "
                             "minutes with meaningless numbers. For validating the pipeline.")
    parser.add_argument(IN_VENV_FLAG, dest="in_venv", action="store_true",
                        help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def selected_stages(args) -> list[str]:
    all_stages = SUITES[args.suite]
    stages = list(all_stages)
    if args.only:
        unknown = set(args.only) - set(all_stages)
        if unknown:
            raise SystemExit(
                f"unknown stage(s) for suite {args.suite!r}: "
                f"{', '.join(sorted(unknown))}\navailable: {' '.join(all_stages)}"
            )
        return [s for s in stages if s in args.only]
    if args.from_stage:
        if args.from_stage not in all_stages:
            raise SystemExit(f"unknown stage: {args.from_stage}")
        stages = stages[all_stages.index(args.from_stage):]
    return [s for s in stages if s not in args.skip]


def main(argv: list[str]) -> None:
    global LOG_PATH

    use_utf8_console()

    args = parse_args(argv)
    outputs = Path(args.outputs).resolve() if args.outputs else OUTPUTS
    outputs.mkdir(parents=True, exist_ok=True)

    if args.serve:
        serve(outputs, args.serve)
        return

    if not args.in_venv and not args.no_venv:
        bootstrap([a for a in argv if a != IN_VENV_FLAG])
        return

    LOG_PATH = outputs / "run.log"
    sys.path.insert(0, str(ROOT / "src"))

    preset = PRESETS[args.preset]
    coco_root = Path(args.coco_root).resolve() if args.coco_root else ROOT / "data" / "coco"
    questions_path = (
        Path(args.clean_questions).resolve() if args.clean_questions
        else ROOT / "data" / "clean_questions.json"
    )
    python = sys.executable
    stages = selected_stages(args)

    state_path = outputs / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.force:
        state = {}

    banner(f"LFM2.5-VL interpretability - suite '{args.suite}', preset '{args.preset}'")
    log(f"{preset['purpose']}, {preset['estimate']} on one modern GPU")
    log(f"host      : {platform.node()} ({platform.system()} {platform.machine()})")
    log(f"python    : {sys.version.split()[0]} at {python}")
    log(f"outputs   : {outputs}")
    log(f"stages    : {', '.join(stages)}")
    if args.tiny:
        log("MODE      : --tiny (random weights; results are meaningless)")

    device = check_gpu(args.allow_cpu, args.tiny)
    if device == "cpu":
        args.device = "cpu"

    manifest = {
        "preset": args.preset,
        "model_id": "LiquidAI/LFM2.5-VL-3B",
        "device": args.device,
        "tiny": args.tiny,
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "stages": {},
    }

    tiny_flag = ["--tiny"] if args.tiny else []
    splits = state.get("splits", {"main_split": "train2017", "lens_split": "val2017"})
    # Written up front and refreshed by the data stage once splits are known, so that
    # running a subset of stages -- or resuming past a cached data stage -- still finds
    # a config rather than a path that was never created.
    config_path = write_run_config(args, preset, splits, coco_root, questions_path, outputs)

    # The perception suite has its own config; --config overrides either suite's default.
    perception_config = (
        Path(args.config) if args.config else ROOT / "configs" / "synthetic.yaml"
    )
    if args.suite == "perception":
        if not perception_config.exists():
            raise SystemExit(f"perception config not found: {perception_config}")
        # Redirect its outputs into this run's directory so the bundle picks them up.
        perception_config = write_perception_config(args, outputs, perception_config)
    started_all = time.time()

    def record(name: str, status: str, elapsed: float, note: str = "") -> None:
        manifest["stages"][name] = {
            "status": status, "duration": human(elapsed) if elapsed else "—", "note": note
        }
        state.setdefault("completed", [])
        if status == "done" and name not in state["completed"]:
            state["completed"].append(name)
        state["splits"] = splits
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    for index, stage in enumerate(stages, 1):
        if stage in state.get("completed", []) and stage not in ("analyze", "figures",
                                                                 "report", "bundle"):
            log(f"\n[{index}/{len(stages)}] {stage}: already complete (--force to rerun)")
            record(stage, "cached", 0)
            continue

        banner(f"Stage {index}/{len(stages)} - {stage}")
        started = time.time()

        if args.suite == "perception" and stage != "bundle":
            status = run_perception_stage(
                stage, args, python, perception_config, outputs, tiny_flag,
                limit=args.limit or preset.get("perception_samples"),
                per_cell=preset.get("perception_per_cell"),
            )
            record(stage, status, time.time() - started)
            log(f"\n-> {stage} finished in {human(time.time() - started)}")
            continue

        if stage == "data":
            fetch = not (args.no_download or (args.tiny and args.coco_root))
            if not fetch:
                log("downloading disabled: using whatever is already at --coco-root")
            splits = stage_data(args, coco_root, questions_path, fetch=fetch)
            config_path = write_run_config(
                args, preset, splits, coco_root, questions_path, outputs
            )
            if fetch and not args.full_zips:
                run([python, "scripts/fetch_coco_images.py", "--config", config_path,
                     "--split", splits["main_split"], "--lens-split", splits["lens_split"],
                     "--budget", str(preset["image_budget"])])

        elif stage == "verify":
            run([python, "-c", VERIFY_SNIPPET.format(tiny=int(args.tiny), device=args.device)],
                capture=outputs / "tables" / "00_model_check.txt")

        elif stage == "dataset":
            command = [python, "scripts/build_dataset.py", "--config", config_path,
                       "--limit", str(preset["ablation_images"]), *tiny_flag]
            if args.tiny:
                # A random-weight model never names the object, so the control would
                # reject every image and leave the later stages with nothing to do.
                command.append("--skip-hallucination-control")
            run(command)

        elif stage == "meanvec":
            run([python, "scripts/compute_mean_visual_token.py", "--config", config_path,
                 "--num-images", str(preset["mean_images"]), *tiny_flag])

        elif stage == "ablation":
            run([python, "scripts/ablation_experiment.py", "--config", config_path,
                 "--attn-implementation", "sdpa", *tiny_flag])

        elif stage == "vqa":
            command = [python, "scripts/ablation_experiment_vqa.py", "--config", config_path,
                       "--attn-implementation", "sdpa", *tiny_flag]
            if preset["vqa_limit"]:
                command += ["--limit", str(preset["vqa_limit"])]
            run(command)

        elif stage == "knockout":
            command = [python, "scripts/attention_knockout.py", "--config", config_path,
                       "--attn-implementation", "eager", *tiny_flag]
            if preset["knockout_limit"]:
                command += ["--limit", str(preset["knockout_limit"])]
            run(command)

        elif stage == "lens_quant":
            run([python, "scripts/logit_lens/quantify_object_tokens.py", "--config", config_path,
                 "--split", splits["lens_split"], *tiny_flag],
                capture=outputs / "tables" / "03_logit_lens.txt")

        elif stage == "lens_pages":
            image_dir = coco_root / splits["lens_split"]
            if not image_dir.exists():
                log(f"skipping: {image_dir} not present")
                record(stage, "skipped", time.time() - started, "image folder missing")
                continue
            run([python, "scripts/logit_lens/create_logit_lens.py", "--config", config_path,
                 "--image-folder", str(image_dir), "--limit", str(preset["lens_pages"]),
                 *tiny_flag])
            run([python, "scripts/logit_lens/generate_overview.py", "--config", config_path])

        elif stage == "analyze":
            run([python, "scripts/analyze_results.py", "--config", config_path,
                 "--split", splits["main_split"], "--lens-split", splits["lens_split"]],
                capture=outputs / "tables" / "01_tables.txt", check=False)

        elif stage == "figures":
            build_figures(outputs, splits)

        elif stage == "report":
            build_report(outputs, splits, manifest)

        elif stage == "bundle":
            manifest["finished"] = datetime.now().isoformat(timespec="seconds")
            manifest["total_duration"] = human(time.time() - started_all)
            (outputs / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            archive = bundle(outputs, manifest)
            record(stage, "done", time.time() - started)
            retrieval_instructions(archive)
            log(f"\ntotal runtime: {human(time.time() - started_all)}")
            return

        record(stage, "done", time.time() - started)
        log(f"\n-> {stage} finished in {human(time.time() - started)}")

    log(f"\ntotal runtime: {human(time.time() - started_all)}")


VERIFY_SNIPPET = """
import sys, torch
sys.path.insert(0, "src")
from PIL import Image
if {tiny}:
    from tiny_model import build_tiny_hooked_model
    model = build_tiny_hooked_model()
    print("tiny model built (random weights)")
else:
    from hooked_lfm2vl import HookedLFM2VL
    model = HookedLFM2VL(device="{device}", attn_implementation="sdpa")
    print("model loaded:", model.model_id)
cfg = model.model.config
print("LM layers      :", cfg.text_config.num_hidden_layers)
print("attention at   :", model.attention_layers)
print("vision layers  :", cfg.vision_config.num_hidden_layers)
print("image token id :", cfg.image_token_id)
import prompts as P
prompt = P.describe_prompt(model.processor)
image = Image.new("RGB", (640, 480), (120, 130, 140))
inputs = model.prepare(image, prompt)
grid = model.single_grid(inputs, (480, 640))
print("640x480 -> grid %dx%d = %d visual tokens" % (grid.n_rows, grid.n_cols, grid.n_tokens))
print("caption:", model.generate(image, prompt, max_new_tokens=30).strip()[:200])
"""


GATED_EXIT_CODE = 3


def write_perception_config(args, outputs: Path, source: Path) -> Path:
    """Derive a perception config whose outputs live inside this run's directory.

    Left at its defaults the synthetic config writes into ``results/synthetic`` in the
    source tree, so a run's artefacts would sit outside ``--outputs`` and never reach the
    bundle. Same failure the replication suite had.
    """
    import yaml

    with open(source, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["device"] = args.device
    config["paths"]["results_dir"] = str(outputs / "synthetic")
    config["paths"]["viewer"] = str(outputs / "viewer.html")
    config["paths"]["figures_dir"] = str(outputs / "figures")

    path = outputs / "perception_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return path


PERCEPTION_SCRIPTS = {
    "generate": ("scripts/synthetic/build_dataset.py", []),
    "baseline": ("scripts/synthetic/baseline.py", []),
    "answer_lens": ("scripts/synthetic/answer_lens.py", []),
    "attribute_lens": ("scripts/synthetic/attribute_lens.py", []),
    # Patching needs no attention mask surgery, so sdpa is fine and faster.
    "patch": ("scripts/synthetic/patching.py", []),
    # Knockout does need an additive float mask.
    "attn_knockout": ("scripts/synthetic/knockout.py", ["--attn-implementation", "eager"]),
    "probe": ("scripts/synthetic/probes.py", []),
    "viewer": ("scripts/synthetic/build_viewer.py", []),
    "analyze": ("scripts/synthetic/analyze.py", []),
}

#: Stages that do not build a model and therefore must not receive --tiny or --limit.
_NO_MODEL_STAGES = {"viewer", "analyze"}


def run_perception_stage(
    stage: str, args, python: str, config_path: Path, outputs: Path,
    tiny_flag: list[str], limit: int | None = None, per_cell: int | None = None,
) -> str:
    """Dispatch one stage of the synthetic perception suite."""
    if stage == "verify":
        run([python, "-c", VERIFY_SNIPPET.format(tiny=int(args.tiny), device=args.device)],
            capture=outputs / "tables" / "00_model_check.txt")
        return "done"

    script, extra = PERCEPTION_SCRIPTS[stage]
    command = [python, script, "--config", str(config_path), *extra]
    if stage == "generate" and per_cell:
        command += ["--per-cell", str(per_cell)]
    if stage not in _NO_MODEL_STAGES:
        command += tiny_flag
        if limit:
            command += ["--limit", str(limit)]
    elif stage == "viewer" and limit:
        command += ["--limit", str(limit)]

    capture = outputs / "tables" / f"{stage}.txt" if stage != "viewer" else None
    # Exit code 3 means the stage ran but its gate left nothing to do -- a legitimate
    # outcome that must not abort the stages after it.
    code = run(command, capture=capture, check=False)
    if code == GATED_EXIT_CODE:
        log(f"\n{stage}: nothing to do (gated) - continuing")
        return "gated"
    if code != 0:
        raise SystemExit(f"\n{stage} failed with exit code {code}")
    return "done"


def build_figures(outputs: Path, splits: dict) -> None:
    import plots

    raw = outputs / "raw"
    paths = {
        "ablation": raw / f"ablation_{splits['main_split']}.json",
        "vqa": raw / f"ablation_vqa_{splits['main_split']}.json",
        "knockout": raw / f"attention_knockout_{splits['main_split']}.json",
        "logit_lens": raw / f"logit_lens_{splits['lens_split']}.json",
    }
    written = plots.build_all(paths, outputs / "figures")
    for name, path in written.items():
        log(f"  figure: {name} -> {path.name}")
    if not written:
        log("  no figures generated (no results yet)")


def build_report(outputs: Path, splits: dict, manifest: dict) -> None:
    import aggregate as A
    import report as R

    raw = outputs / "raw"
    paths = {
        "ablation": raw / f"ablation_{splits['main_split']}.json",
        "vqa": raw / f"ablation_vqa_{splits['main_split']}.json",
        "knockout": raw / f"attention_knockout_{splits['main_split']}.json",
        "logit_lens": raw / f"logit_lens_{splits['lens_split']}.json",
    }
    results = {k: (A.load(p) if p.exists() else None) for k, p in paths.items()}
    figures = {
        name: outputs / "figures" / f"{name}.png"
        for name in ("ablation", "logit_lens", "knockout")
        if (outputs / "figures" / f"{name}.png").exists()
    }
    lens_dir = outputs / "logit_lens"
    pages = sorted(p.name for p in lens_dir.glob("*.html")) if lens_dir.exists() else []

    manifest["split"] = splits["main_split"]
    path = R.build(outputs / "report.html", manifest, results, figures, pages)
    log(f"  report: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
