"""End-to-end smoke tests: every script runs to completion with ``--tiny``.

These use a random-weight model on synthetic COCO data, so the *numbers* are
meaningless. What they verify is that each script's full code path executes -- argument
parsing, config, dataset filtering, grids, hooks, integrated gradients, results
serialisation and the analysis tables -- without a GPU or the 3B checkpoint.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fake_coco import build_fake_coco, build_fake_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
SPLIT = "val2017"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("fakecoco")
    paths = build_fake_coco(root, split=SPLIT)
    paths["config"] = build_fake_config(root, split=SPLIT)
    return paths


def run(script: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{script.name} failed ({result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# Ordered: each test consumes what the previous produced.


@pytest.mark.order(1)
def test_build_dataset(dataset):
    run(
        SCRIPTS / "build_dataset.py",
        "--config", str(dataset["config"]),
        "--tiny",
        "--skip-hallucination-control",
    )
    manifest = dataset["root"] / "results" / f"dataset_{SPLIT}.json"
    assert manifest.exists()
    data = load(manifest)
    assert len(data["images"]) == 4
    assert all(record["kept"] for record in data["images"].values())
    assert data["meta"]["hallucination_control"] is False


@pytest.mark.order(2)
def test_build_dataset_with_hallucination_control(dataset):
    """The control path must run; with random weights nothing is expected to pass it."""
    out = dataset["root"] / "results" / "dataset_control.json"
    run(
        SCRIPTS / "build_dataset.py",
        "--config", str(dataset["config"]),
        "--tiny",
        "--results", str(out),
    )
    data = load(out)
    assert data["meta"]["hallucination_control"] is True
    for record in data["images"].values():
        assert "names_original" in record
        assert isinstance(record["kept"], bool)


@pytest.mark.order(3)
def test_compute_mean_visual_token(dataset):
    import torch

    run(
        SCRIPTS / "compute_mean_visual_token.py",
        "--config", str(dataset["config"]),
        "--tiny",
        "--num-images", "3",
    )
    vector_path = dataset["root"] / "mean_visual_token.pt"
    assert vector_path.exists()
    vector = torch.load(vector_path, map_location="cpu", weights_only=True)
    assert vector.ndim == 1
    assert vector.numel() == 64  # tiny model hidden size
    assert torch.isfinite(vector).all()


@pytest.mark.order(4)
def test_ablation_experiment(dataset):
    run(
        SCRIPTS / "ablation_experiment.py",
        "--config", str(dataset["config"]),
        "--tiny",
        "--limit", "2",
    )
    results = load(dataset["root"] / "results" / f"ablation_{SPLIT}.json")
    assert len(results["images"]) == 2

    record = next(iter(results["images"].values()))
    conditions = record["conditions"]
    assert "no_ablation" in conditions
    for expected in ("object_0", "object_1", "register", "random_5", "gradient_5"):
        assert expected in conditions, f"missing condition {expected}"
        assert isinstance(conditions[expected]["generative"], bool)
        assert isinstance(conditions[expected]["polling"], bool)

    # object+1 must ablate strictly more tokens than object+0.
    assert conditions["object_1"]["n_ablated"] > conditions["object_0"]["n_ablated"]
    # Baseline counts are capped at the grid size.
    assert conditions["random_5"]["n_ablated"] == 5
    # Every index is inside the visual span.
    n_tokens = record["n_visual_tokens"]
    for indices in record["indices"].values():
        assert all(isinstance(i, int) for i in indices)
        assert len(set(indices)) == len(indices)
        assert max(indices, default=0) < n_tokens + 64  # span starts a few tokens in


@pytest.mark.order(5)
def test_ablation_experiment_vqa(dataset):
    run(
        SCRIPTS / "ablation_experiment_vqa.py",
        "--config", str(dataset["config"]),
        "--tiny",
        "--limit", "2",
    )
    results = load(dataset["root"] / "results" / f"ablation_vqa_{SPLIT}.json")
    assert len(results["images"]) == 2
    record = next(iter(results["images"].values()))
    baseline = record["conditions"]["no_ablation"]
    assert set(baseline) >= {
        "is_correct",
        "generated_token",
        "generated_token_prob",
        "correct_token_prob",
    }
    assert 0.0 <= baseline["correct_token_prob"] <= 1.0


@pytest.mark.order(6)
def test_attention_knockout(dataset):
    run(
        SCRIPTS / "attention_knockout.py",
        "--config", str(dataset["config"]),
        "--tiny",
        "--limit", "2",
    )
    results = load(dataset["root"] / "results" / f"attention_knockout_{SPLIT}.json")
    assert len(results["images"]) == 2
    meta = results["meta"]
    assert meta["attention_layers"] == [2, 5]
    assert meta["conv_layers"] == [0, 1, 3, 4]
    assert set(meta["windows"]) >= {"early", "mid", "late", "all"}

    record = next(iter(results["images"].values()))
    assert "baseline" in record
    assert record["distance_to_last"] >= 0


@pytest.mark.order(7)
def test_logit_lens_scripts(dataset):
    run(
        SCRIPTS / "logit_lens" / "create_logit_lens.py",
        "--config", str(dataset["config"]),
        "--tiny",
        "--image-folder", str(dataset["image_dir"]),
        "--limit", "2",
    )
    lens_dir = dataset["root"] / "results" / "logit_lens"
    pages = list(lens_dir.glob("*_logit_lens.html"))
    assert len(pages) == 2
    content = pages[0].read_text(encoding="utf-8")
    assert "data:image/png;base64," in content
    assert "__DATA__" not in content

    run(
        SCRIPTS / "logit_lens" / "generate_overview.py",
        "--config", str(dataset["config"]),
    )
    index = lens_dir / "index.html"
    assert index.exists()
    assert pages[0].name in index.read_text(encoding="utf-8")


@pytest.mark.order(8)
def test_quantify_object_tokens(dataset):
    result = run(
        SCRIPTS / "logit_lens" / "quantify_object_tokens.py",
        "--config", str(dataset["config"]),
        "--tiny",
        "--limit", "2",
    )
    assert "object-token -> class-token correspondence" in result.stdout
    assert "mean best layer" in result.stdout

    results = load(dataset["root"] / "results" / f"logit_lens_{SPLIT}.json")
    record = next(iter(results["images"].values()))
    assert len(record["strict"]) == 7  # 6 tiny layers + embeddings
    assert record["n_object_tokens"] > 0


@pytest.mark.order(9)
def test_analyze_results(dataset):
    result = run(
        SCRIPTS / "analyze_results.py",
        "--config", str(dataset["config"]),
        "--split", SPLIT,
    )
    # Table 1 always renders: the generative/polling settings need no correct baseline.
    assert "TABLE 1" in result.stdout
    assert "+ 1 Buffer" in result.stdout
    assert "Int. Gradients (5)" in result.stdout

    # A random-weight model answers nothing correctly, so the VQA and knockout sections
    # legitimately have no usable images. Either the table or the explanation is fine
    # here; test_table_arithmetic below checks the numbers on handcrafted results.
    assert "TABLE 1 (VQA)" in result.stdout or "no images answered correctly" in result.stdout
    assert "TABLE 2" in result.stdout or "no images answered correctly" in result.stdout


def _ablation_results(flags: list[tuple[bool, bool]]) -> dict:
    """Handcrafted ablation results: ``flags[i]`` is (baseline_correct, object_0_correct)."""
    images = {}
    for index, (baseline, ablated) in enumerate(flags):
        images[str(index)] = {
            "class_name": "dog",
            "n_visual_tokens": 234,
            "grid": [13, 18],
            "conditions": {
                "no_ablation": {"generative": baseline, "polling": baseline, "n_ablated": 0},
                "object_0": {"generative": ablated, "polling": ablated, "n_ablated": 12},
                "random_40": {"generative": True, "polling": True, "n_ablated": 40},
            },
            "indices": {},
        }
    return {"meta": {}, "images": images}


def test_table_arithmetic(tmp_path):
    """Table 1 reports the share of *initially correct* answers the ablation destroyed.

    Four images, three correct unablated; object ablation leaves one of those three
    correct, so two of three were destroyed -> 66.67% degradation (the paper's Table 1
    units). The fourth image was wrong to begin with and must be excluded from the
    denominator entirely.
    """
    config = build_fake_config(tmp_path, split=SPLIT)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    flags = [(True, True), (True, False), (True, False), (False, False)]
    (results_dir / f"ablation_{SPLIT}.json").write_text(
        json.dumps(_ablation_results(flags)), encoding="utf-8"
    )

    result = run(SCRIPTS / "analyze_results.py", "--config", str(config), "--split", SPLIT)

    object_line = next(l for l in result.stdout.splitlines() if l.startswith("Object"))
    assert "66.67%" in object_line  # 2 of 3 destroyed
    assert "12.0" in object_line  # average ablated-token count

    # The fully-surviving baseline destroyed nothing, on the same 3-image denominator.
    random_line = next(l for l in result.stdout.splitlines() if l.startswith("Random (40)"))
    assert "0.00%" in random_line


def test_table_two_arithmetic(tmp_path):
    """Table 2 reports the fraction of correct answers retained under blocking."""
    config = build_fake_config(tmp_path, split=SPLIT)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    images = {}
    for index, retained in enumerate([True, True, False, False]):
        images[str(index)] = {
            "class_name": "teddy bear" if index == 0 else "dog",
            "ambiguous_class": index == 0,
            "baseline": {"is_correct": True},
            "knockout": {
                "O+1->LTP@all": {"is_correct": retained},
                "O+1->LVR@all": {"is_correct": True},
            },
        }
    # An image that was already wrong must not enter the denominator.
    images["99"] = {"baseline": {"is_correct": False}, "knockout": {}}

    (results_dir / f"attention_knockout_{SPLIT}.json").write_text(
        json.dumps(
            {
                "meta": {
                    "attention_layers": [2, 5],
                    "num_layers": 6,
                    "windows": {"all": [2, 5]},
                },
                "images": images,
            }
        ),
        encoding="utf-8",
    )

    result = run(SCRIPTS / "analyze_results.py", "--config", str(config), "--split", SPLIT)

    assert "TABLE 2" in result.stdout
    ltp_line = next(l for l in result.stdout.splitlines() if l.startswith("O+1") and "LTP" in l)
    lvr_line = next(l for l in result.stdout.splitlines() if l.startswith("O+1") and "LVR" in l)
    assert "0.50" in ltp_line  # 2 of 4 retained
    assert "1.00" in lvr_line  # the last-visual-row control is untouched

    # Ambiguous single-token class names are surfaced, not silently averaged in.
    assert "1 of the scored images (25%)" in result.stdout
    assert "teddy bear" in result.stdout


def test_analyze_results_errors_without_data(tmp_path, dataset):
    """A clear message beats a stack trace when someone runs analysis too early."""
    config = build_fake_config(tmp_path, split=SPLIT)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "analyze_results.py"), "--config", str(config)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "no results found" in (result.stdout + result.stderr)
