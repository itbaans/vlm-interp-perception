"""The per-sample inspection viewer.

The viewer is the thing that makes a synthetic-data experiment debuggable, so its own
claims need checking: that it is genuinely self-contained, and that the token cells it
draws are the same cells the experiment will actually patch.
"""

from __future__ import annotations

import json
import re

import pytest

import dataviewer as V
import geometry as G
import synthetic as S
import tasks as T


@pytest.fixture(scope="module")
def grid():
    return G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)


@pytest.fixture(scope="module")
def samples():
    out = []
    for task in T.TASKS:
        out.extend(T.generate(task, "L3", n=2, seed=99))
    return out


def _data(html: str) -> list[dict]:
    return json.loads(re.search(r"const DATA = (\[.*?\]);\n", html, re.S).group(1))


# --------------------------------------------------------------------------------------
# Self-containment
# --------------------------------------------------------------------------------------


def test_viewer_is_self_contained(samples, grid, tmp_path):
    path = V.build(samples, tmp_path / "viewer.html", grid)
    html = path.read_text(encoding="utf-8")

    assert path.exists()
    assert "data:image/png;base64," in html
    assert not re.findall(r'(?:src|href)="https?://', html)  # no external assets
    assert not re.findall(r"__[A-Z_]+__", html)  # every placeholder filled


def test_viewer_builds_with_no_results_at_all(samples, grid, tmp_path):
    """Dataset mode must work before any model has run -- that is its whole point."""
    html = V.build(samples, tmp_path / "v.html", grid).read_text(encoding="utf-8")
    assert "dataset mode" in html or "no model run yet" in html
    assert all("result" not in d for d in _data(html))


def test_every_sample_embeds_both_scenes(samples, grid, tmp_path):
    html = V.build(samples, tmp_path / "v.html", grid).read_text(encoding="utf-8")
    records = _data(html)
    assert len(records) == len(samples)
    for record in records:
        assert record["clean_png"].startswith("data:image/png;base64,")
        assert record["cf_png"].startswith("data:image/png;base64,")
        assert record["clean_png"] != record["cf_png"]  # the pair really differs


def test_viewer_handles_an_empty_dataset(grid, tmp_path):
    html = V.build([], tmp_path / "v.html", grid).read_text(encoding="utf-8")
    assert _data(html) == []


# --------------------------------------------------------------------------------------
# The cells it draws must be the cells the experiment patches
# --------------------------------------------------------------------------------------


def test_drawn_shape_cells_match_geometry(samples, grid, tmp_path):
    """If the viewer's overlay disagreed with annotation_to_cells it would be worse than
    useless -- it would confirm a mapping that the experiment does not use."""
    html = V.build(samples, tmp_path / "v.html", grid).read_text(encoding="utf-8")
    by_id = {d["id"]: d for d in _data(html)}

    for sample in samples:
        record = by_id[sample.sample_id]
        for index, shape in enumerate(sample.clean.shapes):
            expected = sorted([r, c] for r, c in S.shape_cells(shape, grid))
            assert record["shapes"][index]["cells"] == expected


def test_token_groups_partition_the_grid(samples, grid):
    """changed / targets / distractors / background must tile the 256 cells exactly."""
    for sample in samples:
        groups = V.token_groups(sample, grid)
        cells = [tuple(c) for group in groups.values() for c in group]
        assert len(cells) == len(set(cells)), "groups overlap"
        assert len(cells) == grid.n_tokens, "groups do not cover the grid"


def test_changed_group_is_exactly_the_changed_shape(samples, grid):
    for sample in samples:
        groups = V.token_groups(sample, grid)
        expected = sorted([r, c] for r, c in S.shape_cells(sample.changed_shape, grid))
        assert groups["changed"] == expected


def test_changed_shape_is_flagged_for_the_outline(samples, grid, tmp_path):
    html = V.build(samples, tmp_path / "v.html", grid).read_text(encoding="utf-8")
    for record in _data(html):
        flagged = [i for i, s in enumerate(record["shapes"]) if s["changed"]]
        assert flagged == [record["changed_index"]]


# --------------------------------------------------------------------------------------
# Prompt and tokenization surfaced on the card
# --------------------------------------------------------------------------------------


def test_answer_token_id_is_shown_when_a_processor_is_given(samples, grid, tmp_path,
                                                            processor):
    html = V.build(samples, tmp_path / "v.html", grid,
                   processor=processor).read_text(encoding="utf-8")
    for record in _data(html):
        assert "answer_token_error" not in record
        assert isinstance(record["answer_token_id"], int)
        task = T.TASKS[record["task"]]
        assert record["answer_token_id"] == task.answer_token_id(processor, record["answer"])


def test_question_and_prefill_are_recorded(samples, grid, tmp_path):
    html = V.build(samples, tmp_path / "v.html", grid).read_text(encoding="utf-8")
    for record in _data(html):
        assert record["question"]
        assert record["prefill"]
        if record["task"] == "count":
            assert record["prefill"].endswith(" ")  # bare-digit convention


# --------------------------------------------------------------------------------------
# Results mode
# --------------------------------------------------------------------------------------


def test_results_are_attached_to_the_right_samples(samples, grid, tmp_path):
    results = {
        samples[0].sample_id: {
            "predicted": "3", "correct": True, "expected": "3",
            "distribution": {"2": 0.1, "3": 0.8, "4": 0.1},
            "answer_by_layer": [0.0, 0.1, 0.4, 0.9],
        }
    }
    html = V.build(samples, tmp_path / "v.html", grid,
                   results=results).read_text(encoding="utf-8")
    records = {d["id"]: d for d in _data(html)}

    assert records[samples[0].sample_id]["result"]["predicted"] == "3"
    assert "result" not in records[samples[1].sample_id]
    assert "scored" in html


def test_unknown_result_ids_are_ignored(samples, grid, tmp_path):
    html = V.build(samples, tmp_path / "v.html", grid,
                   results={"does-not-exist": {"correct": True}}).read_text(encoding="utf-8")
    assert all("result" not in d for d in _data(html))
