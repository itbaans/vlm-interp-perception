"""Logit-lens correctness, especially the final-layer norm convention."""

from __future__ import annotations

import json
import re

import pytest
import torch

import logit_lens as LL
import prompts as P


@pytest.fixture
def prompt(tiny):
    return P.describe_prompt(tiny.processor)


@pytest.fixture
def lens_inputs(tiny, prompt, image):
    inputs = tiny.prepare(image, prompt)
    grid = tiny.single_grid(inputs, (480, 640))
    outputs = tiny.forward(image, prompt, output_hidden_states=True)
    return inputs, grid, outputs


# --------------------------------------------------------------------------------------
# The norm convention
# --------------------------------------------------------------------------------------


def test_final_hidden_state_is_already_normed(tiny, lens_inputs):
    """``Lfm2Model`` applies ``embedding_norm`` before returning, so the last
    ``hidden_states`` entry must be unembedded directly. Re-norming it -- which the
    original repo does uniformly -- would corrupt the final layer of every lens."""
    _, _, outputs = lens_inputs
    last = outputs.hidden_states[-1]
    assert torch.allclose(tiny.lm_head(last), outputs.logits, atol=1e-3)
    assert not torch.allclose(tiny.lm_head(tiny.final_norm(last)), outputs.logits, atol=1e-3)


def test_layer_logits_reproduces_the_model_logits_on_the_last_layer(tiny, lens_inputs):
    _, _, outputs = lens_inputs
    reproduced = LL.layer_logits(
        outputs.hidden_states[-1], tiny.final_norm, tiny.lm_head, is_final_layer=True
    )
    assert torch.allclose(reproduced, outputs.logits[0].float(), atol=1e-3)


def test_hidden_states_has_one_entry_per_layer_plus_embeddings(tiny, lens_inputs):
    _, _, outputs = lens_inputs
    assert len(outputs.hidden_states) == tiny.num_layers + 1


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


def test_decode_topk_shape(tiny, lens_inputs):
    inputs, _, outputs = lens_inputs
    decoded = LL.decode_topk(
        outputs.hidden_states, tiny.final_norm, tiny.lm_head, tiny.processor.tokenizer, top_k=5
    )
    assert len(decoded) == tiny.num_layers + 1
    assert len(decoded[0]) == inputs["input_ids"].shape[1]
    assert len(decoded[0][0]) == 5
    token, prob = decoded[0][0][0]
    assert isinstance(token, str) and 0.0 <= prob <= 1.0


def test_decode_topk_probabilities_are_descending(tiny, lens_inputs):
    _, _, outputs = lens_inputs
    decoded = LL.decode_topk(
        outputs.hidden_states,
        tiny.final_norm,
        tiny.lm_head,
        tiny.processor.tokenizer,
        positions=[0, 10],
        top_k=5,
    )
    for layer in decoded:
        for position in layer:
            probs = [p for _, p in position]
            assert probs == sorted(probs, reverse=True)


def test_positions_argument_restricts_the_output(tiny, lens_inputs):
    _, grid, outputs = lens_inputs
    positions = [grid.start, grid.start + 5, grid.end - 1]
    decoded = LL.decode_topk(
        outputs.hidden_states,
        tiny.final_norm,
        tiny.lm_head,
        tiny.processor.tokenizer,
        positions=positions,
    )
    assert all(len(layer) == len(positions) for layer in decoded)


def test_top1_token_ids_agrees_with_decode_topk(tiny, lens_inputs):
    _, grid, outputs = lens_inputs
    positions = [grid.start, grid.start + 7]
    ids = LL.top1_token_ids(outputs.hidden_states, tiny.final_norm, tiny.lm_head, positions)
    decoded = LL.decode_topk(
        outputs.hidden_states,
        tiny.final_norm,
        tiny.lm_head,
        tiny.processor.tokenizer,
        positions=positions,
        top_k=1,
    )
    assert ids.shape == (tiny.num_layers + 1, len(positions))
    for layer in range(ids.shape[0]):
        for col in range(ids.shape[1]):
            assert tiny.processor.tokenizer.decode([int(ids[layer, col])]) == decoded[layer][col][0][0]


# --------------------------------------------------------------------------------------
# Class matching
# --------------------------------------------------------------------------------------


def test_class_match_ids_covers_spaced_and_bare_forms(tiny):
    tokenizer = tiny.processor.tokenizer
    ids = LL.class_match_ids(tokenizer, "dog")
    assert tokenizer.encode("dog", add_special_tokens=False)[0] in ids
    assert tokenizer.encode(" dog", add_special_tokens=False)[0] in ids


def test_class_match_ids_handles_multiword_classes(tiny):
    tokenizer = tiny.processor.tokenizer
    ids = LL.class_match_ids(tokenizer, "traffic light")
    assert tokenizer.encode(" traffic", add_special_tokens=False)[0] in ids


@pytest.mark.parametrize(
    "token,class_name,expected",
    [
        ("swe", "sweater", True),
        (" diam", "diamond", True),
        ("sweater", "sweater", True),
        ("do", "dog", False),  # shorter than min_prefix
        ("cat", "dog", False),
        ("traffic", "traffic light", True),
        ("", "dog", False),
    ],
)
def test_lenient_match(token, class_name, expected):
    assert LL.is_lenient_match(token, class_name) is expected


# --------------------------------------------------------------------------------------
# Hit rate
# --------------------------------------------------------------------------------------


def _annotation():
    return {
        "bbox": [300, 200, 60, 60],
        "area": 3600,
        "iscrowd": 0,
        "segmentation": [[300, 200, 360, 200, 360, 260, 300, 260]],
    }


def test_hit_rate_reports_one_value_per_layer(tiny, lens_inputs):
    _, grid, outputs = lens_inputs
    result = LL.object_token_hit_rate(
        outputs.hidden_states,
        tiny.final_norm,
        tiny.lm_head,
        tiny.processor.tokenizer,
        grid,
        _annotation(),
        "dog",
    )
    assert result["n_object_tokens"] > 0
    assert len(result["strict"]) == tiny.num_layers + 1
    assert len(result["lenient"]) == tiny.num_layers + 1
    assert all(0.0 <= r <= 1.0 for r in result["strict"])
    assert 0 <= result["best_layer_strict"] <= tiny.num_layers


def test_lenient_rate_is_never_below_strict(tiny, lens_inputs):
    _, grid, outputs = lens_inputs
    result = LL.object_token_hit_rate(
        outputs.hidden_states,
        tiny.final_norm,
        tiny.lm_head,
        tiny.processor.tokenizer,
        grid,
        _annotation(),
        "dog",
    )
    assert all(l >= s for s, l in zip(result["strict"], result["lenient"]))


def test_hit_rate_handles_an_object_outside_the_image(tiny, lens_inputs):
    _, grid, outputs = lens_inputs
    off_image = {
        "bbox": [10000, 10000, 5, 5],
        "area": 25,
        "iscrowd": 0,
        "segmentation": [[10000, 10000, 10005, 10000, 10005, 10005, 10000, 10005]],
    }
    result = LL.object_token_hit_rate(
        outputs.hidden_states,
        tiny.final_norm,
        tiny.lm_head,
        tiny.processor.tokenizer,
        grid,
        off_image,
        "dog",
    )
    assert result["n_object_tokens"] == 0


# --------------------------------------------------------------------------------------
# HTML output
# --------------------------------------------------------------------------------------


def test_html_is_written_and_self_contained(tiny, lens_inputs, image, prompt, tmp_path):
    inputs, grid, outputs = lens_inputs
    path = LL.create_interactive_logit_lens(
        outputs.hidden_states,
        tiny.final_norm,
        tiny.lm_head,
        tiny.processor.tokenizer,
        inputs["input_ids"][0].tolist(),
        image,
        grid,
        prompt,
        tmp_path / "lens.html",
    )
    content = path.read_text(encoding="utf-8")

    assert path.exists()
    assert "data:image/png;base64," in content
    assert "http://" not in content and "https://" not in content  # no external assets
    assert "__DATA__" not in content and "__ROWS__" not in content  # every placeholder filled
    assert f"const gridRows = {grid.n_rows}, gridCols = {grid.n_cols};" in content


def test_html_has_one_row_per_sequence_position(tiny, lens_inputs, image, prompt, tmp_path):
    inputs, grid, outputs = lens_inputs
    path = LL.create_interactive_logit_lens(
        outputs.hidden_states,
        tiny.final_norm,
        tiny.lm_head,
        tiny.processor.tokenizer,
        inputs["input_ids"][0].tolist(),
        image,
        grid,
        prompt,
        tmp_path / "lens.html",
    )
    content = path.read_text(encoding="utf-8")

    labels = json.loads(re.search(r"const tokenLabels = (\[.*?\]);", content, re.S).group(1))
    is_image = json.loads(re.search(r"const isImageToken = (\[.*?\]);", content, re.S).group(1))
    seq_len = inputs["input_ids"].shape[1]
    assert len(labels) == seq_len
    assert sum(is_image) == grid.n_tokens
    # Image labels carry their grid coordinates, so the overlay can be indexed.
    assert labels[grid.start] == "IMG r00c00"
    assert labels[grid.end - 1] == f"IMG r{grid.n_rows - 1:02d}c{grid.n_cols - 1:02d}"
