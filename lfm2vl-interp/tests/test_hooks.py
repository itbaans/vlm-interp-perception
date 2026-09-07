"""Hook behaviour: ablation, attention knockout, activation capture.

Run against the tiny random-weight model. None of these assertions depend on the
weights being meaningful -- they check that the hooks reach the right modules, change
exactly what they should, and refuse what they cannot do.
"""

from __future__ import annotations

import pytest
import torch

import geometry as G
import prompts as P
from hooked_lfm2vl import layer_windows


@pytest.fixture
def prompt(tiny):
    return P.describe_prompt(tiny.processor)


@pytest.fixture
def grid(tiny, prompt, image):
    return tiny.single_grid(tiny.prepare(image, prompt), (480, 640))


def _logits(tiny, image, prompt):
    with torch.no_grad():
        return tiny.forward(image, prompt).logits


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------


def test_hybrid_backbone_is_wired_as_expected(tiny):
    """Attention only on some layers, conv on the rest -- the premise of the port."""
    layers = tiny.language_model.layers
    assert tiny.attention_layers == [2, 5]
    for index, layer in enumerate(layers):
        if index in tiny.attention_layers:
            assert hasattr(layer, "self_attn")
        else:
            assert hasattr(layer, "conv")
            assert not hasattr(layer, "self_attn")


def test_lm_head_is_tied_to_the_input_embeddings(tiny):
    assert tiny.lm_head.weight.data_ptr() == tiny.language_model.embed_tokens.weight.data_ptr()


def test_final_norm_is_embedding_norm(tiny):
    assert tiny.final_norm is tiny.language_model.embedding_norm


# --------------------------------------------------------------------------------------
# Activation capture
# --------------------------------------------------------------------------------------


def test_capture_returns_post_projector_embeddings(tiny, prompt, image):
    inputs = tiny.prepare(image, prompt)
    embeds = tiny.get_inputs_embeds(image, prompt)
    assert embeds.shape[0] == 1
    assert embeds.shape[1] == inputs["input_ids"].shape[1]
    assert embeds.shape[2] == tiny.model.config.text_config.hidden_size


def test_visual_embeddings_differ_from_the_image_token_embedding(tiny, prompt, image, grid):
    """The projector output really is scattered in, not left as the placeholder embedding."""
    embeds = tiny.get_inputs_embeds(image, prompt)
    placeholder = tiny.language_model.embed_tokens(
        torch.tensor([tiny.image_token_id], device=embeds.device)
    )
    visual = embeds[0, grid.start : grid.end, :]
    assert not torch.allclose(visual, placeholder.expand_as(visual))


def test_visual_token_features_match_the_grid(tiny, image, grid):
    features = tiny.visual_token_features(image)
    assert features.shape == (grid.n_tokens, tiny.model.config.text_config.hidden_size)


# --------------------------------------------------------------------------------------
# Input ablation
# --------------------------------------------------------------------------------------


def test_ablation_replaces_exactly_the_targeted_rows(tiny, prompt, image, grid):
    replacement = torch.full((tiny.model.config.text_config.hidden_size,), 0.25)
    targets = [grid.start, grid.start + 1, grid.start + 50]

    baseline = tiny.get_inputs_embeds(image, prompt)
    inputs = tiny.prepare(image, prompt)
    with tiny.ablate_inputs(targets, replacement):
        with tiny.capture_inputs_embeds() as captured:
            with torch.no_grad():
                tiny.model(**inputs)
    ablated = captured[0]

    changed = (~torch.isclose(baseline, ablated, atol=1e-6)).any(dim=-1)[0]
    assert changed.nonzero().flatten().tolist() == targets
    for index in targets:
        assert torch.allclose(ablated[0, index, :], replacement.to(ablated.dtype), atol=1e-6)


def test_ablation_changes_the_logits(tiny, prompt, image, grid):
    replacement = torch.zeros(tiny.model.config.text_config.hidden_size)
    before = _logits(tiny, image, prompt)
    with tiny.ablate_inputs(grid.indices, replacement):
        after = _logits(tiny, image, prompt)
    assert not torch.allclose(before, after)


def test_ablating_nothing_is_a_no_op(tiny, prompt, image):
    replacement = torch.zeros(tiny.model.config.text_config.hidden_size)
    before = _logits(tiny, image, prompt)
    with tiny.ablate_inputs([], replacement):
        after = _logits(tiny, image, prompt)
    assert torch.equal(before, after)


def test_hook_is_removed_after_the_context_exits(tiny, prompt, image, grid):
    replacement = torch.zeros(tiny.model.config.text_config.hidden_size)
    before = _logits(tiny, image, prompt)
    with tiny.ablate_inputs(grid.indices, replacement):
        pass
    assert torch.equal(before, _logits(tiny, image, prompt))


def test_ablation_survives_generation(tiny, prompt, image, grid):
    """Decode steps have sequence length 1 and must be left alone, not crash."""
    replacement = torch.zeros(tiny.model.config.text_config.hidden_size)
    with tiny.ablate_inputs(grid.indices, replacement):
        text = tiny.generate(image, prompt, max_new_tokens=4)
    assert isinstance(text, str)


def test_generation_is_deterministic_by_default(tiny, prompt, image):
    assert tiny.generate(image, prompt, max_new_tokens=6) == tiny.generate(
        image, prompt, max_new_tokens=6
    )


# --------------------------------------------------------------------------------------
# Attention knockout
# --------------------------------------------------------------------------------------


def test_blocking_a_conv_layer_is_refused(tiny, prompt, image, grid):
    """A silent no-op here would invalidate a whole experiment, so it must raise."""
    conv_layer = next(i for i in range(tiny.num_layers) if i not in tiny.attention_layers)
    with pytest.raises(ValueError, match="short-conv"):
        with tiny.block_attention({conv_layer: [(248, grid.start)]}):
            pass


def test_blocking_object_to_last_token_changes_the_logits(tiny, prompt, image, grid):
    inputs = tiny.prepare(image, prompt)
    last = inputs["input_ids"].shape[1] - 1
    pairs = [(last, k) for k in range(grid.start, grid.end)]

    before = _logits(tiny, image, prompt)
    with tiny.block_attention({layer: pairs for layer in tiny.attention_layers}):
        after = _logits(tiny, image, prompt)

    assert not torch.allclose(before[:, -1, :], after[:, -1, :])


def test_blocking_only_affects_the_query_positions_named(tiny, prompt, image, grid):
    """Blocking attention into the last position must not disturb earlier positions."""
    inputs = tiny.prepare(image, prompt)
    last = inputs["input_ids"].shape[1] - 1
    pairs = [(last, k) for k in range(grid.start, grid.end)]

    before = _logits(tiny, image, prompt)
    with tiny.block_attention({layer: pairs for layer in tiny.attention_layers}):
        after = _logits(tiny, image, prompt)

    assert torch.allclose(before[:, :last, :], after[:, :last, :], atol=1e-5)
    assert not torch.allclose(before[:, last, :], after[:, last, :])


def test_blocking_already_masked_pairs_is_a_no_op(tiny, prompt, image, grid):
    """Pairs with query < key are masked by causality already, so blocking changes nothing."""
    before = _logits(tiny, image, prompt)
    pairs = [(grid.start, grid.start + 10)]  # query before key
    with tiny.block_attention({layer: pairs for layer in tiny.attention_layers}):
        after = _logits(tiny, image, prompt)
    assert torch.allclose(before, after, atol=1e-6)


def test_blocking_hooks_are_removed_after_the_context(tiny, prompt, image, grid):
    inputs = tiny.prepare(image, prompt)
    last = inputs["input_ids"].shape[1] - 1
    pairs = [(last, k) for k in range(grid.start, grid.end)]

    before = _logits(tiny, image, prompt)
    with tiny.block_attention({layer: pairs for layer in tiny.attention_layers}):
        pass
    assert torch.allclose(before, _logits(tiny, image, prompt))


def test_more_blocking_moves_the_logits_further(tiny, prompt, image, grid):
    """Blocking both attention layers must not be identical to blocking one."""
    inputs = tiny.prepare(image, prompt)
    last = inputs["input_ids"].shape[1] - 1
    pairs = [(last, k) for k in range(grid.start, grid.end)]

    with tiny.block_attention({tiny.attention_layers[0]: pairs}):
        one = _logits(tiny, image, prompt)
    with tiny.block_attention({layer: pairs for layer in tiny.attention_layers}):
        both = _logits(tiny, image, prompt)

    assert not torch.allclose(one[:, -1, :], both[:, -1, :])


def test_sdpa_backend_also_supports_blocking(processor):
    """Under sdpa the causal mask can be None; the hook must materialise one."""
    from tiny_model import build_tiny_hooked_model

    model = build_tiny_hooked_model(attn_implementation="sdpa")
    from PIL import Image

    img = Image.new("RGB", (640, 480), (100, 110, 120))
    prompt = P.describe_prompt(model.processor)
    grid = model.single_grid(model.prepare(img, prompt), (480, 640))
    last = model.prepare(img, prompt)["input_ids"].shape[1] - 1
    pairs = [(last, k) for k in range(grid.start, grid.end)]

    before = _logits(model, img, prompt)
    with model.block_attention({layer: pairs for layer in model.attention_layers}):
        after = _logits(model, img, prompt)
    assert not torch.allclose(before[:, -1, :], after[:, -1, :])


# --------------------------------------------------------------------------------------
# Layer windows
# --------------------------------------------------------------------------------------


def test_layer_windows_cover_the_real_models_attention_layers():
    """The mapping used for the paper's Table 2 rows, on the real 30-layer backbone."""
    attention_layers = [2, 5, 9, 13, 17, 21, 24, 27]
    windows = layer_windows(attention_layers, num_layers=30)

    assert windows["all"] == attention_layers
    assert set().union(*windows.values()) == set(attention_layers)
    for name, layers in windows.items():
        assert layers, f"window {name} is empty"
        assert set(layers).issubset(attention_layers)
    # Windows advance monotonically through the stack.
    assert min(windows["early"]) < min(windows["mid"]) < min(windows["late"])


def test_layer_windows_only_contain_blockable_layers(tiny):
    windows = layer_windows(tiny.attention_layers, tiny.num_layers)
    for layers in windows.values():
        for layer in layers:
            assert tiny.language_model.layers[layer].is_attention_layer


# --------------------------------------------------------------------------------------
# Register tokens
# --------------------------------------------------------------------------------------


def test_register_indices_stay_inside_the_visual_span(tiny, prompt, image, grid):
    embeds = tiny.get_inputs_embeds(image, prompt)
    visual = embeds[:, grid.start : grid.end, :]
    indices = G.register_indices(visual, grid)
    assert all(grid.start <= i < grid.end for i in indices)


def test_register_indices_finds_planted_outliers(tiny, grid):
    visual = torch.randn(1, grid.n_tokens, 64) * 0.01
    visual[0, 3] *= 500
    visual[0, 42] *= 500
    assert G.register_indices(visual, grid) == [grid.start + 3, grid.start + 42]


# --------------------------------------------------------------------------------------
# Single-token answer matching
# --------------------------------------------------------------------------------------


def test_first_answer_token_is_space_prefixed(tiny):
    """After the "It is a" prefill the answer starts with a space, a different BPE token."""
    tokenizer = tiny.processor.tokenizer
    token_id = P.first_answer_token_id(tiny.processor, "dog")
    assert tokenizer.decode([token_id]) == " dog"


def test_answer_matches_class_accepts_the_first_word_of_a_compound(tiny):
    token_id = P.first_answer_token_id(tiny.processor, "traffic")
    assert P.answer_matches_class(tiny.processor, token_id, "traffic light")


def test_answer_matches_class_rejects_a_different_class(tiny):
    token_id = P.first_answer_token_id(tiny.processor, "cat")
    assert not P.answer_matches_class(tiny.processor, token_id, "dog")


@pytest.mark.parametrize(
    "class_name,ambiguous",
    [("dog", False), ("cat", False), ("bench", False), ("teddy bear", True)],
)
def test_ambiguity_flag(tiny, class_name, ambiguous):
    """"teddy bear" tokenizes to ' t', which cannot identify it on its own."""
    assert P.first_token_is_ambiguous(tiny.processor, class_name) is ambiguous
