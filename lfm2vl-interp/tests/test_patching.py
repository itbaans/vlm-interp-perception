"""Activation patching.

Run against the tiny random-weight model: none of these assertions depend on the weights
being meaningful, only on the hook splicing the right values into the right positions.

The two identity tests are the ones that actually prove correctness. Patching *every*
position at layer 0 must reproduce the clean run exactly, and patching *no* positions must
leave the counterfactual run bit-identical. A hook that is off by a position, or that
patches the wrong tensor, fails one of them.
"""

from __future__ import annotations

import pytest
import torch

import geometry as G
import patching as PA
import synthetic as S
import tasks as T


@pytest.fixture(scope="module")
def grid():
    return G.token_grid_for_image(orig_h=S.CANVAS, orig_w=S.CANVAS)


@pytest.fixture(scope="module")
def sample():
    return T.generate("count", "L3", n=1, seed=123)[0]


@pytest.fixture(scope="module")
def task():
    return T.TASKS["count"]


@pytest.fixture(scope="module")
def run(tiny, sample, task):
    return PA.prepare_run(tiny, sample, task, tiny.processor)


@pytest.fixture(scope="module")
def prompt(tiny, task, sample):
    return T.build_prompt(tiny.processor, task, sample)


# --------------------------------------------------------------------------------------
# The alignment premise
# --------------------------------------------------------------------------------------


def test_pair_tokenizes_to_the_same_length(tiny, sample, prompt):
    """Patching position i with position i is only meaningful under this alignment."""
    clean = tiny.prepare(sample.clean.render(), prompt)
    cf = tiny.prepare(sample.counterfactual.render(), prompt)
    assert clean["input_ids"].shape == cf["input_ids"].shape
    assert torch.equal(clean["input_ids"], cf["input_ids"])  # same prompt, same tokens


def test_prepare_run_caches_one_hidden_state_per_layer(tiny, run):
    assert len(run.clean_hidden) == tiny.num_layers + 1
    assert run.clean_hidden[0].shape[1] == run.seq_len


def test_mismatched_lengths_are_rejected(tiny, task, processor):
    """A pair that renders to different sizes must fail loudly, not silently misalign."""
    sample = T.generate("count", "L3", n=1, seed=5)[0]
    sample.counterfactual.width = 384  # force a different token count
    sample.counterfactual.height = 384
    with pytest.raises(ValueError, match="different lengths"):
        PA.prepare_run(tiny, sample, task, processor)


# --------------------------------------------------------------------------------------
# Identity tests -- these prove the hook
# --------------------------------------------------------------------------------------


def test_patching_every_position_reproduces_the_clean_run(tiny, sample, task, run, prompt):
    """Splice the whole residual stream at layer 0: the counterfactual must become clean.

    Layer 0's input is the post-projector embedding sequence, so replacing all of it
    replaces the image entirely -- the rest of the network then has nothing left of the
    counterfactual to work from.
    """
    with PA.patch_layer(tiny, 0, run.clean_hidden[0], list(range(run.seq_len))):
        outputs = tiny.forward(sample.counterfactual.render(), prompt)
    assert torch.allclose(outputs.logits, run.clean_logits, atol=1e-4)
    assert PA.recovery(run, outputs.logits) == pytest.approx(1.0, abs=1e-3)


def test_patching_nothing_leaves_the_run_untouched(tiny, sample, task, run, prompt):
    with PA.patch_layer(tiny, 0, run.clean_hidden[0], []):
        outputs = tiny.forward(sample.counterfactual.render(), prompt)
    assert torch.allclose(outputs.logits, run.cf_logits, atol=1e-6)
    assert PA.recovery(run, outputs.logits) == pytest.approx(0.0, abs=1e-3)


def test_patching_is_removed_after_the_context(tiny, sample, run, prompt):
    with PA.patch_layer(tiny, 0, run.clean_hidden[0], list(range(run.seq_len))):
        pass
    outputs = tiny.forward(sample.counterfactual.render(), prompt)
    assert torch.allclose(outputs.logits, run.cf_logits, atol=1e-6)


def test_patching_every_position_at_the_last_layer_also_recovers(tiny, sample, run, prompt):
    """Same identity one layer from the end, to catch an off-by-one in the layer index."""
    last = tiny.num_layers - 1
    with PA.patch_layer(tiny, last, run.clean_hidden[last], list(range(run.seq_len))):
        outputs = tiny.forward(sample.counterfactual.render(), prompt)
    assert torch.allclose(outputs.logits, run.clean_logits, atol=1e-4)


def test_patching_a_subset_changes_only_that_subset_downstream(tiny, sample, run, prompt):
    """A partial patch must do something, but not everything."""
    subset = list(range(10, 20))
    with PA.patch_layer(tiny, 0, run.clean_hidden[0], subset):
        outputs = tiny.forward(sample.counterfactual.render(), prompt)
    assert not torch.allclose(outputs.logits, run.cf_logits, atol=1e-6)
    assert not torch.allclose(outputs.logits, run.clean_logits, atol=1e-4)


def test_patch_hook_writes_exactly_the_named_positions(tiny, run):
    """Unit-check the hook itself, independent of the model."""
    source = torch.full_like(run.clean_hidden[0], 7.0)
    hook = PA.PatchHook(source, [3, 5])
    original = torch.zeros_like(source)
    (patched,), _ = hook(None, (original,), {})

    assert torch.all(patched[:, 3, :] == 7.0)
    assert torch.all(patched[:, 5, :] == 7.0)
    untouched = [i for i in range(patched.shape[1]) if i not in (3, 5)]
    assert torch.all(patched[:, untouched, :] == 0.0)
    assert torch.all(original == 0.0)  # the input tensor is not mutated in place


# --------------------------------------------------------------------------------------
# The metric
# --------------------------------------------------------------------------------------


def test_logit_diff_reads_the_last_position():
    logits = torch.zeros(1, 4, 10)
    logits[0, -1, 3] = 2.0
    logits[0, -1, 7] = 0.5
    assert PA.logit_diff(logits, 3, 7) == pytest.approx(1.5)


def test_recovery_endpoints(run):
    assert PA.recovery(run, run.cf_logits) == pytest.approx(0.0, abs=1e-6)
    assert PA.recovery(run, run.clean_logits) == pytest.approx(1.0, abs=1e-6)


def test_recovery_is_nan_when_the_pair_is_not_separated():
    """A sample whose two runs give the same answer cannot be scored; say so explicitly."""
    logits = torch.zeros(1, 2, 10)
    run = PA.PatchRun(
        clean_hidden=(), clean_logits=logits, cf_logits=logits.clone(),
        answer_id=1, answer_cf_id=2, seq_len=2,
    )
    assert run.separation == pytest.approx(0.0)
    assert PA.recovery(run, logits) != PA.recovery(run, logits)  # NaN


# --------------------------------------------------------------------------------------
# Position groups
# --------------------------------------------------------------------------------------


def test_position_groups_cover_the_sequence_without_overlap(tiny, sample, prompt):
    inputs = tiny.prepare(sample.clean.render(), prompt)
    groups = PA.token_position_groups(sample, inputs, tiny)

    positions = [p for group in groups.values() for p in group]
    assert len(positions) == len(set(positions)), "groups overlap"

    seq_len = inputs["input_ids"].shape[1]
    assert all(0 <= p < seq_len for p in positions)
    # Visual groups plus the delimiters, question and last token account for everything
    # except the chat header before the image.
    assert set(range(min(positions), seq_len)) == set(positions)


def test_visual_groups_land_on_image_tokens(tiny, sample, prompt):
    inputs = tiny.prepare(sample.clean.render(), prompt)
    groups = PA.token_position_groups(sample, inputs, tiny)
    input_ids = inputs["input_ids"][0]

    for name in ("changed", "targets", "distractors", "background"):
        for position in groups[name]:
            assert input_ids[position].item() == tiny.image_token_id, name


def test_delimiter_groups_are_the_image_brackets(tiny, sample, prompt):
    inputs = tiny.prepare(sample.clean.render(), prompt)
    groups = PA.token_position_groups(sample, inputs, tiny)
    tokenizer = tiny.processor.tokenizer
    ids = inputs["input_ids"][0].tolist()

    assert tokenizer.decode([ids[groups["image_start"][0]]]) == "<|image_start|>"
    assert tokenizer.decode([ids[groups["image_end"][0]]]) == "<|image_end|>"


def test_last_group_is_the_final_position(tiny, sample, prompt):
    inputs = tiny.prepare(sample.clean.render(), prompt)
    groups = PA.token_position_groups(sample, inputs, tiny)
    assert groups["last"] == [inputs["input_ids"].shape[1] - 1]


# --------------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------------


def test_sweep_returns_a_value_per_group_and_layer(tiny, sample, task, run, prompt):
    inputs = tiny.prepare(sample.clean.render(), prompt)
    groups = PA.token_position_groups(sample, inputs, tiny)
    layers = [0, tiny.num_layers - 1]

    result = PA.sweep(tiny, sample, task, tiny.processor, run, groups, layers=layers)

    assert set(result) == set(groups)
    for name, by_layer in result.items():
        assert sorted(by_layer) == layers
        assert all(isinstance(v, float) for v in by_layer.values())
