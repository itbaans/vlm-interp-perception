"""Activation patching over minimal pairs -- the primary localization tool here.

Run a clean scene and its counterfactual, then splice the clean residual stream into the
counterfactual run at one (layer, token-group) and ask how much of the clean answer comes
back. Where it comes back is where that information was being carried.

**Why this and not attention knockout.** On LFM2.5-VL only 8 of the 30 LM layers have
attention at all; the other 22 are ``Lfm2ShortConv``. Knockout is therefore structurally
blind to 73% of the network, and the distance control in the COCO replication showed its
measured effect was largely proximity-mediated. Patching the residual stream is
architecture-agnostic -- a conv layer's input is spliced exactly like an attention layer's
-- so it reaches all 30 layers and is the right instrument for this model.

**Why minimal pairs make it valid.** Both runs use the same prompt and render at the same
size, so their token positions align one-to-one. Patching position *i* in one run with
position *i* of the other is meaningful only under that alignment, which is why the scene
generator goes to such lengths to change exactly one shape and nothing else.

The metric is the standard normalised logit difference::

    d(run)   = logit(answer_clean) - logit(answer_cf)   at the last position
    recovery = (d(patched) - d(cf)) / (d(clean) - d(cf))

0 means the patch changed nothing; 1 means it fully restored the clean answer. Values
outside [0, 1] are possible and are not clipped -- an over-correcting patch is a real
observation, not an error.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch


class PatchHook:
    """Forward pre-hook replacing chosen positions of a layer's input hidden states.

    ``Lfm2Model.forward`` calls each decoder layer as ``decoder_layer(hidden_states, ...)``,
    so the residual stream arrives as ``args[0]``.
    """

    def __init__(self, source: torch.Tensor, positions: Sequence[int]):
        self.source = source
        self.positions = list(positions)

    def __call__(self, module, args, kwargs):
        if not self.positions:
            return args, kwargs

        hidden = kwargs.get("hidden_states")
        from_kwargs = hidden is not None
        if hidden is None:
            hidden = args[0]

        patched = hidden.clone()
        replacement = self.source.to(dtype=patched.dtype, device=patched.device)
        patched[:, self.positions, :] = replacement[:, self.positions, :]

        if from_kwargs:
            kwargs["hidden_states"] = patched
            return args, kwargs
        return (patched, *args[1:]), kwargs


@contextmanager
def patch_layer(model, layer: int, source: torch.Tensor, positions: Sequence[int]):
    """Patch the input to ``layer`` with ``source`` at ``positions`` for one forward pass."""
    handle = model.language_model.layers[layer].register_forward_pre_hook(
        PatchHook(source, positions), with_kwargs=True
    )
    try:
        yield
    finally:
        handle.remove()


@dataclass
class PatchRun:
    """Cached state from the clean and counterfactual forward passes of one sample."""

    clean_hidden: tuple[torch.Tensor, ...]
    clean_logits: torch.Tensor
    cf_logits: torch.Tensor
    answer_id: int
    answer_cf_id: int
    seq_len: int

    @property
    def clean_diff(self) -> float:
        return logit_diff(self.clean_logits, self.answer_id, self.answer_cf_id)

    @property
    def cf_diff(self) -> float:
        return logit_diff(self.cf_logits, self.answer_id, self.answer_cf_id)

    @property
    def separation(self) -> float:
        """How far apart the two runs are. Near zero means the sample cannot be scored."""
        return self.clean_diff - self.cf_diff


def logit_diff(logits: torch.Tensor, answer_id: int, answer_cf_id: int) -> float:
    """``logit(clean answer) - logit(counterfactual answer)`` at the last position."""
    last = logits[0, -1, :].float()
    return float(last[answer_id] - last[answer_cf_id])


def prepare_run(model, sample, task, processor) -> PatchRun:
    """Run both scenes once and cache what every subsequent patch needs.

    The clean run keeps all hidden states: ``hidden_states[L]`` is exactly the input to
    layer L for L in 0..num_layers-1, which is what the patch hook splices in.
    """
    import tasks as T

    prompt = T.build_prompt(processor, task, sample)
    clean_image = sample.clean.render()
    cf_image = sample.counterfactual.render()

    clean_out = model.forward(clean_image, prompt, output_hidden_states=True)
    cf_out = model.forward(cf_image, prompt)

    clean_len = clean_out.logits.shape[1]
    cf_len = cf_out.logits.shape[1]
    if clean_len != cf_len:
        raise ValueError(
            f"{sample.sample_id}: clean and counterfactual tokenize to different lengths "
            f"({clean_len} vs {cf_len}); patching positions would not correspond"
        )

    return PatchRun(
        clean_hidden=tuple(h.detach() for h in clean_out.hidden_states),
        clean_logits=clean_out.logits.detach(),
        cf_logits=cf_out.logits.detach(),
        answer_id=task.answer_token_id(processor, sample.answer),
        answer_cf_id=task.answer_token_id(processor, sample.answer_cf),
        seq_len=clean_len,
    )


def recovery(run: PatchRun, patched_logits: torch.Tensor) -> float:
    """Normalised recovery: 0 = patch did nothing, 1 = clean answer fully restored."""
    if abs(run.separation) < 1e-6:
        return float("nan")
    patched = logit_diff(patched_logits, run.answer_id, run.answer_cf_id)
    return (patched - run.cf_diff) / run.separation


def patch_and_score(
    model, sample, task, processor, run: PatchRun, layer: int, positions: Sequence[int]
) -> float:
    """Recovery from splicing the clean stream into the counterfactual run at one site."""
    import tasks as T

    prompt = T.build_prompt(processor, task, sample)
    with patch_layer(model, layer, run.clean_hidden[layer], positions):
        outputs = model.forward(sample.counterfactual.render(), prompt)
    return recovery(run, outputs.logits)


def sweep(
    model,
    sample,
    task,
    processor,
    run: PatchRun,
    groups: dict[str, Sequence[int]],
    layers: Sequence[int] | None = None,
) -> dict[str, dict[int, float]]:
    """Recovery for every (token group, layer) combination.

    Returns ``result[group][layer] = recovery``.
    """
    layers = list(layers if layers is not None else range(model.num_layers))
    out: dict[str, dict[int, float]] = {}
    for name, positions in groups.items():
        out[name] = {
            layer: patch_and_score(model, sample, task, processor, run, layer, positions)
            for layer in layers
        }
    return out


def token_position_groups(sample, inputs, model) -> dict[str, list[int]]:
    """Sequence-index groups for patching, combining visual cells and text positions.

    The grid is derived from ``inputs`` rather than taken as an argument, because only
    that grid carries the correct ``start`` -- the visual tokens begin after the chat
    header, not at index 0. Using a start-0 grid here silently points every visual group
    at the wrong positions, which is the kind of error that produces a plausible-looking
    but meaningless heatmap.
    """
    import dataviewer as V
    from geometry import cells_to_indices

    grid = model.single_grid(inputs, (sample.clean.height, sample.clean.width))

    cell_groups = V.token_groups(sample, grid)
    groups: dict[str, list[int]] = {
        name: cells_to_indices([tuple(c) for c in cells], grid)
        for name, cells in cell_groups.items()
    }

    input_ids = inputs["input_ids"][0].tolist()
    seq_len = len(input_ids)
    image_positions = [i for i, t in enumerate(input_ids) if t == model.image_token_id]
    first, last_image = image_positions[0], image_positions[-1]

    groups["image_start"] = [first - 1] if first > 0 else []
    groups["image_end"] = [last_image + 1] if last_image + 1 < seq_len else []
    # Everything after the image block except the final position: the question and the
    # prefilled answer stem.
    groups["question"] = list(range(last_image + 2, seq_len - 1))
    groups["last"] = [seq_len - 1]
    return groups
