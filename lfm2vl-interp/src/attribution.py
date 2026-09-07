"""Integrated gradients over visual tokens (paper Section 3, ablation method 5).

The paper's "High-Gradient Tokens" baseline: prompt the model with
"Is there a [o] in this image?", integrate the gradient of the "Yes" logit with respect
to the input embeddings along a straight path from a fully-ablated baseline to the real
input, then ablate the highest-attribution tokens. It is the strongest baseline in
Table 1 -- the one object-token ablation has to beat.

Ported from ``llava-interp-main/scripts/ablation_experiment.py`` with three fixes:

* ``use_cache=False``. LFM2 builds an ``Lfm2HybridConvCache`` by default, which is
  wasted work here and interacts badly with repeated backward passes.
* Gradients are zeroed per step. The original calls ``model.zero_grad()`` but relies on
  a fresh ``interpolated_embeds`` each iteration for the input gradient; making it
  explicit avoids accumulation bugs if the loop is ever restructured.
* The scaling is applied once at the end as ``(1/steps) * sum(grad) * (input - baseline)``,
  which is the actual integrated-gradients formula. The original writes
  ``integrated_grads *= diff / steps``, which happens to be equivalent, but only because
  the accumulator holds a plain sum.
"""

from __future__ import annotations

import torch

from geometry import TokenGrid


def integrated_gradients(
    model,
    inputs,
    grid: TokenGrid,
    replacement_tensor: torch.Tensor,
    target_token_id: int,
    steps: int = 50,
) -> torch.Tensor:
    """Attribution per sequence position for ``target_token_id`` at the last position.

    Args:
        model: a ``HookedLFM2VL``.
        inputs: processor output for the image and prompt.
        grid: the visual-token grid; only these positions are perturbed, so attribution
            outside the visual span is zero by construction.
        replacement_tensor: the mean visual token, used as the integration baseline.
        target_token_id: the token whose logit is attributed, e.g. the id of "Yes".

    Returns a 1-D tensor of length ``seq_len`` holding |attribution| summed over the
    embedding dimension.
    """
    with model.capture_inputs_embeds() as captured:
        with torch.no_grad():
            model.model(**inputs)
    if not captured:
        raise RuntimeError("forward pass did not reach the language model")
    input_embeds = captured[0].detach()

    baseline = input_embeds.clone()
    baseline[:, grid.start : grid.end, :] = replacement_tensor.to(
        dtype=baseline.dtype, device=baseline.device
    )

    diff = input_embeds - baseline
    accumulated = torch.zeros_like(input_embeds, dtype=torch.float32)

    for alpha in torch.linspace(0, 1, steps):
        interpolated = (baseline + alpha * diff).detach().requires_grad_(True)
        outputs = model.model(inputs_embeds=interpolated, use_cache=False)
        target_logit = outputs.logits[0, -1, target_token_id]

        model.model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(target_logit, interpolated)[0]
        accumulated += grad.float()

        del outputs, interpolated, grad

    attribution = (accumulated / steps) * diff.float()
    return attribution.abs().sum(dim=-1).squeeze(0).detach()


def yes_token_id(processor) -> int:
    """Token id for an affirmative answer, used as the integrated-gradients target."""
    ids = processor.tokenizer.encode(" Yes", add_special_tokens=False)
    if not ids:
        ids = processor.tokenizer.encode("Yes", add_special_tokens=False)
    return ids[0]
