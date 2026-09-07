"""``HookedLFM2VL`` -- the LFM2.5-VL counterpart of ``llava-interp-main/src/HookedLVLM.py``.

The context-manager API is deliberately the same shape as the original
(``ablate_inputs``, ``block_attention``, ``forward``, ``generate``) so the experiment
scripts read like the paper's. What changed, and why:

Module paths
    Old: ``model.language_model.model.layers`` / ``model.language_model.lm_head``.
    New: ``model.model.language_model.layers`` / ``model.lm_head``, and the final norm
    is ``model.model.language_model.embedding_norm`` (not ``.norm``). ``lm_head`` is
    **tied** to the input embeddings here (``tie_word_embeddings: true``).

Attention masking
    The original ``BlockAttentionHook`` writes ``False`` into the attention mask, which
    only makes sense for a boolean mask. LFM2's ``eager_attention_forward`` does
    ``attn_weights + causal_mask``, an **additive float** mask, so blocking means writing
    ``torch.finfo(dtype).min``. Worse, under sdpa/flash the mask can be ``None``
    entirely (the ``is_causal`` fast path), leaving nothing to patch -- hence
    ``attn_implementation="eager"`` for knockout runs, and a materialisation fallback.

Hybrid backbone
    Only 8 of the 30 LM layers have attention at all
    (``layer_types`` -> indices ``[2, 5, 9, 13, 17, 21, 24, 27]``); the other 22 are
    ``Lfm2ShortConv``. Hooking a conv layer's attention would be a silent no-op, so
    ``block_attention`` refuses. Note the conv layers still mix ~2 positions backwards
    each and ignore token-pair masking, so attention knockout on this model is a partial
    cut -- see ``scripts/attention_knockout.py`` for the distance control.

Variable visual tokens
    There is no 576/24x24 constant. ``image_token_spans`` derives the grid from the
    processor's ``spatial_shapes`` and cross-checks it against the actual count of
    image-token positions.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from geometry import TokenGrid, grid_from_spatial_shape

DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-VL-3B"


def _as_image(image_or_path: str | Path | Image.Image) -> Image.Image:
    if isinstance(image_or_path, (str, Path)):
        return Image.open(image_or_path).convert("RGB")
    return image_or_path.convert("RGB")


class BlockAttentionHook:
    """Forward pre-hook that sets chosen (query, key) attention entries to -inf.

    ``pairs`` are ``(query_index, key_index)``: blocking ``(q, k)`` stops position ``q``
    from attending to position ``k``, i.e. stops information flowing from ``k`` to ``q``.
    For a causal LM ``q >= k``.
    """

    def __init__(self, pairs: Sequence[tuple[int, int]]):
        self.pairs = list(pairs)

    def __call__(self, module, args, kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            hidden_states = args[0]
        batch, seq_len, _ = hidden_states.shape

        mask = kwargs.get("attention_mask")
        if mask is None:
            # sdpa/flash took the is_causal fast path, so there is no mask object to
            # edit. Materialise the equivalent additive causal mask and patch that.
            dtype = hidden_states.dtype
            mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=dtype, device=hidden_states.device)
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).tril()
            mask.masked_fill_(~causal, torch.finfo(dtype).min)
        else:
            mask = mask.clone()

        # A boolean mask marks *allowed* positions; a float mask is additive, so blocking
        # means -inf. LFM2's eager path does `attn_weights + causal_mask`, i.e. float.
        blocked = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
        for query_index, key_index in self.pairs:
            if query_index < seq_len and key_index < mask.shape[-1]:
                mask[:, :, query_index, key_index] = blocked

        kwargs["attention_mask"] = mask
        return args, kwargs


class HookedLFM2VL:
    """LFM2.5-VL with the hooks the paper's three experiments need."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda:0",
        dtype: torch.dtype | str = torch.bfloat16,
        attn_implementation: str = "eager",
        do_image_splitting: bool = False,
        model=None,
        processor=None,
    ):
        """
        Args:
            attn_implementation: ``"eager"`` is required for attention knockout.
                ``"sdpa"`` is fine (and faster) for ablation and logit-lens runs.
            do_image_splitting: kept ``False`` so every image yields exactly one visual
                token grid. Typical COCO images are below the splitting threshold
                anyway, so this costs nothing and keeps the geometry unambiguous.
            model, processor: inject pre-built objects, used by the tiny-model tests.
        """
        self.model_id = model_id
        self.do_image_splitting = do_image_splitting

        if processor is None:
            processor = AutoProcessor.from_pretrained(model_id)
        self.processor = processor

        if model is None:
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=device,
                attn_implementation=attn_implementation,
            )
        self.model = model
        self.model.eval()

    # ---------------------------------------------------------------- structure

    @property
    def language_model(self):
        """The ``Lfm2Model`` decoder stack (no lm_head)."""
        return self.model.model.language_model

    @property
    def vision_tower(self):
        return self.model.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.model.multi_modal_projector

    @property
    def final_norm(self):
        """LFM2 calls this ``embedding_norm``, not ``norm``."""
        return self.language_model.embedding_norm

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def num_layers(self) -> int:
        return len(self.language_model.layers)

    @property
    def attention_layers(self) -> list[int]:
        """Indices of layers that actually have attention (8 of 30 in LFM2.5-VL-3B)."""
        return [i for i, layer in enumerate(self.language_model.layers) if layer.is_attention_layer]

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def image_token_id(self) -> int:
        return self.model.config.image_token_id

    # ---------------------------------------------------------------- inputs

    def prepare(self, image_or_path, prompt: str):
        """Run the processor and move everything to the model's device."""
        image = _as_image(image_or_path)
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            do_image_splitting=self.do_image_splitting,
        )
        return inputs.to(self.device)

    def image_token_spans(self, inputs, orig_size: tuple[int, int]) -> list[TokenGrid]:
        """Locate the visual tokens and their grids.

        Args:
            orig_size: ``(height, width)`` of the original image, the frame COCO
                annotations live in.

        Returns one ``TokenGrid`` per tile. With ``do_image_splitting=False`` that is
        always a single grid covering the whole image.

        Raises:
            ValueError: if the grids implied by ``spatial_shapes`` do not account for
                exactly the image-token positions in ``input_ids``. This one check
                catches every geometry regression, so it is deliberately fatal.
        """
        orig_h, orig_w = orig_size
        input_ids = inputs["input_ids"][0]
        positions = torch.nonzero(input_ids == self.image_token_id).flatten().tolist()
        if not positions:
            raise ValueError("no image tokens found in input_ids")

        # Split the positions into maximal contiguous runs; with tiling the runs are
        # separated by row/col and thumbnail marker tokens.
        runs: list[list[int]] = [[positions[0], positions[0] + 1]]
        for pos in positions[1:]:
            if pos == runs[-1][1]:
                runs[-1][1] = pos + 1
            else:
                runs.append([pos, pos + 1])

        spatial_shapes = inputs["spatial_shapes"]
        if len(spatial_shapes) != len(runs):
            raise ValueError(
                f"{len(spatial_shapes)} tiles in spatial_shapes but {len(runs)} runs of "
                f"image tokens in input_ids"
            )

        grids = []
        for (start, end), shape in zip(runs, spatial_shapes):
            grid = grid_from_spatial_shape(shape.tolist(), orig_h=orig_h, orig_w=orig_w, start=start)
            if grid.n_tokens != end - start:
                raise ValueError(
                    f"grid {grid.n_rows}x{grid.n_cols}={grid.n_tokens} does not match the "
                    f"{end - start} image-token positions at [{start}, {end})"
                )
            grids.append(grid)
        return grids

    def single_grid(self, inputs, orig_size: tuple[int, int]) -> TokenGrid:
        """The one visual-token grid, asserting the image was not split into tiles."""
        grids = self.image_token_spans(inputs, orig_size)
        if len(grids) != 1:
            raise ValueError(
                f"expected a single tile but got {len(grids)}; "
                "run with do_image_splitting=False"
            )
        return grids[0]

    # ---------------------------------------------------------------- hooks

    @contextmanager
    def capture_inputs_embeds(self):
        """Capture the LM's input embeddings, i.e. post-projector, post-scatter.

        Replaces the original ``get_text_model_in``/``prompt_hook`` pair. Yields a
        single-element list that is filled once the forward pass runs.
        """
        captured: list[torch.Tensor] = []

        def hook(module, args, kwargs, output):
            embeds = kwargs.get("inputs_embeds")
            if embeds is None and args:
                embeds = args[0]
            if embeds is not None and embeds.shape[-2] > 1:
                captured.append(embeds)
            return output

        handle = self.language_model.register_forward_hook(hook, with_kwargs=True)
        try:
            yield captured
        finally:
            handle.remove()

    def get_inputs_embeds(self, image_or_path, prompt: str) -> torch.Tensor:
        """Convenience wrapper: one forward pass, return the LM input embeddings."""
        inputs = self.prepare(image_or_path, prompt)
        with self.capture_inputs_embeds() as captured:
            with torch.no_grad():
                self.model(**inputs)
        if not captured:
            raise RuntimeError("forward pass did not reach the language model")
        return captured[0]

    @contextmanager
    def ablate_inputs(self, indices: Sequence[int], replacement_tensor: torch.Tensor):
        """Replace the embeddings at ``indices`` with ``replacement_tensor``.

        The prefill pass is patched; decode steps (sequence length 1) are left alone,
        matching the original implementation.
        """
        indices = list(indices)

        def hook(module, args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            if embeds is None or embeds.shape[-2] == 1 or not indices:
                return args, kwargs
            patched = embeds.clone()
            replacement = replacement_tensor.to(dtype=patched.dtype, device=patched.device)
            patched[:, indices, :] = replacement
            kwargs["inputs_embeds"] = patched
            return args, kwargs

        handle = self.language_model.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def block_attention(self, block_dict: dict[int, Sequence[tuple[int, int]]]):
        """Block attention pairs per layer.

        Args:
            block_dict: ``{layer_index: [(query_index, key_index), ...]}``.

        Raises:
            ValueError: if a layer index has no attention sublayer. On this hybrid
                backbone that would otherwise be a silent no-op.
        """
        layers = self.language_model.layers
        bad = [i for i in block_dict if not layers[i].is_attention_layer]
        if bad:
            raise ValueError(
                f"layers {bad} are short-conv layers with no attention to block; "
                f"attention layers are {self.attention_layers}"
            )

        handles = []
        try:
            for layer_index, pairs in block_dict.items():
                handles.append(
                    layers[layer_index].self_attn.register_forward_pre_hook(
                        BlockAttentionHook(pairs), with_kwargs=True
                    )
                )
            yield
        finally:
            for handle in handles:
                handle.remove()

    # ---------------------------------------------------------------- running

    def forward(self, image_or_path, prompt: str, **kwargs):
        inputs = self.prepare(image_or_path, prompt)
        with torch.no_grad():
            return self.model(**inputs, **kwargs)

    def generate(
        self,
        image_or_path,
        prompt: str,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        **kwargs,
    ) -> str:
        """Generate a continuation, returning only the newly generated text.

        ``do_sample`` defaults to False; the original defaults to True, which makes
        ablation comparisons noisy.
        """
        inputs = self.prepare(image_or_path, prompt)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def next_token_distribution(self, image_or_path, prompt: str) -> torch.Tensor:
        """Softmax over the vocabulary at the last position, for the VQA setting."""
        outputs = self.forward(image_or_path, prompt)
        return torch.softmax(outputs.logits[0, -1, :].float(), dim=-1)

    def hidden_states(self, image_or_path, prompt: str):
        """All layer hidden states plus the inputs, for the logit lens."""
        outputs = self.forward(image_or_path, prompt, output_hidden_states=True)
        return outputs.hidden_states

    def visual_token_features(self, image_or_path) -> torch.Tensor:
        """Post-projector visual tokens, ``(n_tokens, hidden_size)``.

        Vision tower + projector only, no LM -- used to accumulate the mean visual
        token for the ablation baseline.
        """
        image = _as_image(image_or_path)
        vision_inputs = self.processor.image_processor(
            [[image]], return_tensors="pt", do_image_splitting=self.do_image_splitting
        ).to(self.device)
        with torch.no_grad():
            features = self.model.model.get_image_features(
                pixel_values=vision_inputs["pixel_values"],
                spatial_shapes=vision_inputs["spatial_shapes"],
                pixel_attention_mask=vision_inputs["pixel_attention_mask"],
            )
        # transformers >= 5 returns a BaseModelOutputWithPooling whose pooler_output
        # holds the per-tile projected features; 4.x returned that list directly.
        if not isinstance(features, (list, tuple)):
            features = features.pooler_output
        return torch.cat(list(features), dim=0)


def layer_windows(attention_layers: Sequence[int], num_layers: int) -> dict[str, list[int]]:
    """The paper's five layer windows, mapped onto the attention layers they contain.

    The paper blocks over windows of consecutive layers (L1-10, L5-14, L11-20, L15-24,
    L21-31) on a 32-layer all-attention model. Here only ``attention_layers`` can be
    blocked, so each window becomes the attention layers falling inside it, with the
    window boundaries rescaled to this model's depth.
    """
    scale = num_layers / 32.0
    spans = {
        "early": (0, 10),
        "early_mid": (4, 14),
        "mid": (10, 20),
        "mid_late": (14, 24),
        "late": (20, 31),
    }
    windows = {
        name: [i for i in attention_layers if lo * scale <= i < hi * scale]
        for name, (lo, hi) in spans.items()
    }
    windows["all"] = list(attention_layers)
    return windows
