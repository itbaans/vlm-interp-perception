"""A tiny random-weight LFM2-VL, so every hook can be exercised on CPU in seconds.

The real checkpoint is ~6.5 GB and needs a GPU to be useful. But the things most likely
to break in this port -- module paths, image-token span arithmetic, the additive-vs-boolean
attention mask, hooking a conv layer by mistake -- do not depend on the weights at all.
So we build a structurally faithful miniature: the same ``Lfm2VlForConditionalGeneration``
class, the same hybrid ``layer_types`` pattern, the same real processor and image-token id,
just narrow and shallow with random weights.

Only the processor (tokenizer + image-processor config, ~10 MB) is downloaded.
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, Lfm2VlConfig, Lfm2VlForConditionalGeneration

from hooked_lfm2vl import DEFAULT_MODEL_ID, HookedLFM2VL

# Same conv/attention alternation as the real model, just 6 layers instead of 30.
# Attention lands at indices 2 and 5, so tests can cover both a hookable layer and
# a conv layer that must be refused.
TINY_LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "conv", "full_attention"]


def build_tiny_config(processor, vocab_size: int | None = None) -> Lfm2VlConfig:
    """A shrunken but structurally faithful LFM2-VL config."""
    if vocab_size is None:
        # Must exceed every token id the real tokenizer can emit.
        vocab_size = 128000

    text_config = {
        "model_type": "lfm2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": len(TINY_LAYER_TYPES),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "layer_types": list(TINY_LAYER_TYPES),
        "conv_L_cache": 3,
        "conv_bias": False,
        "norm_eps": 1e-5,
        "vocab_size": vocab_size,
        "tie_word_embeddings": True,
        "max_position_embeddings": 4096,
        "block_multiple_of": 16,
        "block_dim": 64,
        "conv_dim": 64,
    }
    vision_config = {
        "model_type": "siglip2_vision_model",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "patch_size": 16,
        "num_patches": 256,
        "vision_use_head": False,
    }
    return Lfm2VlConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=processor.image_token_id,
        downsample_factor=2,
        projector_hidden_size=64,
        projector_hidden_act="gelu",
        projector_bias=True,
        tie_word_embeddings=True,
    )


def build_tiny_hooked_model(
    model_id: str = DEFAULT_MODEL_ID,
    attn_implementation: str = "eager",
    seed: int = 0,
) -> HookedLFM2VL:
    """A ``HookedLFM2VL`` wrapping a random-weight miniature, on CPU."""
    torch.manual_seed(seed)
    processor = AutoProcessor.from_pretrained(model_id)
    config = build_tiny_config(processor)
    config._attn_implementation = attn_implementation
    model = Lfm2VlForConditionalGeneration(config)
    model.to(torch.float32).eval()
    return HookedLFM2VL(
        model_id=model_id,
        device="cpu",
        model=model,
        processor=processor,
        do_image_splitting=False,
    )
