"""Prompt construction for LFM2.5-VL.

The paper's repo builds LLaVA prompts by hand::

    prompt = f"USER: <image>\\n{question} ASSISTANT:"

That does not transfer. LFM2.5-VL's chat template emits::

    <|startoftext|><|im_start|>user\\n<image>{question}<|im_end|>\\n<|im_start|>assistant\\n

and the processor then expands the single ``<image>`` placeholder into
``<|image_start|>`` + N image tokens + ``<|image_end|>``, where N depends on the image.
So we must go through ``apply_chat_template`` and must **never** write ``<image>``
into message text ourselves -- the template inserts it, and writing a second one makes
the processor demand a second image.
"""

from __future__ import annotations

DESCRIBE_QUESTION = "Describe this image."
VQA_PREFILL = "It is a"


def _user_message(question: str, with_image: bool = True) -> dict:
    content: list[dict] = []
    if with_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    return {"role": "user", "content": content}


def build_prompt(processor, question: str, prefill: str | None = None, with_image: bool = True) -> str:
    """Render a single-turn prompt, optionally prefilling the assistant's reply.

    Args:
        prefill: text to append after the assistant header, e.g. ``"It is a"`` for the
            paper's VQA setting, where the next generated token is the answer.
    """
    text = processor.apply_chat_template(
        [_user_message(question, with_image=with_image)],
        add_generation_prompt=True,
        tokenize=False,
    )
    if prefill:
        text += prefill
    return text


def describe_prompt(processor) -> str:
    """Generative setting: "Describe this image." (paper Section 3, method 1)."""
    return build_prompt(processor, DESCRIBE_QUESTION)


def polling_prompt(processor, class_name: str) -> str:
    """Polling setting: "Is there a [o] in this image?" (paper Section 3, method 2)."""
    return build_prompt(processor, f"Is there a {class_name} in this image?")


def vqa_prompt(processor, question: str, prefill: str = VQA_PREFILL) -> str:
    """VQA setting: a curated question with "It is a" prefilled (paper Section 3, method 3)."""
    return build_prompt(processor, question, prefill=prefill)


def first_answer_token_id(processor, class_name: str) -> int:
    """Token id the model should emit first when answering with ``class_name``.

    After the ``"It is a"`` prefill the continuation starts with a space, so we tokenize
    ``" dog"`` rather than ``"dog"`` -- these are different tokens in a byte-level BPE.
    """
    ids = processor.tokenizer.encode(f" {class_name.lower()}", add_special_tokens=False)
    if not ids:
        raise ValueError(f"class name {class_name!r} tokenized to nothing")
    return ids[0]


def answer_matches_class(processor, generated_token_id: int, class_name: str) -> bool:
    """Whether a generated token counts as naming ``class_name``.

    The original ``format_and_compare_answers`` had its comparison inverted
    (``class_token.startswith(generated_answer)``); this checks the intended direction,
    accepting the answer when the generated token is the first token of the class name
    or, for multi-word classes, of its first word.
    """
    if generated_token_id == first_answer_token_id(processor, class_name):
        return True
    first_word = class_name.lower().split()[0]
    return generated_token_id == first_answer_token_id(processor, first_word)


def first_token_is_ambiguous(processor, class_name: str, min_chars: int = 3) -> bool:
    """Whether the class's first token is too short to identify it on its own.

    The VQA and knockout settings judge correctness from a *single* generated token, as
    the paper does. That is fine for "dog" -> ' dog', but "teddy bear" tokenizes to
    ' t', which any of "tie", "train" or "toilet" would also produce. Such images are
    still evaluated -- excluding them would bias the sample -- but they are flagged so
    the analysis can report how much of the measurement rests on ambiguous matches.
    """
    token = processor.tokenizer.decode([first_answer_token_id(processor, class_name)])
    return len(token.strip()) < min_chars and len(token.strip()) < len(class_name.strip())


def mentions_class(answer: str, class_name: str) -> bool:
    """Generative-setting check: does the description name the object?

    Matches the paper's ``class_name in answer`` test, but case-insensitively so that
    a sentence-initial "Dog" still counts.
    """
    return class_name.lower() in answer.lower()


def says_yes(answer: str) -> bool:
    """Polling-setting check, matching the paper's ``"yes" in answer.lower()``."""
    return "yes" in answer.lower()
