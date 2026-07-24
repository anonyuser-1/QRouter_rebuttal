from __future__ import annotations

import torch


def encode_generation_prompts(
    tokenizer,
    questions: list[str],
    template: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize question-only prompts for autoregressive evaluation."""
    prompts = [template.format(question=question) for question in questions]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_padding_side
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
