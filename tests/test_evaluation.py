import torch

from qrouter.preprocessing.generation import encode_generation_prompts


class RecordingTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.prompts = None

    def __call__(self, prompts, **kwargs):
        self.prompts = prompts
        assert kwargs["padding"] is True
        assert self.padding_side == "left"
        return {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones(2, 2, dtype=torch.long),
        }


def test_generation_prompts_contain_questions_but_not_answers():
    tokenizer = RecordingTokenizer()
    encoded = encode_generation_prompts(
        tokenizer=tokenizer,
        questions=["What color is the cup?", "How many dogs?"],
        template="Question: {question}\nAnswer:",
        max_length=128,
    )
    assert tokenizer.prompts == [
        "Question: What color is the cup?\nAnswer:",
        "Question: How many dogs?\nAnswer:",
    ]
    assert tokenizer.padding_side == "right"
    assert encoded["input_ids"].shape == (2, 2)
