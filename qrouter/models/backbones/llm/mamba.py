from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class LanguageModelOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class MambaBackbone(nn.Module):
    def __init__(
        self,
        model_id: str = "xiuyul/mamba-2.8b-zephyr",
        max_length: int = 2048,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("Install transformers before constructing the Mamba backbone.")
        self.model_id = model_id
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            model_max_length=max_length,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        self.lora_enabled = bool(use_lora)
        if use_lora:
            if get_peft_model is None or LoraConfig is None:
                raise ImportError("Install peft to enable Mamba LoRA.")
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, config)
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad = True

    @property
    def embed_dim(self) -> int:
        return int(self.model.get_input_embeddings().embedding_dim)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def encode_question(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embed_input_ids(question_input_ids)
        weights = question_attention_mask.unsqueeze(-1).to(embeddings.dtype)
        return (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> LanguageModelOutput:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,
        )
        return LanguageModelOutput(loss=outputs.loss, logits=outputs.logits)

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 32,
    ) -> torch.Tensor:
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
