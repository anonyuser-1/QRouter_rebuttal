from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - exercised in the full training environment
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None


class QuestionPromptAdapter(nn.Module):
    """Convert Qwen2.5-VL features into K independent SAM2 prompt queries.

    One query is decoded into one mask candidate. This avoids using SAM2's three
    native multimask alternatives as if they were a configurable region budget.
    """

    def __init__(
        self,
        model_id: str,
        sam_prompt_dim: int,
        num_prompt_queries: int = 32,
        attention_heads: int = 8,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("Install transformers with Qwen2.5-VL support.")
        if sam_prompt_dim % attention_heads != 0:
            raise ValueError("sam_prompt_dim must be divisible by attention_heads.")

        self.model_id = model_id
        self.num_prompt_queries = int(num_prompt_queries)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
        self.backbone.config.use_cache = False

        hidden_size = self._hidden_size(self.backbone)
        self.token_projection = nn.Linear(hidden_size, sam_prompt_dim)
        self.query_tokens = nn.Parameter(torch.empty(self.num_prompt_queries, sam_prompt_dim))
        self.query_attention = nn.MultiheadAttention(
            embed_dim=sam_prompt_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(sam_prompt_dim)
        self.dense_projection = nn.Sequential(
            nn.Linear(sam_prompt_dim, sam_prompt_dim),
            nn.SiLU(),
            nn.Linear(sam_prompt_dim, sam_prompt_dim),
        )
        self.relevance_head = nn.Linear(sam_prompt_dim, 1)
        self._initialize_lightweight_adapter()

        self.qwen_lora_enabled = bool(use_lora)
        if use_lora:
            if get_peft_model is None or LoraConfig is None:
                raise ImportError("Install peft to enable Qwen LoRA.")
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        self.set_stage("stage1")

    def _initialize_lightweight_adapter(self) -> None:
        """Initialize the learned cross-model adapter."""
        for module in [
            self.token_projection,
            *self.dense_projection,
        ]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.query_attention.in_proj_weight)
        if self.query_attention.in_proj_bias is not None:
            nn.init.zeros_(self.query_attention.in_proj_bias)
        nn.init.xavier_uniform_(self.query_attention.out_proj.weight)
        nn.init.zeros_(self.query_attention.out_proj.bias)
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.zeros_(self.relevance_head.weight)
        nn.init.constant_(self.relevance_head.bias, 2.0)

    @staticmethod
    def initialization_description() -> dict[str, str]:
        return {
            "token_projection": "Xavier uniform; zero bias",
            "query_tokens": "Normal(mean=0, std=0.02)",
            "query_attention": "Xavier uniform projections; zero bias",
            "dense_projection": "Xavier uniform; zero bias",
            "relevance_head": "zero weight; bias=2.0",
            "qwen_base": "pretrained checkpoint; frozen",
            "qwen_lora": "PEFT default LoRA initialization; enabled only in Stage II",
        }

    @staticmethod
    def _hidden_size(model: nn.Module) -> int:
        config = getattr(model, "config", None)
        candidates = [
            getattr(getattr(config, "text_config", None), "hidden_size", None),
            getattr(getattr(config, "language_config", None), "hidden_size", None),
            getattr(config, "hidden_size", None),
        ]
        for candidate in candidates:
            if candidate is not None:
                return int(candidate)
        embeddings = model.get_input_embeddings()
        return int(embeddings.embedding_dim)

    def set_stage(self, stage: str) -> None:
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported stage: {stage}")

        # Qwen base weights remain frozen in both stages.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if stage == "stage2" and self.qwen_lora_enabled:
            for name, parameter in self.backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True

        # The cross-model adapter is lightweight and must not be frozen at its
        # random initialization in Stage I.
        for module in [
            self.token_projection,
            self.query_attention,
            self.query_norm,
            self.dense_projection,
            self.relevance_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.query_tokens.requires_grad = True

    def _prepare_inputs(
        self,
        questions: list[str],
        image_paths: list[str],
    ) -> dict[str, torch.Tensor]:
        if len(questions) != len(image_paths):
            raise ValueError("questions and image_paths must have equal length.")
        rendered_prompts: list[str] = []
        images: list[Image.Image] = []
        for question, image_path in zip(questions, image_paths):
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(path)},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            rendered_prompts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images.append(Image.open(path).convert("RGB"))
        inputs = self.processor(
            text=rendered_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.backbone.parameters()).device
        return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def forward(
        self,
        questions: list[str],
        image_paths: list[str],
        dense_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self._prepare_inputs(questions, image_paths)
        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"].bool()
        projected = self.token_projection(hidden.to(self.token_projection.weight.dtype))

        queries = self.query_tokens.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        attended, _ = self.query_attention(
            query=queries,
            key=projected,
            value=projected,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        prompt_queries = self.query_norm(attended + queries)
        relevance_scores = torch.sigmoid(self.relevance_head(prompt_queries).squeeze(-1).float())

        weights = attention_mask.unsqueeze(-1).to(projected.dtype)
        pooled = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        dense_bias = self.dense_projection(pooled)
        dense_prompt = (
            dense_bias.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                dense_hw[0],
                dense_hw[1],
            )
        )
        return prompt_queries, dense_prompt, relevance_scores
