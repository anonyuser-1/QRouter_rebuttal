from __future__ import annotations

import torch

from qrouter.models.backbones.llm import MambaBackbone
from qrouter.models.backbones.vision import DualVisionBackbone
from qrouter.models.grounding import ConversationalGroundingModule
from qrouter.models.vlms import QRouterModel


def precision_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported precision: {name}")
    return mapping[name]


def build_model(
    config: dict,
    device: str | torch.device,
) -> QRouterModel:
    model_config = config["model"]
    optimization = config["optimization"]
    dtype = precision_dtype(optimization["precision"])
    vision = DualVisionBackbone(
        backbone_id=model_config["vision_backbone_id"],
    )
    language = MambaBackbone(
        model_id=model_config["llm_id"],
        max_length=int(model_config["max_length"]),
        use_lora=bool(model_config["use_mamba_lora"]),
        dtype=dtype,
    )
    grounding = ConversationalGroundingModule(
        sam2_config=model_config["sam2_config"],
        sam2_checkpoint=model_config["sam2_checkpoint"],
        qwen_id=model_config["qwen_id"],
        max_region_candidates=int(model_config["max_region_candidates"]),
        image_resolution=int(model_config["image_resolution"]),
        precision=dtype,
        device=device,
        use_qwen_lora=bool(model_config["use_qwen_lora"]),
    )
    model = QRouterModel(
        grounding_module=grounding,
        vision_backbone=vision,
        language_backbone=language,
        num_region_tokens=int(model_config["num_region_tokens"]),
        num_context_tokens=int(model_config["num_context_tokens"]),
        score_threshold=float(model_config["score_threshold"]),
        routing_hidden_dim=int(model_config["routing_hidden_dim"]),
        context_modulation_init=float(model_config["context_modulation_init"]),
        projector_hidden_dim=int(model_config["projector_hidden_dim"]),
        cis_dice_weight=float(optimization["cis_dice_weight"]),
    )
    return model.to(device)
