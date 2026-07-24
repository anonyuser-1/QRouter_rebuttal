from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from qrouter.models.backbones.llm import MambaBackbone
from qrouter.models.backbones.vision import DualVisionBackbone
from qrouter.models.grounding import ConversationalGroundingModule
from qrouter.models.vlms.losses import cis_patch_prior_loss, mask_to_patch_target
from qrouter.models.vlms.projector import MultimodalProjector
from qrouter.models.vlms.region_tokenizer import (
    RegionTokenizer,
    RegionTokenizerOutput,
)


IGNORE_INDEX = -100


@dataclass
class QRouterOutput:
    loss: torch.Tensor
    logits: torch.Tensor | None
    metrics: dict[str, torch.Tensor]
    region_output: RegionTokenizerOutput


class QRouterModel(nn.Module):
    def __init__(
        self,
        grounding_module: ConversationalGroundingModule,
        vision_backbone: DualVisionBackbone,
        language_backbone: MambaBackbone,
        num_region_tokens: int = 32,
        num_context_tokens: int = 128,
        score_threshold: float = 0.5,
        routing_hidden_dim: int = 1024,
        context_modulation_init: float = 1.0,
        projector_hidden_dim: int = 2048,
        cis_dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.grounding_module = grounding_module
        self.vision_backbone = vision_backbone
        self.language_backbone = language_backbone
        self.region_tokenizer = RegionTokenizer(
            vision_dim=vision_backbone.embed_dim,
            question_dim=language_backbone.embed_dim,
            num_region_tokens=num_region_tokens,
            num_context_tokens=num_context_tokens,
            score_threshold=score_threshold,
            routing_hidden_dim=routing_hidden_dim,
            context_modulation_init=context_modulation_init,
        )
        self.projector = MultimodalProjector(
            vision_dim=vision_backbone.embed_dim,
            language_dim=language_backbone.embed_dim,
            hidden_dim=projector_hidden_dim,
        )
        self.visual_type_embedding = nn.Embedding(3, language_backbone.embed_dim)
        self.cis_dice_weight = float(cis_dice_weight)
        self.stage = "stage1"
        self.set_stage(self.stage)

    def set_stage(self, stage: str) -> None:
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported stage: {stage}")
        self.stage = stage
        self.grounding_module.set_stage(stage)
        for parameter in self.vision_backbone.parameters():
            parameter.requires_grad = False
        for module in [
            self.region_tokenizer,
            self.projector,
            self.visual_type_embedding,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _visual_prefix(
        self,
        batch: dict[str, Any],
        grounding_questions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, RegionTokenizerOutput]:
        vision_output = self.vision_backbone(batch["pixel_values"])
        question_embedding = self.language_backbone.encode_question(
            batch["question_input_ids"],
            batch["question_attention_mask"],
        )
        grounding_output = self.grounding_module(
            image_paths=batch["image_paths"],
            questions=grounding_questions,
        )
        region_output = self.region_tokenizer(
            patch_tokens=vision_output["patch_tokens"],
            patch_hw=vision_output["patch_hw"],
            masks=grounding_output.masks,
            scores=grounding_output.scores,
            question_embedding=question_embedding,
        )
        projected = self.projector(region_output.visual_tokens)
        typed = projected + self.visual_type_embedding(region_output.visual_type_ids)
        typed = typed * region_output.visual_valid_mask.unsqueeze(-1).to(typed.dtype)
        return typed, region_output.visual_valid_mask, region_output

    def _build_qa_inputs(
        self,
        visual_prefix: torch.Tensor,
        visual_valid_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_embeddings = self.language_backbone.embed_input_ids(input_ids)
        inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)
        full_attention = torch.cat(
            [visual_valid_mask.to(attention_mask.dtype), attention_mask],
            dim=1,
        )
        visual_labels = torch.full(
            visual_valid_mask.shape,
            IGNORE_INDEX,
            dtype=labels.dtype,
            device=labels.device,
        )
        full_labels = torch.cat([visual_labels, labels], dim=1)
        return inputs_embeds, full_attention, full_labels

    def forward(
        self,
        batch: dict[str, Any],
    ) -> QRouterOutput:
        grounding_questions = list(batch.get("grounding_questions", batch["questions"]))

        visual_prefix, visual_valid_mask, region_output = self._visual_prefix(
            batch=batch,
            grounding_questions=grounding_questions,
        )
        zero = visual_prefix.float().sum() * 0.0
        total_loss = zero
        logits = None
        has_supervision = False
        metrics: dict[str, torch.Tensor] = {}

        if "input_ids" in batch and "labels" in batch:
            has_supervision = True
            inputs_embeds, attention_mask, labels = self._build_qa_inputs(
                visual_prefix=visual_prefix,
                visual_valid_mask=visual_valid_mask,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            language_output = self.language_backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            if language_output.loss is None:
                raise RuntimeError("The language backbone did not return a QA loss.")
            total_loss = total_loss + language_output.loss
            logits = language_output.logits
            metrics["answer_loss"] = language_output.loss.detach()

        if "ground_truth_masks" in batch:
            has_supervision = True
            target_prior = mask_to_patch_target(
                batch["ground_truth_masks"],
                patch_hw=region_output.patch_hw,
            )
            cis_loss, cis_metrics = cis_patch_prior_loss(
                predicted_prior=region_output.patch_prior,
                target_prior=target_prior,
                dice_weight=self.cis_dice_weight,
                valid_samples=batch.get("has_ground_truth_mask"),
            )
            total_loss = total_loss + cis_loss
            metrics.update(cis_metrics)
            metrics["cis_loss"] = cis_loss.detach()

        if not has_supervision:
            raise RuntimeError("Batch contains neither QA labels nor CIS supervision.")
        return QRouterOutput(
            loss=total_loss,
            logits=logits,
            metrics=metrics,
            region_output=region_output,
        )

    @torch.no_grad()
    def generate(
        self,
        batch: dict[str, Any],
        max_new_tokens: int = 32,
    ) -> tuple[torch.Tensor, RegionTokenizerOutput]:
        grounding_questions = list(batch.get("grounding_questions", batch["questions"]))
        visual_prefix, visual_valid_mask, region_output = self._visual_prefix(
            batch,
            grounding_questions,
        )
        text_embeddings = self.language_backbone.embed_input_ids(batch["input_ids"])
        inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)
        attention_mask = torch.cat(
            [
                visual_valid_mask.to(batch["attention_mask"].dtype),
                batch["attention_mask"],
            ],
            dim=1,
        )
        generated = self.language_backbone.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
        return generated, region_output
