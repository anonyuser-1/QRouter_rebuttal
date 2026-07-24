from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from qrouter.models.grounding.prompt_adapter import QuestionPromptAdapter

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:  # pragma: no cover - exercised in the full training environment
    build_sam2 = None
    SAM2ImagePredictor = None


@dataclass
class GroundingOutput:
    masks: torch.Tensor
    scores: torch.Tensor
    low_resolution_logits: torch.Tensor
    query_relevance_scores: torch.Tensor
    sam_quality_scores: torch.Tensor


class ConversationalGroundingModule(nn.Module):
    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        qwen_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_region_candidates: int = 32,
        image_resolution: int = 384,
        precision: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        use_qwen_lora: bool = True,
    ) -> None:
        super().__init__()
        if build_sam2 is None or SAM2ImagePredictor is None:
            raise ImportError("Install the official SAM2 package before constructing QRouter.")
        self.max_region_candidates = int(max_region_candidates)
        self.image_resolution = int(image_resolution)
        self.device = torch.device(device)
        self.sam_model = build_sam2(
            sam2_config,
            sam2_checkpoint,
            device=str(self.device),
        )
        # Register SAM2 directly as a child nn.Module so its mask-decoder
        # parameters are visible to optimizers, checkpoints, and diagnostics.
        self.predictor = SAM2ImagePredictor(self.sam_model)
        self.sam_model.eval()
        prompt_dim = int(self.sam_model.sam_mask_decoder.transformer_dim)
        self.prompt_adapter = QuestionPromptAdapter(
            model_id=qwen_id,
            sam_prompt_dim=prompt_dim,
            num_prompt_queries=self.max_region_candidates,
            use_lora=use_qwen_lora,
            dtype=precision,
            device=self.device,
        )
        self.stage = "stage1"
        self.set_stage("stage1")

    def set_stage(self, stage: str) -> None:
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported stage: {stage}")
        self.stage = stage
        for parameter in self.sam_model.parameters():
            parameter.requires_grad = False
        self.prompt_adapter.set_stage(stage)
        if stage == "stage2":
            # The image encoder remains frozen. The mask decoder is the
            # lightweight SAM2 component adapted with CIS supervision.
            for parameter in self.sam_model.sam_mask_decoder.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Heavy SAM2 components stay deterministic. Only the Stage-II mask
        # decoder is placed in training mode.
        self.sam_model.eval()
        if mode and self.stage == "stage2":
            self.sam_model.sam_mask_decoder.train(True)
        if self.stage == "stage1":
            self.prompt_adapter.backbone.eval()
        return self

    def _load_rgb(self, path: str) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        image = image.resize(
            (self.image_resolution, self.image_resolution),
            resample=Image.Resampling.BILINEAR,
        )
        return np.asarray(image)

    @staticmethod
    def _sam_quality_to_probability(scores: torch.Tensor) -> torch.Tensor:
        # SAM2 quality predictions are not guaranteed to be bounded.
        return torch.sigmoid(scores.float())

    def forward(
        self,
        image_paths: list[str],
        questions: list[str],
    ) -> GroundingOutput:
        if len(image_paths) != len(questions):
            raise ValueError("image_paths and questions must have equal length.")

        cached_features: list[tuple[torch.Tensor, list[torch.Tensor], tuple[int, int]]] = []
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp_context:
            for image_path in image_paths:
                self.predictor.set_image(self._load_rgb(image_path))
                image_embedding = self.predictor._features["image_embed"][-1].unsqueeze(0).detach().clone()
                high_resolution = [
                    feature[-1].unsqueeze(0).detach().clone() for feature in self.predictor._features["high_res_feats"]
                ]
                original_hw = tuple(self.predictor._orig_hw[-1])
                cached_features.append((image_embedding, high_resolution, original_hw))

            dense_hw = tuple(cached_features[0][0].shape[-2:])
            prompt_queries, dense_prompts, query_scores = self.prompt_adapter(
                questions=questions,
                image_paths=image_paths,
                dense_hw=dense_hw,
            )

            output_masks: list[torch.Tensor] = []
            output_low_resolution: list[torch.Tensor] = []
            output_sam_scores: list[torch.Tensor] = []
            for batch_index, (image_embedding, high_resolution, original_hw) in enumerate(cached_features):
                decoder = self.sam_model.sam_mask_decoder
                decoder_parameter = next(decoder.parameters())
                decoder_device = decoder_parameter.device
                decoder_dtype = decoder_parameter.dtype
                candidate_count = prompt_queries.shape[1]

                # Each question-conditioned query is decoded as a separate
                # SAM2 prompt with a single output mask.
                sparse_prompt = prompt_queries[batch_index].unsqueeze(1)
                dense_prompt = dense_prompts[batch_index : batch_index + 1].expand(
                    candidate_count,
                    -1,
                    -1,
                    -1,
                )
                image_pe = self.sam_model.sam_prompt_encoder.get_dense_pe()
                low_resolution, sam_scores, _, _ = decoder(
                    image_embeddings=image_embedding.to(decoder_device, decoder_dtype),
                    image_pe=image_pe.to(decoder_device, decoder_dtype),
                    sparse_prompt_embeddings=sparse_prompt.to(decoder_device, decoder_dtype),
                    dense_prompt_embeddings=dense_prompt.to(decoder_device, decoder_dtype),
                    multimask_output=False,
                    repeat_image=True,
                    high_res_features=[feature.to(decoder_device, decoder_dtype) for feature in high_resolution],
                )
                full_resolution = self.predictor._transforms.postprocess_masks(
                    low_resolution,
                    original_hw,
                )
                output_masks.append(torch.sigmoid(full_resolution[:, 0]))
                output_low_resolution.append(low_resolution[:, 0])
                output_sam_scores.append(sam_scores[:, 0])

        masks = torch.stack(output_masks, dim=0)
        low_resolution_logits = torch.stack(output_low_resolution, dim=0)
        sam_quality_scores = self._sam_quality_to_probability(torch.stack(output_sam_scores, dim=0))
        # A retained candidate must be both question-relevant and segmentable.
        confidence_scores = torch.sqrt((query_scores.float() * sam_quality_scores.float()).clamp_min(0.0)).clamp(
            0.0, 1.0
        )
        return GroundingOutput(
            masks=masks,
            scores=confidence_scores,
            low_resolution_logits=low_resolution_logits,
            query_relevance_scores=query_scores,
            sam_quality_scores=sam_quality_scores,
        )
