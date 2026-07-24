from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from qrouter.util.nn_utils import batched_index_select


@dataclass
class RegionTokenizerOutput:
    visual_tokens: torch.Tensor
    visual_valid_mask: torch.Tensor
    visual_type_ids: torch.Tensor
    region_tokens: torch.Tensor
    context_tokens: torch.Tensor
    background_token: torch.Tensor
    patch_prior: torch.Tensor
    routing_scores: torch.Tensor
    routing_weights: torch.Tensor
    region_valid_mask: torch.Tensor
    patch_hw: tuple[int, int]


class MaskDownsampler(nn.Module):
    def forward(
        self,
        masks: torch.Tensor,
        patch_hw: tuple[int, int],
    ) -> torch.Tensor:
        if masks.ndim != 4:
            raise ValueError("masks must have shape [B,K0,H,W].")
        batch_size, num_masks = masks.shape[:2]
        flattened = masks.reshape(
            batch_size * num_masks,
            1,
            masks.shape[-2],
            masks.shape[-1],
        )
        pooled = F.adaptive_avg_pool2d(flattened.float(), patch_hw)
        return pooled.reshape(batch_size, num_masks, -1).to(masks.dtype)


def build_noisy_or_prior(
    mask_grid: torch.Tensor,
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Confidence-weighted noisy-OR from Eqs. (3) and (23)."""
    if mask_grid.ndim != 3 or scores.ndim != 2:
        raise ValueError("Expected mask_grid [B,K,N] and scores [B,K].")
    if mask_grid.shape[:2] != scores.shape:
        raise ValueError("mask_grid and scores disagree on [B,K].")
    support = mask_grid.float().clamp(0.0, 1.0) * scores.float().clamp(0.0, 1.0).unsqueeze(-1)
    if valid_mask is not None:
        support = support * valid_mask.float().unsqueeze(-1)
    prior = 1.0 - torch.prod(1.0 - support.clamp(max=1.0 - 1e-6), dim=1)
    return prior.clamp(0.0, 1.0).to(mask_grid.dtype)


class CandidateRetainer(nn.Module):
    def __init__(self, num_regions: int, score_threshold: float) -> None:
        super().__init__()
        self.num_regions = int(num_regions)
        self.score_threshold = float(score_threshold)

    def forward(
        self,
        mask_grid: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, raw_count, num_patches = mask_grid.shape
        scores = scores.float().clamp(0.0, 1.0)
        nonempty = mask_grid.float().sum(dim=-1) > 1e-6
        candidate_valid = (scores >= self.score_threshold) & nonempty

        retained_count = min(self.num_regions, raw_count)
        ranking_scores = scores.masked_fill(~candidate_valid, -1.0)
        _, indices = torch.topk(ranking_scores, k=retained_count, dim=-1)
        retained_masks = batched_index_select(mask_grid, indices)
        retained_scores = torch.gather(scores, dim=1, index=indices)
        retained_valid = torch.gather(candidate_valid, dim=1, index=indices)

        if retained_count < self.num_regions:
            pad = self.num_regions - retained_count
            retained_masks = F.pad(retained_masks, (0, 0, 0, pad))
            retained_scores = F.pad(retained_scores, (0, pad))
            retained_valid = F.pad(retained_valid, (0, pad), value=False)

        retained_masks = retained_masks * retained_valid.unsqueeze(-1).to(retained_masks.dtype)
        retained_scores = retained_scores * retained_valid.to(retained_scores.dtype)
        return retained_masks, retained_scores, retained_valid


class RegionPooler(nn.Module):
    def __init__(self, vision_dim: int, geometry_hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = geometry_hidden_dim or min(vision_dim, 512)
        self.region_projection = nn.Linear(vision_dim, vision_dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vision_dim),
        )

    @staticmethod
    def _geometry(
        mask_grid: torch.Tensor,
        patch_hw: tuple[int, int],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_masks, _ = mask_grid.shape
        height, width = patch_hw
        masks_2d = mask_grid.reshape(batch_size, num_masks, height, width)
        geometry = mask_grid.new_zeros(batch_size, num_masks, 5)
        y_coords = torch.linspace(0.0, 1.0, height, device=mask_grid.device)
        x_coords = torch.linspace(0.0, 1.0, width, device=mask_grid.device)

        for batch_index in range(batch_size):
            for mask_index in range(num_masks):
                if not bool(valid_mask[batch_index, mask_index]):
                    continue
                mask = masks_2d[batch_index, mask_index]
                coordinates = torch.nonzero(mask > 0.05, as_tuple=False)
                if coordinates.numel() == 0:
                    continue
                y = coordinates[:, 0]
                x = coordinates[:, 1]
                geometry[batch_index, mask_index] = torch.stack(
                    [
                        x_coords[x.min()],
                        y_coords[y.min()],
                        x_coords[x.max()],
                        y_coords[y.max()],
                        mask.mean(),
                    ]
                )
        return geometry

    def forward(
        self,
        patch_tokens: torch.Tensor,
        mask_grid: torch.Tensor,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        patch_hw: tuple[int, int],
    ) -> torch.Tensor:
        weighted_masks = mask_grid.float() * scores.float().unsqueeze(-1) * valid_mask.float().unsqueeze(-1)
        normalized = weighted_masks / weighted_masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum(
            "bkn,bnd->bkd",
            normalized.to(patch_tokens.dtype),
            patch_tokens,
        )
        geometry = self.geometry_mlp(self._geometry(mask_grid, patch_hw, valid_mask).to(patch_tokens.dtype))
        tokens = self.region_projection(pooled) + geometry
        return tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)


class RoutingScorer(nn.Module):
    def __init__(self, vision_dim: int, question_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.visual_projection = nn.Linear(vision_dim, hidden_dim)
        self.prior_projection = nn.Linear(1, hidden_dim)
        self.question_projection = nn.Linear(question_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_prior: torch.Tensor,
        question_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.tanh(
            self.visual_projection(patch_tokens)
            + self.prior_projection(patch_prior.unsqueeze(-1))
            + self.question_projection(question_embedding).unsqueeze(1)
        )
        return self.output_projection(hidden).squeeze(-1)


class BackgroundPooler(nn.Module):
    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_prior: torch.Tensor,
        selected_patch_mask_st: torch.Tensor,
    ) -> torch.Tensor:
        # The forward value follows Eq. (8)/(37). The straight-through
        # selection mask supplies gradients to the discrete routing scorer.
        weights = (1.0 - patch_prior.float()) * (1.0 - selected_patch_mask_st.float())
        denominator = weights.sum(dim=-1, keepdim=True) + 1e-6
        normalized = weights / denominator
        return torch.einsum(
            "bn,bnd->bd",
            normalized.to(patch_tokens.dtype),
            patch_tokens,
        )


class RegionTokenizer(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        question_dim: int,
        num_region_tokens: int = 32,
        num_context_tokens: int = 128,
        score_threshold: float = 0.5,
        routing_hidden_dim: int = 1024,
        context_modulation_init: float = 1.0,
    ) -> None:
        super().__init__()
        if num_region_tokens <= 0 or num_context_tokens <= 0:
            raise ValueError("Region and context budgets must be positive.")
        self.num_region_tokens = int(num_region_tokens)
        self.num_context_tokens = int(num_context_tokens)
        self.downsampler = MaskDownsampler()
        self.retainer = CandidateRetainer(
            num_regions=num_region_tokens,
            score_threshold=score_threshold,
        )
        self.region_pooler = RegionPooler(vision_dim=vision_dim)
        self.routing_scorer = RoutingScorer(
            vision_dim=vision_dim,
            question_dim=question_dim,
            hidden_dim=routing_hidden_dim,
        )
        self.context_modulation = nn.Linear(vision_dim, vision_dim, bias=False)
        self.context_scale = nn.Parameter(torch.tensor(float(context_modulation_init)))
        self.background_pooler = BackgroundPooler()

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        masks: torch.Tensor,
        scores: torch.Tensor,
        question_embedding: torch.Tensor,
    ) -> RegionTokenizerOutput:
        batch_size, num_patches, _ = patch_tokens.shape
        if num_patches != patch_hw[0] * patch_hw[1]:
            raise ValueError("patch_hw does not match the number of patch tokens.")

        raw_mask_grid = self.downsampler(masks, patch_hw)

        mask_grid, retained_scores, region_valid_mask = self.retainer(raw_mask_grid, scores)
        patch_prior = build_noisy_or_prior(
            mask_grid,
            retained_scores,
            region_valid_mask,
        )
        region_tokens = self.region_pooler(
            patch_tokens=patch_tokens,
            mask_grid=mask_grid,
            scores=retained_scores,
            valid_mask=region_valid_mask,
            patch_hw=patch_hw,
        )

        routing_scores = self.routing_scorer(
            patch_tokens=patch_tokens,
            patch_prior=patch_prior,
            question_embedding=question_embedding,
        ).float()
        routing_weights = torch.softmax(routing_scores, dim=-1)

        selected_count = min(self.num_context_tokens, num_patches)
        selected_indices = torch.topk(
            routing_scores,
            k=selected_count,
            dim=-1,
        ).indices
        selected_patches = batched_index_select(patch_tokens, selected_indices)
        selected_prior = torch.gather(patch_prior, dim=1, index=selected_indices)

        # Eq. (7)/(35) in the forward pass.
        modulated_context = selected_patches + (
            self.context_scale.to(selected_patches.dtype)
            * selected_prior.unsqueeze(-1).to(selected_patches.dtype)
            * self.context_modulation(selected_patches)
        )

        # Hard Top-L has no derivative through its indices. This unit-valued
        # straight-through factor preserves Eq. (7) numerically while allowing
        # answer-loss gradients to train the RoutingScorer.
        selected_probabilities = torch.gather(
            routing_weights,
            dim=1,
            index=selected_indices,
        )
        straight_through_scale = 1.0 + selected_probabilities - selected_probabilities.detach()
        context_tokens = modulated_context * straight_through_scale.unsqueeze(-1).to(modulated_context.dtype)

        hard_selected_mask = torch.zeros_like(routing_weights)
        hard_selected_mask.scatter_(1, selected_indices, 1.0)
        selected_mask_st = hard_selected_mask + routing_weights - routing_weights.detach()
        background_token = self.background_pooler(
            patch_tokens=patch_tokens,
            patch_prior=patch_prior,
            selected_patch_mask_st=selected_mask_st,
        ).unsqueeze(1)

        context_valid_mask = torch.ones(
            batch_size,
            selected_count,
            dtype=torch.bool,
            device=patch_tokens.device,
        )
        if selected_count < self.num_context_tokens:
            pad_count = self.num_context_tokens - selected_count
            context_tokens = F.pad(context_tokens, (0, 0, 0, pad_count))
            selected_indices = F.pad(selected_indices, (0, pad_count), value=-1)
            context_valid_mask = F.pad(context_valid_mask, (0, pad_count), value=False)

        background_valid_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=patch_tokens.device,
        )
        visual_valid_mask = torch.cat(
            [region_valid_mask, context_valid_mask, background_valid_mask],
            dim=1,
        )
        visual_type_ids = (
            torch.cat(
                [
                    torch.zeros(self.num_region_tokens, dtype=torch.long, device=patch_tokens.device),
                    torch.ones(self.num_context_tokens, dtype=torch.long, device=patch_tokens.device),
                    torch.full((1,), 2, dtype=torch.long, device=patch_tokens.device),
                ],
                dim=0,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        visual_tokens = torch.cat(
            [region_tokens, context_tokens, background_token],
            dim=1,
        )
        visual_tokens = visual_tokens * visual_valid_mask.unsqueeze(-1).to(visual_tokens.dtype)
        return RegionTokenizerOutput(
            visual_tokens=visual_tokens,
            visual_valid_mask=visual_valid_mask,
            visual_type_ids=visual_type_ids,
            region_tokens=region_tokens,
            context_tokens=context_tokens,
            background_token=background_token,
            patch_prior=patch_prior,
            routing_scores=routing_scores,
            routing_weights=routing_weights,
            region_valid_mask=region_valid_mask,
            patch_hw=patch_hw,
        )
