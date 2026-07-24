from __future__ import annotations

import torch
import torch.nn as nn


class MultimodalProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, language_dim),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(visual_tokens)
