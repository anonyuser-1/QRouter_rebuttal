from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize


VISION_BACKBONES = {
    "dinosiglip-vit-so-384px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "siglip": "vit_so400m_patch14_siglip_384",
        "image_size": 384,
    }
}


def _unpack_first(function):
    def wrapper(*args, **kwargs):
        output = function(*args, **kwargs)
        return output[0] if isinstance(output, (tuple, list)) else output

    return wrapper


def _strip_prefix_tokens(tokens: torch.Tensor, expected_patches: int) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,D] patch tokens, got {tuple(tokens.shape)}.")
    extra = tokens.shape[1] - expected_patches
    if extra < 0:
        raise ValueError("Vision backbone returned fewer tokens than its patch grid.")
    if extra > 0:
        tokens = tokens[:, extra:]
    return tokens


@dataclass
class DualVisionTransform:
    dino_transform: Compose
    siglip_transform: Compose

    def __call__(self, image: Image.Image) -> dict[str, torch.Tensor]:
        return {
            "dino": self.dino_transform(image),
            "siglip": self.siglip_transform(image),
        }


class DualVisionBackbone(nn.Module):
    def __init__(
        self,
        backbone_id: str = "dinosiglip-vit-so-384px",
    ) -> None:
        super().__init__()
        if backbone_id not in VISION_BACKBONES:
            raise ValueError(f"Unsupported vision backbone: {backbone_id}")
        specification = VISION_BACKBONES[backbone_id]
        self.identifier = backbone_id
        self.image_size = int(specification["image_size"])

        self.dino = timm.create_model(
            specification["dino"],
            pretrained=True,
            num_classes=0,
            img_size=self.image_size,
        )
        self.siglip = timm.create_model(
            specification["siglip"],
            pretrained=True,
            num_classes=0,
            img_size=self.image_size,
        )
        self.dino.forward = _unpack_first(
            partial(
                self.dino.get_intermediate_layers,
                n={len(self.dino.blocks) - 2},
            )
        )
        self.siglip.forward = _unpack_first(
            partial(
                self.siglip.get_intermediate_layers,
                n={len(self.siglip.blocks) - 2},
            )
        )

        dino_config = timm.data.resolve_model_data_config(self.dino)
        siglip_config = timm.data.resolve_model_data_config(self.siglip)
        dino_config["input_size"] = (3, self.image_size, self.image_size)
        siglip_config["input_size"] = (3, self.image_size, self.image_size)
        dino_transform = timm.data.create_transform(
            **dino_config,
            is_training=False,
        )
        siglip_transform = timm.data.create_transform(
            **siglip_config,
            is_training=False,
        )
        self.image_transform = DualVisionTransform(
            dino_transform=Compose(
                [
                    Resize(
                        (self.image_size, self.image_size),
                        interpolation=dino_transform.transforms[0].interpolation,
                    ),
                    *dino_transform.transforms[1:],
                ]
            ),
            siglip_transform=Compose(
                [
                    Resize(
                        (self.image_size, self.image_size),
                        interpolation=siglip_transform.transforms[0].interpolation,
                    ),
                    *siglip_transform.transforms[1:],
                ]
            ),
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @property
    def embed_dim(self) -> int:
        return int(self.dino.embed_dim + self.siglip.embed_dim)

    @property
    def num_patches(self) -> int:
        dino_count = int(self.dino.patch_embed.num_patches)
        siglip_count = int(self.siglip.patch_embed.num_patches)
        if dino_count != siglip_count:
            raise RuntimeError("DINOv2 and SigLIP patch grids do not match.")
        return dino_count

    @property
    def patch_hw(self) -> tuple[int, int]:
        grid_size = self.dino.patch_embed.grid_size
        if isinstance(grid_size, int):
            return grid_size, grid_size
        return int(grid_size[0]), int(grid_size[1])

    def train(self, mode: bool = True):
        # Frozen encoders stay in evaluation mode throughout training.
        super().train(False)
        self.dino.eval()
        self.siglip.eval()
        return self

    def forward(
        self,
        pixel_values: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | tuple[int, int]]:
        with torch.no_grad():
            dino_tokens = _strip_prefix_tokens(
                self.dino(pixel_values["dino"]),
                self.num_patches,
            )
            siglip_tokens = _strip_prefix_tokens(
                self.siglip(pixel_values["siglip"]),
                self.num_patches,
            )
        return {
            "patch_tokens": torch.cat([dino_tokens, siglip_tokens], dim=-1),
            "patch_hw": self.patch_hw,
            "dino_tokens": dino_tokens,
            "siglip_tokens": siglip_tokens,
        }
