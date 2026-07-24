from __future__ import annotations

from torch.utils.data import ConcatDataset

from qrouter.preprocessing.datasets import ManifestDataset


def get_dataset(manifests: list[str], task: str, model, config: dict) -> ConcatDataset:
    evaluation = config["evaluation"]
    model_config = config["model"]
    return ConcatDataset(
        [
            ManifestDataset(
                manifest_paths=[manifest],
                tokenizer=model.language_backbone.tokenizer,
                vision_transform=model.vision_backbone.image_transform,
                prompt_template=evaluation["prompt_template"],
                image_resolution=int(model_config["image_resolution"]),
                expected_task=task,
                max_length=int(model_config["max_length"]),
            )
            for manifest in manifests
        ]
    )
