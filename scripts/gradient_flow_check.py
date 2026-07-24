from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qrouter.conf import load_config, validate_config
from qrouter.models import build_model
from qrouter.preprocessing import ManifestDataset, collate_manifest_batch
from qrouter.training import load_checkpoint
from qrouter.util import move_to_device


WATCHED_PREFIXES = {
    "question_to_sam_adapter": "grounding_module.prompt_adapter.",
    "qwen_lora": "grounding_module.prompt_adapter.backbone.",
    "sam_mask_decoder": "grounding_module.sam_model.sam_mask_decoder.",
    "routing_scorer": "region_tokenizer.routing_scorer.",
    "context_modulation": "region_tokenizer.context_modulation.",
    "multimodal_projector": "projector.",
    "mamba": "language_backbone.",
}


def gradient_report(model: torch.nn.Module) -> dict:
    report = {}
    named_parameters = list(model.named_parameters())
    for label, prefix in WATCHED_PREFIXES.items():
        eligible = [
            (name, parameter)
            for name, parameter in named_parameters
            if name.startswith(prefix) and parameter.requires_grad
        ]
        with_gradient = [(name, parameter) for name, parameter in eligible if parameter.grad is not None]
        gradient_norm = sum(float(parameter.grad.detach().float().norm()) for _, parameter in with_gradient)
        report[label] = {
            "trainable_tensor_count": len(eligible),
            "tensors_with_gradient": len(with_gradient),
            "gradient_norm_sum": gradient_norm,
            "has_task_gradient": gradient_norm > 0.0,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", choices=["qa", "cis"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config, allow_placeholders=False)
    device = torch.device(args.device)
    model = build_model(config, device)
    model.set_stage(args.stage)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, strict=True)
    dataset = ManifestDataset(
        manifest_paths=[args.manifest],
        tokenizer=model.language_backbone.tokenizer,
        vision_transform=model.vision_backbone.image_transform,
        prompt_template=config["evaluation"]["prompt_template"],
        image_resolution=int(config["model"]["image_resolution"]),
        expected_task=args.task,
        max_length=int(config["model"]["max_length"]),
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_manifest_batch)))
    batch = move_to_device(batch, device)
    model.zero_grad(set_to_none=True)
    output = model(batch)
    output.loss.backward()
    report = {
        "stage": args.stage,
        "task": args.task,
        "loss": float(output.loss.detach()),
        "groups": gradient_report(model),
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
