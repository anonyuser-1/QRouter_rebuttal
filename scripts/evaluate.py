from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qrouter.conf import load_config, validate_config
from qrouter.models import build_model
from qrouter.preprocessing import (
    ManifestDataset,
    collate_manifest_batch,
    encode_generation_prompts,
)
from qrouter.training import load_checkpoint
from qrouter.util import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate QRouter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", default="default")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalize_answer(answer: str) -> str:
    return answer.strip().lower().strip(" \t\r\n.,!?;:\"'()[]{}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config, allow_placeholders=False)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = build_model(config, device)
    load_checkpoint(args.checkpoint, model, strict=True)
    model.eval()

    template = (
        config["evaluation"]["gqa_prompt_template"]
        if args.benchmark.lower() == "gqa"
        else config["evaluation"]["prompt_template"]
    )
    dataset = ManifestDataset(
        manifest_paths=[args.manifest],
        tokenizer=model.language_backbone.tokenizer,
        vision_transform=model.vision_backbone.image_transform,
        prompt_template=template,
        image_resolution=int(config["model"]["image_resolution"]),
        expected_task="qa",
        max_length=int(config["model"]["max_length"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_manifest_batch,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for batch in loader:
            generation_inputs = encode_generation_prompts(
                tokenizer=model.language_backbone.tokenizer,
                questions=batch["questions"],
                template=template,
                max_length=int(config["model"]["max_length"]),
            )
            batch["input_ids"] = generation_inputs["input_ids"]
            batch["attention_mask"] = generation_inputs["attention_mask"]
            batch = move_to_device(batch, device)
            generated, _ = model.generate(
                batch,
                max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
            )
            predictions = model.language_backbone.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
            for index, prediction in enumerate(predictions):
                normalized_prediction = normalize_answer(prediction)
                normalized_answer = normalize_answer(batch["answers"][index])
                record = {
                    "id": batch["sample_ids"][index],
                    "prediction": normalized_prediction,
                    "answer": normalized_answer,
                    "answer_correct": normalized_prediction == normalized_answer,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
