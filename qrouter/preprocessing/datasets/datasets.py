from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def _read_jsonl(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_manifest"] = str(path.resolve())
                record["_line"] = line_number
                records.append(record)
    return records


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_paths: list[str | Path],
        tokenizer,
        vision_transform: Callable[[Image.Image], dict[str, torch.Tensor]],
        prompt_template: str = "Question: {question}\nAnswer:",
        image_resolution: int = 384,
        expected_task: str | None = None,
        max_length: int = 2048,
    ) -> None:
        self.records = _read_jsonl(manifest_paths)
        self.tokenizer = tokenizer
        self.vision_transform = vision_transform
        self.prompt_template = prompt_template
        self.image_resolution = int(image_resolution)
        self.expected_task = expected_task
        self.max_length = int(max_length)
        for record in self.records:
            task = record.get("task", "qa")
            if expected_task is not None and task != expected_task:
                raise ValueError(
                    f"Expected {expected_task} row, found {task} at " f"{record['_manifest']}:{record['_line']}."
                )

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize_question(self, question: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            question,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"][0], encoded["attention_mask"][0]

    def _tokenize_qa(
        self,
        question: str,
        answer: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self.prompt_template.format(question=question)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"][0]
        answer_text = " " + answer.strip()
        if self.tokenizer.eos_token:
            answer_text += self.tokenizer.eos_token
        answer_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, self.max_length - prompt_ids.numel()),
            return_tensors="pt",
        )["input_ids"][0]
        input_ids = torch.cat([prompt_ids, answer_ids], dim=0)[: self.max_length]
        labels = input_ids.clone()
        labels[: min(prompt_ids.numel(), labels.numel())] = IGNORE_INDEX
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, labels

    def _load_mask(self, path_value: str) -> torch.Tensor:
        mask = Image.open(path_value).convert("L")
        tensor = torch.from_numpy(np.asarray(mask, dtype="float32").copy()) / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            tensor,
            size=(self.image_resolution, self.image_resolution),
            mode="nearest",
        )
        return resized[0]

    @staticmethod
    def _resolve_record_path(record: dict[str, Any], path_value: str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = Path(record["_manifest"]).parent / path
        return path.resolve()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = str(self._resolve_record_path(record, record["image"]))
        image = Image.open(image_path).convert("RGB")
        question = str(record["question"])
        question_ids, question_mask = self._tokenize_question(question)
        item: dict[str, Any] = {
            "sample_id": str(record.get("id", index)),
            "task": str(record.get("task", "qa")),
            "image_path": image_path,
            "question": question,
            "grounding_question": str(record.get("grounding_question", question)),
            "pixel_values": self.vision_transform(image),
            "question_input_ids": question_ids,
            "question_attention_mask": question_mask,
            "pad_token_id": int(self.tokenizer.pad_token_id or 0),
        }
        if item["task"] == "qa":
            if "answer" not in record:
                raise ValueError(f"QA row lacks answer: {record['_manifest']}:{record['_line']}")
            input_ids, attention_mask, labels = self._tokenize_qa(
                question,
                str(record["answer"]),
            )
            item.update(
                {
                    "answer": str(record["answer"]),
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
            if "mask" in record:
                item["ground_truth_mask"] = self._load_mask(str(self._resolve_record_path(record, record["mask"])))
        elif item["task"] == "cis":
            if "mask" not in record:
                raise ValueError(f"CIS row lacks mask: {record['_manifest']}:{record['_line']}")
            item["ground_truth_mask"] = self._load_mask(str(self._resolve_record_path(record, record["mask"])))
        else:
            raise ValueError(f"Unsupported task: {item['task']}")
        return item


def _pad_1d(
    tensors: list[torch.Tensor],
    pad_value: int,
) -> torch.Tensor:
    max_length = max(tensor.numel() for tensor in tensors)
    output = tensors[0].new_full((len(tensors), max_length), pad_value)
    for index, tensor in enumerate(tensors):
        output[index, : tensor.numel()] = tensor
    return output


def collate_manifest_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    tasks = {item["task"] for item in items}
    if len(tasks) != 1:
        raise ValueError("QA and CIS samples must be drawn by separate loaders.")
    output: dict[str, Any] = {
        "task": items[0]["task"],
        "sample_ids": [item["sample_id"] for item in items],
        "image_paths": [item["image_path"] for item in items],
        "questions": [item["question"] for item in items],
        "grounding_questions": [item["grounding_question"] for item in items],
        "pixel_values": {
            key: torch.stack([item["pixel_values"][key] for item in items]) for key in items[0]["pixel_values"]
        },
        "question_input_ids": _pad_1d(
            [item["question_input_ids"] for item in items],
            pad_value=items[0]["pad_token_id"],
        ),
        "question_attention_mask": _pad_1d(
            [item["question_attention_mask"] for item in items],
            pad_value=0,
        ),
    }
    if items[0]["task"] == "qa":
        output.update(
            {
                "answers": [item["answer"] for item in items],
                "input_ids": _pad_1d(
                    [item["input_ids"] for item in items],
                    pad_value=items[0]["pad_token_id"],
                ),
                "attention_mask": _pad_1d(
                    [item["attention_mask"] for item in items],
                    pad_value=0,
                ),
                "labels": _pad_1d(
                    [item["labels"] for item in items],
                    pad_value=IGNORE_INDEX,
                ),
            }
        )
        if all("ground_truth_mask" in item for item in items):
            output["ground_truth_masks"] = torch.stack([item["ground_truth_mask"] for item in items])
            output["has_ground_truth_mask"] = torch.ones(
                len(items),
                dtype=torch.bool,
            )
    else:
        output["ground_truth_masks"] = torch.stack([item["ground_truth_mask"] for item in items])
        output["has_ground_truth_mask"] = torch.ones(len(items), dtype=torch.bool)
    return output
