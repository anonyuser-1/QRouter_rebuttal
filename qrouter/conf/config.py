from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PLACEHOLDER_MARKERS = ("TODO_PATH", "YOUR_PATH", "<PATH>")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    config["_config_path"] = str(config_path.resolve())
    validate_config(config, allow_placeholders=True)
    return config


def find_placeholders(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_placeholders(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_placeholders(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and any(marker in value for marker in PLACEHOLDER_MARKERS):
        found.append(prefix)
    return found


def validate_config(
    config: dict[str, Any],
    allow_placeholders: bool = False,
) -> None:
    for section in ("experiment", "model", "data", "optimization", "evaluation"):
        if section not in config:
            raise ValueError(f"Configuration is missing the '{section}' section.")
    model = config["model"]
    optimization = config["optimization"]
    if int(model["max_region_candidates"]) < int(model["num_region_tokens"]):
        raise ValueError("max_region_candidates must be >= num_region_tokens.")
    expected_tokens = int(model["num_region_tokens"]) + int(model["num_context_tokens"]) + 1
    if bool(config["experiment"].get("paper_default", False)):
        expected = {
            "max_region_candidates": 32,
            "num_region_tokens": 32,
            "num_context_tokens": 128,
            "image_resolution": 384,
        }
        mismatches = {
            key: (model.get(key), value) for key, value in expected.items() if int(model.get(key, -1)) != value
        }
        if mismatches:
            raise ValueError(f"Paper-default model settings disagree: {mismatches}")
        if expected_tokens != 161:
            raise ValueError(f"Paper default must form 161 visual tokens, got {expected_tokens}.")
        paper_optimization = {
            "stage1_steps": 20000,
            "stage2_steps": 10000,
            "global_batch_size": 128,
        }
        optimization_mismatches = {
            key: (optimization.get(key), value)
            for key, value in paper_optimization.items()
            if int(optimization.get(key, -1)) != value
        }
        if optimization_mismatches:
            raise ValueError("Paper-default optimization settings disagree: " f"{optimization_mismatches}")
        if abs(float(optimization["learning_rate"]) - 2.0e-5) > 1e-12:
            raise ValueError("The paper specifies learning_rate=2e-5.")
        if abs(float(optimization["weight_decay"]) - 0.1) > 1e-12:
            raise ValueError("The paper specifies weight_decay=0.1.")
        if abs(float(optimization["warmup_ratio"]) - 0.03) > 1e-12:
            raise ValueError("The paper specifies warmup_ratio=0.03.")
    if float(model["score_threshold"]) != 0.5:
        raise ValueError("The paper specifies score_threshold=0.5.")
    if optimization["schedule"] != "cosine":
        raise ValueError("The paper specifies cosine learning-rate decay.")
    if optimization.get("distributed_strategy") not in {"fsdp", "ddp", "single"}:
        raise ValueError("distributed_strategy must be fsdp, ddp, or single.")
    ratio = config["data"].get("qa_to_cis_ratio")
    if list(ratio) != [2, 1]:
        raise ValueError("The paper specifies a Stage-II QA:CIS ratio of 2:1.")
    if not allow_placeholders:
        placeholders = find_placeholders(config)
        if placeholders:
            raise ValueError("Resolve placeholder fields before execution: " + ", ".join(placeholders))
