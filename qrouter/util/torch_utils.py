from __future__ import annotations

from typing import Any

import torch


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [move_to_device(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(child, device) for child in value)
    return value


def trainable_parameter_summary(model: torch.nn.Module) -> dict:
    names = []
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            names.append({"name": name, "count": count})
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / max(total, 1),
        "parameters": names,
    }


def adamw_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split trainable parameters into AdamW decay and no-decay groups."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower_name = name.lower()
        use_no_decay = (
            parameter.ndim <= 1 or lower_name.endswith(".bias") or "norm" in lower_name or "embedding" in lower_name
        )
        if use_no_decay:
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise RuntimeError("No trainable parameters were found.")
    report = {
        "decay": {
            "tensor_count": len(decay),
            "parameter_count": sum(parameter.numel() for parameter in decay),
            "weight_decay": float(weight_decay),
            "names": decay_names,
        },
        "no_decay": {
            "tensor_count": len(no_decay),
            "parameter_count": sum(parameter.numel() for parameter in no_decay),
            "weight_decay": 0.0,
            "names": no_decay_names,
        },
    }
    return groups, report
