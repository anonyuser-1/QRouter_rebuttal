from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch

from qrouter.training import sha256_file


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint payload.")
    return checkpoint


def optimizer_groups(checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict):
        return []
    report = []
    for index, group in enumerate(optimizer.get("param_groups", [])):
        entry = {key: json_safe(value) for key, value in group.items() if key != "params"}
        entry["group_index"] = index
        entry["parameter_entry_count"] = len(group.get("params", []))
        report.append(entry)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export immutable checkpoint metadata and SHA-256.")
    parser.add_argument("checkpoint")
    parser.add_argument("--output")
    parser.add_argument(
        "--absolute-path",
        action="store_true",
        help="Include the local absolute path in the exported metadata.",
    )
    args = parser.parse_args()

    path = Path(args.checkpoint).resolve()
    checkpoint = load_trusted_checkpoint(path)
    state = checkpoint.get("model", {})
    dtype_counts: Counter[str] = Counter()
    total_numel = 0
    tensor_count = 0
    if isinstance(state, dict):
        for tensor in state.values():
            if torch.is_tensor(tensor):
                tensor_count += 1
                total_numel += tensor.numel()
                dtype_counts[str(tensor.dtype)] += 1

    metadata = {
        "checkpoint": str(path) if args.absolute_path else path.name,
        "checkpoint_sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "top_level_keys": list(checkpoint.keys()),
        "step": json_safe(checkpoint.get("step")),
        "epoch": json_safe(checkpoint.get("epoch")),
        "args": json_safe(checkpoint.get("args")),
        "metrics": json_safe(checkpoint.get("metrics")),
        "model_state": {
            "number_of_entries": len(state) if isinstance(state, dict) else None,
            "tensor_count": tensor_count,
            "total_numel": total_numel,
            "dtype_counts": dict(dtype_counts),
        },
        "stored_training_states": {
            "optimizer": checkpoint.get("optimizer") is not None,
            "scheduler": checkpoint.get("scheduler") is not None,
            "scaler": checkpoint.get("scaler") is not None,
        },
        "optimizer_parameter_groups": optimizer_groups(checkpoint),
    }
    output = Path(args.output) if args.output else path.with_suffix(path.suffix + ".metadata.json")
    output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
