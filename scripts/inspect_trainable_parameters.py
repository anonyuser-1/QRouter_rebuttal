from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qrouter.conf import load_config, validate_config
from qrouter.models import build_model
from qrouter.util import trainable_parameter_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config, allow_placeholders=False)
    model = build_model(config, torch.device(args.device))
    model.set_stage(args.stage)
    report = {
        "stage": args.stage,
        "qwen_to_sam_initialization": (model.grounding_module.prompt_adapter.initialization_description()),
        **trainable_parameter_summary(model),
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
