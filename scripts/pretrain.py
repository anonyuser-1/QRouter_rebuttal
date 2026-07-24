from __future__ import annotations

import argparse
import contextlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from qrouter.conf import load_config, validate_config
from qrouter.models import build_model
from qrouter.preprocessing import collate_manifest_batch, get_dataset
from qrouter.training import (
    DistributedContext,
    get_train_strategy,
    initialize_distributed,
    load_checkpoint,
    restore_optimizer_scheduler,
    save_checkpoint,
    shutdown_distributed,
    write_json,
)
from qrouter.util import (
    adamw_parameter_groups,
    move_to_device,
    trainable_parameter_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train QRouter.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["stage1", "stage2"])
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--init-checkpoint",
        help="Load model weights only, e.g. Stage-I best -> Stage II.",
    )
    checkpoint_group.add_argument(
        "--resume",
        help="Resume the same stage, including optimizer and scheduler state.",
    )
    return parser.parse_args()


def set_seed(seed: int, rank: int) -> None:
    effective_seed = int(seed) + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)


def make_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    context: DistributedContext,
) -> DataLoader:
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            drop_last=True,
        )
        if context.world_size > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=context.device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=True,
        collate_fn=collate_manifest_batch,
    )


def infinite_batches(loader: DataLoader):
    epoch = 0
    while True:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def scheduler_for(
    optimizer: torch.optim.Optimizer,
    steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    warmup_steps = int(steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def validate_runtime_batch(config: dict, context: DistributedContext) -> None:
    optimization = config["optimization"]
    resolved = (
        int(optimization["per_device_batch_size"])
        * int(optimization["gradient_accumulation_steps"])
        * context.world_size
    )
    declared = int(optimization["global_batch_size"])
    if resolved != declared:
        raise ValueError(
            f"Resolved global batch is {resolved}, but config declares {declared}. "
            "global_batch = per_device_batch x accumulation x world_size."
        )


def resolved_checkpoint_args(
    config: dict,
    stage: str,
    context: DistributedContext,
    init_checkpoint: str | None,
    resume: str | None,
) -> dict:
    experiment = config["experiment"]
    model = config["model"]
    data = config["data"]
    optimization = config["optimization"]
    resolution = int(model["image_resolution"])
    return {
        "stage": stage,
        "stage1_max_steps": int(optimization["stage1_steps"]),
        "stage2_max_steps": int(optimization["stage2_steps"]),
        "stage_step": int(optimization[f"{stage}_steps"]),
        "total_optimization_steps": int(optimization["stage1_steps"]) + int(optimization["stage2_steps"]),
        "qa_to_cis_ratio": list(data["qa_to_cis_ratio"]),
        "qa_data_jsonl": list(data[f"{stage}_qa_manifests"]),
        "cis_data_jsonl": list(data["stage2_cis_manifests"]),
        "init_checkpoint": init_checkpoint,
        "resume": resume,
        "output_dir": str(Path(experiment["output_dir"]) / stage),
        "llm_id": model["llm_id"],
        "vision_backbone_id": model["vision_backbone_id"],
        "grounding_qwen_id": model["qwen_id"],
        "sam2_config": model["sam2_config"],
        "sam2_checkpoint": model["sam2_checkpoint"],
        "distributed_strategy": optimization["distributed_strategy"],
        "world_size": context.world_size,
        "batch_size": int(optimization["per_device_batch_size"]),
        "grad_accum_steps": int(optimization["gradient_accumulation_steps"]),
        "global_batch_size": int(optimization["global_batch_size"]),
        "num_workers": int(data["num_workers"]),
        "optimizer": "AdamW",
        "learning_rate": float(optimization["learning_rate"]),
        "weight_decay": float(optimization["weight_decay"]),
        "scheduler": optimization["schedule"],
        "warmup_ratio": float(optimization["warmup_ratio"]),
        "max_steps": int(optimization[f"{stage}_steps"]),
        "save_every": int(optimization["save_every"]),
        "log_every": int(optimization["log_every"]),
        "image_resolution": [resolution, resolution],
        "max_length": int(model["max_length"]),
        "projector_layers": 2,
        "projector_hidden_dim": int(model["projector_hidden_dim"]),
        "num_region_tokens": int(model["num_region_tokens"]),
        "num_context_tokens": int(model["num_context_tokens"]),
        "num_background_tokens": 1,
        "visual_prefix_length": int(model["num_region_tokens"]) + int(model["num_context_tokens"]) + 1,
        "max_retained_masks": int(model["num_region_tokens"]),
        "raw_candidate_mask_count": int(model["max_region_candidates"]),
        "confidence_threshold": float(model["score_threshold"]),
        "lambda_dice": float(optimization["cis_dice_weight"]),
        "sam2_image_encoder_frozen": True,
        "grounding_trainable_scope": "LoRA and cross-modal projections",
        "precision": optimization["precision"],
        "gradient_checkpointing": bool(optimization["gradient_checkpointing"]),
        "seed": int(experiment["seed"]),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config, allow_placeholders=False)
    context = initialize_distributed()
    set_seed(int(config["experiment"]["seed"]), context.rank)
    validate_runtime_batch(config, context)

    optimization = config["optimization"]
    model = build_model(config, device=context.device)
    model.set_stage(args.stage)
    if bool(optimization.get("gradient_checkpointing", True)):
        model.language_backbone.enable_gradient_checkpointing()

    loaded_checkpoint = None
    checkpoint_path = args.init_checkpoint or args.resume
    if checkpoint_path:
        loaded_checkpoint = load_checkpoint(
            checkpoint_path,
            model,
            optimizer=None,
            scheduler=None,
            strict=True,
        )
        if args.resume and loaded_checkpoint.get("stage") != args.stage:
            raise ValueError(
                "--resume requires a checkpoint from the same stage; use "
                "--init-checkpoint for Stage-I -> Stage-II initialization."
            )
        if args.init_checkpoint:
            del loaded_checkpoint
            loaded_checkpoint = None

    trainable_report = trainable_parameter_summary(model)
    if context.is_main:
        stage_dir = Path(config["experiment"]["output_dir"]) / args.stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        public_config = {key: value for key, value in config.items() if key != "_config_path"}
        write_json(stage_dir / "resolved_config.json", public_config)
        write_json(stage_dir / "trainable_parameters.json", trainable_report)

    data_config = config["data"]
    qa_dataset = get_dataset(
        data_config[f"{args.stage}_qa_manifests"],
        "qa",
        model,
        config,
    )
    qa_iterator = infinite_batches(
        make_loader(
            qa_dataset,
            int(optimization["per_device_batch_size"]),
            int(data_config["num_workers"]),
            context,
        )
    )
    cis_iterator = None
    if args.stage == "stage2":
        cis_dataset = get_dataset(
            data_config["stage2_cis_manifests"],
            "cis",
            model,
            config,
        )
        cis_iterator = infinite_batches(
            make_loader(
                cis_dataset,
                int(optimization["per_device_batch_size"]),
                int(data_config["num_workers"]),
                context,
            )
        )

    model.train()
    model = get_train_strategy(
        strategy=str(optimization["distributed_strategy"]),
        model=model,
        context=context,
        precision=str(optimization["precision"]),
    )
    parameter_groups, optimizer_group_report = adamw_parameter_groups(
        model,
        weight_decay=float(optimization["weight_decay"]),
    )
    optimizer = AdamW(
        parameter_groups,
        lr=float(optimization["learning_rate"]),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    if context.is_main:
        write_json(
            Path(config["experiment"]["output_dir"]) / args.stage / "optimizer_parameter_groups.json",
            optimizer_group_report,
        )

    max_steps = int(optimization[f"{args.stage}_steps"])
    scheduler = scheduler_for(
        optimizer,
        steps=max_steps,
        warmup_ratio=float(optimization["warmup_ratio"]),
    )
    start_step = 0
    if args.resume and loaded_checkpoint is not None:
        restore_optimizer_scheduler(
            loaded_checkpoint,
            model,
            optimizer,
            scheduler,
        )
        start_step = int(loaded_checkpoint.get("step", 0))
        del loaded_checkpoint

    gradient_accumulation = int(optimization["gradient_accumulation_steps"])
    qa_count, cis_count = [int(value) for value in data_config["qa_to_cis_ratio"]]
    stage2_pattern = ["qa"] * qa_count + ["cis"] * cis_count
    precision = str(optimization["precision"])
    checkpoint_args = resolved_checkpoint_args(
        config=config,
        stage=args.stage,
        context=context,
        init_checkpoint=args.init_checkpoint,
        resume=args.resume,
    )
    train_loss_ema: float | None = None

    for update_step in range(start_step, max_steps):
        optimizer.zero_grad(set_to_none=True)
        output = None
        for micro_step in range(gradient_accumulation):
            schedule_index = update_step * gradient_accumulation + micro_step
            task = "qa" if args.stage == "stage1" else stage2_pattern[schedule_index % len(stage2_pattern)]
            batch = next(qa_iterator if task == "qa" else cis_iterator)
            batch = move_to_device(batch, context.device)
            is_last_micro_step = micro_step == gradient_accumulation - 1
            sync_context = (
                contextlib.nullcontext() if is_last_micro_step or not hasattr(model, "no_sync") else model.no_sync()
            )
            with sync_context:
                with autocast_context(context.device, precision):
                    output = model(batch)
                    loss = output.loss / gradient_accumulation
                loss.backward()

        optimizer.step()
        scheduler.step()
        completed_step = update_step + 1
        assert output is not None
        train_loss = float(output.loss.detach())
        train_loss_ema = train_loss if train_loss_ema is None else 0.99 * train_loss_ema + 0.01 * train_loss

        if context.is_main and completed_step % int(optimization["log_every"]) == 0:
            metrics = " ".join(f"{key}={float(value):.5f}" for key, value in output.metrics.items())
            print(
                f"stage={args.stage} optimization_step={completed_step} "
                f"last_micro_task={task} loss={float(output.loss):.5f} {metrics}",
                flush=True,
            )

        should_save = completed_step % int(optimization["save_every"]) == 0 or completed_step == max_steps
        if should_save:
            output_path = (
                Path(config["experiment"]["output_dir"]) / args.stage / "checkpoints" / f"step-{completed_step:06d}.pt"
            )
            save_checkpoint(
                output_path,
                model,
                optimizer,
                scheduler,
                completed_step,
                args.stage,
                config,
                rank=context.rank,
                epoch=None,
                scaler=None,
                metrics={
                    "train_loss": train_loss,
                    "train_loss_ema": train_loss_ema,
                    **{key: float(value) for key, value in output.metrics.items()},
                },
                checkpoint_args=checkpoint_args,
            )

    shutdown_distributed()


if __name__ == "__main__":
    main()
