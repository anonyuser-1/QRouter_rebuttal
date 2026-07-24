from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    stage: str,
    config: dict,
    rank: int = 0,
    epoch: int | None = None,
    scaler: Any | None = None,
    metrics: dict[str, Any] | None = None,
    checkpoint_args: dict[str, Any] | None = None,
) -> str | None:
    target = Path(path)
    is_fsdp = False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        is_fsdp = isinstance(model, FSDP)
    except ImportError:
        FSDP = None

    if is_fsdp:
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
    else:
        unwrapped = model.module if hasattr(model, "module") else model
        model_state = unwrapped.state_dict()
        optimizer_state = optimizer.state_dict()

    if rank != 0:
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    public_config = {key: value for key, value in config.items() if key != "_config_path"}
    stored_args = checkpoint_args if checkpoint_args is not None else {"stage": stage, **public_config}
    payload = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "step": int(step),
        "epoch": None if epoch is None else int(epoch),
        "args": stored_args,
        "metrics": {} if metrics is None else metrics,
    }
    torch.save(payload, temporary)
    os.replace(temporary, target)
    checksum = sha256_file(target)
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{checksum}  {target.name}\n",
        encoding="utf-8",
    )
    return checksum


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    strict: bool = True,
) -> dict:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    unwrapped = model.module if hasattr(model, "module") else model
    incompatible = unwrapped.load_state_dict(checkpoint["model"], strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            "Checkpoint load produced incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def restore_optimizer_scheduler(
    checkpoint: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None = None,
) -> None:
    if "optimizer" in checkpoint:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(model, FSDP):
                optimizer_state = FSDP.optim_state_dict_to_load(
                    model,
                    optimizer,
                    checkpoint["optimizer"],
                )
            else:
                optimizer_state = checkpoint["optimizer"]
        except ImportError:
            optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
