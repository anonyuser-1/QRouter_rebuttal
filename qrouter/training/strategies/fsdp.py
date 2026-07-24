from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    world_size: int
    rank: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return DistributedContext(world_size, rank, local_rank, device)


def wrap_distributed(
    model: nn.Module,
    strategy: str,
    context: DistributedContext,
    precision: str,
) -> nn.Module:
    if context.world_size == 1:
        if strategy not in {"single", "ddp", "fsdp"}:
            raise ValueError(f"Unsupported distributed strategy: {strategy}")
        return model

    if strategy == "ddp":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            find_unused_parameters=True,
        )
    if strategy != "fsdp":
        raise ValueError("Multi-process execution requires distributed_strategy=ddp or fsdp.")

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000,
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=context.device,
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )


def shutdown_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
