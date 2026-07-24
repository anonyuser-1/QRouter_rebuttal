from __future__ import annotations

import torch.nn as nn

from qrouter.training.strategies import DistributedContext, wrap_distributed


def get_train_strategy(
    strategy: str,
    model: nn.Module,
    context: DistributedContext,
    precision: str,
) -> nn.Module:
    return wrap_distributed(
        model=model,
        strategy=strategy,
        context=context,
        precision=precision,
    )
