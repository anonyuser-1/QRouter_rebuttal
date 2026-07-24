from __future__ import annotations

import torch


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select dimension 1 independently for each batch element."""
    if values.ndim < 2 or indices.ndim != 2:
        raise ValueError("Expected values [B,N,...] and indices [B,K].")
    if values.shape[0] != indices.shape[0]:
        raise ValueError("Batch dimensions do not match.")
    view_shape = [indices.shape[0], indices.shape[1]] + [1] * (values.ndim - 2)
    expand_shape = [indices.shape[0], indices.shape[1], *values.shape[2:]]
    gather_index = indices.view(*view_shape).expand(*expand_shape)
    return torch.gather(values, dim=1, index=gather_index)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(eps)
    return numerator / denominator
