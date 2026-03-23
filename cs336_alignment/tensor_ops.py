from __future__ import annotations

import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Average tensor values over the elements selected by ``mask``.

    Args:
        tensor: Tensor to reduce.
        mask: Boolean or 0/1 tensor with the same shape as ``tensor``.
        dim: Dimension to reduce over. If None, reduce across all dimensions.

    Returns:
        Tensor containing the masked mean reduction.
    """
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    counts = mask.sum(dim=dim)
    return masked_tensor.sum(dim=dim) / counts


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum tensor values under a mask and divide by a normalization constant.

    Args:
        tensor: Tensor to reduce.
        mask: Boolean or 0/1 tensor with the same shape as `tensor`.
        dim: Dimension to reduce over. If None, reduce across all dimensions.
        normalize_constant: Scalar divisor used for normalization.

    Returns:
        Tensor containing the masked, normalized reduction.
    """
    mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / normalize_constant
