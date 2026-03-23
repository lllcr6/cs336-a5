from __future__ import annotations

import torch


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
    # Scaffold only:
    # - In the real implementation, this should multiply by `mask`, reduce
    #   along the requested dimension, and then divide by `normalize_constant`.
    # - This helper is intentionally kept separate because future RL code can
    #   reuse the same reduction logic.
    raise NotImplementedError
