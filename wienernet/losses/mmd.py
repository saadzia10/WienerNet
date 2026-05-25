"""Maximum Mean Discrepancy (MMD) loss with multi-bandwidth Gaussian kernel.

Moved verbatim from piae_sde/mmd_loss.py with cleaner naming and types.
Used as a distributional loss on NEE, bNEE, and the noise term.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    """Multi-bandwidth Gaussian MMD between two distributions.

    The bandwidth is set adaptively from the median pairwise squared distance,
    then scaled across `kernel_num` log-spaced bandwidths and summed
    (the same recipe as the reference implementation in piae_sde/mmd_loss.py).

    Args:
        kernel_mul: factor multiplied between consecutive bandwidths.
        kernel_num: number of kernels combined (odd values keep the median
            bandwidth at the centre of the geometric series).
        fix_sigma: if set, overrides the adaptive bandwidth.
    """

    def __init__(
        self,
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: float | None = None,
    ) -> None:
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        n = total.size(0)

        total0 = total.unsqueeze(0).expand(n, n, total.size(1))
        total1 = total.unsqueeze(1).expand(n, n, total.size(1))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)

        bandwidth = bandwidth / self.kernel_mul ** (self.kernel_num // 2)
        bandwidths = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bw) for bw in bandwidths]
        return sum(kernel_val)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = source.size(0)
        kernels = self._gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        return torch.mean(XX + YY - XY - YX)
