"""实现 RMSNorm 以及与 residual 融合的变体。"""

import torch
from torch import nn


class RMSNorm(nn.Module):

    """对最后一个维度做 RMS 归一化，并乘以可学习权重。"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """只执行 RMSNorm，不融合 residual。"""

        orig_dtype = x.dtype
        # 先转成 float 计算方差，避免低精度下数值不稳定。
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """先与 residual 相加，再执行 RMSNorm。"""

        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """根据是否提供 residual 选择融合或非融合路径。"""

        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
