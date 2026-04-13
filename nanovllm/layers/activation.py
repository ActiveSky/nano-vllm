"""实现 Qwen3 MLP 中使用的激活函数。"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    """将输入拆成两半，前半做 SiLU 后与后半逐元素相乘。"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 SwiGLU / SiLU-and-multiply 变换。"""

        x, y = x.chunk(2, -1)
        return F.silu(x) * y
