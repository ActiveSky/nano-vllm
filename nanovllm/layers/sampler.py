"""实现基于 temperature 的随机采样。"""

import torch
from torch import nn


class Sampler(nn.Module):

    """把 logits 转换成最终采样 token id。"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """先做温度缩放，再通过随机扰动选出 token。"""

        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # 这里等价于 Gumbel-max 风格的随机采样，返回每行概率最大的扰动结果。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
