"""定义生成阶段的采样参数。"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """描述一次生成请求的温度、长度和 EOS 行为。"""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        """禁止温度过低的退化 greedy 采样。"""

        assert self.temperature > 1e-10, "greedy sampling is not permitted"
