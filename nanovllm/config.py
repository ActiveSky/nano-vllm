"""定义推理引擎和模型初始化所需的配置对象。"""

import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class Config:
    """推理引擎启动时使用的静态配置。"""

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        """校验配置并从 HuggingFace 读取模型元信息。"""

        assert os.path.isdir(self.model)
        # KV cache block size 需要和底层块管理逻辑对齐。
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 真实可用的上下文长度不能超过模型本身的最大位置编码上限。
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
