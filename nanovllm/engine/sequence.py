"""定义单条请求在调度器中的状态和序列元数据。"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列在调度队列中的生命周期状态。"""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """封装一条请求的 token 序列，以及 KV cache 和调度所需元数据。"""

    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        """根据 prompt token 和采样参数创建序列对象。"""

        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回当前序列的 token 总数。"""

        return self.num_tokens

    def __getitem__(self, key):
        """支持按索引或切片访问 token_ids。"""

        return self.token_ids[key]

    @property
    def is_finished(self):
        """判断序列是否已经完成生成。"""

        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """返回已经生成的 completion token 数量。"""

        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回 prompt 对应的 token 切片。"""

        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回模型生成的 completion token 切片。"""

        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """返回已经命中 prefix cache 的块数量。"""

        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """返回当前序列总共需要多少个 KV cache 块。"""

        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """返回最后一个块中实际使用的 token 数。"""

        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个块对应的 token 切片。"""

        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """向序列末尾追加一个新 token。"""

        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """序列对象在进程间传输时只保留必要状态。"""

        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """根据精简状态恢复序列对象。"""

        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
