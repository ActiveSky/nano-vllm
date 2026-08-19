from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列生命周期状态机。

    状态流转：
        WAITING  --[scheduler 调度到 prefill 完成]--> RUNNING
        RUNNING  --[preempt 显存不足回退]-----------> WAITING
        RUNNING  --[遇到 eos 或达到 max_tokens]----> FINISHED
    """

    WAITING = auto()  # 在 waiting 队列，等 prefill
    RUNNING = auto()  # 在 running 队列，正在 decode
    FINISHED = auto()  # 生成结束，等回收


class Sequence:
    """一条推理请求的完整状态载体。

    在系统里的位置：
        用户 prompt -> LLMEngine.add_request -> Sequence 诞生
        -> scheduler.add(seq) -> waiting 队列
        -> scheduler.schedule() 产出 [seq, ...] -> model_runner 前向 + 采样
        -> scheduler.postprocess 写回 seq.append_token
        -> 达到 max_tokens 或遇到 eos -> status = FINISHED

    它是调度器、模型执行器、block manager 三方共同的操作对象：
      - 调度器读 status 决定排谁
      - 模型执行器读 token_ids / last_token 做前向
      - block manager 根据 block_table 管理 KV cache

    封装 4 类信息：
      1. 身份与状态：seq_id + status
      2. token 序列：token_ids + 各种长度计数 + prefill/decode 阶段标记
      3. KV cache 映射：block_table 记录占用哪些物理块
      4. 采样控制：从 SamplingParams 拆出的 temperature/max_tokens/ignore_eos
    """

    # 类属性：所有实例共享。block_size 由 LLMEngine.__init__ 阶段 2 设置，
    # block_manager 后续按这个尺寸切分 KV cache 块。
    block_size = 256
    # 类属性：itertools.count() 是个无限自增迭代器，
    # next(Sequence.counter) 给每个新序列发全局唯一 id（全局 id 生成器模式）。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        # —— 1. 身份与状态 ——
        self.seq_id = next(Sequence.counter)  # 全局唯一 id
        self.status = SequenceStatus.WAITING  # 初始为 WAITING，进 waiting 队列

        # —— 2. token 序列 ——
        self.token_ids = copy(
            token_ids
        )  # 完整 token 列表（prompt + 已生成），copy 防外部修改
        self.last_token = token_ids[-1]  # 末尾 token，decode 阶段前向只输入它，省传输
        self.num_tokens = len(self.token_ids)  # 当前总长度（随 append_token 增长）
        self.num_prompt_tokens = len(
            token_ids
        )  # prompt 长度（固定不变，作为 prompt/completion 分界）
        self.num_cached_tokens = 0  # 已经算过 prefill 的 token 数（调度进度）
        self.num_scheduled_tokens = (
            0  # 本轮要算的 token 数（由 scheduler.schedule 写入）
        )
        self.is_prefill = True  # True=prefill 阶段, False=decode 阶段

        # —— 3. KV cache 块映射 ——
        self.block_table = []  # 这条序列占用了哪些 KV cache 块的索引列表

        # —— 4. 采样控制（从 SamplingParams 拆出来，便于前向时直接取用）——
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回当前序列的 token 总数。让 len(seq) 可用，符合 Python 序列协议。"""
        return self.num_tokens

    def __getitem__(self, key):
        """按下标访问 token。让 seq[i] / seq[3:7] 可用，符合 Python 序列协议。"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """序列是否已进入 FINISHED 状态。step() 用它过滤出已完成序列返回给用户。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的补全 token 数（不含 prompt）。用于判断是否达到 max_tokens 停止条件。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回 prompt 部分的 token 切片（[:num_prompt_tokens]，固定不变）。"""
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回已生成补全部分的 token 切片（[num_prompt_tokens:]，随生成增长）。"""
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_blocks(self):
        """返回当前序列总共需要多少个 KV cache 块（向上取整）。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个未写满的块里已有的 token 数。block_manager 用它判断能否再 append。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个 block_size 长度的 token 切片，供 block_manager 做块级哈希或写入。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """追加一个补全 token 并更新末尾 token 与总长度。scheduler.postprocess 每步 decode 后调用。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """进程间传输时的精简状态：prefill 传整段 token_ids，decode 仅传 last_token。

        多卡时主进程通过共享内存 pickle 序列给从进程。完整 token_ids 太大，没必要每次都传：
          - prefill 阶段：从进程需要完整 token_ids 做前向
          - decode 阶段：从进程的 KV cache 已就位，前向只需 last_token 一个 token
        这样大幅减少共享内存传输量。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        )

    def __setstate__(self, state):
        """根据精简状态恢复序列：last_state 为 list 时重建整段 token_ids，否则仅恢复末尾 token。"""
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        ) = state
        if isinstance(last_state, list):
            # prefill：恢复整段 token_ids
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode：不需要历史 token（KV cache 已在从进程 GPU 上），只恢复末尾 token
            self.token_ids = []
            self.last_token = last_state
