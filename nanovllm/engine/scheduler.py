from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """调度器：continuous batching 的大脑。

    在系统里的位置：
        LLMEngine.generate
          → add_request(seq) → scheduler.waiting
          → while not is_finished:
                step() → scheduler.schedule()    产出本轮要算的序列
                       → model_runner.run(...)    前向 + 采样
                       → scheduler.postprocess()  写回 token、判断完成、回收块

    核心是 waiting/running 双队列模型：
      - waiting: 刚提交或被 preempt 回退的序列，等 prefill
      - running: 已 prefill 完，每步 decode 生成 1 个 token

    序列状态机：
      WAITING → (prefill 完成) → RUNNING → (eos/max_tokens) → FINISHED
      RUNNING → (preempt 显存不足) → WAITING  （要重算 prompt）

    调度策略：prefill 优先（支持 chunked prefill + prefix cache 复用），
    无 prefill 时才 decode；显存不足时 preempt running 队尾序列腾出块。
    通过 BlockManager 间接管理 KV cache 块的分配/释放/哈希。
    """

    def __init__(self, config: Config):
        self.max_num_seqs = (
            config.max_num_seqs
        )  # 每步最多并行的序列数（batch 上限），防 OOM
        self.max_num_batched_tokens = (
            config.max_num_batched_tokens
        )  # prefill 每步最多处理的 token 数，支持 chunked prefill
        self.eos = config.eos  # EOS token id，postprocess 用它判断生成结束
        self.block_size = config.kvcache_block_size  # KV cache 块大小（256）
        # BlockManager 负责具体的块分配/释放/哈希，Scheduler 只决定「调度谁」
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size
        )
        # continuous batching 的核心双队列
        self.waiting: deque[Sequence] = (
            deque()
        )  # 待 prefill 的序列（含被 preempt 回退的）
        self.running: deque[Sequence] = deque()  # prefill 完成、正在 decode 的序列

    def is_finished(self):
        """没有待处理请求时返回 True。waiting 和 running 都空了才停，供 generate 主循环判断。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """把新序列加入 waiting 队列尾部。LLMEngine.add_request 调用。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """挑选本轮要执行的序列：优先 prefill（支持 chunked prefill），无 prefill 时再 decode。

        返回 (scheduled_seqs, is_prefill)：is_prefill=True 表示本步是 prefill，False 是 decode。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # =================================================================
        # 阶段 A. prefill 调度（prefill 优先）
        # =================================================================
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[
                0
            ]  # 窥队首不弹出：只有 prefill 全部算完才弹出转 RUNNING，
            # chunked prefill 中途序列留在 waiting 队首，下一步继续算。
            remaining = (
                self.max_num_batched_tokens - num_batched_tokens
            )  # 本步剩余 token 预算
            if remaining == 0:  # token 预算耗尽，跳出
                break
            # —— 计算这个 seq 本步需要算多少 token ——
            if not seq.block_table:  # 新序列，未分配块
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:  # 空闲块不足，跳出（等下次或 preempt）
                    break
                # 减去 prefix cache 命中的块数：命中前缀的 prompt 不用重算
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:  # chunked prefill 续算（之前算了一部分）
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # chunked prefill 只对队首序列开放：预算不够且不是第一个就跳出，
            # 让队首独占剩余预算做 chunked，避免后续序列饿死。
            if (
                remaining < num_tokens and scheduled_seqs
            ):  # only allow chunked prefill for the first seq
                break
            # —— 分配块（新序列才分配，续算的已分配过）——
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(
                num_tokens, remaining
            )  # 本步实际算的 token 数
            num_batched_tokens += seq.num_scheduled_tokens
            # prefill 全部算完才转 RUNNING 进 decode 队列
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()  # 算完才弹出 waiting
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # prefill 阶段调度到任何序列就直接返回，实现「prefill 优先」
        if scheduled_seqs:
            return scheduled_seqs, True

        # =================================================================
        # 阶段 B. decode 调度（无 prefill 时才 decode）
        # =================================================================
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()  # 弹出 running 队首
            # —— 检查能否追加 1 个 token 的 KV cache ——
            # while-else 语法：while 正常结束（条件为假）才进 else，break 跳出时不进。
            # 所以只有 can_append 通过才进 else 分支调度该序列。
            while not self.block_manager.can_append(seq):
                # 显存不足，需要抢占其他序列的块
                if self.running:
                    self.preempt(
                        self.running.pop()
                    )  # 抢占 running 队尾的序列（牺牲队尾，保队首）
                else:
                    self.preempt(seq)  # 只剩自己，抢占自己后跳出
                    break
            else:
                seq.num_scheduled_tokens = 1  # decode 每步固定 1 token
                seq.is_prefill = False  # 标记进入 decode 阶段
                self.block_manager.may_append(seq)  # 跨入新块才分配（token 跨块边界时）
                scheduled_seqs.append(seq)
        assert scheduled_seqs  # 至少调度到一个序列（否则前面会 break）
        # 放回 running 队首，保持原顺序：reversed 后 extendleft 等价于原顺序追加到队首
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """显存不足时把序列回退到 waiting 队列并释放其 KV cache 块，等待重新 prefill。

        代价：下次要重新 prefill（重算 prompt），但保证了正在 decode 的队首序列能继续。
        appendleft 让被 preempt 的序列下次优先 prefill，避免饿死。
        """
        seq.status = SequenceStatus.WAITING  # 回到 WAITING
        seq.is_prefill = True  # 重新标记为 prefill（要重算）
        self.block_manager.deallocate(seq)  # 释放 KV cache 块
        self.waiting.appendleft(seq)  # 放回 waiting 队首（优先重算）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """写回采样 token：先结算本步调度量与块哈希，prefill 未结束则不 append，结束后判断完成条件。

        LLMEngine.step() 在前向+采样后调用，把本步每个 seq 采到的 token_id 写回。
        """
        for seq, token_id in zip(seqs, token_ids):
            # —— 1. 给本轮新写满的块算链式哈希，纳入 prefix cache ——
            # 这样下次有相同前缀的 prompt 可复用这些块，省重算。
            self.block_manager.hash_blocks(seq)
            # —— 2. 累计已算 token 数 + 复位调度量 ——
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # —— 3. prefill 没算完不 append token ——
            # chunked prefill 中途采样的 token 是中间 logits 算出来的，不是「下一个 token」，丢弃。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # —— 4. prefill 完成或 decode 阶段：追加采样 token ——
            seq.append_token(token_id)
            # —— 5. 判断完成条件 ——
            # 遇到 eos（且 ignore_eos=False）或达到 max_tokens，标记 FINISHED 并回收块。
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 回收 KV cache 块
                self.running.remove(seq)  # 从 running 移除
