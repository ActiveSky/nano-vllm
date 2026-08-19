from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """单个物理 KV cache 块的 Python 侧元数据，不保存真实 K/V 张量。"""

    def __init__(self, block_id):
        self.block_id = block_id  # 对应 GPU KV cache 第几个物理块
        self.ref_count = 0  # 正在引用该块的 Sequence 数；为 0 时可被覆盖
        self.hash = -1  # 完整 token block 的链式哈希；-1 表示还不能用于 prefix cache
        self.token_ids = []  # 与 hash 对应的完整 token 列表，用于二次确认避免 hash 碰撞

    def update(self, hash: int, token_ids: list[int]):
        """在一个逻辑块写满后，记录它的 token 内容和 prefix cache 索引键。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """把即将被覆盖的空闲块初始化给一个新的 Sequence 使用。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """管理 KV cache 块的分配/释放，以及基于哈希的 prefix cache 复用。"""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size  # 一个逻辑 / 物理块能存放的 token 数
        # 固定大小的物理块元数据表；下标就是 GPU KV cache 的物理 block id。
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 链式 hash → 物理 block id：prefix cache 的查询索引。
        self.hash_to_block_id: dict[int, int] = dict()
        # 当前没有活跃 Sequence 引用的块。它们仍可能保留旧 K/V 与 hash 供 prefix cache 复用。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 当前至少被一个活跃 Sequence 引用的块，绝不能重新分配或覆盖。
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """为一个完整逻辑块生成链式 prefix hash。

        H_i = hash(H_(i-1) + 当前块 token)，而非只 hash 当前块。因此即使当前块
        token 相同，只要前面 prompt 不同，hash 也不同；命中代表完整前缀一致。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """取一个空闲物理块供新数据写入；覆盖前删除它遗留的 prefix cache 索引。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0  # free 队列中的块不应被任何活跃 Sequence 使用
        # 空闲块可保留旧 hash 以供复用；一旦要覆盖其 K/V，旧 hash 映射必须失效。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()  # 设 ref_count=1，清掉即将被覆盖的旧 token/hash 元数据
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """将引用归零的块放回空闲池，但保留旧 K/V/hash，以便后续 prefix cache 命中。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """检查新序列能复用多少完整 prefix block，并确认剩余 KV block 是否足够。

        返回命中的完整块数；返回 -1 表示无法为该序列准备足够的 block_table。最后一块
        即使当前已满也不参与 prefix cache，因为它可能很快因后续 decode 被继续写入。
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks  # 假设完全不命中时，整个序列需要的物理块数
        # 只扫描除最后一块外的完整块。prefix cache 只能安全共享、不可继续写入的完整块。
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # hash 命中后还要比较 token 列表，防止极低概率的 xxhash 碰撞导致错误复用。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 已被活跃序列占用的命中块不能从 free 池中再取，故少需要一个新空闲块。
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # 命中但当前空闲的块仍需从 free 池“拿走”，故 num_new_blocks 保持不变。
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """建立 seq.block_table：先共享命中的前缀块，再为未命中部分分配可写的新块。"""
        assert not seq.block_table
        h = -1
        # 命中块的 K/V 已存在：将其加入本序列映射，并增加引用（或从 free 池重新激活）。
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1  # 多条活跃序列共享同一完整前缀 KV block
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)  # 保留的 prefix cache 被新序列重新引用
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 未命中的块没有有效 K/V，必须获取独占、可写的物理块供 prefill/decode 填充。
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # prefix 命中的 token 无需再 prefill，后续 prepare_prefill 将从这里开始取输入。
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放一条 Sequence 对 KV blocks 的引用；归零块回空闲池但仍保留 prefix cache 数据。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0  # 若是抢占，未来会从头重新 prefill（或重新命中 prefix）
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """判断 decode 前能否为“下一个 token”预留所需的 KV block。

        append_token 在 postprocess 后才会让 len(seq) 增加 1；因此当前长度模 block_size
        等于 1，意味着上轮刚把一个 token 写进新逻辑块，下一轮 decode 需要新物理块。
        若不跨块则右侧为 False（数值 0），只要 0 个空闲块即可继续。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """真正为 decode 需要的新逻辑块分配物理块；仅在跨 block 边界时执行。"""
        # 例如 block_size=256、当前 len(seq)=257：逻辑第 1 块已有第一个 token，
        # 本轮模型会为它写 KV，因此需要把新的可写物理块追加进 block_table。
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """登记本轮刚写满的 KV blocks，使它们可被未来相同 prompt 前缀复用。

        当前未写满的最后一块不会入 cache：它仍可能被当前序列的 decode 继续追加，
        不能安全共享。该函数在 postprocess 的 append_token 前调用，因此只考虑本轮
        prefill 已计算完的 token 范围。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:  # 本轮没有任何逻辑块从“未完成”变成“完整”，无需更新索引
            return
        # 链式 hash 从上一个完整块继续；第 0 块没有前驱，prefix 使用 -1。
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
