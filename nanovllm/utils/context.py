"""保存一次前向过程中各个算子共享的上下文信息。

设计动机（学习重点）：
- 一次推理 step 中，attention 等算子需要很多「调度侧准备的数据」：
  slot_mapping（K/V 写哪）、cu_seqlens（序列边界）、block_tables（块映射）等。
- 如果通过函数参数层层传递，模型代码（Qwen3Attention.forward 等）就必须
  感知这些推理引擎细节，耦合严重。
- 因此采用「进程级全局上下文」：ModelRunner 准备好后统一 set_context()，
  attention 等算子需要时直接 get_context() 读取——算子与调度解耦。

注意：这个全局状态是每进程一份的（多卡 spawn 的子进程各自独立），
单步 step 内先 set 再使用，step 结束后 reset，不会跨 step 泄漏。
"""

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    """单次推理 step 需要在算子之间传递的辅助张量集合。

    字段含义（全部由 ModelRunner.prepare_* 填充）：
    - is_prefill    ：本轮是 prefill（True）还是 decode（False），
                      attention 据此选择 varlen / kvcache 两条执行路径。
    - cu_seqlens_q/k：各序列在拼接大 batch 中的起止边界（前加 0 的累加和），
                      让 flash_attn_varlen 在一个 kernel 里处理不等长序列。
    - max_seqlen_q/k：本轮所有序列的最大长度，varlen 接口需要的上限参数。
    - slot_mapping  ：每个 token 的 K/V 应写入 KV cache 的物理槽位。
    - context_lens  ：decode 时各序列当前已有 KV 历史长度。
    - block_tables  ：逻辑块 → 物理块的映射表（PagedAttention），
                      prefix cache 命中或 decode 时传给 flash-attention 使用。
    """

    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


# 进程内只保留一个当前上下文，供 attention 等算子直接读取。
_CONTEXT = Context()


def get_context() -> Context:
    """返回当前全局上下文。"""

    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
) -> None:
    """更新当前全局上下文，供后续算子共享读取。

    直接重建一个新的 Context 对象而不是原地修改字段：
    这样 get_context() 拿到的引用天然不可变，算子不会意外污染共享状态。
    """

    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context() -> None:
    """将全局上下文重置为空状态。

    在 ModelRunner.run 的末尾调用，避免上一轮 step 的 slot_mapping 等
    过期数据被下一轮意外读到。
    """

    global _CONTEXT
    _CONTEXT = Context()
