"""保存一次前向过程中各个算子共享的上下文信息。"""

from dataclasses import dataclass

import torch


@dataclass
class Context:
    """单次推理 step 需要在算子之间传递的辅助张量集合。"""

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
    """更新当前全局上下文，供后续算子共享读取。"""

    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context() -> None:
    """将全局上下文重置为空状态。"""

    global _CONTEXT
    _CONTEXT = Context()
