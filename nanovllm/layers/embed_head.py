"""实现词表并行的 embedding 和输出头。

学习重点：词表并行是「按输出维度切分」的列并行思想的另一个实例。
- vocab 维度（十几万）是权重矩阵里最大的一维，必须分摊到各卡：
  每卡只保存 1/tp_size 的词表向量（embedding）或输出行（lm_head）。
- VocabParallelEmbedding 查表后靠 all_reduce 还原完整向量（每个 token
  只有一张卡有非零贡献，求和即还原）。
- ParallelLMHead 与 embedding 权重布局完全相同（tie weights 时共享同一
  份权重），但输出的是完整词表的 logits，只能靠 gather+cat 还原，且
  只有 rank 0 需要（采样只在 rank 0 做）。
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """把词表按 tensor parallel rank 切分，只负责本地词段。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 词表必须能被 tp_size 整除，否则无法均匀切分。
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 本卡负责的连续词段 [vocab_start_idx, vocab_end_idx)，
        # 例如 tp=2 时卡 0 管 [0, V/2)，卡 1 管 [V/2, V)。
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        # 给参数挂上自定义 loader：加载权重时只取本卡词段（见 utils/loader.py）。
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """只加载当前 rank 对应的词表分片。"""

        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """计算本地 embedding，并在多卡时做 all-reduce。

        mask 技巧（学习重点）：
        每张卡只持有词表的一个连续区间，但输入 token id 可能落在任意区间。
        处理办法是「不属于本卡区间的 token 一律置零」：
        1. mask = 是否属于本卡词段；
        2. 属于的 token 减去段起点，得到本卡表内的行号；不属于的置 0
           （查表得到零向量，反正之后会被 mask 掉）；
        3. 查本地表后，把不属于本卡的向量再次置零；
        4. all_reduce 求和：每个 token 全局只有一张卡有非零向量，
           求和后恰好还原出正确的完整 embedding，且所有卡都拿到结果。
        """

        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)  # 求和还原：只有本卡命中的 token 贡献非零
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """词表并行的语言模型输出头。

    继承 VocabParallelEmbedding（学习重点）：
    - lm_head 与 embedding 的权重形状、切分方式、weight_loader 完全一致
      （[vocab_size/tp, hidden_size]），继承即可全部复用；
    - 尤其 tie_word_embeddings 时两者共享同一份权重数据，布局必须严格对齐；
    - 唯一区别是 forward：查表（F.embedding）→ 矩阵乘（F.linear），
      以及结果的聚合方式：all_reduce（求和）→ gather+cat（拼接）。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """在 prefill 阶段只取最后一个位置，在多卡时聚合 logits。

        prefill 只取最后位置（学习重点）：
        推理是自回归的，每步只需要「当前最后一个 token 位置」的 logits
        来决定下一个 token。prefill 一次算了整段 prompt，但只有每条序列
        末尾位置的结果会被采样使用，中间位置的 logits 直接丢弃，
        提前裁剪可以省掉 lm_head 在那些位置上的巨大计算量
        （词表维矩阵乘的 FLOPs 与位置数成正比）。
        - last_indices = cu_seqlens_q[1:] - 1：cu_seqlens 记录每条序列的
          结束边界，减 1 正是该序列最后一个 token 在拼接 batch 中的下标。
        - decode 阶段每序列只有 1 个 token，无需裁剪，直接算即可。

        多卡聚合（学习重点）：
        每卡只算了本卡词表分片的 logits（形状 [bs, vocab/tp]），要采样
        必须拿到完整词表的 logits → gather 收集到 rank 0 后按词表维拼接。
        注意与 embedding 的 all_reduce 不同：这里是「拼接还原」而不是
        「求和还原」（同一位置需要所有词表分量，而非只有一个非零贡献）；
        且只有 rank 0 需要结果（采样在 rank 0），其他 rank 返回 None。
        """

        context = get_context()
        if context.is_prefill:
            # prefill 阶段只需要每条序列最后一个 token 的 logits。
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 只有 rank 0 分配接收缓冲区；gather 是点对点收集语义，
            # 源张量在各卡必须形状一致（[bs, vocab/tp]）。
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            # 按词表维拼接：各卡分片 [bs, vocab/tp] → [bs, vocab]。
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
