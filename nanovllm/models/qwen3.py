"""定义 Qwen3 模型结构以及与 Nano-vLLM 组件的组装方式。

整体设计（张量并行视角，学习重点）：
1. 权重全部按「张量并行」切分：QKV/gate_up/embedding 用列并行（按输出/head 切），
   O/down/lm_head 用行并行（按输入切），「列→行」配对保证每层只通信一次
   （O/down 输出前的 all_reduce）。
2. attention 内部是纯 per-head 运算（QK^T、softmax、PV 都不跨 head），天然适配
   head 分片，无需跨卡通信；KV cache 也因此按 head 分片、各卡自存。
3. 权重存储是「合并式」的：q/k/v 合成 QKVParallelLinear 一个参数，gate/up 合成
   MergedColumnParallelLinear 一个参数，加载时靠 packed_modules_mapping 映射回
   HF 官方的独立矩阵。
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3 的单层自注意力模块，张量并行的核心单元。

    并行设计要点（学习重点）：
    1. head 切分：总 head 数按 tp_size 均分，每张卡只持有 num_heads 个 Q head
       和 num_kv_heads 个 KV head。attention 的每个算子都是 per-head 的，
       因此分片后全程无需跨卡通信（唯一的通信在 O 投影的 all_reduce）。
    2. QKV 合并存储：q/k/v 三个投影合成一个权重参数（QKVParallelLinear），
       一次 GEMM 同时算出三段，减少 kernel 启动；加载时按 q/k/v 区段写入。
    3. QK-Norm：Qwen3 在无 attention bias 时对 q/k 逐 head 做 RMSNorm，
       用于稳定注意力分数的数值范围。
    4. RoPE 只旋转 q/k 不旋转 v：位置信息只应影响注意力权重（QK^T），
       v 代表内容，保持原样。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        # —— 张量并行：head 按卡均分 ——
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # 本卡持有的 Q head 数
        self.total_num_kv_heads = num_kv_heads
        # GQA 下 KV head 数较少，必须能被 tp_size 整除才能按 head 均匀分片，
        # 否则会出现某张卡拿不到完整 KV head 分片的情况（需要跨卡通信）。
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 本卡持有的 KV head 数
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 本卡一段 q/k/v 的字节长度。Q 段长于 KV 段——这正是 GQA 的体现：
        # 多个 Q head 共享少数 KV head。
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5  # 1/sqrt(d)，防止 QK^T 点积数值过大
        self.qkv_bias = qkv_bias

        # QKV 合并的列并行投影：一次 GEMM 同时算出本卡的 q/k/v 分片。
        # 输出形状 [num_tokens, (num_heads + 2*num_kv_heads) * head_dim]，
        # 每张卡只持有自己的 head 分片（列并行，输出维切分，前向无通信）。
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # O 投影用行并行：输入是本卡 head 的输出分片，输出前 all_reduce 汇总，
        # 与前面的列并行组成「列→行」配对，整层只通信这一次。
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # Qwen3 的 rope_scaling 配置里可能携带 rope_theta，优先取用。
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,  # 由 config.max_position_embeddings 决定，即 RoPE 覆盖的最大位置
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # QK-Norm：Qwen3 无 attention bias 时，对每个 head 的 q/k 单独做 RMSNorm。
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """完成 QKV 投影、RoPE、注意力和输出投影。

        Q 的存储形态变化（学习重点）：
        [num_tokens, hidden_size]                  ← 输入
        → [num_tokens, (Q+2KV)*head_dim]           ← 合并投影后，q/k/v 挨着存
        → 三段 split → 每段 view 成 [num_tokens, num_heads, head_dim]
        → RoPE 原地旋转 q/k
        → attention 输出仍是 [num_tokens, num_heads, head_dim]
        → flatten(1, -1) 拼回 hidden 维 → O 投影
        """

        # 1. 初始形态：q/k/v 合并在一个张量里，按 q 段、k 段、v 段顺序排布。
        qkv = self.qkv_proj(hidden_states)
        # 2. 按三段长度切分（split 是同一块显存上的视图，零拷贝）。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 3. 重排成多头格式 [num_tokens, num_heads, head_dim]。
        #    view 只是元数据操作，不复制数据。
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # QK-Norm：无 bias 时对 q/k 逐 head 归一化，稳定注意力分数。
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # RoPE 只作用在 q/k 上：位置信息注入 QK^T 的点积（编码相对位置）；
        # v 不旋转，因为 v 是「内容」而不是「位置匹配」。
        q, k = self.rotary_emb(positions, q, k)
        # FlashAttention：prefill / decode 由 Attention 内部根据全局 Context 选择，
        # 并负责 KV cache 的写入与读取。
        o = self.attn(q, k, v)
        # 4. flatten(1, -1) 把 [num_tokens, num_heads, head_dim] 的 head 维与
        #    head_dim 维拼回一个向量，喂给行并行的 O 投影（内部 all_reduce）。
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """Qwen3 的前馈网络，使用 gate/up 合并投影。

    结构与 attention 同理：gate_up_proj（列并行）→ SiLU 门控 → down_proj
    （行并行），「列→行」配对，整层只通信一次（down 的 all_reduce）。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # gate 和 up 合并成一个列并行参数：一次 GEMM 同时算出 gate 与 up 两个
        # 通道（输出形状 [num_tokens, 2 * intermediate_size/tp]），
        # 相比两个独立投影省一次 kernel 启动和一份中间结果。
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        # SiluAndMul：对 gate 通道做 silu 再与 up 逐元素相乘（融合成一个算子）。
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """先做门控投影，再通过下投影回到 hidden size。"""

        gate_up = self.gate_up_proj(x)
        # 非线性夹在两次 GEMM 之间，所以两个权重矩阵无法预先合并成一个。
        x = self.act_fn(gate_up)  # silu(gate) * up
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 的单个 decoder block（attention + MLP + 残差）。

    残差设计（学习重点）：
    - Pre-Norm：先 RMSNorm 再进子层，主流做法，训练更稳定。
    - residual 以「流式」方式传入传出一个张量，而不是在块内原地相加：
      残差分支不经过子层计算，避免每层多一次加法与中间张量驻留（省显存）。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 所有结构参数（head 数、rope_theta、attention_bias 等）都直接取自
        # HF config，保证与官方权重的结构/推理行为一致。
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """执行 attention、残差融合和 MLP。"""

        if residual is None:
            # 第一层：residual 从当前 hidden_states 开始。
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 非首层：norm 直接消费上层的残差流，融合加法和 norm。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """由 embedding、多层 decoder 和最终 norm 组成的主体模型。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词表并行 embedding：vocab 维度很大（十几万），按词表维切分到各卡，
        # 每卡只保存自己那部分词表向量（列并行：输出维切分，前向无通信）。
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """从 token id 和位置编码出发，逐层计算 hidden states。"""

        hidden_states = self.embed_tokens(input_ids)
        residual = None
        # 逐层前向：每层只通信一次（O 投影的 all_reduce），层与层之间零通信。
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """带语言模型头的 Qwen3 封装。

    packed_modules_mapping（学习重点）：
    HF 官方权重里 q_proj/k_proj/v_proj 是三个独立矩阵，而本实现把它们合并
    进了 qkv_proj 一个参数。这张映射表告诉权重加载器：加载 q_proj 的权重时，
    应写入 qkv_proj 参数的 "q" 区段（k/v 同理；gate/up 则写入 gate_up_proj
    的 0/1 区段）。这是「合并存储 + 映射加载」的解耦设计。
    """

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # lm_head 同样按词表并行切分，与 embed_tokens 对称。
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # tie_word_embeddings：共享 embedding 与 lm_head 的权重（Qwen3 默认开启），
        # 让两者指向同一份数据，省一份参数。
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """只返回最后一层 hidden states，logits 由 compute_logits 负责。"""

        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """把 hidden states 投影到词表空间。"""

        return self.lm_head(hidden_states)
