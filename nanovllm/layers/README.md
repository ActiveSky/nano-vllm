# nanovllm/layers —— 算子层

> 本目录是模型的"零件库"：所有张量并行线性层、注意力、归一化、激活、位置编码和采样算子。
> 学习顺序建议：`linear.py` → `attention.py` → `rotary_embedding.py` → `embed_head.py` → 其余小算子。

## 文件作用一览

| 文件                  | 作用                | 一句话定位                                                             |
| --------------------- | ------------------- | ---------------------------------------------------------------------- |
| `linear.py`           | 张量并行线性层      | Column/Row/Merged/QKV 四种并行投影 + 分片权重加载                      |
| `attention.py`        | 注意力算子          | FlashAttention 封装（varlen/kvcache 双接口）+ Triton KV cache 写入内核 |
| `rotary_embedding.py` | RoPE 位置编码       | cos/sin **预计算缓存**，前向按位置索引，只旋转 Q/K                     |
| `layernorm.py`        | RMSNorm             | 归一化 + 残差融合的两种变体                                            |
| `activation.py`       | SiLU 门控激活       | SwiGLU 的融合算子                                                      |
| `embed_head.py`       | 词表并行嵌入/输出头 | embedding（mask+all_reduce）与 lm_head（gather+cat）                   |
| `sampler.py`          | 采样器              | Gumbel-max 风格的随机采样                                              |

## 核心加速技巧（学习重点）

### ★★★ `linear.py`：列/行并行的零通信衔接

- **列并行**（Column）：按输出维切分，每卡算出输出的一部分，前向无通信。
- **行并行**（Row）：按输入维切分，每卡算部分和，输出前 `all_reduce` 一次。
- **配对规则**：列的输出分片 → 直接做行的输入分片，中间零通信；Transformer 里 `QKV→O`、`gate_up→down` 都是这个配对。
- **QKVParallelLinear**：q/k/v 合并成一个参数（GQA 下三段长度不同），按 `shard_id`（"q"/"k"/"v"）加载权重。
- **MergedColumnParallelLinear**：gate/up 合并，一次 GEMM 算两个通道。

### ★★★ `attention.py`：FlashAttention + Triton

- `flash_attn_varlen_func`（prefill）：用 `cu_seqlens` 处理不等长序列，**免 padding**。
- `flash_attn_with_kvcache`（decode）：**直接读 KV cache**，历史 K/V 不重复计算。
- `store_kvcache_kernel`（自研 Triton）：每个 token 一个 program 并行写入 cache，向量化读写。
- FlashAttention 内部：tiling + online softmax + 融合，中间矩阵不出显存。

### ★★ `rotary_embedding.py`：预计算缓存

- 初始化时把 `max_position_embeddings` 个位置的 cos/sin 全部算好存进 `cos_sin_cache`；
- 前向只做 `cos_sin_cache[positions]` **索引**——每步省掉三角函数；
- `@lru_cache` 保证多层共享同一个 RoPE 实例；
- 注意：未实现 YaRN/NTK scaling（`rope_scaling` 只取了 `rope_theta`）。

### ★★ `embed_head.py`：词表并行的两种还原

- **embedding**：mask 置零 → 查本地表 → `all_reduce` **求和**还原（每 token 只有一个非零贡献）。
- **lm_head**：`gather+cat` **拼接**还原（需要完整词表才能采样），且只给 rank 0（采样只在 rank 0）。
- **prefill 只取最后位置**：`cu_seqlens_q[1:]-1` 裁剪中间位置的 logits，省掉词表维大矩阵的巨额 FLOPs。

### ★★ `sampler.py`：Gumbel-max

- `probs / exponential(1)` 再 `argmax` ≈ Gumbel-max 采样：一次向量化操作替代逐 token 累积分布采样，GPU 友好。

### ★ `layernorm.py` / `activation.py`：融合小技巧

- `add_rms_forward`：残差加法与 RMSNorm 融合成一个 kernel（少一次显存往返）；方差用 float 计算防数值不稳。
- `SiluAndMul`：gate 的 SiLU 与 up 的乘法融合成一个算子。

### 全局：`@torch.compile`

`rotary_embedding.py`、`layernorm.py`、`activation.py`、`sampler.py` 都用 `@torch.compile` 做图编译，配合引擎的 `warmup_model()` 在预热期触发编译。

## 与模型的关系

`models/qwen3.py` 把这些算子按 Qwen3 结构组装：`linear`（QKV/O/gate_up/down）+ `layernorm`（RMSNorm）+ `rotary_embedding`（RoPE）+ `attention`（FlashAttention）+ `embed_head`（embed/lm_head）+ `sampler`（采样）。
