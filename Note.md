# Nano-vLLM 学习笔记：Tensor Parallel 分片计算

## 1. `tensor_parallel_size` 控制什么

`Config.tensor_parallel_size` 控制 **Tensor Parallel（张量并行，简称 TP）的并行规模**，即一份模型由多少张 GPU 共同承载并完成计算。

例如：

| 配置                     | 进程与 GPU                  |
| ------------------------ | --------------------------- |
| `tensor_parallel_size=1` | rank 0 使用 GPU 0           |
| `tensor_parallel_size=2` | rank 0、1 分别使用 GPU 0、1 |
| `tensor_parallel_size=4` | rank 0～3 分别使用 GPU 0～3 |

在 `LLMEngine` 中，主进程负责 rank 0，并额外创建 rank 1 到 rank N-1：

```python
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()

self.model_runner = ModelRunner(config, 0, self.events)
```

在 `ModelRunner` 中，每个 rank 绑定对应的 GPU，并加入同一个 NCCL 进程组：

```python
self.world_size = config.tensor_parallel_size

dist.init_process_group(
    "nccl",
    "tcp://localhost:2333",
    world_size=self.world_size,
    rank=rank,
)
torch.cuda.set_device(rank)
```

因此，TP 不是简单地把不同请求分给不同 GPU：

```text
数据并行：
请求 A → GPU 0 上的一份完整模型
请求 B → GPU 1 上的一份完整模型

张量并行：
同一个请求 → GPU 0 计算模型的一部分
          → GPU 1 计算模型的另一部分
          → 集合通信合并局部结果
```

在本项目中，模型的线性层权重、Attention Head、词表权重和 KV Cache 都会按照 TP rank 进行切分。

---

## 2. 线性层的数学形式与命名

PyTorch 线性层的权重形状为：

```text
W.shape = [out_features, in_features]
```

`F.linear` 执行：

$$
Y = XW^T
$$

需要注意，Tensor Parallel 中 `Column Parallel` 和 `Row Parallel` 的命名来自经典写法 $Y=XA$，其中 $A$ 的形状为 `[in_features, out_features]`。PyTorch 保存的是转置后的 $W=A^T$，所以名称与代码中看到的物理行列方向容易混淆。

| TP 名称                | 逻辑切分维度 | PyTorch 权重 `[out, in]` 的实际切法 | 结果                              |
| ---------------------- | ------------ | ----------------------------------- | --------------------------------- |
| `ColumnParallelLinear` | 输出特征维度 | 切 `dim=0`，看起来是按行切          | 每个 GPU 产生一部分输出特征       |
| `RowParallelLinear`    | 输入特征维度 | 切 `dim=1`，看起来是按列切          | 每个 GPU 计算完整输出的一部分贡献 |

因此后文中的“列并行”和“行并行”均指 TP 的标准逻辑名称，不是单纯看 PyTorch 权重打印出来的行列外观。

---

## 3. 双 GPU 的实际矩阵计算

假设完整输入为：

$$
X=[1,2,3,4]
$$

第一层权重为：

$$
W_1=
\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
1&1&0&0\\
0&0&1&1
\end{bmatrix}
$$

单 GPU 计算：

$$
Y=XW_1^T=[4,6,3,7]
$$

具体为：

```text
y[0] = 1×1 + 2×0 + 3×1 + 4×0 = 4
y[1] = 1×0 + 2×1 + 3×0 + 4×1 = 6
y[2] = 1×1 + 2×1 + 3×0 + 4×0 = 3
y[3] = 1×0 + 2×0 + 3×1 + 4×1 = 7
```

### 3.1 第一层：Column Parallel

当 `tensor_parallel_size=2` 时，按照输出维度切分第一层权重。在 PyTorch 的 `[out, in]` 存储形式中，也就是切分权重的行。

GPU 0 保存：

$$
W_1^{(0)}=
\begin{bmatrix}
1&0&1&0\\
0&1&0&1
\end{bmatrix}
$$

GPU 1 保存：

$$
W_1^{(1)}=
\begin{bmatrix}
1&1&0&0\\
0&0&1&1
\end{bmatrix}
$$

两张 GPU 分别计算：

$$
Y^{(0)}=X(W_1^{(0)})^T=[4,6]
$$

$$
Y^{(1)}=X(W_1^{(1)})^T=[3,7]
$$

完整输出在逻辑上是：

$$
Y=[Y^{(0)},Y^{(1)}]=[4,6,3,7]
$$

但是此时通常不会立即合并，真实的数据状态是：

```text
GPU 0 持有 Y⁽⁰⁾ = [4, 6]
GPU 1 持有 Y⁽¹⁾ = [3, 7]
```

这对应项目中的 `ColumnParallelLinear`：

```python
tp_size = dist.get_world_size()
super().__init__(input_size, divide(output_size, tp_size), bias, 0)
```

加载模型时，每个 rank 也只加载自己的输出分片：

```python
shard_size = param_data.size(self.tp_dim)
start_idx = self.tp_rank * shard_size
loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
param_data.copy_(loaded_weight)
```

`ColumnParallelLinear.forward()` 只执行本地矩阵乘法，没有立即执行 `all_gather`：

```python
return F.linear(x, self.weight, self.bias)
```

---

## 4. 第二层为什么改用 Row Parallel

假设第二层完整权重为：

$$
W_2=
\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
1&1&1&1\\
1&-1&1&-1
\end{bmatrix}
$$

完整计算为：

$$
Z=YW_2^T=[7,13,20,-6]
$$

关键问题是，第二层开始前没有任何一张 GPU 持有完整的 $Y$：

```text
GPU 0 只有 Y⁽⁰⁾ = [4, 6]
GPU 1 只有 Y⁽¹⁾ = [3, 7]
```

### 4.1 如果第二层继续使用 Column Parallel

如果第二层继续按照输出维度切分，那么每个 GPU 虽然只负责一部分输出，但每一个输出都依赖完整的输入 $Y$。

例如第一个输出：

$$
Z_0=4\times1+6\times0+3\times1+7\times0=7
$$

其中：

- `4、6` 位于 GPU 0；
- `3、7` 位于 GPU 1。

如果 GPU 0 负责计算这个输出，它必须先取得 GPU 1 上的输入分片。因此，第二层继续使用 Column Parallel 并非数学上不可行，而是需要先执行 `all_gather`：

```text
操作前：
GPU 0：[4, 6]
GPU 1：[3, 7]

             All-Gather
                 ↓

操作后：
GPU 0：[4, 6, 3, 7]
GPU 1：[4, 6, 3, 7]
```

随后每张 GPU 才能使用完整输入，计算自己负责的输出分片。

这种方案的代价是：

1. 两层之间增加一次 GPU 集合通信；
2. 每张 GPU 都要临时保存完整中间激活；
3. Transformer 的中间维度通常很大，通信和显存开销明显；
4. 第二层输出依然是分片状态，后续层仍可能需要通信。

### 4.2 第二层使用 Row Parallel

Row Parallel 按照输入维度切分权重，即在 PyTorch 权重 `[out, in]` 上切 `dim=1`。这样第二层的权重分片与第一层的输出分片恰好对齐：

```text
GPU 0：持有输入特征 0、1，以及 W₂ 对应的前两列
GPU 1：持有输入特征 2、3，以及 W₂ 对应的后两列
```

GPU 0 保存：

$$
W_2^{(0)}=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
1&-1
\end{bmatrix}
$$

GPU 1 保存：

$$
W_2^{(1)}=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
1&-1
\end{bmatrix}
$$

本例中两部分恰好数值相同，但它们代表完整权重中不同位置的列。

GPU 0 直接使用本地数据计算：

$$
Z^{(0)}=Y^{(0)}(W_2^{(0)})^T=[4,6,10,-2]
$$

GPU 1 直接使用本地数据计算：

$$
Z^{(1)}=Y^{(1)}(W_2^{(1)})^T=[3,7,10,-4]
$$

这里两张 GPU 计算的不是不同输出，而是**所有输出的不同局部贡献**。完整结果是局部贡献之和：

$$
Z=Z^{(0)}+Z^{(1)}
$$

$$
Z=[4,6,10,-2]+[3,7,10,-4]=[7,13,20,-6]
$$

项目通过 `all_reduce` 完成逐元素求和：

```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
if self.tp_size > 1:
    dist.all_reduce(y)
return y
```

执行 `all_reduce` 后，每个 rank 上通常都有相同的完整输出：

```text
GPU 0：[7, 13, 20, -6]
GPU 1：[7, 13, 20, -6]
```

---

## 5. 两种方案的本质区别

### 方案 A：连续使用 Column Parallel

```text
第一层输出按特征分片
          ↓
      All-Gather
          ↓
每张 GPU 获得完整输入
          ↓
第二层每张 GPU 计算不同的输出分片
```

特点：

> 每个 GPU 负责不同输出，但每个 GPU 都需要完整输入。

### 方案 B：Column Parallel → Row Parallel

```text
第一层输出按特征分片
          ↓
第二层直接消费本地输入分片
          ↓
每张 GPU 计算所有输出的局部贡献
          ↓
      All-Reduce
          ↓
得到完整输出
```

特点：

> 每个 GPU 根据本地输入计算局部贡献，最后将贡献相加。

标准 Tensor Parallel 通常选择方案 B。最核心的原因是：

> 第一层按输出维度切分，是为了产生分片输出；第二层按输入维度切分，是为了直接消费这些分片输出，避免在两层之间先复制完整中间激活。

---

## 6. `all_gather` 与 `all_reduce`

### `all_gather`

收集各个 rank 上不同的分片，并按指定维度拼接：

```text
操作前：
GPU 0：[a, b]
GPU 1：[c, d]

操作后：
GPU 0：[a, b, c, d]
GPU 1：[a, b, c, d]
```

它适合将分片张量恢复为完整张量。

### `all_reduce`

将各个 rank 上形状相同的局部结果逐元素归约。本项目使用默认的求和操作：

```text
操作前：
GPU 0：[1, 2, 3]
GPU 1：[4, 5, 6]

操作后：
GPU 0：[5, 7, 9]
GPU 1：[5, 7, 9]
```

它适合合并 Row Parallel 产生的局部贡献。

---

## 7. 在 Transformer Attention 中的对应关系

Qwen3 Attention 中使用：

```python
self.qkv_proj = QKVParallelLinear(...)
self.o_proj = RowParallelLinear(...)
```

执行过程是：

```text
完整隐藏状态 X
        ↓
QKV 投影按输出维度切分
        ↓
GPU 0 负责一部分 Attention Head
GPU 1 负责另一部分 Attention Head
        ↓
每张 GPU 独立计算本地 Attention
        ↓
Attention 输出仍按 Head/特征分片
        ↓
O Projection 按输入维度切分
        ↓
每个 GPU 计算本地 Head 对最终输出的贡献
        ↓
All-Reduce
        ↓
每张 GPU 得到完整隐藏状态
```

代码中 Attention Head 也会按照 TP 数量平均分配：

```python
tp_size = dist.get_world_size()
self.num_heads = self.total_num_heads // tp_size
self.num_kv_heads = self.total_num_kv_heads // tp_size
```

例如模型有 32 个 Query Head、8 个 KV Head，且 `tensor_parallel_size=4`，那么每个 rank 负责：

```text
8 个 Query Head
2 个 KV Head
```

---

## 8. 在 Transformer MLP 中的对应关系

Qwen3 MLP 中使用：

```python
self.gate_up_proj = MergedColumnParallelLinear(...)
self.down_proj = RowParallelLinear(...)
```

执行过程是：

```text
完整隐藏状态 X，维度 d
        ↓
Gate/Up Projection 按输出维度切分
        ↓
中间激活按特征分布在多个 GPU 上
        ↓
SwiGLU 是局部逐元素运算，无需通信
        ↓
Down Projection 按输入维度切分
        ↓
每张 GPU 计算局部贡献
        ↓
All-Reduce
        ↓
完整输出，维度 d
```

典型 Transformer MLP 的中间维度通常大于隐藏维度。例如：

```text
隐藏维度 d       = 4096
中间维度约 4d    = 16384
```

如果第二层继续使用 Column Parallel，需要在两层之间 `all_gather` 较大的中间激活。改用 Row Parallel，则可以在第二层结束后对较小的输出执行 `all_reduce`。这通常更有利于控制通信量和中间激活显存。

---

## 9. 词表与 KV Cache 的分片

### 词表并行

`VocabParallelEmbedding` 把词表平均分配给不同 rank：

```python
self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
```

例如词表大小为 128000，TP 为 4，则每张 GPU 保存 32000 个 token 对应的 embedding 权重。

### KV Cache 分片

每张 GPU 只分配自己负责的 KV Head 对应的缓存：

```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size
```

因此，提高 TP 数量不仅会切分模型权重和计算，也会切分部分 KV Cache。

---

## 10. 可运行的双 GPU 演示

下面的示例用两张 GPU 实现前面的 `Column Parallel → Row Parallel` 计算：

```python
# tp_demo.py
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    assert world_size == 2

    # 两个 rank 都持有相同的初始输入。
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        device=device,
    )

    # 第一层完整权重。
    w1 = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        device=device,
    )

    # Column Parallel：按输出维度切分，即切分 PyTorch 权重的行。
    w1_shard = torch.chunk(w1, world_size, dim=0)[rank]
    y_shard = F.linear(x, w1_shard)

    print(f"rank={rank}, column output={y_shard.tolist()}")

    # 仅用于演示完整第一层结果；真实的 TP 配对中通常不需要这一步。
    gathered_y = [torch.empty_like(y_shard) for _ in range(world_size)]
    dist.all_gather(gathered_y, y_shard)
    full_y = torch.cat(gathered_y, dim=-1)

    if rank == 0:
        print("完整第一层输出:", full_y.tolist())

    # 第二层完整权重。
    w2 = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ],
        device=device,
    )

    # Row Parallel：按输入维度切分，即切分 PyTorch 权重的列。
    w2_shard = torch.chunk(w2, world_size, dim=1)[rank]

    # 直接消费本地的上一层输出分片。
    z_partial = F.linear(y_shard, w2_shard)
    print(f"rank={rank}, row partial={z_partial.tolist()}")

    # 对所有 GPU 的局部贡献逐元素求和。
    dist.all_reduce(z_partial, op=dist.ReduceOp.SUM)
    print(f"rank={rank}, final output={z_partial.tolist()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

在至少有两张 CUDA GPU 且支持 NCCL 的 Linux/WSL2 环境中运行：

```bash
torchrun --standalone --nproc_per_node=2 tp_demo.py
```

预期关键结果为：

```text
rank=0, column output=[[4.0, 6.0]]
rank=1, column output=[[3.0, 7.0]]

完整第一层输出=[[4.0, 6.0, 3.0, 7.0]]

rank=0, row partial=[[4.0, 6.0, 10.0, -2.0]]
rank=1, row partial=[[3.0, 7.0, 10.0, -4.0]]

rank=0, final output=[[7.0, 13.0, 20.0, -6.0]]
rank=1, final output=[[7.0, 13.0, 20.0, -6.0]]
```

由于两个进程并发打印，实际输出顺序可能不同。

---

## 11. TP 数量的约束与权衡

本项目限制：

```python
assert 1 <= self.tensor_parallel_size <= 8
```

此外，切分维度必须能被 TP 数量整除，例如：

```python
assert self.total_num_heads % tp_size == 0
assert self.total_num_kv_heads % tp_size == 0
assert num_embeddings % self.tp_size == 0
```

TP 数量增加的优点：

1. 降低每张 GPU 的模型权重占用；
2. 降低每张 GPU 的部分 KV Cache 占用；
3. 允许多张 GPU 并行完成大型矩阵运算；
4. 使单张 GPU 无法容纳的模型能够运行。

TP 数量增加的代价：

1. `all_reduce` 等 GPU 间通信增加；
2. 小模型、小 batch 下，通信开销可能超过并行收益；
3. 一条请求仍会同时占用整个 TP 组；
4. 需要满足 Head 数、KV Head 数和词表大小等整除约束；
5. 需要足够的可用 GPU 和高效的卡间互连。

因此，`tensor_parallel_size` 并不是越大越好，而是在单卡显存、模型规模、本地计算量和 GPU 通信开销之间取平衡。

---

## 12. 核心结论

1. `tensor_parallel_size=N` 表示同一个模型由 N 个 rank/N 张 GPU 共同计算。
2. Column Parallel 按输出特征切分，每张 GPU 产生一部分输出。
3. Row Parallel 按输入特征切分，每张 GPU 计算完整输出的一部分贡献。
4. 第一层 Column Parallel 的分片输出，能够直接作为第二层 Row Parallel 的本地输入。
5. 第二层继续使用 Column Parallel 不是不能计算，而是需要先 `all_gather` 完整中间激活。
6. `Column Parallel → Row Parallel` 通过最后一次 `all_reduce` 合并贡献，避免在两层之间复制完整中间激活。
7. 该模式同时应用在 Transformer 的 `QKV Projection → O Projection` 和 `Gate/Up Projection → Down Projection` 中。

一句话记忆：

> **Column Parallel 负责把输出拆开，Row Parallel 负责直接吃掉这些分片并把局部贡献加回来。**

---

# 2026-08-09：推理执行链路、KV Cache 与加速技巧

> 今日主题：从「模型定义」走向「推理引擎」，覆盖张量并行细节、KV cache 全生命周期、
> 多进程执行架构，以及贯穿全项目的推理加速点。与上面 1~12 节的 TP 基础互补。

## 1. `max_position_embeddings` 与 `max_model_len`

```python
# config.py
self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

- `max_position_embeddings` 是**模型作者写入 `config.json`** 的参数，规定 RoPE 可以覆盖的最大位置范围（它不是 RoPE 公式中的直接变量，而是「位置能取到多大」的上限）。
- 推理引擎用 `min()` 约束实际上下文长度：用户配的 `max_model_len` 再大，也不会超过模型原生上限。
- 注意：直接改 `config.json` 里的值**不等于**模型真能支持更长上下文，RoPE 需要配套 scaling（YaRN/NTK）与长上下文训练。

## 2. RoPE 只作用于 Q/K，不作用于 V（联网核实过）

- 依据：RoFormer 原论文（旋转 $q_m, k_n$）、HuggingFace 官方文档（"rotates query and key vectors"）、主流实现签名均为 `apply_rotary_pos_emb(q, k, cos, sin) -> (q, k)`。
- 原因：位置信息影响**注意力权重**（QK^T），旋转 Q/K 即可编码相对位置；V 是"匹配后取出的内容"，保持原样。
- 当前代码：`q, k = self.rotary_emb(positions, q, k)`，V 不参与。

## 3. attention 输出的 flatten

```python
output = self.o_proj(o.flatten(1, -1))
```

- `o` 形状 `[num_tokens, num_heads, head_dim]`；
- `flatten(1, -1)` 把 head 维和 head_dim 维合并成 `num_heads * head_dim`（保留 token 维）；
- 等价于 `o.reshape(num_tokens, num_heads * head_dim)`，喂给 O 投影。

## 4. 为什么 QKV 列并行、O 行并行（「先列后行」）

- **接口配对**：列并行输出分片、行并行输入分片 → 两者无缝衔接，中间**零通信**；整层只在最后 `all_reduce` 一次。
- 反过来「先行后列」需要先通信切分输入，层内层间都要通信，开销翻倍，实践中不采用。
- MLP 同理：`gate_up`（列）→ `down_proj`（行）。
- **命名迷惑点**：命名依据数学记法 $W \in \mathbb{R}^{in \times out}$——Column=切输出维（W 的列），Row=切输入维（W 的行）；PyTorch 存的是转置 `[out, in]`，所以代码里看着"反了"。

## 5. 为什么不能把「列+行」融合成一个大 block

- **数学上不可能**：两个线性层中间隔着非线性（attention 的 softmax、MLP 的 SiLU），$W_{eff}=W_2W_1$ 的预乘合并只在纯线性串联时成立。
- **工程上的融合粒度**是「整段计算」：FlashAttention（QKV→attention→输出一个 kernel）、`SiluAndMul`（GEMM+激活融合）、`add_rms_forward`（残差+norm 融合）；两个 GEMM 强行塞进一个 kernel 收益小、灵活性差。

## 6. attention 是 per-head 的 → KV cache 无需跨卡同步

- QK^T、softmax、PV **都不跨 head**（softmax 是 per-row：在每个 query 位置自己的键上归一化），所以 head 分片后 attention 无需任何跨卡通信。
- KV cache **按 head 分片**存储：`num_kv_heads // world_size`（见 `allocate_kv_cache`），每卡存「所有 token × 自己分片的 KV head」。
- 所有 rank 处理相同 token，每个 token 的 K/V 分片**天然在每张卡上**，decode 时 `flash_attn_with_kvcache` 只读本卡 cache——跨 token 与跨 GPU 两个维度正交。
- 唯一跨卡通信是 O 投影的 `all_reduce`（传输计算结果，不是 KV）。
- 破坏对称性的情况：GQA 的 `num_kv_heads` 不能被 tp_size 整除（代码中 `assert` 防线）；MLA 等低秩 KV 需要解压，TP 下必须跨卡通信（反例）。

## 7. KV cache 不在模型代码里：它的完整生命周期

`qwen3.py` 里没有任何 KV cache 字样，逻辑在 `ModelRunner` + `Attention` 两处：

```text
ModelRunner.allocate_kv_cache()
  → 按剩余显存算块数，一次性分配 [2, L, blocks, block_size, kv_heads, head_dim]
  → 遍历模型模块，把每层 Attention 的 k_cache/v_cache 占位替换成切片
ModelRunner.prepare_prefill / prepare_decode
  → 算 slot_mapping（物理槽位）、cu_seqlens、context_lens、block_tables
  → set_context() 放进全局 Context
Attention.forward
  → store_kvcache(k, v, k_cache, v_cache, slot_mapping)   # Triton 内核并行写入
  → prefill:  flash_attn_varlen_func(...)                 # 用本次 K/V
  → decode:   flash_attn_with_kvcache(q, k_cache, v_cache, ...)  # 只读历史
```

**slot_mapping 公式**：`slot = block_table[i] * block_size + 块内偏移`（PagedAttention 的物理寻址）。

## 8. ModelRunner：多进程拓扑与初始化顺序

- **拓扑**：rank 0 留在主进程，rank>0 各起一个 spawn 子进程；主进程通过「共享内存（4 字节长度 + pickle 负载）+ Event」广播方法调用，NCCL 负责卡间集合通信。
- **初始化顺序有因果链**（不能换）：

```text
warmup_model（触发编译缓存、让显存到峰值）
  → allocate_kv_cache（按 warmup 后的剩余显存算块数）
  → capture_cudagraph（KV cache 地址固定后才能录制）
```

- **CUDA Graph**：decode 用（每步只有 1 个 token，kernel 启动开销占比高，录制后一次 replay）；prefill 不用（compute-bound）。代价是输入地址固定 → 预分配 `graph_vars` 缓冲区，每步拷贝数据再 replay；按 batch 档位（1,2,4,8,16...）录制。
- `pin_memory + non_blocking`：页锁定内存 + 异步拷贝，与 GPU 计算重叠。

## 9. Context 全局上下文模式（utils/context.py）

- 动机：attention 等算子需要的 slot_mapping/cu_seqlens 等如果层层传参，模型代码会被推理引擎细节污染。
- 方案：进程级全局 `_CONTEXT`，`ModelRunner` 准备后 `set_context()`，算子 `get_context()` 读取，step 结束 `reset_context()`。
- 每进程一份（多卡 spawn 子进程各自独立）；`set_context` 重建对象而非原地改（引用不可变）。

## 10. 权重加载：packed_modules_mapping（utils/loader.py）

- HF 官方权重里 q/k/v 是三个独立矩阵，本实现合并进 `qkv_proj` 一个参数。
- `packed_modules_mapping`：`"q_proj" -> ("qkv_proj", "q")`，加载时按 `shard_id` 交给参数自定义的 `weight_loader`（linear.py 里 `weight.weight_loader = ...`），完成"切分 + 写本卡分片"。
- `safe_open(..., "cpu")`：先落主机内存再拷贝，避免 GPU 显存峰值翻倍。
- `for...else` 语法区分「命中打包映射」与「普通参数」两条分支。

## 11. 词表并行：embed_head.py

- 词表（十几万）是最大的一维，按 `tp_size` 连续切分，每卡存 `[vocab/tp, hidden]`。
- **embedding 的 mask 技巧**：mask 标记本卡词段 → 不属于的置 0（查表得零向量）→ 查本地表 → 再置零 → `all_reduce` 求和还原（每个 token 全局只有一个非零贡献）。
- **lm_head 用 gather+cat 而非 all_reduce**：同一位置需要完整词表的 logits（拼接还原而非求和）；且只有 rank 0 需要（采样在 rank 0），省通信。
- **prefill 只取每条序列最后位置**：`last_indices = cu_seqlens_q[1:] - 1`，省掉中间位置在词表维上的巨大 FLOPs（自回归每步只需要末尾位置的 logits）。
- **ParallelLMHead 继承 VocabParallelEmbedding**：权重布局/切分/loader 完全复用；tie_word_embeddings 时两者共享同一份权重，布局必须严格对齐；区别只在 forward（查表 vs 线性）和聚合方式（all_reduce vs gather）。

## 12. 推理加速点地图（按收益排序）

| 优先级 | 文件                         | 加速点                                                                                                               |
| ------ | ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| ★★★    | `engine/block_manager.py`    | PagedAttention 块管理（按块分配/回收，消灭显存碎片）+ **prefix cache 哈希复用**（`compute_hash` 为键）               |
| ★★★    | `engine/scheduler.py`        | **连续批处理**：每步动态调度 prefill/decode，请求完成立即释放块、新请求立即插入，GPU 永不空转；chunked prefill；抢占 |
| ★★★    | `layers/attention.py`        | FlashAttention（varlen 免 padding / kvcache 免重算）+ Triton `store_kvcache` 并行写缓存                              |
| ★★     | `layers/rotary_embedding.py` | cos/sin **预计算缓存**，前向只做 `cos_sin_cache[positions]` 索引，省三角函数                                         |
| ★★     | `layers/sampler.py`          | **Gumbel-max 采样**：`probs / exponential(1)` 再 argmax，一次向量化替代逐 token 采样循环                             |
| ★      | `layers/layernorm.py`        | residual+norm 融合（少一次显存往返）；方差用 float 精度计算                                                          |
| ★      | `layers/activation.py`       | SiluAndMul：gate/up 融合成一个算子                                                                                   |
| ★      | `engine/model_runner.py`     | CUDA Graph、pin_memory 异步拷贝、warmup                                                                              |

另外 `activation.py`、`rotary_embedding.py`、`layernorm.py`、`sampler.py` 都用了 `@torch.compile`（PyTorch 2 图编译），配合 `warmup_model` 在预热期触发编译。

## 13. 学习进度清单（注释完成情况）

- ✅ `models/qwen3.py`（模型组装、TP 全貌）
- ✅ `engine/model_runner.py`（执行器、KV cache、CUDA Graph）
- ✅ `utils/context.py`、`utils/loader.py`
- ✅ `layers/embed_head.py`、`layers/linear.py`、`layers/attention.py`（此前已带注释）
- ⬜ `engine/scheduler.py`、`engine/block_manager.py`（第一梯队，建议下一步）
- ⬜ `layers/rotary_embedding.py`、`layers/sampler.py`、`layers/layernorm.py`、`layers/activation.py`
- ⬜ `engine/sequence.py`、`engine/llm_engine.py`（部分已有注释）、`llm.py`、`sampling_params.py`

每个子目录下有 README.md 提供"文件作用 + 核心加速技巧"的分层速查表。
