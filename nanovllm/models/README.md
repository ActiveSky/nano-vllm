# nanovllm/models —— 模型定义

> 本目录定义具体的大模型结构（目前是 Qwen3）。
> 它只做"结构组装"，不关心推理引擎细节：KV cache、调度、采样都在 engine/ 层。
> 学习要点：看它如何把 layers/ 的算子按张量并行规则拼成完整模型。

## 文件作用一览

| 文件       | 作用           | 一句话定位                                                                              |
| ---------- | -------------- | --------------------------------------------------------------------------------------- |
| `qwen3.py` | Qwen3 完整模型 | `Qwen3Attention` / `Qwen3MLP` / `Qwen3DecoderLayer` / `Qwen3Model` / `Qwen3ForCausalLM` |

## 核心设计（学习重点）

### 1. 张量并行：head 分片 + 列→行配对

- **head 按卡均分**：`num_heads = total // tp_size`，每卡只持有一部分 head；
- **QKV 列并行 → O 行并行**、**gate_up 列并行 → down 行并行**：每层只通信一次（O/down 输出前的 `all_reduce`）；
- **per-head 天然适配**：attention 的 QK^T、softmax、PV 都不跨 head，分片后无需跨卡通信——这是整个 TP 设计能成立的前提；
- 防线：`assert total_num_kv_heads % tp_size == 0`（GQA 下 KV head 少，必须能整除）。

### 2. 合并存储 + 映射加载

- q/k/v 合并进 `qkv_proj` 一个参数，gate/up 合并进 `gate_up_proj`；
- `packed_modules_mapping` 告诉加载器：`q_proj → (qkv_proj, "q")`，配合 `utils/loader.py` 完成分片加载。

### 3. Q 的存储形态变化（forward 里的数据流）

```text
[num_tokens, hidden_size]
→ [num_tokens, (Q+2KV)*head_dim]      # qkv_proj 合并投影
→ split 成 q/k/v 三段（零拷贝视图）
→ view 成 [num_tokens, num_heads, head_dim]
→ q_norm/k_norm（QK-Norm，Qwen3 无 bias 时启用）
→ RoPE 旋转 q/k（v 不动）
→ FlashAttention → [num_tokens, num_heads, head_dim]
→ flatten(1,-1) → [num_tokens, num_heads*head_dim]
→ o_proj（行并行 + all_reduce）
```

### 4. 其他要点

- **QK-Norm**：Qwen3 无 attention bias 时对 q/k 逐 head 做 RMSNorm，稳定注意力分数；
- **流式残差**：Pre-Norm + residual 以流式传出（省中间张量驻留）；
- **RoPE 只旋转 Q/K**：位置信息进注意力权重，V 是内容不动；
- **tie_word_embeddings**：lm_head 与 embedding 共享权重（`lm_head.weight.data = embed_tokens.weight.data`）；
- 结构参数全部取自 HF `Qwen3Config`，与官方权重/推理行为对齐。

## 与引擎的接口

- `forward(input_ids, positions)`：只返回最后一层 hidden states（logits 不在这里算）；
- `compute_logits(hidden_states)`：单独投影到词表空间——这样 `engine/model_runner.py` 可以灵活选择"算完整前向"还是"CUDA Graph replay + 补算 logits"。
