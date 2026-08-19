# nanovllm/utils —— 工具与基础设施

> 本目录提供被 engine/layers/models 共用的"基础设施"：跨算子传参的全局上下文、
> 以及 safetensors 权重加载。
> 学习要点：两个文件都不长，但分别代表两种重要设计——"全局上下文解耦"与"合并参数映射加载"。

## 文件作用一览

| 文件         | 作用                 | 一句话定位                                                           |
| ------------ | -------------------- | -------------------------------------------------------------------- |
| `context.py` | 进程级全局上下文     | 一次 step 中调度侧 → 算子侧的数据传递（slot_mapping、cu_seqlens 等） |
| `loader.py`  | safetensors 权重加载 | 遍历权重分片文件，按映射拆回合并参数，分片写入各 rank                |

## 核心设计（学习重点）

### `context.py`：为什么用全局变量而不是传参？

- **问题**：attention 等算子需要 slot_mapping（K/V 写哪）、cu_seqlens（序列边界）、block_tables（块映射）等调度侧数据。如果层层传参，模型代码会被推理引擎细节污染。
- **方案**：`ModelRunner.prepare_*` 算好后 `set_context()` 写入全局 `_CONTEXT`，算子 `get_context()` 直接读——**算子与调度解耦**。
- 三个函数：`get_context` / `set_context` / `reset_context`（step 结束清空，防过期数据泄漏）。
- 细节：`set_context` 重建对象而非原地改字段（引用不可变，防污染）；每进程一份（多卡 spawn 子进程各自独立）。

字段速查：

| 字段             | 含义                                                           |
| ---------------- | -------------------------------------------------------------- |
| `is_prefill`     | 本轮 prefill 还是 decode（attention 选 varlen / kvcache 路径） |
| `cu_seqlens_q/k` | 各序列在拼接 batch 中的起止边界（varlen 接口用）               |
| `max_seqlen_q/k` | 本轮最大序列长度                                               |
| `slot_mapping`   | 每个 token 的 K/V 应写入 KV cache 的物理槽位                   |
| `context_lens`   | decode 时各序列已有 KV 历史长度                                |
| `block_tables`   | 逻辑块 → 物理块映射（prefix cache / decode 用）                |

### `loader.py`：合并参数的映射加载

- **背景**：HF 官方权重里 q/k/v 是独立矩阵，本项目的并行层把它们合并成一个参数。
- **机制**：`packed_modules_mapping`（定义在 `models/qwen3.py`）把 `q_proj → ("qkv_proj", "q")`，加载时命中映射的权重交给参数自定义的 `weight_loader(param, tensor, shard_id)`，按区段（"q"/"k"/"v"、0/1）切分写入；未命中的普通参数走默认整块拷贝。
- `for...else` 语法区分两条分支。
- **为什么加载到 CPU**（`safe_open(..., "cpu")`）：先落主机内存再由各参数拷到自己设备/分片，避免 GPU 显存峰值翻倍。
- `glob` 遍历所有 `*.safetensors` 分片文件（大模型多分片场景）。

## 被谁使用

```text
context.py ← engine/model_runner.py（写）、layers/attention.py、layers/embed_head.py（读）
loader.py  ← engine/model_runner.py 的 load_model(self.model, config.model)
```
