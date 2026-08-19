# nanovllm —— Nano-vLLM 包总览

> 一个用于学习的最小 vLLM 实现：连续批处理 + PagedAttention + 张量并行 + FlashAttention。
> 本 README 是**分层学习地图**：从顶层入口出发，逐层深入各子目录。

## 包结构总览

```text
nanovllm/
├── config.py              # 静态配置（模型路径、batch 上限、TP 规模、KV cache 块大小）
├── llm.py                 # 用户级 API（LLM 类，对标 vLLM 的 LLM.generate）
├── sampling_params.py     # 采样参数（temperature 等）
├── engine/                # ★ 推理引擎核心（调度 + 块管理 + 执行器）→ 见 engine/README.md
├── layers/                # ★ 算子库（并行线性层、attention、RoPE、norm、采样）→ 见 layers/README.md
├── models/                # ★ 模型定义（Qwen3 组装）→ 见 models/README.md
└── utils/                 # 基础设施（全局上下文、权重加载）→ 见 utils/README.md
```

## 顶层文件作用

| 文件                 | 作用                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `config.py`          | `Config` 数据类：启动参数校验 + 读取 HF `config.json`；其中 `max_model_len = min(配置值, max_position_embeddings)` |
| `llm.py`             | 对外 API：`LLM(prompt, sampling_params)`，内部创建 `LLMEngine` 并驱动 `generate`                                   |
| `sampling_params.py` | `SamplingParams`：采样配置（temperature、max_tokens 等），被 `Sequence` 引用                                       |

## 分层学习地图（推荐阅读顺序）

```text
第 1 层：入口
  llm.py → engine/llm_engine.py（引擎编排，step 主循环）

第 2 层：调度与显存（vLLM 的灵魂）
  engine/block_manager.py（PagedAttention 块管理 + prefix cache）
  → engine/scheduler.py（连续批处理、chunked prefill、抢占）
  → engine/sequence.py（请求状态机）

第 3 层：执行器
  engine/model_runner.py（输入打包、KV cache 分配、前向、CUDA Graph）
  → utils/context.py（算子间数据传递）
  → utils/loader.py（权重加载）

第 4 层：模型与算子
  models/qwen3.py（模型组装）
  → layers/linear.py（列/行并行线性层）
  → layers/attention.py（FlashAttention + Triton KV 写入）
  → layers/rotary_embedding.py / layernorm.py / activation.py / embed_head.py / sampler.py

第 5 层：学习笔记
  Note.md（TP 数学基础 + 每日学习记录）
```

## 一条请求的完整旅程

```text
LLM.generate(prompts, sampling_params)
  → LLMEngine.generate：add_request 入队
  → step() 循环：
      scheduler.schedule()        # 决定本轮跑谁、跑 prefill 还是 decode，分配 KV 块
      model_runner.run(seqs)      # 打包输入 → Qwen3 前向 → 采样出 token
      scheduler.postprocess()     # 写回 token、释放完成请求的块、判断结束
  → 按 seq_id 排序返回结果
```

## 加速点速查（详见各子目录 README）

| 层   | 加速点                                                                                  |
| ---- | --------------------------------------------------------------------------------------- |
| 调度 | 连续批处理（GPU 永不空转）、chunked prefill、抢占                                       |
| 显存 | PagedAttention 按块分配、prefix cache 哈希复用                                          |
| 内核 | FlashAttention（varlen/kvcache）、Triton 写缓存、RoPE 预计算、Gumbel-max 采样、算子融合 |
| 执行 | CUDA Graph、pin_memory 异步拷贝、warmup 预编译                                          |
| 并行 | 张量并行列→行零通信衔接、词表并行、KV cache 按 head 分片                                |
