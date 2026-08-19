# nanovllm/engine —— 推理引擎核心

> 本目录是整个推理引擎的心脏：从"请求进来"到"token 吐出"的所有执行逻辑。
> 学习顺序建议：`block_manager.py` → `scheduler.py` → `model_runner.py` → `llm_engine.py` → `sequence.py`。

## 文件作用一览

| 文件               | 作用             | 一句话定位                                                                                     |
| ------------------ | ---------------- | ---------------------------------------------------------------------------------------------- |
| `sequence.py`      | 单条请求的状态机 | 记录 token 序列、`block_table`、采样参数、进度（`num_cached_tokens` / `num_scheduled_tokens`） |
| `block_manager.py` | KV cache 块管理  | PagedAttention 的显存调度：按块分配/回收、**prefix cache 哈希复用**                            |
| `scheduler.py`     | 请求调度器       | **连续批处理**：每步决定哪些序列跑 prefill 还是 decode，显存不足时抢占                         |
| `model_runner.py`  | GPU 执行器       | 权重加载、KV cache 分配、输入打包（slot_mapping 等）、前向、采样、CUDA Graph                   |
| `llm_engine.py`    | 引擎编排入口     | tokenizer、spawn 子进程、`step()` 主循环（调度→前向→后处理→吞吐统计）                          |

## 核心加速技巧（学习重点）

### ★★★ `block_manager.py`：PagedAttention 块管理

- **按块分配**：KV cache 不再"每条序列预分配整块"，而是按需分配固定大小的块（`block_size=256`），消灭显存碎片和预分配浪费——这是 vLLM 的成名作。
- **prefix cache**：`compute_hash()` 用 token 序列哈希作为块的键，相同前缀（系统提示词、多轮对话）直接复用已算好的 KV 块，省掉重复 prefill。
- **软性显存限制**：`can_allocate` / `may_append` 在显存不足时返回失败而非 OOM，让调度器有机会抢占。

### ★★★ `scheduler.py`：连续批处理（continuous batching）

- **动态调度**：不再等整批请求全部完成才释放 GPU——每个 decode 步都检查"有请求完成吗？有显存吗？"，完成即释放、新请求即插入，GPU 永不空转。
- **chunked prefill**：长 prompt 拆成多轮调度，避免 prefill 独占 GPU 太久饿死 decode 请求。
- **抢占（preempt）**：显存不够时把长序列的块还回去，先让短请求跑完（以空间换时间）。

### ★★★ `model_runner.py`：执行层加速

- **CUDA Graph**：decode 每步只 1 个 token，kernel 启动开销占比高 → 把整条前向录制成一个 Graph，每步 `replay()` 一次搞定。
- **warmup → allocate → capture 顺序**：warmup 触发编译缓存并测显存峰值 → 按剩余显存分 KV cache → cache 地址固定后捕获 Graph（顺序不能换）。
- **pin_memory + non_blocking**：主机到设备拷贝与 GPU 计算重叠。
- **slot_mapping**：把调度器的逻辑块映射翻译成物理槽位（PagedAttention 的寻址核心）。

## 数据流全景

```text
LLMEngine.generate()
  → scheduler.add() 入队（waiting 队列）
  → step() 循环：
      scheduler.schedule()      # 选序列、决定 prefill/decode、分配块
      model_runner.run()        # 打包输入 → 模型前向 → 采样
      scheduler.postprocess()   # 写回 token、推进进度、判断结束/释放块
  → 输出收集
```
