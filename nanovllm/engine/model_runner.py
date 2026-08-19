"""GPU 进程内的模型执行器：加载权重、维护 KV cache、执行 prefill/decode 前向与采样。

进程拓扑（学习重点）：
- rank 0 运行在主进程（LLMEngine 所在进程），其余 rank 各起一个子进程。
- 子进程通过「共享内存 + Event」接收主进程的方法调用指令（read_shm/write_shm），
  模型前向则通过 NCCL 完成跨卡同步（all_reduce 等）。
- 所有 rank 加载的是同一模型的不同权重分片（张量并行），执行同一份前向代码。
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """在 GPU 进程中加载模型、维护 KV cache 并执行 prefill/decode 前向与采样。

    核心职责（学习重点）：
    1. 初始化：建 NCCL 进程组、加载并行权重、warmup 预热、按剩余显存分配
       KV cache、捕获 CUDA Graph。
    2. 数据准备：把调度器选出的 Sequence 列表翻译成 GPU 张量（input_ids、
       positions、slot_mapping、block_tables 等），并通过全局 Context 传给
       attention 等算子——这是「调度状态 → 算子输入」的桥梁。
    3. 执行与采样：跑模型前向得到 logits，再采样出下一个 token id。
    4. 多卡协调：rank 0 通过共享内存把方法调用广播给所有从进程，保证
       各 rank 执行同一个方法、同一份前向。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # —— 初始化 NCCL 进程组（张量并行的通信底座）——
        # tcp://localhost:2333 是本机单机多卡场景的简易初始化方式；
        # rank 0 先建组，其他 rank 加入。之后 all_reduce 等算子才能工作。
        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        # 默认 dtype 与默认设备在构造期间临时切到 cuda + 模型精度，
        # 让模型参数和 KV cache 直接在 GPU 上创建，构造完再恢复。
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 实例化并行模型结构，并从磁盘加载权重（每个 rank 只加载自己的分片）。
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 初始化顺序（学习重点）：
        # warmup 在前：先跑一次最大 batch 的假 prefill，触发 cuDNN/编译缓存，
        #   并让显存分配达到峰值，为下面「按剩余显存算 KV cache 块数」提供依据。
        # allocate 在中：按 warmup 后的剩余显存分配 KV cache，并挂载到各层。
        # capture 在后：KV cache 地址确定后，才能捕获 CUDA Graph（地址必须固定）。
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # —— 多进程拓扑：rank 0 留在主进程，rank>0 进入从进程服务循环 ——
        # 主进程创建共享内存；从进程打开同一块共享内存后进入 loop()，
        # 阻塞等待主进程通过 Event 唤醒并派发方法调用。
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 等所有从进程就绪
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 从进程在此循环直到收到 exit，不返回

    def exit(self):
        """释放共享内存与 CUDA Graph 资源并销毁进程组。"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()  # 只有创建者能 unlink
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """rank>0 的从进程主循环：持续从共享内存读取并分发方法调用。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """从进程等待事件后从共享内存反序列化出方法名与参数。

        协议（学习重点）：
        - 前 4 字节：小端整数，记录本次负载长度 n；
        - 之后 n 字节：pickle 序列化的 [method_name, *args]。
        主进程 set() Event 唤醒等待中的从进程，从进程读完 clear() 复位。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """主进程把方法调用序列化写入共享内存并唤醒所有从进程。"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()  # 唤醒每一个从进程

    def call(self, method_name, *args):
        """同步分发方法调用：多卡时先广播给从进程，再在本地执行。

        学习重点：为什么用共享内存而不直接跨进程调函数？
        - 所有 rank 必须执行「同一个方法、同一份参数」，前向才能对齐
          （NCCL 集合通信要求所有 rank 都参与，缺一不可）。
        - 共享内存 + Event 比套接字/RPC 开销小、实现简单，适合本机多进程。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """用最大 batch 的假数据跑一次 prefill，触发 cudnn/编译缓存并统计峰值显存。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        # 构造「一条最长的序列」或「多条较短序列」，凑满 max_num_batched_tokens。
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据剩余显存与目标利用率计算 KV cache 块数，并为每层 attention 挂载 k/v 缓存。

        学习重点：
        - 显存公式：能用的 KV cache 显存 = 目标占用 - 当前已用 - 峰值波动，
          再除以单块的字节数，得到块数（PagedAttention 的「分块」思想）。
        - 按 head 分片：num_kv_heads 除以 world_size，每张卡只存自己那部分
          KV head——与 QKV 列并行切分完全对齐，attention 只读本卡 cache。
        - 挂载方式：遍历模型模块，把每层 Attention 的空 cache 占位替换成
          这个大张量的切片（通过 hasattr 判断，模型代码保持「零 cache 逻辑」）。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free  # 当前已用显存
        # warmup 时的峰值显存：KV cache 必须躲开这个峰值，否则推理时会 OOM。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 每卡只分到 1/world_size 的 KV head（与 QKV 列并行一致）。
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        # 单块字节数：K/V 两个 × 层数 × 块内 token 数 × KV head × head_dim × dtype。
        # 注意与 kv_cache 张量的维度一一对应：2, L, blocks, block_size, kv_heads, head_dim。
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        # 可用显存预算 = 目标总占用 - 模型已用 - 峰值余量（+current 把峰值里
        # 已释放的部分加回来），整除单块字节数得到总块数。
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        # 一次性分配整块 KV cache（连续显存，CUDA Graph 友好的固定地址）：
        # [K/V, 层, 块, 块内位置, KV head, head_dim]
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        # 逐层挂载：把第 layer_id 层的 k/v 切片注入对应的 Attention 模块，
        # 之后 attention 前向里 store_kvcache / flash_attn_* 就能直接读写它。
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把各序列的 block_table 补齐到同一长度后打包成 GPU 张量。

        学习重点：GPU kernel 要求所有序列等长，但不同序列占用的块数不同，
        所以用 -1 填充到最长（-1 在 flash-attention 中表示无效块），
        打包成 [num_seqs, max_blocks] 的二维张量。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """为 prefill 阶段组装 input_ids/positions/cu_seqlens 与逐块 slot_mapping，支持 chunked prefill 与 prefix cache。

        学习重点（PagedAttention 的核心：slot_mapping）：
        - slot = 物理槽位编号，表示「这个 token 的 K/V 应写入 KV cache 的哪里」，
          由 block_table（逻辑块 → 物理块映射）和块内偏移共同算出。
        - cu_seqlens 让 flash_attn_varlen 知道每个序列从哪里开始到哪里结束，
          从而在一个 kernel 里处理不等长序列（无需 padding）。
        - 支持 chunked prefill：start 从 num_cached_tokens 开始，只算本轮
          调度的部分 token；prefix cache 命中时 block_tables 非空。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # start = 已缓存的 token 数（chunked prefill 时 > 0），
            # seqlen_q = 本轮要处理的 token 数，end = 本轮结束位置。
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))  # 位置编号是全局连续的
            # cu_seqlens 逐序列累加，形成 [0, s1, s1+s2, ...] 的区间边界。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup 阶段还没有分配块
                continue
            # 遍历本轮涉及的所有逻辑块，把「块首物理槽位 + 块内偏移」翻译成
            # 每个 token 的物理 slot。slot = block_table[i] * block_size + 块内偏移。
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size  # 首个块可能有部分偏移
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = (
                        seq.block_table[i] * self.block_size + end - i * self.block_size
                    )
                slot_mapping.extend(range(slot_start, slot_end))
        # prefix cache 场景：K/V 长度大于本轮 Q 长度，flash_attn_varlen 需要
        # block_table 从 cache 中读取历史 K/V（本段 K/V 已由前面的 prefill 写入）。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # pin_memory + non_blocking：先把数据钉在页锁定内存，再异步拷贝到 GPU，
        # 与 GPU 计算重叠，减少主机到设备的传输延迟。
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        # 通过全局 Context 传递给 attention：算子不需要显式接收这些参数，
        # 直接从 get_context() 读取——避免在模型代码里层层传参。
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """为 decode 阶段组装单 token 的输入、位置与 slot 映射，并打包 block_tables。"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # decode 每步每个序列只处理 1 个 token：最后一个已生成 token。
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)  # 位置 = 序列当前长度 - 1
            context_lens.append(len(seq))  # 该序列的完整 KV 历史长度
            # 新 token 的写入槽位：最后一块的块首 + 块内已有 token 数。
            # last_block_num_tokens 由调度器维护（写入时递增）。
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """把每个序列的采样温度打包成 GPU 张量。"""
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        """执行模型前向并返回 logits：prefill/大 batch 走 eager，小 decode 走 CUDA Graph replay。

        学习重点（为什么 decode 用 CUDA Graph）：
        - decode 每步只算 1 个 token/序列，kernel 启动开销占比很高。
        - CUDA Graph 把整条前向的 kernel 依赖关系提前「录制」成一张图，
          之后每次 replay 只提交一次启动，省掉逐 kernel 启动的 CPU 开销。
        - 但 Graph 要求输入/输出地址固定（捕获时锁定），所以每步先把新数据
          拷进预分配的 graph_vars 缓冲区，再 replay。
        - prefill 是 compute-bound，kernel 启动开销占比低，且序列长度不定，
          不适合 Graph，直接 eager 执行。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 选出不小于当前 batch 的最小预录制尺寸（Graph 以固定 batch 捕获）。
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 把本步的真实数据写入预分配缓冲区（Graph 内读的是这些固定地址）。
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)  # 无效槽位先填 -1
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()  # 一次性重放整张前向图
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """一次完整推理：准备输入 → 前向 → 采样，返回 token id 列表（仅 rank 0）。

        学习重点：所有 rank 都执行 run，但只有 rank 0 采样返回 token id；
        从进程的返回值为 None（不会被使用，它们只负责参与前向计算）。
        """
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        # 清空全局 Context，避免下次 step 读到过期的 slot_mapping 等状态。
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """为若干固定 batch 尺寸录制 decode 前向的 CUDA Graph，供 run_model 重放。

        学习重点（捕获流程）：
        - 先 warmup 一次：让 cuDNN 等完成自选型并分配 workspace。
        - 再在 torch.cuda.graph 上下文里跑一次前向：这次把 kernel 依赖录制下来。
        - 用独立内存池（graph.pool）保证捕获与重放期间地址稳定。
        - 预分配一份固定大小的 graph_vars 缓冲区，重放时往里面拷数据。
        """
        config = self.config
        hf_config = config.hf_config
        # 预分配固定大小的输入/输出缓冲区（Graph 捕获后地址不能再变）。
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 覆盖常见 batch 尺寸：1,2,4,8,16,32,...（run_model 取不小于 bs 的最小档）。
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
