import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """推理引擎：串联 tokenizer、调度器与多进程模型执行器，对外提供 generate 入口。"""

    def __init__(self, model, **kwargs):
        # =================================================================
        # 阶段 1. 构造 Config 并过滤 kwargs
        # =================================================================
        # 用 dataclasses.fields 内省出 Config 的全部字段名，作为「合法参数白名单」。
        # 这样 LLMEngine 不必逐个枚举 Config 的字段，新增字段时自动透传，实现两层解耦。
        config_fields = {field.name for field in fields(Config)}
        # 过滤 kwargs：只保留 Config 认得的键，丢弃拼写错误或无关参数，
        # 避免 Config(**kwargs) 因收到未知字段而 TypeError。
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 构造 Config；__post_init__ 会校验参数、从 HF 读取模型元信息并收敛 max_model_len。
        config = Config(model, **config_kwargs)

        # =================================================================
        # 阶段 2. 设置 Sequence 类的全局 block_size
        # =================================================================
        # 把 block_size 挂到 Sequence 类属性上，所有 Sequence 实例共享，
        # block_manager 后续按这个尺寸切分 KV cache 块。
        Sequence.block_size = config.kvcache_block_size

        # =================================================================
        # 阶段 3. 拉起张量并行子进程
        # =================================================================
        # 进程拓扑：rank 0 留在主进程（self.model_runner），rank 1..N-1 各起一个独立子进程。
        # 子进程跑各自的 ModelRunner，通过 NCCL 通信 + 共享内存接收主进程指令。
        self.ps = []  # 子进程列表（不含主进程的 rank 0）
        self.events = []  # 每个子进程对应一个 Event，用于主从同步
        # 用 spawn 而非 fork：CUDA 状态和多线程锁不能安全地被 fork 继承，
        # spawn 会起一个全新的解释器进程，干净但需要重新 import + 序列化参数。
        ctx = mp.get_context("spawn")
        # range(1, tensor_parallel_size)：单卡时 range(1, 1) 为空，不拉任何子进程。
        for i in range(1, config.tensor_parallel_size):
            # 每个子进程一个 Event：主进程 set() 唤醒它读共享内存，子进程读完后 clear() 复位。
            event = ctx.Event()
            # 子进程入口是 ModelRunner(config, rank=i, event)；
            # 它构造完模型后会进入 loop()，阻塞在 event.wait() 上等主进程发指令。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()  # 启动子进程（此处不阻塞，立刻返回）
            self.ps.append(process)
            self.events.append(event)

        # =================================================================
        # 阶段 4. 构造主进程的 ModelRunner（整个构造里最重的一步）
        # =================================================================
        # 内部会：1) 初始化 NCCL 进程组 2) 设 CUDA 设备/默认 dtype
        #         3) 实例化 Qwen3 模型并加载权重 4) warmup 预热 5) 分配 KV cache
        #         6) 非 enforce_eager 时捕获 CUDA Graph
        # 多卡时还会创建 SharedMemory("nanovllm") 并 barrier 等所有从进程就绪。
        self.model_runner = ModelRunner(config, 0, self.events)

        # =================================================================
        # 阶段 5. 加载 tokenizer 并回填 eos
        # =================================================================
        # 加载 HuggingFace fast tokenizer，供 add_request 编码、generate 解码使用。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 把 Config.eos（默认 -1）回填成真实的 EOS token id，
        # Scheduler.postprocess 用它判断序列是否生成结束。
        config.eos = self.tokenizer.eos_token_id

        # =================================================================
        # 阶段 6. 构造 Scheduler 并注册退出钩子
        # =================================================================
        # 构造调度器：内部建 BlockManager(num_kvcache_blocks, block_size)
        # 并初始化 waiting/running 两个 deque，这是 continuous batching 的核心调度结构。
        self.scheduler = Scheduler(config)
        # 注册退出钩子：解释器退出时自动调 self.exit()，
        # 通知所有子进程退出、回收共享内存与 NCCL 进程组，避免孤儿进程和资源泄漏。
        atexit.register(self.exit)

    def exit(self):
        """通知模型进程退出并等待所有子进程结束。"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """把单条 prompt 编码成 Sequence 后交给调度器入队。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """核心函数，执行一次调度、前向与后处理，并返回已完成序列的输出与本步 token 吞吐量。"""
        # 1. Scheduler 决定本轮要运行哪些请求，以及统一执行 prefill 还是 decode。
        #    它只更新调度状态和 KV cache 分配，不进行模型计算，也不会生成 token。
        seqs, is_prefill = self.scheduler.schedule()

        # 2. 记录本轮 token 数，供 generate() 分别统计 prefill / decode 吞吐量。
        #    prefill 一条序列可能处理多个 token，故累加 num_scheduled_tokens；
        #    decode 每条序列固定生成一个 token，使用负号作为“这是 decode 轮”的标记。
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        )

        # 3. 真正执行模型：准备 GPU 输入和 KV cache 映射 → Transformer 前向 → 采样。
        #    返回值与 seqs 一一对应，每个元素是该序列本轮采样得到的下一个 token id。
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 4. 将新 token 写回 Sequence，推进 prefill 进度，更新 prefix cache；
        #    若命中 EOS 或达到 max_tokens，则释放该序列的 KV cache 并标记为 FINISHED。
        self.scheduler.postprocess(seqs, token_ids, is_prefill)

        # 5. 仅收集恰好在本轮结束的序列；未完成序列仍留在 Scheduler 的 running 队列。
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        """判断引擎是否已无待处理请求。"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """批量生成文本，逐条入队后循环 step，返回每个请求的 text 与 token_ids。

        本质是「入队 + 循环 step + 收集结果」的驱动循环，把 add_request / step /
        is_finished 三个底层方法串成用户友好的批量生成 API。完成顺序可能与提交顺序
        不同（短请求先完成），最后按 seq_id 排序还原，对用户透明。
        """
        # =================================================================
        # 阶段 1. 初始化 tqdm 进度条
        # =================================================================
        # total=len(prompts)：进度条总数 = 请求条数（不是 token 数），每个请求完成时 update 1。
        # dynamic_ncols：终端宽度自适应；disable=not use_tqdm：日志场景可关掉进度条。
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )

        # =================================================================
        # 阶段 2. 归一化 sampling_params（单条 → 列表）
        # =================================================================
        # 支持两种调用方式：
        #   A. 所有请求共享同一组采样参数：generate(prompts, SamplingParams(...))
        #      → 内部展开成 [sp, sp, sp, ...]
        #   B. 每条请求独立采样参数：generate(prompts, [sp1, sp2, ...])
        #      → 保持原样
        # [sp] * len(prompts) 是浅拷贝复用——所有请求共享同一个 SamplingParams 对象，
        # 但 Sequence.__init__ 只读其字段不会修改，所以共享是安全的。
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # =================================================================
        # 阶段 3. 逐条入队（只入队不执行）
        # =================================================================
        # add_request 内部：str → tokenizer.encode → Sequence → scheduler.waiting。
        # 所有请求先进 waiting 队列，后续 step 循环才真正调度执行。
        # 这就是 continuous batching 的入口：批量提交，调度器自己决定每步算哪些、算多少。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # =================================================================
        # 阶段 4. 主循环：反复 step 直到全部完成
        # =================================================================
        # outputs 用 dict 按 seq_id 收集，因为请求完成顺序 ≠ 提交顺序（短请求可能先完成，
        # prefill 与 decode 交叉进行）。seq_id 是 Sequence.counter 发的全局自增整数，
        # 最后 sorted(outputs.keys()) 升序就是提交顺序。
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        # is_finished: waiting 和 running 都空了才停。
        while not self.is_finished():
            t = perf_counter()
            # step 返回 (outputs, num_tokens)：
            #   outputs = 本步完成序列的 [(seq_id, completion_token_ids)]
            #   num_tokens > 0  → prefill 步，本轮处理的 token 总数（可能多序列 + chunked）
            #   num_tokens < 0  → decode 步，-len(seqs)，即本轮 decode 的序列数
            # 用正负号区分 prefill/decode 步，循环里据此分流到对应吞吐量统计。
            output, num_tokens = self.step()
            # —— 吞吐量统计：prefill 和 decode 语义不同、瓶颈不同 ——
            # prefill 是 compute-bound（大量 token 一次算），衡量「吃 prompt 速度」
            # decode 是 memory-bound（每步 1 token），衡量「吐 token 速度」
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix(
                {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
            )
            # 收集本步完成的序列到 outputs dict，并推进进度条。
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()

        # =================================================================
        # 阶段 5. 按 seq_id 排序还原提交顺序 + decode 成文本
        # =================================================================
        # dict → 按 seq_id 升序的 list，保证返回顺序和 prompts 输入顺序一致，
        # 屏蔽 continuous batching 内部乱序完成的复杂性。
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # token_ids → text，包成 {"text": ..., "token_ids": ...} 返回。
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        return outputs
