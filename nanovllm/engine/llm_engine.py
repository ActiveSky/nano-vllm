"""封装模型进程管理、调度和文本生成的高层推理引擎。"""

import atexit
from dataclasses import fields
from time import perf_counter
from typing import TypedDict

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class GenerationOutput(TypedDict):
    """一次生成结果中返回的文本和 token 信息。"""

    text: str
    token_ids: list[int]


class LLMEngine:

    """负责进程初始化、请求调度和对外生成接口。"""

    def __init__(self, model: str, **kwargs):
        """根据模型路径和配置参数启动整个推理引擎。"""

        # 只保留 Config dataclass 中定义的字段，避免把无关参数传给 Config。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 为每个张量并行 rank 启动一个子进程。
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self) -> None:
        """释放模型进程和相关资源。"""

        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> None:
        """把一个 prompt 转成序列对象并加入调度器。"""

        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """执行一次调度、前向和后处理，并返回已完成的输出。"""

        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self) -> bool:
        """判断整个引擎是否已经没有待处理请求。"""

        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[GenerationOutput]:
        """批量生成文本，并返回每个请求的文本和 token id。"""

        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # 逐条将请求放入等待队列，随后由调度器统一处理。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        completed_token_ids = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        generation_outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in completed_token_ids]
        if use_tqdm:
            pbar.close()
        return generation_outputs
