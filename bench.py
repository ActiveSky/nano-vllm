"""基准测试脚本，用随机 token 序列评估 Nano-vLLM 的吞吐性能。"""

import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams

# from vllm import LLM, SamplingParams


def main():
    """构造随机请求并统计生成阶段的吞吐量。"""
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # 默认使用本地下载的 Qwen3 模型权重目录。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 用随机 token 模拟不同长度的 prompt，便于压测调度与缓存逻辑。
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # 如果要对比 vLLM，可以把 prompt 包装成字典格式。
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
