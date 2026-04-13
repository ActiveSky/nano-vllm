"""提供与 vLLM 风格接近的高层 `LLM` 入口。"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """`LLMEngine` 的对外别名，保持更简洁的使用体验。"""

    pass
