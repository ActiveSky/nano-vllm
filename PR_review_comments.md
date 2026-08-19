# vLLM PR review comments

## PR #52199

建议评论在 `vllm/entrypoints/openai/completion/serving.py` 的 `last_final_res` 附近：

```text
Could this be an issue for Completion requests with multiple prompts? `prompt_tokens`
is aggregated across all prompts, but `prompt_tokens_details` appears to be read
only from `last_final_res` (and the streaming path keeps only the first result).
If so, the cache statistics would describe only one prompt rather than the whole
request, and streaming values could depend on result order. Would it make sense
to aggregate cached/local/external tokens once per prompt (without multiplying
by `n`) and add a multi-prompt regression test?
```

## PR #52201

建议评论在 `vllm/entrypoints/openai/completion/serving.py` 的 `last_final_res` 附近：

```text
Could this be an issue for Completion requests with multiple prompts? `prompt_tokens`
is aggregated across all prompts, but `prompt_tokens_details` appears to be read
only from `last_final_res` (and the streaming path keeps only the first result).
If so, the cache statistics would describe only one prompt rather than the whole
request, and streaming values could depend on result order. Would it make sense
to aggregate cached/local/external tokens once per prompt (without multiplying
by `n`) and add a multi-prompt regression test?
```
