"""Qwen3-4B greedy engine: packed weights, static KV cache, SDPA attention."""

from __future__ import annotations

import torch

from model import QwenRunner


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.runner = QwenRunner(model_path)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        batch = len(input_ids)
        prompt_len = len(input_ids[0])
        runner = self.runner
        runner.prepare(batch, prompt_len + max_new_tokens)

        ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
        with torch.inference_mode():
            runner.prefill(ids)
            yield runner.tokens_to_host()
            for t in range(max_new_tokens - 1):
                runner.token.copy_(runner.out_ids)
                runner.decode_step(prompt_len + t)
                yield runner.tokens_to_host()
