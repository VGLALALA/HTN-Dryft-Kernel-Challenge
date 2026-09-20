"""Qwen3-4B greedy engine: packed weights, static KV, CUDA-graph decode.

Graph attention uses SDPA over the padded cache with an additive mask.
"""

from __future__ import annotations

import torch

from model import QwenRunner


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.runner = QwenRunner(model_path)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        batch = len(input_ids)
        prompt_len = len(input_ids[0])
        runner = self.runner
        runner.prepare(batch, prompt_len + max_new_tokens)

        ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
        with torch.inference_mode():
            runner.prefill(ids)
            yield runner.tokens_to_host()
            use_graph = runner.graph is not None
            for t in range(max_new_tokens - 1):
                pos = prompt_len + t
                runner.token.copy_(runner.out_ids)
                if use_graph:
                    runner.bind_pos(pos)
                    runner.replay()
                else:
                    runner.decode_step_eager(pos)
                yield runner.tokens_to_host()
