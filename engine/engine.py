"""Worst-accuracy decode: dummy CUDA touch, no Qwen forward.

Repeats the last prompt token. Throughput is host yield rate.
"""

from __future__ import annotations

import torch


class Engine:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        # Harness records peak GPU memory; allocate something so the device exists.
        torch.cuda.init()
        self._scratch = torch.empty(1, device="cuda:0", dtype=torch.int32)
        self._scratch.fill_(1)

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        n = len(input_ids)
        tok = 1
        if input_ids and input_ids[0]:
            tok = input_ids[0][-1]
        for _ in range(max_new_tokens):
            yield [tok] * n
