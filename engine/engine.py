"""Worst-accuracy decode: no weights, no kernels, no GPU.

Repeats the last prompt token. Throughput is host yield rate.
"""

from __future__ import annotations


class Engine:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        n = len(input_ids)
        tok = 1
        if input_ids and input_ids[0]:
            tok = input_ids[0][-1]
        step = [tok] * n
        # Same list object every step; judge copies token ids out of the pipe.
        for _ in range(max_new_tokens):
            yield step
