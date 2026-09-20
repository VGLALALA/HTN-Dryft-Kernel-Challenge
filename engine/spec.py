"""Exact greedy drafts: n-grams from the prompt plus a Jacobi leftover window.

Verification is the model's own argmax, so every emitted token is native greedy.
"""

from __future__ import annotations

MAX_DRAFT = 8
MAX_OCC = 4


def ngram_cont(seq: list[int], max_draft: int) -> list[int]:
    if max_draft <= 0 or len(seq) < 2:
        return []
    for n in (3, 2):
        if len(seq) < n:
            continue
        needle = seq[-n:]
        # Last occurrence strictly before the current suffix.
        limit = len(seq) - n
        for i in range(limit - 1, -1, -1):
            if seq[i : i + n] == needle:
                cont = seq[i + n : i + n + max_draft]
                if cont:
                    return list(cont)
    return []


class Draft:
    __slots__ = ("seq", "window")

    def __init__(self, seq: list[int]) -> None:
        self.seq = seq
        self.window: list[int] = []

    def propose(self, k: int) -> list[int]:
        if k <= 0:
            return []
        cands: list[list[int]] = []
        if self.window:
            cands.append(self.window[:k])
        ng = ngram_cont(self.seq, k)
        if ng:
            cands.append(ng)
        if not cands:
            return []
        return max(cands, key=len)

    def consume(self, emitted: list[int], leftover: list[int]) -> None:
        self.seq.extend(emitted)
        self.window = leftover
