"""Qwen3-4B greedy engine with exact n-gram speculative verify.

Drafts are prompt n-grams. The packed model teacher-forces them and emits
only its own argmax tokens. T=1 CUDA-graph decode is the no-draft path.
"""

from __future__ import annotations

import torch

from model import QwenRunner
from spec import MAX_DRAFT, Draft


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

        drafts = [Draft(list(seq)) for seq in input_ids]
        ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
        remaining = max_new_tokens

        with torch.inference_mode():
            runner.prefill(ids)
            first = runner.tokens_to_host()
            for i, tok in enumerate(first):
                drafts[i].consume([tok], [])
            yield first
            remaining -= 1
            pos = prompt_len  # cache slot for the token just emitted

            while remaining > 0:
                k = min(MAX_DRAFT, remaining)
                proposals = [d.propose(k) for d in drafts]
                depth = min((len(p) for p in proposals), default=0)

                if depth == 0:
                    runner.token.copy_(runner.out_ids)
                    if runner.graph is not None:
                        runner.bind_pos(pos)
                        runner.replay()
                    else:
                        runner.decode_step_eager(pos)
                    step = runner.tokens_to_host()
                    for i, tok in enumerate(step):
                        drafts[i].consume([tok], [])
                    yield step
                    remaining -= 1
                    pos += 1
                    continue

                # Teacher-force last greedy token + drafted next tokens.
                # Cache write starts at `pos` (slot of last emitted token).
                width = depth + 1
                tokens = torch.empty(batch, width, dtype=torch.int64, device="cuda:0")
                tokens[:, 0] = runner.out_ids
                for b, p in enumerate(proposals):
                    tokens[b, 1:] = torch.tensor(p[:depth], dtype=torch.int64, device="cuda:0")

                greedy = runner.verify(tokens, pos).tolist()
                draft = tokens[:, 1:].tolist()

                # greedy[b][0] is the true next token after last emit.
                # Accept drafted token j if greedy[b][j] == draft[b][j] for all b.
                # greedy[b][0] is always the true next token (cache-consistent).
                # Accept drafted token j if it equals greedy[b][j] for every seq.
                matched = 0
                for j in range(depth):
                    if any(greedy[b][j] != draft[b][j] for b in range(batch)):
                        break
                    matched += 1
                n_emit = min(matched + 1, remaining)  # +1 bonus token after last match, or just greedy[0]

                for col in range(n_emit):
                    step = [greedy[b][col] for b in range(batch)]
                    for b in range(batch):
                        leftover = draft[b][col:] if col == n_emit - 1 and matched == depth else []
                        drafts[b].consume([step[b]], leftover)
                    yield step

                remaining -= n_emit
                pos += n_emit
                runner.out_ids.copy_(
                    torch.tensor([greedy[b][n_emit - 1] for b in range(batch)], device="cuda:0")
                )
