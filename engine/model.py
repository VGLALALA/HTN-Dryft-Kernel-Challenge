"""Custom Qwen3-4B greedy runner: packed BF16 weights, static KV, SDPA.

Numerics follow Transformers 4.51.3 / Qwen3-4B-Instruct-2507:
RMSNorm reduces in fp32 then casts to BF16 *before* the weight multiply,
NeoX RoPE, GQA=4, SwiGLU, tied embeddings, scale 1/sqrt(128).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

N_LAYERS = 36
HIDDEN = 2560
INTERMEDIATE = 9728
N_Q = 32
N_KV = 8
HEAD_DIM = 128
GQA = 4
Q_DIM = N_Q * HEAD_DIM
KV_DIM = N_KV * HEAD_DIM
EPS = 1e-6
ROPE_THETA = 5_000_000.0
SM_SCALE = HEAD_DIM**-0.5
PREFILL_CHUNK = 2048
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Match Qwen3RMSNorm: fp32 reduce, BF16 cast, then weight."""
    input_dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + EPS)
    return weight * y.to(input_dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX half-rotate. cos/sin duplicate the 64 frequencies."""
    x1, x2 = x[..., :64], x[..., 64:]
    c, s = cos[..., :64], sin[..., :64]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    kwargs = dict(dropout_p=0.0, scale=SM_SCALE, is_causal=is_causal, attn_mask=attn_mask)
    try:
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **kwargs)
    except TypeError:
        if k.shape[1] != q.shape[1]:
            k = k.repeat_interleave(GQA, dim=1)
            v = v.repeat_interleave(GQA, dim=1)
        return F.scaled_dot_product_attention(q, k, v, **kwargs)


@dataclass
class PackedLayer:
    qkv: torch.Tensor
    o: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor


def _build_rope(max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        ROPE_THETA
        ** (torch.arange(0, HEAD_DIM, 2, device=DEVICE, dtype=torch.float32) / HEAD_DIM)
    )
    t = torch.arange(max_len, device=DEVICE, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(DTYPE), emb.sin().to(DTYPE)


def load_packed(model_path: str) -> tuple[torch.Tensor, list[PackedLayer], torch.Tensor]:
    with torch.inference_mode():
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=DTYPE,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to(DEVICE)
        )
        base = model.model
        embed = base.embed_tokens.weight.detach().clone().contiguous()
        layers: list[PackedLayer] = []
        for block in base.layers:
            attn = block.self_attn
            mlp = block.mlp
            qkv = torch.cat(
                (
                    attn.q_proj.weight.detach(),
                    attn.k_proj.weight.detach(),
                    attn.v_proj.weight.detach(),
                ),
                dim=0,
            ).contiguous()
            layers.append(
                PackedLayer(
                    qkv=qkv,
                    o=attn.o_proj.weight.detach().clone().contiguous(),
                    gate=mlp.gate_proj.weight.detach().clone().contiguous(),
                    up=mlp.up_proj.weight.detach().clone().contiguous(),
                    down=mlp.down_proj.weight.detach().clone().contiguous(),
                    attn_norm=block.input_layernorm.weight.detach().clone().contiguous(),
                    ffn_norm=block.post_attention_layernorm.weight.detach().clone().contiguous(),
                    q_norm=attn.q_norm.weight.detach().clone().contiguous(),
                    k_norm=attn.k_norm.weight.detach().clone().contiguous(),
                )
            )
        final_norm = base.norm.weight.detach().clone().contiguous()
        del model
    gc.collect()
    torch.cuda.empty_cache()
    return embed, layers, final_norm


class QwenRunner:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        self.embed, self.layers, self.final_norm = load_packed(model_path)
        self.batch = 0
        self.max_len = 0
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        self.cos: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None
        self.token = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        self.out_ids = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        self.pinned_out = torch.empty(0, dtype=torch.int64, pin_memory=True)

    def prepare(self, batch: int, max_len: int) -> None:
        if batch == self.batch and max_len <= self.max_len and self.k_cache:
            return
        self.batch = batch
        self.max_len = max_len
        self.k_cache = [
            torch.zeros(batch, N_KV, max_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
            for _ in range(N_LAYERS)
        ]
        self.v_cache = [
            torch.zeros(batch, N_KV, max_len, HEAD_DIM, device=DEVICE, dtype=DTYPE)
            for _ in range(N_LAYERS)
        ]
        self.cos, self.sin = _build_rope(max_len)
        self.token = torch.zeros(batch, device=DEVICE, dtype=torch.int64)
        self.out_ids = torch.zeros(batch, device=DEVICE, dtype=torch.int64)
        self.pinned_out = torch.empty(batch, dtype=torch.int64, pin_memory=True)

    def decode_step(self, pos: int) -> None:
        b = self.batch
        slen = pos + 1
        x = F.embedding(self.token, self.embed)
        cos = self.cos[pos].view(1, 1, HEAD_DIM)
        sin = self.sin[pos].view(1, 1, HEAD_DIM)
        for i, layer in enumerate(self.layers):
            n = rms_norm(x, layer.attn_norm)
            qkv = F.linear(n, layer.qkv)
            q = rms_norm(qkv[:, :Q_DIM].view(b, N_Q, HEAD_DIM), layer.q_norm)
            k = rms_norm(qkv[:, Q_DIM : Q_DIM + KV_DIM].view(b, N_KV, HEAD_DIM), layer.k_norm)
            v = qkv[:, Q_DIM + KV_DIM :].view(b, N_KV, HEAD_DIM)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            self.k_cache[i][:, :, pos].copy_(k)
            self.v_cache[i][:, :, pos].copy_(v)
            attn = sdpa(
                q.unsqueeze(2),
                self.k_cache[i][:, :, :slen],
                self.v_cache[i][:, :, :slen],
            )
            x = x + F.linear(attn.squeeze(2).reshape(b, Q_DIM), layer.o)
            n = rms_norm(x, layer.ffn_norm)
            x = x + F.linear(F.silu(F.linear(n, layer.gate)) * F.linear(n, layer.up), layer.down)
        logits = F.linear(rms_norm(x, self.final_norm), self.embed)
        self.out_ids.copy_(torch.argmax(logits, dim=-1))

    def _prefill_chunk(self, hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
        bsz, chunk, _ = hidden.shape
        positions = torch.arange(start, end, device=DEVICE)
        cos = self.cos.index_select(0, positions).view(1, chunk, 1, HEAD_DIM)
        sin = self.sin.index_select(0, positions).view(1, chunk, 1, HEAD_DIM)
        x = hidden
        for i, layer in enumerate(self.layers):
            residual = x
            n = rms_norm(x, layer.attn_norm)
            qkv = F.linear(n, layer.qkv)
            q, k, v = qkv.split((Q_DIM, KV_DIM, KV_DIM), dim=-1)
            q = apply_rope(rms_norm(q.view(bsz, chunk, N_Q, HEAD_DIM), layer.q_norm), cos, sin)
            k = apply_rope(rms_norm(k.view(bsz, chunk, N_KV, HEAD_DIM), layer.k_norm), cos, sin)
            v = v.view(bsz, chunk, N_KV, HEAD_DIM)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            self.k_cache[i][:, :, start:end].copy_(k)
            self.v_cache[i][:, :, start:end].copy_(v)
            if start == 0:
                attn = sdpa(q, k, v, is_causal=True)
            else:
                q_pos = torch.arange(start, end, device=DEVICE)
                k_pos = torch.arange(end, device=DEVICE)
                banned = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                attn_mask = torch.zeros(1, 1, chunk, end, device=DEVICE, dtype=DTYPE)
                attn_mask = attn_mask.masked_fill(banned.view(1, 1, chunk, end), float("-inf"))
                attn = sdpa(
                    q,
                    self.k_cache[i][:, :, :end],
                    self.v_cache[i][:, :, :end],
                    attn_mask=attn_mask,
                )
            x = residual + F.linear(attn.transpose(1, 2).contiguous().view(bsz, chunk, Q_DIM), layer.o)
            residual = x
            n = rms_norm(x, layer.ffn_norm)
            x = residual + F.linear(F.silu(F.linear(n, layer.gate)) * F.linear(n, layer.up), layer.down)
        return x

    def prefill(self, input_ids: torch.Tensor) -> None:
        seqlen = input_ids.shape[1]
        hidden = None
        for start in range(0, seqlen, PREFILL_CHUNK):
            end = min(seqlen, start + PREFILL_CHUNK)
            hidden = F.embedding(input_ids[:, start:end], self.embed)
            hidden = self._prefill_chunk(hidden, start, end)
        logits = F.linear(rms_norm(hidden[:, -1, :], self.final_norm), self.embed)
        self.out_ids.copy_(torch.argmax(logits, dim=-1))

    def tokens_to_host(self) -> list[int]:
        self.pinned_out.copy_(self.out_ids, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.pinned_out.tolist()
