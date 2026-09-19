"""Custom Qwen3-4B greedy runner: packed weights, static KV, CUDA-graph decode.

Numerics follow Transformers 4.51.3 / Qwen3-4B-Instruct-2507:
RMSNorm fp32 reduce then BF16-cast-before-weight, NeoX RoPE, GQA=4,
SwiGLU, tied embeddings, attention scale 1/sqrt(128).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from kernels.attention import DecodeAttnWorkspace, decode_gqa_attention
from kernels.decode import apply_rope, apply_rope_out, gather_rope, store_kv
from kernels.rmsnorm import rms_norm, rms_norm_out

N_LAYERS = 36
HIDDEN = 2560
INTERMEDIATE = 9728
N_Q = 32
N_KV = 8
HEAD_DIM = 128
GQA = 4
VOCAB = 151936
Q_DIM = N_Q * HEAD_DIM  # 4096
KV_DIM = N_KV * HEAD_DIM  # 1024
QKV_DIM = Q_DIM + 2 * KV_DIM  # 6144
GU_DIM = 2 * INTERMEDIATE  # 19456
EPS = 1e-6
ROPE_THETA = 5_000_000.0
SM_SCALE = HEAD_DIM**-0.5
PREFILL_CHUNK = 8192
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
    kwargs.setdefault("scale", SM_SCALE)
    kwargs.setdefault("dropout_p", 0.0)
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
    gu: torch.Tensor
    down: torch.Tensor
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor


def _build_rope(max_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        ROPE_THETA
        ** (
            torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32)
            / HEAD_DIM
        )
    )
    t = torch.arange(max_len, device=device, dtype=torch.float32)
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
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(DEVICE)
        )
        base = model.model
        embed = base.embed_tokens.weight.detach().contiguous()
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
            gu = torch.cat(
                (mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()),
                dim=0,
            ).contiguous()
            layers.append(
                PackedLayer(
                    qkv=qkv,
                    o=attn.o_proj.weight.detach().contiguous(),
                    gu=gu,
                    down=mlp.down_proj.weight.detach().contiguous(),
                    attn_norm=block.input_layernorm.weight.detach().contiguous(),
                    ffn_norm=block.post_attention_layernorm.weight.detach().contiguous(),
                    q_norm=attn.q_norm.weight.detach().contiguous(),
                    k_norm=attn.k_norm.weight.detach().contiguous(),
                )
            )
        final_norm = base.norm.weight.detach().contiguous()
        # Drop the Transformers tree; packed tensors hold the live storage.
        for block in base.layers:
            block.self_attn.q_proj.weight = None
            block.self_attn.k_proj.weight = None
            block.self_attn.v_proj.weight = None
            block.self_attn.o_proj.weight = None
            block.mlp.gate_proj.weight = None
            block.mlp.up_proj.weight = None
            block.mlp.down_proj.weight = None
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
        self.device = DEVICE

        self.batch = 0
        self.max_len = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.use_graph = False
        self.workspace: DecodeAttnWorkspace | None = None

        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.cos: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None

        self.pos = torch.zeros((), device=DEVICE, dtype=torch.int64)
        self.token = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        self.out_ids = torch.zeros(1, device=DEVICE, dtype=torch.int64)

        self.x = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.n = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.qkv = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.q = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.k = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.q_rot = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.k_rot = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.attn = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.attn_flat = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.gu = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.mid = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.logits = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.cos_row = torch.empty(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.sin_row = torch.empty(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.pinned_out = torch.empty(0, dtype=torch.int64, pin_memory=True)

    def prepare(self, batch: int, max_len: int) -> None:
        if (
            batch == self.batch
            and max_len <= self.max_len
            and self.graph is not None
        ):
            return

        self.graph = None
        self.use_graph = False
        self.batch = batch
        self.max_len = max_len

        self.k_cache = torch.empty(
            N_LAYERS, batch, N_KV, max_len, HEAD_DIM, device=DEVICE, dtype=DTYPE
        )
        self.v_cache = torch.empty_like(self.k_cache)
        self.cos, self.sin = _build_rope(max_len, DEVICE)
        self.workspace = DecodeAttnWorkspace(batch, max_len, DEVICE)

        self.token = torch.zeros(batch, device=DEVICE, dtype=torch.int64)
        self.out_ids = torch.zeros(batch, device=DEVICE, dtype=torch.int64)
        self.pinned_out = torch.empty(batch, dtype=torch.int64, pin_memory=True)

        self.x = torch.empty(batch, HIDDEN, device=DEVICE, dtype=DTYPE)
        self.n = torch.empty_like(self.x)
        self.qkv = torch.empty(batch, QKV_DIM, device=DEVICE, dtype=DTYPE)
        self.q = torch.empty(batch, N_Q, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.k = torch.empty(batch, N_KV, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.q_rot = torch.empty_like(self.q)
        self.k_rot = torch.empty_like(self.k)
        self.attn = torch.empty(batch, N_Q, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.attn_flat = self.attn.view(batch, Q_DIM)
        self.gu = torch.empty(batch, GU_DIM, device=DEVICE, dtype=DTYPE)
        self.mid = torch.empty(batch, INTERMEDIATE, device=DEVICE, dtype=DTYPE)
        self.logits = torch.empty(batch, VOCAB, device=DEVICE, dtype=DTYPE)
        self.cos_row = torch.empty(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.sin_row = torch.empty(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.cos_b = self.cos_row.view(1, 1, HEAD_DIM)
        self.sin_b = self.sin_row.view(1, 1, HEAD_DIM)

        self._warmup_kernels()
        self.token.zero_()
        self.pos.zero_()
        self.capture_graph()

    def _warmup_kernels(self) -> None:
        dummy_x = self.x.zero_()
        dummy_n = self.n
        rms_norm_out(dummy_x, self.layers[0].attn_norm, dummy_n, EPS)
        self.q.zero_()
        self.k.zero_()
        self.attn.zero_()
        self.pos.fill_(0)
        store_kv(self.k, self.k, self.k_cache[0], self.v_cache[0], self.pos)
        decode_gqa_attention(
            self.q, self.k_cache[0], self.v_cache[0], self.pos, self.attn, self.workspace
        )
        rms_norm_out(self.q, self.layers[0].q_norm, self.q, EPS)
        rms_norm_out(self.k, self.layers[0].k_norm, self.k, EPS)
        gather_rope(self.cos, self.sin, self.pos, self.cos_row, self.sin_row)
        apply_rope_out(self.q, self.cos_b, self.sin_b, self.q_rot)
        torch.cuda.synchronize()

    def _decode_layers(self) -> None:
        gather_rope(self.cos, self.sin, self.pos, self.cos_row, self.sin_row)
        x = self.x
        n = self.n
        batch = self.batch
        for i, layer in enumerate(self.layers):
            rms_norm_out(x, layer.attn_norm, n, EPS)
            torch.mm(n, layer.qkv.t(), out=self.qkv)
            q_raw = self.qkv[:, :Q_DIM].view(batch, N_Q, HEAD_DIM)
            k_raw = self.qkv[:, Q_DIM : Q_DIM + KV_DIM].view(batch, N_KV, HEAD_DIM)
            v_raw = self.qkv[:, Q_DIM + KV_DIM :].view(batch, N_KV, HEAD_DIM)
            rms_norm_out(q_raw, layer.q_norm, self.q, EPS)
            rms_norm_out(k_raw, layer.k_norm, self.k, EPS)
            apply_rope_out(self.q, self.cos_b, self.sin_b, self.q_rot)
            apply_rope_out(self.k, self.cos_b, self.sin_b, self.k_rot)
            store_kv(self.k_rot, v_raw, self.k_cache[i], self.v_cache[i], self.pos)
            decode_gqa_attention(
                self.q_rot,
                self.k_cache[i],
                self.v_cache[i],
                self.pos,
                self.attn,
                self.workspace,
            )
            torch.mm(self.attn_flat, layer.o.t(), out=n)
            x.add_(n)
            rms_norm_out(x, layer.ffn_norm, n, EPS)
            torch.mm(n, layer.gu.t(), out=self.gu)
            gate = self.gu[:, :INTERMEDIATE]
            up = self.gu[:, INTERMEDIATE:]
            torch.silu(gate, out=self.mid)
            self.mid.mul_(up)
            torch.mm(self.mid, layer.down.t(), out=n)
            x.add_(n)
        rms_norm_out(x, self.final_norm, n, EPS)
        torch.mm(n, self.embed.t(), out=self.logits)
        torch.argmax(self.logits, dim=-1, out=self.out_ids)

    def decode_step(self) -> None:
        torch.index_select(self.embed, 0, self.token, out=self.x)
        self._decode_layers()

    def capture_graph(self) -> None:
        if self.graph is not None:
            return
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.decode_step()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.decode_step()
        torch.cuda.current_stream().wait_stream(s)
        self.use_graph = True
        torch.cuda.synchronize()

    def replay(self) -> None:
        if self.use_graph and self.graph is not None:
            self.graph.replay()
        else:
            self.decode_step()

    def _prefill_chunk(
        self,
        hidden: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        bsz, chunk, _ = hidden.shape
        positions = torch.arange(start, end, device=DEVICE)
        cos = self.cos.index_select(0, positions).view(1, chunk, 1, HEAD_DIM)
        sin = self.sin.index_select(0, positions).view(1, chunk, 1, HEAD_DIM)
        x = hidden
        for i, layer in enumerate(self.layers):
            residual = x
            n = rms_norm(x, layer.attn_norm, EPS)
            qkv = F.linear(n, layer.qkv)
            q, k, v = qkv.split((Q_DIM, KV_DIM, KV_DIM), dim=-1)
            q = q.view(bsz, chunk, N_Q, HEAD_DIM)
            k = k.view(bsz, chunk, N_KV, HEAD_DIM)
            v = v.view(bsz, chunk, N_KV, HEAD_DIM)
            q = rms_norm(q, layer.q_norm, EPS)
            k = rms_norm(k, layer.k_norm, EPS)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            self.k_cache[i, :, :, start:end].copy_(k)
            self.v_cache[i, :, :, start:end].copy_(v)
            if start == 0:
                attn = _sdpa(q, k, v, is_causal=True)
            else:
                q_pos = torch.arange(start, end, device=DEVICE)
                k_pos = torch.arange(end, device=DEVICE)
                banned = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                attn_mask = torch.zeros(1, 1, chunk, end, device=DEVICE, dtype=x.dtype)
                attn_mask = attn_mask.masked_fill(banned.view(1, 1, chunk, end), float("-inf"))
                attn = _sdpa(
                    q,
                    self.k_cache[i, :, :, :end],
                    self.v_cache[i, :, :, :end],
                    attn_mask=attn_mask,
                    is_causal=False,
                )
            attn = attn.transpose(1, 2).contiguous().view(bsz, chunk, Q_DIM)
            x = residual + F.linear(attn, layer.o)
            residual = x
            n = rms_norm(x, layer.ffn_norm, EPS)
            gu = F.linear(n, layer.gu)
            gate, up = gu.split((INTERMEDIATE, INTERMEDIATE), dim=-1)
            x = residual + F.linear(F.silu(gate) * up, layer.down)
        return x

    def prefill(self, input_ids: torch.Tensor) -> None:
        bsz, seqlen = input_ids.shape
        hidden = None
        for start in range(0, seqlen, PREFILL_CHUNK):
            end = min(seqlen, start + PREFILL_CHUNK)
            hidden = F.embedding(input_ids[:, start:end], self.embed)
            hidden = self._prefill_chunk(hidden, start, end)
        last = rms_norm(hidden[:, -1, :], self.final_norm, EPS)
        torch.mm(last, self.embed.t(), out=self.logits)
        torch.argmax(self.logits, dim=-1, out=self.out_ids)

    def tokens_to_host(self) -> list[int]:
        self.pinned_out.copy_(self.out_ids, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.pinned_out.tolist()
