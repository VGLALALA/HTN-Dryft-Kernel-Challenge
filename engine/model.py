"""Custom Qwen3-4B greedy runner: packed BF16 weights, static KV, CUDA-graph decode.

Warmup generate stays on the proven eager/SDPA path. After it finishes we
capture a CUDA graph over a padded-KV decode step and replay it for the
measured samples. Capture failure falls back to eager (still correct).

Numerics follow Transformers 4.51.3:
RMSNorm fp32 reduce then BF16-cast-before-weight, NeoX RoPE, GQA=4,
SwiGLU, tied embeddings, scale 1/sqrt(128).
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
NEG_INF = torch.finfo(DTYPE).min


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


def graph_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """T=1 GQA attention over a padded cache. CUDA-graph friendly.

    q: [B, Hq, D], k/v: [B, Hkv, S, D], mask: [1, 1, 1, S] additive.
    Softmax in fp32, matching Transformers eager_attention_forward.
    """
    b = q.shape[0]
    qg = q.view(b, N_KV, GQA, HEAD_DIM)
    scores = torch.matmul(qg, k.transpose(-1, -2)) * SM_SCALE
    scores = scores + mask
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    ctx = torch.matmul(probs, v)
    return ctx.reshape(b, N_Q, HEAD_DIM)


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
            gu = torch.cat(
                (mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()),
                dim=0,
            ).contiguous()
            layers.append(
                PackedLayer(
                    qkv=qkv,
                    o=attn.o_proj.weight.detach().clone().contiguous(),
                    gu=gu,
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
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_failed = False
        self.k_cache: list[torch.Tensor] = []
        self.v_cache: list[torch.Tensor] = []
        self.cos: torch.Tensor | None = None
        self.sin: torch.Tensor | None = None
        self.token = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        self.out_ids = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        self.pos_index = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        self.cos_row = torch.empty(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.sin_row = torch.empty(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.attn_mask = torch.empty(0, device=DEVICE, dtype=DTYPE)
        self.pinned_out = torch.empty(0, dtype=torch.int64, pin_memory=True)

    def prepare(self, batch: int, max_len: int) -> None:
        if batch == self.batch and max_len <= self.max_len and self.k_cache:
            return
        self.graph = None
        self.capture_failed = False
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
        self.pos_index = torch.zeros(1, device=DEVICE, dtype=torch.int64)
        self.cos_row = torch.empty(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.sin_row = torch.empty(HEAD_DIM, device=DEVICE, dtype=DTYPE)
        self.attn_mask = torch.zeros(1, 1, 1, max_len, device=DEVICE, dtype=DTYPE)
        self.try_capture()

    def bind_pos(self, pos: int) -> None:
        """Host-side: point the graph inputs at ``pos``. Not captured."""
        self.pos_index.fill_(pos)
        self.cos_row.copy_(self.cos[pos])
        self.sin_row.copy_(self.sin[pos])
        self.attn_mask.fill_(0)
        nxt = pos + 1
        if nxt < self.max_len:
            self.attn_mask[..., nxt:].fill_(NEG_INF)

    def _mlp(self, x: torch.Tensor, layer: PackedLayer) -> torch.Tensor:
        n = rms_norm(x, layer.ffn_norm)
        gu = F.linear(n, layer.gu)
        gate, up = gu.split((INTERMEDIATE, INTERMEDIATE), dim=-1)
        return x + F.linear(F.silu(gate) * up, layer.down)

    def decode_step_eager(self, pos: int) -> None:
        """Sliced-KV SDPA. Used for warmup (and if graph capture fails)."""
        b = self.batch
        slen = pos + 1
        x = F.embedding(self.token, self.embed)
        cos = self.cos[pos].view(1, 1, HEAD_DIM)
        sin = self.sin[pos].view(1, 1, HEAD_DIM)
        for i, layer in enumerate(self.layers):
            n = rms_norm(x, layer.attn_norm)
            qkv = F.linear(n, layer.qkv)
            q = apply_rope(rms_norm(qkv[:, :Q_DIM].view(b, N_Q, HEAD_DIM), layer.q_norm), cos, sin)
            k = apply_rope(
                rms_norm(qkv[:, Q_DIM : Q_DIM + KV_DIM].view(b, N_KV, HEAD_DIM), layer.k_norm),
                cos,
                sin,
            )
            v = qkv[:, Q_DIM + KV_DIM :].view(b, N_KV, HEAD_DIM)
            self.k_cache[i][:, :, pos].copy_(k)
            self.v_cache[i][:, :, pos].copy_(v)
            attn = sdpa(
                q.unsqueeze(2),
                self.k_cache[i][:, :, :slen],
                self.v_cache[i][:, :, :slen],
            )
            x = x + F.linear(attn.squeeze(2).reshape(b, Q_DIM), layer.o)
            x = self._mlp(x, layer)
        self.out_ids.copy_(torch.argmax(F.linear(rms_norm(x, self.final_norm), self.embed), dim=-1))

    def decode_step_graph(self) -> None:
        """Fixed-shape decode. KV written via index_copy; unused slots masked."""
        b = self.batch
        x = F.embedding(self.token, self.embed)
        cos = self.cos_row.view(1, 1, HEAD_DIM)
        sin = self.sin_row.view(1, 1, HEAD_DIM)
        idx = self.pos_index
        mask = self.attn_mask
        for i, layer in enumerate(self.layers):
            n = rms_norm(x, layer.attn_norm)
            qkv = F.linear(n, layer.qkv)
            q = apply_rope(rms_norm(qkv[:, :Q_DIM].view(b, N_Q, HEAD_DIM), layer.q_norm), cos, sin)
            k = apply_rope(
                rms_norm(qkv[:, Q_DIM : Q_DIM + KV_DIM].view(b, N_KV, HEAD_DIM), layer.k_norm),
                cos,
                sin,
            )
            v = qkv[:, Q_DIM + KV_DIM :].view(b, N_KV, HEAD_DIM)
            self.k_cache[i].index_copy_(2, idx, k.unsqueeze(2))
            self.v_cache[i].index_copy_(2, idx, v.unsqueeze(2))
            attn = graph_attn(q, self.k_cache[i], self.v_cache[i], mask)
            x = x + F.linear(attn.reshape(b, Q_DIM), layer.o)
            x = self._mlp(x, layer)
        self.out_ids.copy_(torch.argmax(F.linear(rms_norm(x, self.final_norm), self.embed), dim=-1))

    def try_capture(self) -> None:
        if self.graph is not None or self.capture_failed:
            return
        self.token.zero_()
        self.bind_pos(0)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.decode_step_graph()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self.decode_step_graph()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = g
        except Exception as exc:
            self.graph = None
            self.capture_failed = True
            print(f"[engine] cuda graph capture skipped: {type(exc).__name__}: {exc}", flush=True)

    def replay(self) -> None:
        if self.graph is not None:
            self.graph.replay()
        else:
            self.decode_step_graph()

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
                attn_mask = attn_mask.masked_fill(banned.view(1, 1, chunk, end), NEG_INF)
                attn = sdpa(
                    q,
                    self.k_cache[i][:, :, :end],
                    self.v_cache[i][:, :, :end],
                    attn_mask=attn_mask,
                )
            x = residual + F.linear(attn.transpose(1, 2).contiguous().view(bsz, chunk, Q_DIM), layer.o)
            x = self._mlp(x, layer)
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
