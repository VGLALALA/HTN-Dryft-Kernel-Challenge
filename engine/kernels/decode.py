"""Decode helpers: NeoX RoPE, KV scatter.

RoPE matches Transformers 4.51.3:

    rotate_half(x) = cat(-x2, x1)
    (x * cos) + (rotate_half(x) * sin)

cos/sin duplicate the 64 frequencies, so the half-split form is exact.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 128


@triton.jit
def _store_kv_kernel(
    k_ptr,
    v_ptr,
    kc_ptr,
    vc_ptr,
    pos_ptr,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_kcb,
    stride_kch,
    stride_kcs,
    stride_kcd,
    stride_vcb,
    stride_vch,
    stride_vcs,
    stride_vcd,
    D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs = tl.arange(0, D)
    pos = tl.load(pos_ptr).to(tl.int64)
    k = tl.load(k_ptr + b * stride_kb + h * stride_kh + offs * stride_kd)
    v = tl.load(v_ptr + b * stride_vb + h * stride_vh + offs * stride_vd)
    tl.store(
        kc_ptr + b * stride_kcb + h * stride_kch + pos * stride_kcs + offs * stride_kcd,
        k,
    )
    tl.store(
        vc_ptr + b * stride_vcb + h * stride_vch + pos * stride_vcs + offs * stride_vcd,
        v,
    )


@triton.jit
def _gather_rope_kernel(
    cos_ptr,
    sin_ptr,
    pos_ptr,
    cos_out_ptr,
    sin_out_ptr,
    stride_table,
    D: tl.constexpr,
):
    offs = tl.arange(0, D)
    pos = tl.load(pos_ptr).to(tl.int64)
    base = pos * stride_table
    tl.store(cos_out_ptr + offs, tl.load(cos_ptr + base + offs))
    tl.store(sin_out_ptr + offs, tl.load(sin_ptr + base + offs))


def store_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> None:
    """Write K/V ``[B, Hkv, D]`` into cache ``[B, Hkv, S, D]`` at device index ``pos``."""
    bsz, n_kv, dim = k.shape
    _store_kv_kernel[(bsz, n_kv)](
        k,
        v,
        k_cache,
        v_cache,
        pos,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        D=dim,
        num_warps=4,
    )


def gather_rope(
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    pos: torch.Tensor,
    cos_out: torch.Tensor,
    sin_out: torch.Tensor,
) -> None:
    """Load ``cos/sin[pos]`` into length-``HEAD_DIM`` buffers (graph-safe)."""
    _gather_rope_kernel[(1,)](
        cos_table,
        sin_table,
        pos,
        cos_out.view(-1),
        sin_out.view(-1),
        cos_table.stride(0),
        D=HEAD_DIM,
        num_warps=1,
    )


def apply_rope_out(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write NeoX RoPE into ``out``. ``x`` and ``out`` must not alias."""
    x1 = x[..., :64]
    x2 = x[..., 64:]
    c = cos[..., :64]
    s = sin[..., :64]
    o1 = out[..., :64]
    o2 = out[..., 64:]
    torch.mul(x1, c, out=o1)
    o1.addcmul_(x2, s, value=-1.0)
    torch.mul(x2, c, out=o2)
    o2.addcmul_(x1, s)
    return out


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Allocating RoPE for prefill. ``cos``/``sin`` broadcast on the last dim."""
    x1 = x[..., :64]
    x2 = x[..., 64:]
    c = cos[..., :64]
    s = sin[..., :64]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
