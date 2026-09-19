"""Qwen3 RMSNorm in Triton, matching Transformers 4.51.3 exactly.

Reference (Qwen3RMSNorm.forward):

    hidden_states = hidden_states.to(float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * rsqrt(variance + eps)
    return self.weight * hidden_states.to(input_dtype)

The BF16 cast happens *before* the weight multiply. Do not "improve" that.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_BLOCK = 8192


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    row_stride_x,
    row_stride_y,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normed = x * tl.math.rsqrt(variance + eps)
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(
        y_ptr + row * row_stride_y + cols,
        normed.to(y_ptr.dtype.element_ty) * weight,
        mask=mask,
    )


def _launch(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float) -> torch.Tensor:
    n_cols = x.shape[-1]
    rows_x = x.reshape(-1, n_cols)
    rows_y = out.reshape(-1, n_cols)
    n_rows = rows_x.shape[0]
    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK:
        raise ValueError(f"a row must fit in one block; {n_cols} columns does not")
    _rms_norm_kernel[(n_rows,)](
        rows_x,
        weight,
        rows_y,
        rows_x.stride(0),
        rows_y.stride(0),
        n_cols,
        eps,
        BLOCK=block,
        num_warps=max(4, min(16, block // 256)),
    )
    return out


def rms_norm_out(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm over the last dimension, writing into ``out`` (graph-safe)."""
    if out.shape != x.shape:
        raise ValueError(f"out shape {tuple(out.shape)} != x shape {tuple(x.shape)}")
    return _launch(x, weight, out, eps)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension, matching ``Qwen3RMSNorm.forward``."""
    out = torch.empty_like(x)
    return _launch(x, weight, out, eps)
