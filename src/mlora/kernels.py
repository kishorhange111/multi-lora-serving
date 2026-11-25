"""BGMV in Triton: shrink (x → A[idx] x) and expand (y += s · B[idx] t), with no gathering.

Decode batches have one token per request, so each LoRA term is two mat-vecs whose
weights differ per row. Both kernels give each program one row (and, for expand, one block
of output columns). The program loads its adapter index, and an index of −1 writes zeros
(shrink) or skips the row (expand). The kernels are memory-bound: per row they read
``r·d_in + d_out·r`` adapter values, so throughput is set by how well those reads coalesce.
Here they are contiguous along d_in (A) and along r (B).

The reductions use ``tl.sum`` of elementwise products, not ``tl.dot``: with one row per
program the "matmul" is a mat-vec, which tensor cores would not help with anyway.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _shrink(X, A, IDX, T, d_in, R: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    a = tl.load(IDX + row)
    valid = a >= 0
    a = tl.maximum(a, 0)  # index −1: read adapter 0 under a false mask, write zeros
    rr = tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)
    base = A + a.to(tl.int64) * R * d_in
    for k0 in range(0, d_in, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        km = (kk < d_in) & valid
        x = tl.load(X + row * d_in + kk, mask=km, other=0.0).to(tl.float32)
        w = tl.load(base + rr[:, None] * d_in + kk[None, :], mask=km[None, :], other=0.0)
        acc += tl.sum(w.to(tl.float32) * x[None, :], axis=1)
    tl.store(T + row * R + rr, acc)


@triton.jit
def _expand(T, B, IDX, Y, d_out, scale, R: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    nb = tl.program_id(1)
    a = tl.load(IDX + row)
    valid = a >= 0
    a = tl.maximum(a, 0)
    nn = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = (nn < d_out) & valid  # rows without an adapter are left untouched
    rr = tl.arange(0, R)
    t = tl.load(T + row * R + rr)
    w = tl.load(
        B + a.to(tl.int64) * d_out * R + nn[:, None] * R + rr[None, :], mask=nm[:, None], other=0.0
    )
    y = tl.sum(w.to(tl.float32) * t[None, :], axis=1) * scale
    ptr = Y + row * d_out + nn
    old = tl.load(ptr, mask=nm, other=0.0)
    tl.store(ptr, (old.to(tl.float32) + y).to(old.dtype), mask=nm)


def triton_bgmv(x, A, B, idx, scale: float = 1.0, out: torch.Tensor | None = None):
    """LoRA term for rows ``x (n, d_in)``; adds into ``out`` (n, d_out) if given.

    The rank r must be a power of two (Triton block shapes); 8, 16, 32 and 64 are typical.
    """
    n, d_in = x.shape
    N, r, _ = A.shape
    d_out = B.shape[1]
    assert r & (r - 1) == 0, "rank must be a power of two"
    x, A, B = x.contiguous(), A.contiguous(), B.contiguous()
    idx = idx.to(torch.int32).contiguous()
    t = torch.empty(n, r, dtype=torch.float32, device=x.device)
    _shrink[(n,)](x, A, idx, t, d_in, R=r, BLOCK_K=128)
    if out is None:
        out = torch.zeros(n, d_out, dtype=x.dtype, device=x.device)
    block_n = 64
    _expand[(n, triton.cdiv(d_out, block_n))](t, B, idx, out, d_out, scale, R=r, BLOCK_N=block_n)
    return out
