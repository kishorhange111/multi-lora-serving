"""Batched LoRA with a different adapter per row: the core op of multi-tenant LoRA serving.

One base weight W is shared by every request; request i also carries adapter a_i with
low-rank factors A[a_i] (r × d_in) and B[a_i] (d_out × r):

    y_i = W x_i + s · B[a_i] (A[a_i] x_i)

Adapters are stacked into ``A (N, r, d_in)`` and ``B (N, d_out, r)``. ``idx[i] = -1`` means
"base model only". The four ways to compute the LoRA term are:

- ``lora_loop``: one small matmul pair per distinct adapter in the batch, the obvious
  approach. The cost grows with the number of *distinct* adapters, and each launch is tiny.
- ``lora_gather`` (BGMV, Punica): gather each row's factors, then one batched mat-vec. A
  constant number of kernels regardless of how many adapters are mixed, but it materialises
  ``(n, r, d_in)`` of gathered weights.
- ``lora_segment`` (SGMV): sort rows by adapter and run one matmul per contiguous segment.
  This is the prefill-friendly form, since a long prompt is many rows with the same adapter.
- ``triton_bgmv`` (in :mod:`mlora.kernels`): BGMV fused into two Triton kernels, shrink
  (x → r) and expand (r → d_out). Each program reads its row's adapter index and streams that
  adapter's factors straight from the stacked tensors, so nothing is gathered.

Merging, ``W + s·B[a]A[a]``, is exact and free at inference for a *single* adapter, but a
mixed batch would need one merged copy of W per adapter. That is the case these ops exist
for.
"""

from __future__ import annotations

import torch


def _check(x, A, B, idx):
    n, d_in = x.shape
    N, r, d_in2 = A.shape
    assert d_in == d_in2 and B.shape[0] == N and B.shape[2] == r and idx.shape == (n,)


def lora_reference(x, A, B, idx, scale: float = 1.0) -> torch.Tensor:
    """Row by row, in fp32 (or fp64 for fp64 inputs): the definition."""
    _check(x, A, B, idx)
    dt = torch.promote_types(x.dtype, torch.float32)
    out = torch.zeros(x.shape[0], B.shape[1], dtype=dt, device=x.device)
    for i, a in enumerate(idx.tolist()):
        if a >= 0:
            out[i] = scale * (B[a].to(dt) @ (A[a].to(dt) @ x[i].to(dt)))
    return out


def lora_loop(x, A, B, idx, scale: float = 1.0) -> torch.Tensor:
    """One (A, B) matmul pair per distinct adapter present in the batch."""
    _check(x, A, B, idx)
    out = torch.zeros(x.shape[0], B.shape[1], dtype=x.dtype, device=x.device)
    for a in torch.unique(idx).tolist():
        if a < 0:
            continue
        rows = (idx == a).nonzero(as_tuple=True)[0]
        out[rows] = scale * ((x[rows] @ A[a].T) @ B[a].T)
    return out


def lora_gather(x, A, B, idx, scale: float = 1.0) -> torch.Tensor:
    """BGMV by gathering: ``A[idx] (n, r, d_in)`` and ``B[idx] (n, d_out, r)`` then bmm."""
    _check(x, A, B, idx)
    valid = idx >= 0
    safe = idx.clamp(min=0)
    s = torch.bmm(A[safe], x[:, :, None])  # (n, r, 1)
    y = torch.bmm(B[safe], s)[:, :, 0]  # (n, d_out)
    return y * (scale * valid[:, None].to(y.dtype))


def segments(idx: torch.Tensor):
    """Sort rows by adapter. Returns ``(order, [(adapter, start, end), ...])`` over sorted rows."""
    order = torch.argsort(idx, stable=True)
    sorted_idx = idx[order]
    vals, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
    bounds, start = [], 0
    for a, c in zip(vals.tolist(), counts.tolist(), strict=True):
        bounds.append((a, start, start + c))
        start += c
    return order, bounds


def lora_segment(x, A, B, idx, scale: float = 1.0) -> torch.Tensor:
    """SGMV: rows sorted by adapter, one matmul pair per contiguous segment."""
    _check(x, A, B, idx)
    order, bounds = segments(idx)
    xs = x[order]
    ys = torch.zeros(x.shape[0], B.shape[1], dtype=x.dtype, device=x.device)
    for a, s0, s1 in bounds:
        if a >= 0:
            ys[s0:s1] = scale * ((xs[s0:s1] @ A[a].T) @ B[a].T)
    out = torch.empty_like(ys)
    out[order] = ys
    return out


def merged_weight(W: torch.Tensor, A, B, a: int, scale: float = 1.0) -> torch.Tensor:
    """``W + s·B[a]A[a]`` for a weight stored as (d_out, d_in)."""
    return W + scale * (B[a] @ A[a]).to(W.dtype)


IMPLS = {"loop": lora_loop, "gather": lora_gather, "segment": lora_segment}
