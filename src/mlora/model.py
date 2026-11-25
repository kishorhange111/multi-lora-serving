"""GPT-2 with every attention and MLP projection carrying N stacked LoRA adapters.

``attach(model, n_adapters, r)`` wraps each ``Conv1D`` (``c_attn``, ``attn.c_proj``,
``c_fc``, ``mlp.c_proj``). Before a forward pass the caller sets one adapter index per
*request* with ``set_adapters(model, idx)``. Each wrapper expands it to one index per token
row and adds the LoRA term with the chosen implementation. HF's Conv1D stores its weight
as (d_in, d_out), so the base product is ``x @ W + b``.
"""

from __future__ import annotations

import torch
from torch import nn

from . import ops


class AdapterState:
    """Shared by every wrapped layer of one model: per-request adapter ids and the impl."""

    def __init__(self, impl: str = "gather"):
        self.idx: torch.Tensor | None = None
        self.impl = impl


def lora_fn(impl: str):
    if impl == "triton":
        from .kernels import triton_bgmv

        return triton_bgmv
    return ops.IMPLS[impl]


class MultiLoRA(nn.Module):
    def __init__(self, base, n_adapters: int, r: int, state: AdapterState, scale=1.0, seed=0):
        super().__init__()
        self.base, self.state, self.scale = base, state, scale
        d_in, d_out = base.weight.shape
        g = torch.Generator().manual_seed(seed)
        dev, dt = base.weight.device, base.weight.dtype
        # Random adapters with a visible effect: A ~ N(0, 1/d_in), B ~ N(0, 0.02²).
        self.A = nn.Parameter(
            (torch.randn(n_adapters, r, d_in, generator=g) / d_in**0.5).to(dev, dt),
            requires_grad=False,
        )
        self.B = nn.Parameter(
            (torch.randn(n_adapters, d_out, r, generator=g) * 0.02).to(dev, dt),
            requires_grad=False,
        )

    def forward(self, x):
        y = self.base(x)
        idx = self.state.idx
        if idx is None:
            return y
        b, t, d = x.shape
        rows = idx.to(x.device).repeat_interleave(t)
        flat = x.reshape(b * t, d)
        if self.state.impl == "triton":
            from .kernels import triton_bgmv

            yf = y.reshape(b * t, -1).contiguous()
            triton_bgmv(flat, self.A, self.B, rows, self.scale, out=yf)
            return yf.view_as(y)
        return y + lora_fn(self.state.impl)(flat, self.A, self.B, rows, self.scale).view_as(y)


def attach(model, n_adapters: int, r: int = 16, impl: str = "gather", scale: float = 1.0):
    state = AdapterState(impl)
    seed = 0
    for block in model.transformer.h:
        for parent, name in (
            (block.attn, "c_attn"),
            (block.attn, "c_proj"),
            (block.mlp, "c_fc"),
            (block.mlp, "c_proj"),
        ):
            setattr(
                parent, name, MultiLoRA(getattr(parent, name), n_adapters, r, state, scale, seed)
            )
            seed += 1
    model._adapter_state = state
    return model


def set_adapters(model, idx: torch.Tensor | None, impl: str | None = None):
    model._adapter_state.idx = idx
    if impl is not None:
        model._adapter_state.impl = impl


def wrapped(model):
    return [m for m in model.modules() if isinstance(m, MultiLoRA)]


@torch.no_grad()
def merge_adapter(model, a: int):
    """Fold adapter ``a`` into every base weight (in place) and return a function undoing it."""
    saved = []
    for m in wrapped(model):
        saved.append(m.base.weight.clone())
        delta = (m.B[a] @ m.A[a]).T * m.scale  # (d_in, d_out), Conv1D layout
        m.base.weight += delta.to(m.base.weight.dtype)

    @torch.no_grad()
    def undo():
        for m, w in zip(wrapped(model), saved, strict=True):
            m.base.weight.copy_(w)

    return undo


@torch.no_grad()
def greedy(model, prompts: torch.Tensor, new_tokens: int) -> torch.Tensor:
    """Greedy decoding with the KV cache; the adapter ids stay fixed per request."""
    out = model(prompts, use_cache=True)
    past, tok = out.past_key_values, out.logits[:, -1].argmax(-1)
    toks = [tok]
    for _ in range(new_tokens - 1):
        out = model(tok[:, None], past_key_values=past, use_cache=True)
        past, tok = out.past_key_values, out.logits[:, -1].argmax(-1)
        toks.append(tok)
    return torch.stack(toks, 1)
