import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mlora import ops
from mlora.model import attach, greedy, merge_adapter, set_adapters

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def factors(N=5, r=4, d_in=24, d_out=40, n=17, seed=0, device="cpu", dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(N, r, d_in, generator=g, dtype=dtype).to(device)
    B = torch.randn(N, d_out, r, generator=g, dtype=dtype).to(device)
    x = torch.randn(n, d_in, generator=g, dtype=dtype).to(device)
    idx = torch.randint(-1, N, (n,), generator=g).to(device)
    return x, A, B, idx


@pytest.mark.parametrize("impl", list(ops.IMPLS))
def test_every_impl_matches_reference_including_no_adapter_rows(impl):
    x, A, B, idx = factors()
    idx[0] = -1
    got = ops.IMPLS[impl](x, A, B, idx, scale=0.5)
    want = ops.lora_reference(x, A, B, idx, scale=0.5).double()
    torch.testing.assert_close(got, want)
    assert got[idx < 0].abs().max() == 0


def test_segments_cover_rows_once_and_are_sorted():
    idx = torch.tensor([3, 1, 3, -1, 1, 1, 0])
    order, bounds = ops.segments(idx)
    assert sorted(order.tolist()) == list(range(7))
    assert [a for a, _, _ in bounds] == [-1, 0, 1, 3]
    for a, s0, s1 in bounds:
        assert (idx[order[s0:s1]] == a).all()
    assert bounds[-1][2] == 7


def test_merging_one_adapter_is_exact():
    x, A, B, _ = factors()
    W = torch.randn(40, 24, dtype=torch.float64)
    idx = torch.full((x.shape[0],), 2)
    torch.testing.assert_close(
        x @ ops.merged_weight(W, A, B, 2, 0.7).T, x @ W.T + ops.lora_gather(x, A, B, idx, 0.7)
    )


def tiny_gpt2():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=50, n_positions=32, n_embd=32, n_layer=2, n_head=2)
    return GPT2LMHeadModel(cfg).double().eval()


def test_attach_without_adapters_is_the_base_model():
    base = tiny_gpt2()
    m = attach(tiny_gpt2(), n_adapters=4, r=4)
    x = torch.randint(0, 50, (3, 7))
    set_adapters(m, None)
    torch.testing.assert_close(m(x).logits, base(x).logits)


@pytest.mark.parametrize("impl", list(ops.IMPLS))
def test_mixed_batch_equals_each_request_with_its_merged_adapter(impl):
    m = attach(tiny_gpt2(), n_adapters=4, r=4)
    x = torch.randint(0, 50, (4, 6))
    idx = torch.tensor([2, -1, 0, 2])
    set_adapters(m, idx, impl)
    mixed = m(x).logits
    set_adapters(m, None)
    for i, a in enumerate(idx.tolist()):
        undo = merge_adapter(m, a) if a >= 0 else (lambda: None)
        torch.testing.assert_close(mixed[i], m(x[i : i + 1]).logits[0])
        undo()
    # undo restored the base weights exactly
    torch.testing.assert_close(m(x).logits, tiny_gpt2()(x).logits)


def test_adapters_change_the_output_and_greedy_decodes():
    m = attach(tiny_gpt2(), n_adapters=2, r=4)
    x = torch.randint(0, 50, (2, 5))
    set_adapters(m, torch.tensor([0, 1]))
    a = m(x).logits
    set_adapters(m, torch.tensor([1, 0]))
    assert not torch.allclose(a, m(x).logits)
    assert greedy(m, x, 4).shape == (2, 4)


@cuda
@pytest.mark.parametrize("r", [8, 16, 64])
def test_triton_bgmv_matches_reference(r):
    from mlora.kernels import triton_bgmv

    x, A, B, idx = factors(N=7, r=r, d_in=768, d_out=2304, n=33, device="cuda", dtype=torch.float16)
    want = ops.lora_reference(x, A, B, idx, scale=0.5)
    torch.testing.assert_close(triton_bgmv(x, A, B, idx, 0.5).float(), want, atol=0.15, rtol=2e-2)
    base = torch.randn(33, 2304, device="cuda", dtype=torch.float16)
    out = triton_bgmv(x, A, B, idx, 0.5, out=base.clone())
    torch.testing.assert_close(out.float(), base.float() + want, atol=0.15, rtol=2e-2)


@cuda
def test_triton_in_model_matches_gather():
    m = attach(tiny_gpt2().float().cuda(), n_adapters=4, r=4)
    x = torch.randint(0, 50, (4, 6), device="cuda")
    idx = torch.tensor([2, -1, 0, 3], device="cuda")
    set_adapters(m, idx, "gather")
    a = m(x).logits
    set_adapters(m, idx, "triton")
    torch.testing.assert_close(m(x).logits, a, atol=1e-4, rtol=1e-4)
