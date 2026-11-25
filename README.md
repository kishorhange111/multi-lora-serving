# multi-lora-serving

[![CI](https://github.com/kishorhange111/multi-lora-serving/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/multi-lora-serving/actions/workflows/ci.yml)

Serve many LoRA fine-tunes of one base model **in the same batch**, with a different
adapter for every request. This is the problem Punica (BGMV/SGMV kernels) and S-LoRA solve,
and the reason a multi-tenant LoRA endpoint can be much cheaper than one deployment per
fine-tune. Here it is built from the op up: PyTorch reference implementations, a
hand-written **Triton BGMV** kernel pair, and GPT-2 decoding with 32 adapters in one batch.

## The op ([`ops.py`](src/mlora/ops.py), [`kernels.py`](src/mlora/kernels.py))

Request i has adapter `a_i`, and `a_i = −1` means base model only:

    y_i = W x_i + s · B[a_i] (A[a_i] x_i),   A: (N, r, d_in),  B: (N, d_out, r)

| implementation | how | cost |
|---|---|---|
| `loop` | one matmul pair per *distinct* adapter in the batch | 2 launches per adapter; grows with adapter diversity |
| `gather` (BGMV) | gather `A[idx]`, `B[idx]`, then `bmm` | constant launches, but materialises `n·r·(d_in + d_out)` gathered weights |
| `segment` (SGMV) | sort rows by adapter, one matmul pair per segment | prefill-friendly: a prompt is many rows with one adapter |
| `triton` (BGMV) | **shrink** kernel `t = A[idx]·x` then **expand** kernel `y += s·B[idx]·t`; each program reads its row's index and streams that adapter's factors | 2 launches, no gather, adds into `y` in place |
| merge | `W + s·B[a]A[a]` | exact and free for *one* adapter; a mixed batch would need a copy of W per adapter |

## Model ([`model.py`](src/mlora/model.py))

`attach(gpt2, n_adapters, r)` wraps all four projections in every block (`c_attn`,
`attn.c_proj`, `c_fc`, `mlp.c_proj`). `set_adapters(model, idx)` sets one adapter id per
request; every layer expands it to one id per token row. `merge_adapter` folds an adapter
into the weights and returns an exact undo, which gives the single-adapter baseline.

## Tests

There are 14. The 10 CPU tests run in CI, in fp64:
- every implementation matches the row-by-row definition, including `−1` rows;
- segments cover every row exactly once;
- merging is exact;
- the wrapped model with no adapters equals the base model;
- **a mixed batch equals each request run on its own with its adapter merged**, for every
  implementation, and the undo restores the weights.

The 4 GPU tests check the Triton kernels against the reference at ranks 8, 16 and 64, both
fresh and accumulating, and check Triton against `gather` inside the model.

```
pytest -q                            # CPU tests; GPU tests skip without CUDA
python -m mlora.bench --out results  # T4: op-level decode/prefill + GPT-2 end to end
```

**Results are pending** from the Colab run.
