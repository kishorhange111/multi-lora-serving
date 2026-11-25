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

## Results (Tesla T4, Colab)

All 14 tests pass on the T4, including the 4 Triton tests
([`results/pytest.txt`](results/pytest.txt)). The tables below come from
[`results/bench_Tesla_T4.md`](results/bench_Tesla_T4.md): fp16, rank 16, median of 3.

**Decode, 64 rows (one per request), `c_attn` 768 → 2304. Time of the LoRA term, in ms:**

| distinct adapters | base matmul | loop | gather | segment | triton |
|---|---|---|---|---|---|
| 1 | 0.027 | 0.388 | 0.265 | 0.429 | **0.101** |
| 4 | 0.043 | 1.091 | 0.263 | 0.733 | **0.099** |
| 16 | 0.044 | 3.903 | 0.387 | 2.542 | **0.129** |
| 64 | 0.043 | 14.065 | 0.262 | 6.385 | **0.100** |

- **The Triton kernel's cost does not depend on how many adapters the batch mixes:**
  0.10–0.13 ms from 1 to 64 adapters. `loop` grows linearly with them, reaching 14 ms, or
  140× slower, at 64.
- `gather` is also flat, but 2.6× slower than Triton because it materialises the gathered
  factors first.
- The LoRA term still costs about 2.3× the base matmul. At 64 rows both ops are tiny, so
  launch overhead, not FLOPs, most likely sets both times.

**Prefill, 16 requests × 256 tokens (4,096 rows), in ms:**

| distinct adapters | base matmul | loop | gather | segment | triton |
|---|---|---|---|---|---|
| 1 | 0.720 | 0.948 | 4.562 | 1.036 | 0.964 |
| 4 | 0.726 | 1.352 | 4.596 | 1.094 | 0.979 |
| 16 | 0.763 | 4.703 | 4.635 | 2.211 | 1.048 |

- **`gather` is the worst choice for prefill** at 4.6 ms. It copies a rank-16 `A` and `B` for
  every one of the 4,096 rows, even though only 16 distinct pairs exist.
- `segment` (SGMV) avoids that copy, but still pays one matmul pair per segment.
- Triton stays within 1.3–1.4× of the base matmul at every adapter count.

**End to end: GPT-2, 32 requests × 64 greedy tokens, each request with its own adapter:**

| strategy | tokens/s | greedy-token agreement with merged weights |
|---|---|---|
| base model only | 3233 | – |
| multi-LoRA (loop) | 85 | 0.949 |
| multi-LoRA (gather) | 1088 | 0.958 |
| multi-LoRA (segment) | 165 | 0.949 |
| multi-LoRA (triton) | **1519** | 0.943 |
| merge + run per adapter | 97 | – |

- **Batching all 32 adapters together with the Triton kernels is 15.7× faster than serving
  each adapter on its own** (1519 vs 97 tokens/s). It reaches 47% of the base model's
  throughput, while every request gets its own fine-tune.
- `loop` and `segment` are slower than merging here: with one request per adapter, they
  launch 2 kernels per adapter in each of 48 wrapped layers at every decode step.
- **Agreement with the merged weights is 94–96% of greedy tokens.** The CPU tests show the
  math is exact in fp64. In fp16, `W x + B(A x)` and `(W + BA) x` round differently. Once a
  near-tie flips one greedy token, the rest of that sequence diverges, so this counts
  diverged suffixes, not wrong logits.
