GPU **Tesla T4**, fp16, rank 16, c_attn shape 768 → 2304. Times in ms (median).

### Decode: 64 rows, one per request

| distinct adapters | base matmul | loop | gather | segment | triton |
|---|---|---|---|---|---|
| 1 | 0.027 | 0.388 | 0.265 | 0.429 | 0.101 |
| 4 | 0.043 | 1.091 | 0.263 | 0.733 | 0.099 |
| 16 | 0.044 | 3.903 | 0.387 | 2.542 | 0.129 |
| 64 | 0.043 | 14.065 | 0.262 | 6.385 | 0.100 |

### Prefill: 16 requests × 256 tokens

| distinct adapters | base matmul | loop | gather | segment | triton |
|---|---|---|---|---|---|
| 1 | 0.720 | 0.948 | 4.562 | 1.036 | 0.964 |
| 4 | 0.726 | 1.352 | 4.596 | 1.094 | 0.979 |
| 16 | 0.763 | 4.703 | 4.635 | 2.211 | 1.048 |

### End to end: GPT-2, 32 requests × 64 greedy tokens, 32 adapters

| strategy | tokens/s | greedy-token agreement with merged weights |
|---|---|---|
| base model only | 3233 | – |
| multi-LoRA (loop) | 85 | 0.949 |
| multi-LoRA (gather) | 1088 | 0.958 |
| multi-LoRA (segment) | 165 | 0.949 |
| multi-LoRA (triton) | 1519 | 0.943 |
| merge + run per adapter | 97 | – |
