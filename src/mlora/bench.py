"""T4 benchmark of multi-adapter LoRA.

    python -m mlora.bench --out results

1. **Op level, decode** (one row per request): GPT-2's ``c_attn`` shape (768 → 2304),
   rank 16, 64 requests, spread over 1, 4, 16 or 64 distinct adapters. The table reports
   the time of the LoRA term next to the shared base matmul.
2. **Op level, prefill:** 16 requests × 256 tokens (4,096 rows), with the same spread.
3. **End to end:** GPT-2 (124M) greedy decoding of 64 tokens for 32 requests with 32
   different adapters. The serving strategies compared are base model only, multi-LoRA with
   each implementation, and "merge and run per adapter": fold each adapter into the weights
   and decode its requests as a separate batch, which is what single-adapter serving does.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from . import ops
from .model import attach, greedy, merge_adapter, set_adapters


def _ms(fn, reps: int = 50, rounds: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / reps * 1e3)
    return sorted(ts)[len(ts) // 2]


def _impls():
    impls = dict(ops.IMPLS)
    try:
        from .kernels import triton_bgmv

        impls["triton"] = triton_bgmv
    except ImportError:
        pass
    return impls


def op_bench(n_rows: int, req: int, distinct: list[int], d_in=768, d_out=2304, r=16, N=64):
    dev, dt = "cuda", torch.float16
    g = torch.Generator(device=dev).manual_seed(0)
    A = torch.randn(N, r, d_in, device=dev, dtype=dt, generator=g) / d_in**0.5
    B = torch.randn(N, d_out, r, device=dev, dtype=dt, generator=g) * 0.02
    W = torch.randn(d_in, d_out, device=dev, dtype=dt, generator=g) / d_in**0.5
    x = torch.randn(n_rows, d_in, device=dev, dtype=dt, generator=g)
    rows = []
    for k in distinct:
        per_req = torch.arange(req, device=dev) % k  # k distinct adapters, round-robin
        idx = per_req.repeat_interleave(n_rows // req)
        ref = ops.lora_reference(x, A, B, idx)
        r_ = {"rows": n_rows, "distinct_adapters": k, "base_matmul_ms": _ms(lambda: x @ W)}
        for name, fn in _impls().items():
            err = (fn(x, A, B, idx).float() - ref).abs().max().item()
            r_[f"{name}_ms"] = _ms(lambda fn=fn, idx=idx: fn(x, A, B, idx))
            r_[f"{name}_err"] = err
        rows.append(r_)
        print(json.dumps(r_), flush=True)
    return rows


def e2e_bench(n_req=32, n_adapters=32, prompt_len=32, new_tokens=64, r=16):
    from transformers import AutoModelForCausalLM

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).to(dev).eval()
    attach(model, n_adapters, r)
    g = torch.Generator(device=dev).manual_seed(0)
    prompts = torch.randint(0, 50000, (n_req, prompt_len), device=dev, generator=g)
    idx = torch.arange(n_req, device=dev) % n_adapters
    tokens = n_req * new_tokens

    def timed(fn):
        fn()  # warm-up (and Triton compile)
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[1]

    res = {}
    set_adapters(model, None)
    res["base model only"] = timed(lambda: greedy(model, prompts, new_tokens))
    outs = {}
    for impl in _impls():
        set_adapters(model, idx, impl)
        res[f"multi-LoRA ({impl})"] = timed(lambda: greedy(model, prompts, new_tokens))
        outs[impl] = greedy(model, prompts, new_tokens)

    def merged_serving():
        set_adapters(model, None)
        out = torch.empty(n_req, new_tokens, dtype=torch.long, device=dev)
        for a in range(n_adapters):
            sel = (idx == a).nonzero(as_tuple=True)[0]
            if len(sel) == 0:
                continue
            undo = merge_adapter(model, a)
            out[sel] = greedy(model, prompts[sel], new_tokens)
            undo()
        return out

    res["merge + run per adapter"] = timed(merged_serving)
    outs["merged"] = merged_serving()
    # agreement of greedy tokens with the merged (exact-weights) path; fp16 can flip near-ties
    agree = {k: float((v == outs["merged"]).float().mean()) for k, v in outs.items()}
    rows = [
        {"strategy": k, "seconds": v, "tokens_per_s": tokens / v, "token_agreement": None}
        for k, v in res.items()
    ]
    for row in rows:
        for impl, a in agree.items():
            if row["strategy"] == f"multi-LoRA ({impl})":
                row["token_agreement"] = a
    for row in rows:
        print(json.dumps(row), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.cuda.get_device_name()
    decode = op_bench(64, 64, [1, 4, 16, 64])
    prefill = op_bench(4096, 16, [1, 4, 16])
    e2e = e2e_bench()
    names = [k[:-3] for k in decode[0] if k.endswith("_ms") and k != "base_matmul_ms"]

    def table(rows, title):
        out = [
            f"### {title}",
            "",
            "| distinct adapters | base matmul | " + " | ".join(names) + " |",
            "|---|---|" + "---|" * len(names),
        ]
        for r in rows:
            out.append(
                f"| {r['distinct_adapters']} | {r['base_matmul_ms']:.3f} | "
                + " | ".join(f"{r[n + '_ms']:.3f}" for n in names)
                + " |"
            )
        return out

    lines = [f"GPU **{dev}**, fp16, rank 16, c_attn shape 768 → 2304. Times in ms (median).", ""]
    lines += table(decode, "Decode: 64 rows, one per request")
    lines += [""] + table(prefill, "Prefill: 16 requests × 256 tokens")
    lines += [
        "",
        "### End to end: GPT-2, 32 requests × 64 greedy tokens, 32 adapters",
        "",
        "| strategy | tokens/s | greedy-token agreement with merged weights |",
        "|---|---|---|",
    ]
    for r in e2e:
        ag = "–" if r["token_agreement"] is None else f"{r['token_agreement']:.3f}"
        lines.append(f"| {r['strategy']} | {r['tokens_per_s']:.0f} | {ag} |")
    text = "\n".join(lines) + "\n"
    tag = dev.replace(" ", "_")
    with open(os.path.join(a.out, f"bench_{tag}.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(a.out, f"bench_{tag}.json"), "w") as f:
        json.dump({"decode": decode, "prefill": prefill, "e2e": e2e}, f, indent=1)
    print(text)


if __name__ == "__main__":
    main()
