# MoE Layer Benchmark Results

**Date:** 2026-06-20
**CPU:** Intel Core i5-4200M @ 2.50 GHz (Haswell, AVX2, FMA)
**Rust:** nightly
**Criterion:** v0.5.1, 10 samples, 200 ms warm-up, 500 ms measurement

## Throughput — Token Count Sweep

Config: 32 experts, top-4, hidden_dim=2048, intermediate_dim=1408

| num_tokens | Latency | Throughput |
|-----------:|--------:|-----------:|
| 1 | 34.65 ms | 28.86 elem/s |
| 2 | 67.34 ms | 29.70 elem/s |
| 4 | 143.9 ms | 27.80 elem/s |
| 8 | 279.7 ms | 28.60 elem/s |
| 16 | 565.7 ms | 28.29 elem/s |
| 32 | 1.161 s | 27.56 elem/s |

Throughput is stable (~28 elem/s) across batch sizes, indicating MoE compute is the bottleneck.

## Throughput — Expert Count Sweep

Config: 16 tokens, top-4, hidden_dim=2048, intermediate_dim=1408

| num_experts | Latency | Throughput |
|------------:|--------:|-----------:|
| 8 | 492.6 ms | 32.48 elem/s |
| 16 | 511.0 ms | 31.31 elem/s |
| 32 | 527.2 ms | 30.35 elem/s |
| 64 | 558.7 ms | 28.64 elem/s |

Modest scaling — doubling experts increases latency by ~13% due to routing overhead.

## Throughput — Top-K Sweep

Config: 16 tokens, 32 experts, hidden_dim=2048, intermediate_dim=1408

| top_k | Latency | Throughput |
|------:|--------:|-----------:|
| 2 | 281.2 ms | 56.90 elem/s |
| 4 | 555.1 ms | 28.83 elem/s |
| 6 | 885.4 ms | 18.07 elem/s |
| 8 | 1.184 s | 13.51 elem/s |

Near-linear scaling with top-k: top_k=2 is ~2× faster than top_k=4, as expected.

## Throughput — Dimension Comparison

Config: 16 tokens, 32 experts (clamped to 8 for toy), top-4 (clamped to 2 for toy)

| Dim | Latency | Throughput |
|-----|--------:|-----------:|
| Toy (64×256) | 4.262 ms | 240.3 Kelem/s |
| Real (2048×1408) | 614.1 ms | 53.36 Kelem/s |

The real dim is ~144× slower than toy, reflecting the 32× larger hidden dim × 5.5× larger intermediate dim.

## Sub-Stage Breakdown — Real Dim

Config: 16 tokens, 32 experts, top-4, hidden_dim=2048, intermediate_dim=1408
Measured via single-shot timing, averaged over 10 runs.

| Sub-stage | μs | % of total |
|-----------|----|-----------|
| Gate matmul | 216.85 µs | 0.0% |
| Top-k sorting | 4.18 µs | 0.0% |
| Softmax + binning | 26.21 µs | 0.0% |
| Expert compute (3× fwd_simd + SwiGLU) | 597.93 ms | 100.0% |
| **Total** | **598.18 ms** | **100%** |

**Key finding:** Expert compute dominates (>99.9%). Routing overhead (gate matmul + top-k + softmax) accounts for <0.1% of total time.

## Sub-Stage Breakdown — Toy Dim

Config: 16 tokens, 32 experts (clamped to 8), top-4 (clamped to 2), hidden_dim=64, intermediate_dim=256

| Sub-stage | μs | % of total |
|-----------|----|-----------|
| Gate matmul | 27.27 µs | 0.6% |
| Top-k sorting | 4.45 µs | 0.1% |
| Softmax + binning | 7.90 µs | 0.2% |
| Expert compute (3× fwd_simd + SwiGLU) | 4.33 ms | 99.1% |
| **Total** | **4.37 ms** | **100%** |

Expert compute still dominates at 99%, but routing overhead becomes measurable at 0.9% for toy dimensions.

## Comparison: Standard vs Baselines

Config: 16 tokens, 32 experts, top-4, hidden_dim=2048, intermediate_dim=1408

| Variant | Latency | vs Standard |
|---------|---------:|------------:|
| Standard (binned routing) | 621.77 ms | 1.00× |
| Oracle (all experts, no routing) | 4376.54 ms | 7.04× |
| Naive loop (no binning) | 504.73 ms | 0.81× |
| FP16 matmul (unquantized) | 118.83 ms | 0.19× |

**Analysis:**
- **Oracle** is 7× slower because it processes all 32 experts per token (32×16=512 invocations) vs top-4 (4×16=64 invocations) — an 8× difference, close to the measured 7×.
- **Naive loop** is slightly faster than binned routing with synthetic data, suggesting the binning overhead outweighs its cache benefit when tokens are evenly distributed.
- **FP16 matmul** is 5.2× faster than packed 2-bit, confirming candle's optimized BLAS matmul significantly outperforms our scalar fused dequant-GEMV loop in raw throughput.

## Criterion Raw Measurements

Full criterion output is saved in `moe_layer_raw.txt`.

## Key Takeaways

1. **Expert compute dominates** the MoE layer forward pass (>99.9% for real dimensions). Routing overhead (gate matmul, top-k, softmax) is negligible.
2. **Throughput is stable** across batch sizes (~28 elem/s), indicating the kernel is compute-bound rather than memory-bound.
3. **Top-k scaling is near-linear** — throughput halves when top-k doubles.
4. **Packed 2-bit is 5.2× slower than FP16 matmul** on CPU, which is expected — the packed format trades compute for memory savings (16× fewer weight bits).
5. **Oracle baseline confirms** that routing is essential: using all experts is 7× slower than top-4 routing.
