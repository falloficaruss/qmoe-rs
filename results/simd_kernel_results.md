# SIMD Kernel Benchmark Results

**Date:** 2026-06-19
**CPU:** Intel Core i5-4200M @ 2.50 GHz (Haswell, AVX2, FMA)
**Rust:** nightly 1.97.0
**Criterion:** v0.5.1, 50 samples, 500 ms warm-up, 2 s measurement

## Throughput Comparison

| in_features | fused_f32x4 | fused_f32x8 | scalar | unpack_then_dot |
|------------:|------------:|------------:|-------:|----------------:|
| 64          | 953 Melem/s | 1.36 Gelem/s | 682 Melem/s | 358 Melem/s |
| 128         | 980 Melem/s | 1.48 Gelem/s | 700 Melem/s | 380 Melem/s |
| 256         | 1.00 Gelem/s | 1.56 Gelem/s | 699 Melem/s | 393 Melem/s |
| 512         | 1.03 Gelem/s | 1.61 Gelem/s | 735 Melem/s | 388 Melem/s |
| 1024        | 1.02 Gelem/s | 1.62 Gelem/s | 719 Melem/s | 393 Melem/s |
| 2048        | 1.01 Gelem/s | 1.69 Gelem/s | 699 Melem/s | 412 Melem/s |
| 4096        | 1.06 Gelem/s | 1.66 Gelem/s | 734 Melem/s | 408 Melem/s |
| **Geomean** | **1.01 Gelem/s** | **1.58 Gelem/s** | **712 Melem/s** | **395 Melem/s** |

## Latency Comparison

| in_features | fused_f32x4 | fused_f32x8 | scalar | unpack_then_dot |
|------------:|------------:|------------:|-------:|----------------:|
| 64          | 67.1 ns     | 47.0 ns     | 93.8 ns | 179 ns |
| 128         | 131 ns      | 86.4 ns     | 183 ns | 337 ns |
| 256         | 255 ns      | 165 ns      | 366 ns | 651 ns |
| 512         | 495 ns      | 319 ns      | 697 ns | 1.32 µs |
| 1024        | 999 ns      | 633 ns      | 1.42 µs | 2.61 µs |
| 2048        | 2.02 µs     | 1.21 µs     | 2.93 µs | 4.97 µs |
| 4096        | 3.86 µs     | 2.47 µs     | 5.58 µs | 10.0 µs |

## Speedup Summary (geometric mean across all dims)

| Comparison | Speedup |
|-----------|--------|
| fused_f32x4 vs scalar | **1.42×** |
| fused_f32x4 vs unpack_then_dot | **2.56×** |
| fused_f32x8 vs fused_f32x4 | **1.57×** |
| fused_f32x8 vs scalar | **2.22×** |
| fused_f32x8 vs unpack_then_dot | **3.99×** |

## Analysis

1. **SIMD vectorization (f32x4 vs scalar):** 1.42× speedup. The scalar baseline is already
   relatively efficient (the compiler auto-vectorizes partially), but explicit SIMD gives
   a measurable edge through tighter register control.

2. **Fusion benefit (fused vs unpack_then_dot):** 2.56×. Writing unpacked f32 weights to
   memory and reading them back costs ~2.5× the compute time. Keeping dequantized values
   in SIMD registers is the critical optimization.

3. **Wider SIMD (f32x8 vs f32x4):** 1.57×. The Haswell CPU has two 128-bit FMA units
   that can combine to handle 256-bit vectors efficiently. The wider kernel reduces
   loop overhead and instruction count per element.

4. **Dimension scaling:** All variants show stable throughput across the full 64–4096
   range, indicating the working set stays within L1/L2 cache.
