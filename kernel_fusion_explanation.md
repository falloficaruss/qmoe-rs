# Kernel Fusion in QMoE.rs: Fused Decompression and GEMV

## What is Kernel Fusion?

In high-performance computing and machine learning, **Kernel Fusion** refers to the technique of combining multiple operations into a single execution kernel. Instead of running Operation A (which writes its output to memory) followed by Operation B (which reads that output from memory), fused kernels perform both operations sequentially within the processor's registers or fast cache (SRAM/L1).

This drastically reduces **memory bandwidth bottlenecks**—the most common limiting factor in large language model (LLM) inference.

## The Problem: Memory Wall in Quantized LLMs

In Quantized Mixture of Experts (QMoE), weights are stored in a highly compressed format (e.g., 2 bits per weight). A naive approach to matrix multiplication (GEMM or GEMV) would involve two distinct steps:
1. **Decompression:** Read packed weights from memory, decode them into `f32` or `f16`, and write them to a temporary buffer in RAM.
2. **Matrix Multiplication:** Read the temporary uncompressed weights from RAM, multiply them by the input activations, and store the result.

This approach is inefficient. Writing the uncompressed weights back to memory only to immediately read them for multiplication wastes precious memory bandwidth and introduces latency.

## The Solution: Fused Decompression + GEMV

The `qmoe.rs` repository solves this by merging the decompression step directly into the inner loop of the Generalized Matrix-Vector Multiplication (GEMV) calculation. This ensures that **uncompressed parameters never leave the fast registers/L1 cache**, meaning they are never written back to main memory.

### Implementation Details in `qmoe.rs`

The fusion happens primarily across two files: `src/simd.rs` and `src/tensor.rs`.

#### 1. The Core Kernel: `simd.rs`
The `fused_dequantize_and_dot` function is the heart of the fusion. It processes the compressed matrix weights and the uncompressed `f32` input activations simultaneously using SIMD (Single Instruction, Multiple Data) instructions.

```rust
pub fn fused_dequantize_and_dot(
    packed_weights: &[u8],
    activations: &[f32],
    scale: f32,
) -> f32 { ... }
```

**How it works inside the loop:**
1. **Load:** It reads a single byte from `packed_weights` which contains 4 packed weights (2 bits each).
2. **Unpack (Decompress):** It extracts the four 2-bit values (`w0`, `w1`, `w2`, `w3`) using bitwise operations (`&` and `>>`).
3. **Dequantize:** It maps the 2-bit values (which represent integers `0, 1, 2`) into real `f32` weights (`-1.0, 0.0, 1.0`).
4. **Vectorize:** It loads these 4 `f32` weights into a SIMD register (`f32x4`). It simultaneously loads 4 continuous input activations into another SIMD register.
5. **Multiply-Add (GEMV):** It performs a Fused Multiply-Add (FMA) instruction (`sum += w_vec * a_vec`), directly updating the accumulator in the SIMD register.

At the end of the loop, it performs a horizontal sum of the SIMD register and multiplies by a block `scale` factor.

#### 2. The Matrix-Vector Loop: `tensor.rs`
The `PackedQMoETensor::forward_simd` function wraps the core kernel to process the entire matrix row-by-row.

```rust
for i in 0..out_features {
    let start = i * bytes_per_row;
    let end = start + bytes_per_row;
    let row_packed = &raw[start..end];
    let scale = scales_vec[i];
    
    // The Fused Kernel is called here
    let dot = crate::simd::fused_dequantize_and_dot(row_packed, x_vec, scale);
    out_data.push(dot);
}
```
For each output feature (i.e., each row of the weight matrix):
1. It slices out the **packed** bytes for that row.
2. It fetches the scale factor for that row.
3. It calls the `fused_dequantize_and_dot` kernel against the input vector `x_vec`.

## Why is this so fast?

1. **Zero Intermediate Memory:** Uncompressed `f32` weights only exist inside the CPU's vector registers (`f32x4`). They are created, multiplied, and discarded instantly.
2. **Maximized Bandwidth:** The memory bus only transfers 2-bit weights from RAM, making memory bandwidth utilization 16x more efficient than transferring uncompressed `f32` weights.
3. **L1 Cache Utilization:** Because there is no temporary uncompressed buffer bloating memory usage, the hot paths of the matrix multiply easily fit within the processor's ultra-fast L1 cache.
