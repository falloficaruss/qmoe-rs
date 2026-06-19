#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::simd::prelude::*;

// ---------------------------------------------------------------------------
// Baseline 1: Fully scalar unpack + dot (no SIMD vectors)
// ---------------------------------------------------------------------------
fn scalar_dequantize_and_dot(packed: &[u8], activations: &[f32], scale: f32) -> f32 {
    let num_bytes = packed.len().min(activations.len() / 4);
    let mut sum = 0.0f32;
    for i in 0..num_bytes {
        let byte = packed[i];
        let w0 = ((byte & 0b0000_0011) as f32) - 1.0;
        let w1 = (((byte >> 2) & 0b0000_0011) as f32) - 1.0;
        let w2 = (((byte >> 4) & 0b0000_0011) as f32) - 1.0;
        let w3 = (((byte >> 6) & 0b0000_0011) as f32) - 1.0;
        let a_idx = i * 4;
        sum += w0 * activations[a_idx]
            + w1 * activations[a_idx + 1]
            + w2 * activations[a_idx + 2]
            + w3 * activations[a_idx + 3];
    }
    sum * scale
}

// ---------------------------------------------------------------------------
// Baseline 2: Two-pass — unpack all weights to f32 buffer, then dot
// Measures the memory-bandwidth cost of writing then reading the unpacked buffer.
// ---------------------------------------------------------------------------
fn unpack_then_dot(packed: &[u8], activations: &[f32], scale: f32) -> f32 {
    let num_bytes = packed.len().min(activations.len() / 4);
    let mut unpacked = Vec::with_capacity(num_bytes * 4);
    for &byte in &packed[..num_bytes] {
        unpacked.push(((byte & 0b0000_0011) as f32) - 1.0);
        unpacked.push((((byte >> 2) & 0b0000_0011) as f32) - 1.0);
        unpacked.push((((byte >> 4) & 0b0000_0011) as f32) - 1.0);
        unpacked.push((((byte >> 6) & 0b0000_0011) as f32) - 1.0);
    }
    let mut sum = 0.0f32;
    for i in 0..unpacked.len() {
        sum += unpacked[i] * activations[i];
    }
    sum * scale
}

// ---------------------------------------------------------------------------
// Baseline 3: Wider SIMD (f32x8) — processes 2 bytes = 8 weights per iteration
// Requires in_features to be a multiple of 8.
// ---------------------------------------------------------------------------
fn fused_dequantize_and_dot_f32x8(packed: &[u8], activations: &[f32], scale: f32) -> f32 {
    debug_assert!(
        activations.len() % 8 == 0,
        "f32x8 variant requires in_features % 8 == 0"
    );
    let mut sum = f32x8::splat(0.0);
    let num_pairs = packed.len().min(activations.len() / 8) / 2;

    for i in 0..num_pairs {
        let b0 = packed[i * 2];
        let b1 = packed[i * 2 + 1];

        let w0 = (b0 & 0b0000_0011) as i32;
        let w1 = ((b0 >> 2) & 0b0000_0011) as i32;
        let w2 = ((b0 >> 4) & 0b0000_0011) as i32;
        let w3 = ((b0 >> 6) & 0b0000_0011) as i32;
        let w4 = (b1 & 0b0000_0011) as i32;
        let w5 = ((b1 >> 2) & 0b0000_0011) as i32;
        let w6 = ((b1 >> 4) & 0b0000_0011) as i32;
        let w7 = ((b1 >> 6) & 0b0000_0011) as i32;

        let w_vec = f32x8::from_array([
            (w0 as f32) - 1.0,
            (w1 as f32) - 1.0,
            (w2 as f32) - 1.0,
            (w3 as f32) - 1.0,
            (w4 as f32) - 1.0,
            (w5 as f32) - 1.0,
            (w6 as f32) - 1.0,
            (w7 as f32) - 1.0,
        ]);
        let a_vec = f32x8::from_slice(&activations[i * 8..i * 8 + 8]);
        sum += w_vec * a_vec;
    }

    sum.reduce_sum() * scale
}

// ---------------------------------------------------------------------------
// Baseline 4: Candle FP16 matmul (full-precision baseline)
// Compares our 2-bit packed GEMV against standard FP16 matrix-vector multiply.
// ---------------------------------------------------------------------------
fn candle_hf16_matmul(
    weight: &candle_core::Tensor,
    activations: &candle_core::Tensor,
) -> candle_core::Result<f32> {
    let result = weight.matmul(activations)?;
    result.to_vec0::<f32>()
}

// ---------------------------------------------------------------------------
// Deterministic test-data generator
// ---------------------------------------------------------------------------
fn generate_test_data(in_features: usize) -> (Vec<u8>, Vec<f32>) {
    let num_bytes = in_features / 4;
    // Deterministic pseudo-random: each byte alternates patterns
    let packed: Vec<u8> = (0..num_bytes)
        .map(|i| {
            let phase = i % 4;
            match phase {
                0 => 0b01011000, // [-1, 1, 0, 0]
                1 => 0b10100111, // [ 1,-1, 1, 1]
                2 => 0b00101110, // [ 0, 1,-1, 1]
                _ => 0b11100100, // [ 1, 1, 0,-1]
            }
        })
        .collect();

    // Activations: slowly varying sinusoid
    let activations: Vec<f32> = (0..in_features)
        .map(|i| (i as f32 * 0.1).sin() + 0.5)
        .collect();

    (packed, activations)
}

fn generate_candle_fixtures(
    in_features: usize,
    out_features: usize,
    device: &candle_core::Device,
) -> (candle_core::Tensor, candle_core::Tensor) {
    use candle_core::Tensor;

    // Create FP16 weight matrix
    let weight_data: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i as f32) * 0.07).cos())
        .collect();
    let weight = Tensor::from_vec(weight_data, (out_features, in_features), device)
        .unwrap()
        .to_dtype(candle_core::DType::F16)
        .unwrap();

    // Activation vector
    let act_data: Vec<f32> = (0..in_features)
        .map(|i| (i as f32 * 0.1).sin() + 0.5)
        .collect();
    let activations = Tensor::from_vec(act_data, (in_features, 1), device).unwrap();

    (weight, activations)
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

fn bench_simd_kernel(c: &mut Criterion) {
    let dims = [64, 128, 256, 512, 1024, 2048, 4096];

    let mut group = c.benchmark_group("simd_kernel/throughput");
    for &in_features in &dims {
        let (packed, activations) = generate_test_data(in_features);
        let scale = 1.0f32;

        group.throughput(Throughput::Elements(in_features as u64));

        // Current SIMD (f32x4) — the production kernel
        group.bench_with_input(
            BenchmarkId::new("fused_f32x4", in_features),
            &(&packed, &activations, scale),
            |b, (p, a, s)| {
                b.iter(|| {
                    qmoe_engine::simd::fused_dequantize_and_dot(
                        black_box(p),
                        black_box(a),
                        black_box(*s),
                    )
                })
            },
        );

        // Scalar baseline
        group.bench_with_input(
            BenchmarkId::new("scalar", in_features),
            &(&packed, &activations, scale),
            |b, (p, a, s)| {
                b.iter(|| scalar_dequantize_and_dot(black_box(p), black_box(a), black_box(*s)))
            },
        );

        // Two-pass unpack-then-dot baseline
        group.bench_with_input(
            BenchmarkId::new("unpack_then_dot", in_features),
            &(&packed, &activations, scale),
            |b, (p, a, s)| {
                b.iter(|| unpack_then_dot(black_box(p), black_box(a), black_box(*s)))
            },
        );

        // Wider SIMD (f32x8) — only for dimensions divisible by 8
        if in_features % 8 == 0 {
            group.bench_with_input(
                BenchmarkId::new("fused_f32x8", in_features),
                &(&packed, &activations, scale),
                |b, (p, a, s)| {
                    b.iter(|| {
                        fused_dequantize_and_dot_f32x8(
                            black_box(p),
                            black_box(a),
                            black_box(*s),
                        )
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_simd_candle_comparison(c: &mut Criterion) {
    let device = candle_core::Device::Cpu;
    let dims = [64, 128, 256, 512, 1024, 2048, 4096];
    let out_features = 256; // fixed small output dim for FP16 matmul

    let mut group = c.benchmark_group("simd_kernel/vs_fp16_matmul");
    for &in_features in &dims {
        let (packed, activations) = generate_test_data(in_features);
        let scale = 1.0f32;

        // Our packed kernel (single row)
        group.bench_with_input(
            BenchmarkId::new("packed_2bit_row", in_features),
            &(&packed, &activations, scale),
            |b, (p, a, s)| {
                b.iter(|| {
                    qmoe_engine::simd::fused_dequantize_and_dot(
                        black_box(p),
                        black_box(a),
                        black_box(*s),
                    )
                })
            },
        );

        // FP16 matmul: does out_features rows in one call (amortized overhead)
        let (weight_t, act_t) = generate_candle_fixtures(in_features, out_features, &device);
        group.bench_with_input(
            BenchmarkId::new("fp16_matmul_x256", in_features),
            &(&weight_t, &act_t),
            |b, (w, a)| {
                b.iter(|| candle_hf16_matmul(black_box(w), black_box(a)))
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = simd_kernel;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
        .sample_size(50);
    targets = bench_simd_kernel, bench_simd_candle_comparison
}

criterion_main!(simd_kernel);
