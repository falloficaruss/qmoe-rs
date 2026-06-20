use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use candle_core::{Device, Tensor, DType};
use qmoe_engine::moe::{MoEConfig, PackedExpert, QMoELayer};
use qmoe_engine::tensor::PackedQMoETensor;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Deterministic test-data generators
// ---------------------------------------------------------------------------

fn generate_packed_data(out_features: usize, in_features: usize) -> (Vec<u8>, Vec<f32>) {
    let bytes_per_row = in_features / 4;
    let total_bytes = out_features * bytes_per_row;
    let packed: Vec<u8> = (0..total_bytes)
        .map(|i| {
            let phase = i % 4;
            match phase {
                0 => 0b01011000,
                1 => 0b10100111,
                2 => 0b00101110,
                _ => 0b11100100,
            }
        })
        .collect();
    let scales: Vec<f32> = (0..out_features)
        .map(|i| (i as f32 * 0.05).sin() + 1.0)
        .collect();
    (packed, scales)
}

fn generate_gate_weight(num_experts: usize, hidden_dim: usize) -> Tensor {
    let data: Vec<f32> = (0..num_experts * hidden_dim)
        .map(|i| ((i as f32) * 0.07).cos())
        .collect();
    Tensor::from_vec(data, (num_experts, hidden_dim), &Device::Cpu).unwrap()
}

fn generate_experts(
    config: &MoEConfig,
) -> Vec<PackedExpert> {
    let mut experts = Vec::with_capacity(config.num_experts);
    for _ in 0..config.num_experts {
        let (gate_data, gate_scales) = generate_packed_data(config.intermediate_dim, config.hidden_dim);
        let (up_data, up_scales) = generate_packed_data(config.intermediate_dim, config.hidden_dim);
        let (down_data, down_scales) = generate_packed_data(config.hidden_dim, config.intermediate_dim);

        let gate_proj = PackedQMoETensor::from_bytes(
            gate_data,
            (config.intermediate_dim, config.hidden_dim),
            Tensor::from_vec(gate_scales, config.intermediate_dim, &Device::Cpu).unwrap(),
        );
        let up_proj = PackedQMoETensor::from_bytes(
            up_data,
            (config.intermediate_dim, config.hidden_dim),
            Tensor::from_vec(up_scales, config.intermediate_dim, &Device::Cpu).unwrap(),
        );
        let down_proj = PackedQMoETensor::from_bytes(
            down_data,
            (config.hidden_dim, config.intermediate_dim),
            Tensor::from_vec(down_scales, config.hidden_dim, &Device::Cpu).unwrap(),
        );

        experts.push(PackedExpert { gate_proj, up_proj, down_proj });
    }
    experts
}

fn generate_input(num_tokens: usize, hidden_dim: usize) -> Tensor {
    let data: Vec<f32> = (0..num_tokens * hidden_dim)
        .map(|i| ((i as f32) * 0.1).sin() + 0.5)
        .collect();
    Tensor::from_vec(data, (num_tokens, hidden_dim), &Device::Cpu).unwrap()
}

fn make_moe_layer(
    num_experts: usize,
    top_k: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> QMoELayer {
    let config = MoEConfig {
        num_experts,
        top_k,
        hidden_dim,
        intermediate_dim,
    };
    let gate_weight = generate_gate_weight(num_experts, hidden_dim);
    let experts = generate_experts(&config);
    QMoELayer::new(config, gate_weight, experts)
}

// ---------------------------------------------------------------------------
// Breakdown measurement
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ForwardBreakdown {
    gate_matmul_us: f64,
    topk_sorting_us: f64,
    softmax_binning_us: f64,
    expert_compute_us: f64,    // includes 3× forward_simd + SwiGLU + weighted combine
    total_us: f64,
}

fn forward_with_breakdown(layer: &QMoELayer, xs: &Tensor) -> (Tensor, ForwardBreakdown) {
    let (seq_len, hidden_dim) = xs.dims2().unwrap();
    let device = xs.device();
    let config = &layer.config;

    let t0 = Instant::now();

    // 1. Gate matmul
    let gate_logits = xs.matmul(&layer.gate_weight.t().unwrap()).unwrap();
    let t1 = Instant::now();

    // 2. Gate data to vec for CPU routing
    let gate_data = gate_logits.to_vec2::<f32>().unwrap();
    let t2 = Instant::now();

    // 3. Top-k routing + softmax + binning
    let mut expert_bins: Vec<Vec<(usize, f32)>> = vec![Vec::new(); config.num_experts];

    for t in 0..seq_len {
        let logits = &gate_data[t];
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .cloned()
            .enumerate()
            .collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_val = indexed_logits.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
        let all_sum_exp: f32 = indexed_logits.iter().map(|x| (x.1 - max_val).exp()).sum();
        let top_k_elements = &indexed_logits[0..config.top_k];

        for &(expert_idx, score) in top_k_elements {
            let weight = (score - max_val).exp() / all_sum_exp;
            expert_bins[expert_idx].push((t, weight));
        }
    }
    let t3 = Instant::now();

    // 4. Expert compute + weighted combine
    let mut output_buf = vec![0.0f32; seq_len * hidden_dim];
    let xs_data = xs.to_vec2::<f32>().unwrap();

    for (expert_idx, bin) in expert_bins.iter().enumerate() {
        if bin.is_empty() {
            continue;
        }
        let expert = &layer.experts[expert_idx];

        for &(token_idx, weight) in bin {
            let token_vector = &xs_data[token_idx];
            let token_tensor = Tensor::from_vec(token_vector.clone(), (1, hidden_dim), device).unwrap();

            let gate_out = expert.gate_proj.forward_simd(&token_tensor).unwrap();
            let up_out = expert.up_proj.forward_simd(&token_tensor).unwrap();

            let gate_out_vec = gate_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let up_out_vec = up_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let mut activated = Vec::with_capacity(up_out_vec.len());
            for i in 0..up_out_vec.len() {
                let g = gate_out_vec[i];
                let u = up_out_vec[i];
                let swish = g * (1.0 / (1.0 + (-g).exp()));
                activated.push(swish * u);
            }

            let activated_tensor =
                Tensor::from_vec(activated, (1, config.intermediate_dim), device).unwrap();
            let down_out = expert.down_proj.forward_simd(&activated_tensor).unwrap();
            let down_out_vec = down_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let out_offset = token_idx * hidden_dim;
            for h in 0..hidden_dim {
                output_buf[out_offset + h] += down_out_vec[h] * weight;
            }
        }
    }
    let t4 = Instant::now();

    let result = Tensor::from_vec(output_buf, (seq_len, hidden_dim), device).unwrap();

    let breakdown = ForwardBreakdown {
        gate_matmul_us: (t1 - t0).as_secs_f64() * 1e6,
        topk_sorting_us: (t2 - t1).as_secs_f64() * 1e6,
        softmax_binning_us: (t3 - t2).as_secs_f64() * 1e6,
        expert_compute_us: (t4 - t3).as_secs_f64() * 1e6,
        total_us: (t4 - t0).as_secs_f64() * 1e6,
    };

    (result, breakdown)
}

// ---------------------------------------------------------------------------
// Oracle: all experts always on, uniform weight — no routing
// ---------------------------------------------------------------------------

fn forward_oracle(
    config: &MoEConfig,
    experts: &[PackedExpert],
    gate_weight: &Tensor,
    xs: &Tensor,
) -> Tensor {
    let (seq_len, hidden_dim) = xs.dims2().unwrap();
    let device = xs.device();

    if experts.is_empty() {
        return Tensor::zeros((seq_len, hidden_dim), xs.dtype(), device).unwrap();
    }

    // Still compute gate logits for API parity, but use uniform weights
    let _gate_logits = xs.matmul(&gate_weight.t().unwrap()).unwrap();

    let mut output_buf = vec![0.0f32; seq_len * hidden_dim];
    let xs_data = xs.to_vec2::<f32>().unwrap();
    let uniform_weight = 1.0 / config.num_experts as f32;

    for (_expert_idx, expert) in experts.iter().enumerate() {
        for t in 0..seq_len {
            let token_vector = &xs_data[t];
            let token_tensor =
                Tensor::from_vec(token_vector.clone(), (1, hidden_dim), device).unwrap();

            let gate_out = expert.gate_proj.forward_simd(&token_tensor).unwrap();
            let up_out = expert.up_proj.forward_simd(&token_tensor).unwrap();

            let gate_out_vec = gate_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let up_out_vec = up_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let mut activated = Vec::with_capacity(up_out_vec.len());
            for i in 0..up_out_vec.len() {
                let g = gate_out_vec[i];
                let u = up_out_vec[i];
                let swish = g * (1.0 / (1.0 + (-g).exp()));
                activated.push(swish * u);
            }

            let activated_tensor =
                Tensor::from_vec(activated, (1, config.intermediate_dim), device).unwrap();
            let down_out = expert.down_proj.forward_simd(&activated_tensor).unwrap();
            let down_out_vec = down_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let out_offset = t * hidden_dim;
            for h in 0..hidden_dim {
                output_buf[out_offset + h] += down_out_vec[h] * uniform_weight;
            }
        }
    }

    Tensor::from_vec(output_buf, (seq_len, hidden_dim), device).unwrap()
}

// ---------------------------------------------------------------------------
// Naive loop: per-token processing without binning
// ---------------------------------------------------------------------------

fn forward_naive(layer: &QMoELayer, xs: &Tensor) -> Tensor {
    let (seq_len, hidden_dim) = xs.dims2().unwrap();
    let device = xs.device();

    if layer.experts.is_empty() {
        return Tensor::zeros((seq_len, hidden_dim), xs.dtype(), device).unwrap();
    }

    let gate_logits = xs.matmul(&layer.gate_weight.t().unwrap()).unwrap();
    let gate_data = gate_logits.to_vec2::<f32>().unwrap();
    let xs_data = xs.to_vec2::<f32>().unwrap();

    let mut output_buf = vec![0.0f32; seq_len * hidden_dim];

    for t in 0..seq_len {
        let logits = &gate_data[t];
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .cloned()
            .enumerate()
            .collect();
        indexed_logits
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_val = indexed_logits
            .iter()
            .map(|x| x.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let all_sum_exp: f32 = indexed_logits.iter().map(|x| (x.1 - max_val).exp()).sum();
        let top_k_elements = &indexed_logits[0..layer.config.top_k];

        let token_vector = &xs_data[t];
        let token_tensor =
            Tensor::from_vec(token_vector.clone(), (1, hidden_dim), device).unwrap();

        for &(expert_idx, score) in top_k_elements {
            let weight = (score - max_val).exp() / all_sum_exp;
            let expert = &layer.experts[expert_idx];

            let gate_out = expert.gate_proj.forward_simd(&token_tensor).unwrap();
            let up_out = expert.up_proj.forward_simd(&token_tensor).unwrap();

            let gate_out_vec = gate_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let up_out_vec = up_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let mut activated = Vec::with_capacity(up_out_vec.len());
            for i in 0..up_out_vec.len() {
                let g = gate_out_vec[i];
                let u = up_out_vec[i];
                let swish = g * (1.0 / (1.0 + (-g).exp()));
                activated.push(swish * u);
            }

            let activated_tensor =
                Tensor::from_vec(activated, (1, layer.config.intermediate_dim), device).unwrap();
            let down_out = expert.down_proj.forward_simd(&activated_tensor).unwrap();
            let down_out_vec = down_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let out_offset = t * hidden_dim;
            for h in 0..hidden_dim {
                output_buf[out_offset + h] += down_out_vec[h] * weight;
            }
        }
    }

    Tensor::from_vec(output_buf, (seq_len, hidden_dim), device).unwrap()
}

// ---------------------------------------------------------------------------
// FP16 baseline: unquantized FP16 weights with candle matmul
// ---------------------------------------------------------------------------

struct FP16ExpertWeights {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

fn generate_fp16_experts(
    num_experts: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<FP16ExpertWeights> {
    (0..num_experts)
        .map(|_| {
            let gate_data: Vec<f32> = (0..intermediate_dim * hidden_dim)
                .map(|i| ((i as f32) * 0.07).cos())
                .collect();
            let up_data: Vec<f32> = (0..intermediate_dim * hidden_dim)
                .map(|i| ((i as f32) * 0.11).sin())
                .collect();
            let down_data: Vec<f32> = (0..hidden_dim * intermediate_dim)
                .map(|i| ((i as f32) * 0.13).cos())
                .collect();

            let device = Device::Cpu;
            FP16ExpertWeights {
                gate: Tensor::from_vec(gate_data, (intermediate_dim, hidden_dim), &device)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap(),
                up: Tensor::from_vec(up_data, (intermediate_dim, hidden_dim), &device)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap(),
                down: Tensor::from_vec(down_data, (hidden_dim, intermediate_dim), &device)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap(),
            }
        })
        .collect()
}

fn forward_fp16(
    gate_weight: &Tensor,
    fp16_experts: &[FP16ExpertWeights],
    config: &MoEConfig,
    xs: &Tensor,
) -> Tensor {
    let (seq_len, hidden_dim) = xs.dims2().unwrap();
    let device = xs.device();

    if fp16_experts.is_empty() {
        return Tensor::zeros((seq_len, hidden_dim), xs.dtype(), device).unwrap();
    }

    // Gate logits and routing (same as standard forward)
    let gate_logits = xs.matmul(&gate_weight.t().unwrap()).unwrap();
    let gate_data = gate_logits.to_vec2::<f32>().unwrap();
    let xs_data = xs.to_vec2::<f32>().unwrap();

    let mut expert_bins: Vec<Vec<(usize, f32)>> = vec![Vec::new(); config.num_experts];

    for t in 0..seq_len {
        let logits = &gate_data[t];
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed_logits
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_val = indexed_logits
            .iter()
            .map(|x| x.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let all_sum_exp: f32 = indexed_logits.iter().map(|x| (x.1 - max_val).exp()).sum();
        let top_k_elements = &indexed_logits[0..config.top_k];

        for &(expert_idx, score) in top_k_elements {
            let weight = (score - max_val).exp() / all_sum_exp;
            expert_bins[expert_idx].push((t, weight));
        }
    }

    // Expert compute using FP16 matmul
    let mut output_buf = vec![0.0f32; seq_len * hidden_dim];

    for (expert_idx, bin) in expert_bins.iter().enumerate() {
        if bin.is_empty() {
            continue;
        }
        let exp = &fp16_experts[expert_idx];

        for &(token_idx, weight) in bin {
            let token_vector = &xs_data[token_idx];
            let token_f16 = Tensor::from_vec(token_vector.to_vec(), (1, hidden_dim), device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();

            // FP16 matmul: [interm, hidden] @ [hidden, 1] → [interm, 1]
            let gate_out = exp.gate.matmul(&token_f16.reshape((hidden_dim, 1)).unwrap()).unwrap();
            let up_out = exp.up.matmul(&token_f16.reshape((hidden_dim, 1)).unwrap()).unwrap();

            let gate_vec = gate_out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let up_vec = up_out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let mut activated = Vec::with_capacity(up_vec.len());
            for i in 0..up_vec.len() {
                let g = gate_vec[i];
                let u = up_vec[i];
                let swish = g * (1.0 / (1.0 + (-g).exp()));
                activated.push(swish * u);
            }

            let activated_t = Tensor::from_vec(activated, (1, config.intermediate_dim), device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let down_out = exp.down.matmul(&activated_t.reshape((config.intermediate_dim, 1)).unwrap()).unwrap();
            let down_vec = down_out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let out_offset = token_idx * hidden_dim;
            for h in 0..hidden_dim {
                output_buf[out_offset + h] += down_vec[h] * weight;
            }
        }
    }

    Tensor::from_vec(output_buf, (seq_len, hidden_dim), device).unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark: forward throughput — sweep all dimension combinations
// ---------------------------------------------------------------------------

const DEFAULT_NUM_TOKENS: usize = 16;
const DEFAULT_NUM_EXPERTS: usize = 32;
const DEFAULT_TOP_K: usize = 4;
const DEFAULT_HIDDEN_DIM: usize = 2048;
const DEFAULT_INTERMEDIATE_DIM: usize = 1408;

const TOY_HIDDEN_DIM: usize = 64;
const TOY_INTERMEDIATE_DIM: usize = 256;

fn bench_forward_throughput(c: &mut Criterion) {
    // Sweep num_tokens
    {
        let mut group = c.benchmark_group("moe_layer/throughput_tokens");
        for &num_tokens in &[1, 2, 4, 8, 16, 32] {
            let layer = make_moe_layer(
                DEFAULT_NUM_EXPERTS,
                DEFAULT_TOP_K,
                DEFAULT_HIDDEN_DIM,
                DEFAULT_INTERMEDIATE_DIM,
            );
            let input = generate_input(num_tokens, DEFAULT_HIDDEN_DIM);
            group.throughput(Throughput::Elements(num_tokens as u64));
            group.bench_with_input(
                BenchmarkId::new("tokens", num_tokens),
                &(&layer, &input),
                |b, (l, i)| b.iter(|| l.forward(black_box(i)).unwrap()),
            );
        }
        group.finish();
    }

    // Sweep num_experts
    {
        let mut group = c.benchmark_group("moe_layer/throughput_experts");
        for &num_experts in &[8, 16, 32, 64] {
            let top_k = num_experts.min(DEFAULT_TOP_K);
            let layer = make_moe_layer(
                num_experts,
                top_k,
                DEFAULT_HIDDEN_DIM,
                DEFAULT_INTERMEDIATE_DIM,
            );
            let input = generate_input(DEFAULT_NUM_TOKENS, DEFAULT_HIDDEN_DIM);
            group.throughput(Throughput::Elements(DEFAULT_NUM_TOKENS as u64));
            group.bench_with_input(
                BenchmarkId::new("experts", num_experts),
                &(&layer, &input),
                |b, (l, i)| b.iter(|| l.forward(black_box(i)).unwrap()),
            );
        }
        group.finish();
    }

    // Sweep top_k
    {
        let mut group = c.benchmark_group("moe_layer/throughput_topk");
        for &top_k in &[2, 4, 6, 8] {
            let layer = make_moe_layer(
                DEFAULT_NUM_EXPERTS,
                top_k,
                DEFAULT_HIDDEN_DIM,
                DEFAULT_INTERMEDIATE_DIM,
            );
            let input = generate_input(DEFAULT_NUM_TOKENS, DEFAULT_HIDDEN_DIM);
            group.throughput(Throughput::Elements(DEFAULT_NUM_TOKENS as u64));
            group.bench_with_input(
                BenchmarkId::new("topk", top_k),
                &(&layer, &input),
                |b, (l, i)| b.iter(|| l.forward(black_box(i)).unwrap()),
            );
        }
        group.finish();
    }

    // Compare toy dim vs real dim
    {
        let mut group = c.benchmark_group("moe_layer/throughput_dim");
        for &(hidden_dim, intermediate_dim, label) in
            &[(TOY_HIDDEN_DIM, TOY_INTERMEDIATE_DIM, "toy"), (DEFAULT_HIDDEN_DIM, DEFAULT_INTERMEDIATE_DIM, "real")]
        {
            let layer = make_moe_layer(
                DEFAULT_NUM_EXPERTS.min(8),
                DEFAULT_TOP_K.min(4),
                hidden_dim,
                intermediate_dim,
            );
            let input = generate_input(DEFAULT_NUM_TOKENS, hidden_dim);
            group.throughput(Throughput::Elements((DEFAULT_NUM_TOKENS * hidden_dim) as u64));
            group.bench_with_input(
                BenchmarkId::new("dim", label),
                &(&layer, &input),
                |b, (l, i)| b.iter(|| l.forward(black_box(i)).unwrap()),
            );
        }
        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Benchmark: sub-stage breakdown for a fixed config
// ---------------------------------------------------------------------------

fn bench_forward_breakdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_layer/breakdown");
    group.sample_size(30);
    let layer = make_moe_layer(
        DEFAULT_NUM_EXPERTS,
        DEFAULT_TOP_K,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_INTERMEDIATE_DIM,
    );
    let input = generate_input(DEFAULT_NUM_TOKENS, DEFAULT_HIDDEN_DIM);
    group.bench_function("full", |b| {
        b.iter(|| {
            forward_with_breakdown(black_box(&layer), black_box(&input))
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: standard vs oracle (no routing)
// ---------------------------------------------------------------------------

fn bench_comparison_oracle(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_layer/vs_oracle");
    let config = MoEConfig {
        num_experts: DEFAULT_NUM_EXPERTS,
        top_k: DEFAULT_TOP_K,
        hidden_dim: DEFAULT_HIDDEN_DIM,
        intermediate_dim: DEFAULT_INTERMEDIATE_DIM,
    };
    let gate_weight = generate_gate_weight(DEFAULT_NUM_EXPERTS, DEFAULT_HIDDEN_DIM);
    let experts = generate_experts(&config);
    let layer = QMoELayer::new(config.clone(), gate_weight.clone(), experts);
    let input = generate_input(DEFAULT_NUM_TOKENS, DEFAULT_HIDDEN_DIM);

    group.throughput(Throughput::Elements(DEFAULT_NUM_TOKENS as u64));

    group.bench_with_input(
        BenchmarkId::new("standard", DEFAULT_NUM_TOKENS),
        &(&layer, &input),
        |b, (l, i)| b.iter(|| l.forward(black_box(i)).unwrap()),
    );

    let oracle_experts = generate_experts(&config);
    group.bench_with_input(
        BenchmarkId::new("oracle", DEFAULT_NUM_TOKENS),
        &(&config, &oracle_experts, &gate_weight, &input),
        |b, (cfg, exp, gw, xs)| {
            b.iter(|| forward_oracle(black_box(cfg), black_box(exp), black_box(gw), black_box(xs)))
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: standard vs naive loop (no binning)
// ---------------------------------------------------------------------------

fn bench_comparison_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_layer/vs_naive");
    let layer = make_moe_layer(
        DEFAULT_NUM_EXPERTS,
        DEFAULT_TOP_K,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_INTERMEDIATE_DIM,
    );
    let input = generate_input(DEFAULT_NUM_TOKENS, DEFAULT_HIDDEN_DIM);

    group.throughput(Throughput::Elements(DEFAULT_NUM_TOKENS as u64));

    group.bench_with_input(
        BenchmarkId::new("standard", DEFAULT_NUM_TOKENS),
        &(&layer, &input),
        |b, (l, i)| b.iter(|| l.forward(black_box(i)).unwrap()),
    );

    group.bench_with_input(
        BenchmarkId::new("naive_loop", DEFAULT_NUM_TOKENS),
        &(&layer, &input),
        |b, (l, i)| b.iter(|| forward_naive(black_box(l), black_box(i))),
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: standard packed vs FP16 matmul
// ---------------------------------------------------------------------------

fn bench_comparison_fp16(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_layer/vs_fp16");
    let config = MoEConfig {
        num_experts: DEFAULT_NUM_EXPERTS,
        top_k: DEFAULT_TOP_K,
        hidden_dim: DEFAULT_HIDDEN_DIM,
        intermediate_dim: DEFAULT_INTERMEDIATE_DIM,
    };
    let gate_weight = generate_gate_weight(DEFAULT_NUM_EXPERTS, DEFAULT_HIDDEN_DIM);
    let experts = generate_experts(&config);
    let layer = QMoELayer::new(config.clone(), gate_weight.clone(), experts);
    let input = generate_input(DEFAULT_NUM_TOKENS, DEFAULT_HIDDEN_DIM);

    group.throughput(Throughput::Elements(DEFAULT_NUM_TOKENS as u64));

    group.bench_with_input(
        BenchmarkId::new("packed_2bit", DEFAULT_NUM_TOKENS),
        &(&layer, &input),
        |b, (l, i)| b.iter(|| l.forward(black_box(i)).unwrap()),
    );

    let fp16_experts = generate_fp16_experts(
        DEFAULT_NUM_EXPERTS,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_INTERMEDIATE_DIM,
    );
    group.bench_with_input(
        BenchmarkId::new("fp16_matmul", DEFAULT_NUM_TOKENS),
        &(&gate_weight, &fp16_experts, &config, &input),
        |b, (gw, exp, cfg, xs)| {
            b.iter(|| forward_fp16(black_box(gw), black_box(exp), black_box(cfg), black_box(xs)))
        },
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Secondary benchmarks: breakdown + comparisons across multiple token counts
// ---------------------------------------------------------------------------

fn bench_breakdown_varied(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_layer/breakdown_varied");

    for &num_tokens in &[1, 4, 16, 32] {
        let layer = make_moe_layer(
            DEFAULT_NUM_EXPERTS,
            DEFAULT_TOP_K,
            DEFAULT_HIDDEN_DIM,
            DEFAULT_INTERMEDIATE_DIM,
        );
        let input = generate_input(num_tokens, DEFAULT_HIDDEN_DIM);
        group.bench_with_input(
            BenchmarkId::new("breakdown", num_tokens),
            &(&layer, &input),
            |b, (l, i)| b.iter(|| forward_with_breakdown(black_box(l), black_box(i))),
        );
    }

    group.finish();
}

fn bench_breakdown_toy(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_layer/breakdown_toy");

    for &num_tokens in &[1, 4, 16, 32] {
        let layer = make_moe_layer(
            DEFAULT_NUM_EXPERTS.min(8),
            DEFAULT_TOP_K.min(4),
            TOY_HIDDEN_DIM,
            TOY_INTERMEDIATE_DIM,
        );
        let input = generate_input(num_tokens, TOY_HIDDEN_DIM);
        group.bench_with_input(
            BenchmarkId::new("breakdown_toy", num_tokens),
            &(&layer, &input),
            |b, (l, i)| b.iter(|| forward_with_breakdown(black_box(l), black_box(i))),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Manual single-shot analysis for breakdown tables (printed to stdout)
// ---------------------------------------------------------------------------

fn print_breakdown_table() {
    println!("\n## MoE Layer Breakdown — Single-Shot Timing\n");
    println!("Config: {} tokens, {} experts, top-{}, hidden_dim={}, intermediate_dim={}",
             DEFAULT_NUM_TOKENS, DEFAULT_NUM_EXPERTS, DEFAULT_TOP_K,
             DEFAULT_HIDDEN_DIM, DEFAULT_INTERMEDIATE_DIM);
    println!();
    println!("| Sub-stage | μs | % of total |");
    println!("|-----------|----|-----------|");

    let layer = make_moe_layer(
        DEFAULT_NUM_EXPERTS,
        DEFAULT_TOP_K,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_INTERMEDIATE_DIM,
    );
    let input = generate_input(DEFAULT_NUM_TOKENS, DEFAULT_HIDDEN_DIM);

    // Run multiple times and average
    const SAMPLES: usize = 10;
    let mut total_bd = ForwardBreakdown {
        gate_matmul_us: 0.0,
        topk_sorting_us: 0.0,
        softmax_binning_us: 0.0,
        expert_compute_us: 0.0,
        total_us: 0.0,
    };

    for _ in 0..SAMPLES {
        let (_, bd) = forward_with_breakdown(&layer, &input);
        total_bd.gate_matmul_us += bd.gate_matmul_us;
        total_bd.topk_sorting_us += bd.topk_sorting_us;
        total_bd.softmax_binning_us += bd.softmax_binning_us;
        total_bd.expert_compute_us += bd.expert_compute_us;
        total_bd.total_us += bd.total_us;
    }

    let avg = ForwardBreakdown {
        gate_matmul_us: total_bd.gate_matmul_us / SAMPLES as f64,
        topk_sorting_us: total_bd.topk_sorting_us / SAMPLES as f64,
        softmax_binning_us: total_bd.softmax_binning_us / SAMPLES as f64,
        expert_compute_us: total_bd.expert_compute_us / SAMPLES as f64,
        total_us: total_bd.total_us / SAMPLES as f64,
    };

    let stages: [(&str, f64); 5] = [
        ("Gate matmul", avg.gate_matmul_us),
        ("Top-k sorting", avg.topk_sorting_us),
        ("Softmax + binning", avg.softmax_binning_us),
        ("Expert compute (3× fwd_simd + SwiGLU)", avg.expert_compute_us),
        ("Overhead & combine", avg.total_us - avg.gate_matmul_us - avg.topk_sorting_us
            - avg.softmax_binning_us - avg.expert_compute_us),
    ];

    for (name, us) in &stages {
        let pct = if avg.total_us > 0.0 { (us / avg.total_us) * 100.0 } else { 0.0 };
        if *us >= 1000.0 {
            println!("| {} | {:.2} µs ({:.2} ms) | {:.1}% |", name, us, us / 1000.0, pct);
        } else {
            println!("| {} | {:.2} µs | {:.1}% |", name, us, pct);
        }
    }
    println!("| **Total** | **{:.2} µs** | **100%** |", avg.total_us);

    // Also run toy dim
    println!("\n### Toy Dim (hidden_dim=64, intermediate_dim=256)\n");
    println!("| Sub-stage | μs | % of total |");
    println!("|-----------|----|-----------|");

    let layer_toy = make_moe_layer(
        DEFAULT_NUM_EXPERTS.min(8),
        DEFAULT_TOP_K.min(4),
        TOY_HIDDEN_DIM,
        TOY_INTERMEDIATE_DIM,
    );
    let input_toy = generate_input(DEFAULT_NUM_TOKENS, TOY_HIDDEN_DIM);

    let mut tt_bd = ForwardBreakdown {
        gate_matmul_us: 0.0,
        topk_sorting_us: 0.0,
        softmax_binning_us: 0.0,
        expert_compute_us: 0.0,
        total_us: 0.0,
    };
    for _ in 0..SAMPLES {
        let (_, bd) = forward_with_breakdown(&layer_toy, &input_toy);
        tt_bd.gate_matmul_us += bd.gate_matmul_us;
        tt_bd.topk_sorting_us += bd.topk_sorting_us;
        tt_bd.softmax_binning_us += bd.softmax_binning_us;
        tt_bd.expert_compute_us += bd.expert_compute_us;
        tt_bd.total_us += bd.total_us;
    }

    let avg_toy = ForwardBreakdown {
        gate_matmul_us: tt_bd.gate_matmul_us / SAMPLES as f64,
        topk_sorting_us: tt_bd.topk_sorting_us / SAMPLES as f64,
        softmax_binning_us: tt_bd.softmax_binning_us / SAMPLES as f64,
        expert_compute_us: tt_bd.expert_compute_us / SAMPLES as f64,
        total_us: tt_bd.total_us / SAMPLES as f64,
    };

    let t_stages: [(&str, f64); 5] = [
        ("Gate matmul", avg_toy.gate_matmul_us),
        ("Top-k sorting", avg_toy.topk_sorting_us),
        ("Softmax + binning", avg_toy.softmax_binning_us),
        ("Expert compute (3× fwd_simd + SwiGLU)", avg_toy.expert_compute_us),
        ("Overhead & combine", avg_toy.total_us - avg_toy.gate_matmul_us
            - avg_toy.topk_sorting_us - avg_toy.softmax_binning_us - avg_toy.expert_compute_us),
    ];

    for (name, us) in &t_stages {
        let pct = if avg_toy.total_us > 0.0 { (us / avg_toy.total_us) * 100.0 } else { 0.0 };
        if *us >= 1000.0 {
            println!("| {} | {:.2} µs ({:.2} ms) | {:.1}% |", name, us, us / 1000.0, pct);
        } else {
            println!("| {} | {:.2} µs | {:.1}% |", name, us, pct);
        }
    }
    println!("| **Total** | **{:.2} µs** | **100%** |", avg_toy.total_us);

    println!();
}

fn print_comparison_tables() {
    println!("\n## Routing Overhead — Standard vs Oracle\n");
    println!("Config: {} tokens, {} experts, top-{}, hidden_dim={}",
             DEFAULT_NUM_TOKENS, DEFAULT_NUM_EXPERTS, DEFAULT_TOP_K,
             DEFAULT_HIDDEN_DIM);

    let config = MoEConfig {
        num_experts: DEFAULT_NUM_EXPERTS,
        top_k: DEFAULT_TOP_K,
        hidden_dim: DEFAULT_HIDDEN_DIM,
        intermediate_dim: DEFAULT_INTERMEDIATE_DIM,
    };
    let gate_weight = generate_gate_weight(DEFAULT_NUM_EXPERTS, DEFAULT_HIDDEN_DIM);
    let experts = generate_experts(&config);
    let layer = QMoELayer::new(config.clone(), gate_weight.clone(), experts);
    let input = generate_input(DEFAULT_NUM_TOKENS, DEFAULT_HIDDEN_DIM);

    const SAMPLES: usize = 5;

    // Standard forward
    let start = Instant::now();
    for _ in 0..SAMPLES {
        let _ = layer.forward(&input).unwrap();
    }
    let std_us = start.elapsed().as_secs_f64() / SAMPLES as f64 * 1e6;

    // Oracle forward
    let oracle_experts = generate_experts(&config);
    let start = Instant::now();
    for _ in 0..SAMPLES {
        let _ = forward_oracle(&config, &oracle_experts, &gate_weight, &input);
    }
    let oracle_us = start.elapsed().as_secs_f64() / SAMPLES as f64 * 1e6;

    // Naive forward
    let start = Instant::now();
    for _ in 0..SAMPLES {
        let _ = forward_naive(&layer, &input);
    }
    let naive_us = start.elapsed().as_secs_f64() / SAMPLES as f64 * 1e6;

    // FP16 forward
    let fp16_experts = generate_fp16_experts(DEFAULT_NUM_EXPERTS, DEFAULT_HIDDEN_DIM, DEFAULT_INTERMEDIATE_DIM);
    let start = Instant::now();
    for _ in 0..SAMPLES {
        let _ = forward_fp16(&gate_weight, &fp16_experts, &config, &input);
    }
    let fp16_us = start.elapsed().as_secs_f64() / SAMPLES as f64 * 1e6;

    println!();
    println!("| Variant | Latency (μs) | vs Standard |");
    println!("|---------|-------------|-------------|");
    print_row("Standard (binned routing)", std_us, std_us);
    print_row("Oracle (all experts, no routing)", oracle_us, std_us);
    print_row("Naive loop (no binning)", naive_us, std_us);
    print_row("FP16 matmul (unquantized)", fp16_us, std_us);
    println!();
}

fn print_row(name: &str, us: f64, baseline: f64) {
    let ratio = if baseline > 0.0 { us / baseline } else { 1.0 };
    if us >= 1000.0 {
        println!("| {} | {:.2} µs ({:.2} ms) | {:.2}× |", name, us, us / 1000.0, ratio);
    } else {
        println!("| {} | {:.2} µs | {:.2}× |", name, us, ratio);
    }
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group! {
    name = moe_layer;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
        .sample_size(50);
    targets =
        bench_forward_throughput,
        bench_forward_breakdown,
        bench_comparison_oracle,
        bench_comparison_naive,
        bench_comparison_fp16,
        bench_breakdown_varied,
        bench_breakdown_toy
}

fn main() {
    // Run criterion benchmarks
    moe_layer();

    // Print analysis tables
    print_breakdown_table();
    print_comparison_tables();
}
