use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{VarBuilder, VarMap};
use qmoe_engine::model::{MlaAttention, ModelConfig, KVCache, SharedExpert, precompute_rope_freqs, apply_rope};
use qmoe_engine::moe::{MoEConfig, PackedExpert, QMoELayer};
use qmoe_engine::tensor::PackedQMoETensor;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Model config constants
// ---------------------------------------------------------------------------
const HIDDEN_SIZE: usize = 2048;
const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 16;
const QK_NOPE_HEAD_DIM: usize = 128;
const QK_ROPE_HEAD_DIM: usize = 64;
const V_HEAD_DIM: usize = 128;
const KV_LORA_RANK: usize = 512;

const NUM_EXPERTS: usize = 32;
const TOP_K: usize = 6;
const INTERMEDIATE_DIM: usize = 1408;

// ---------------------------------------------------------------------------
// Test data generators
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

fn generate_experts(config: &MoEConfig) -> Vec<PackedExpert> {
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

fn generate_shared_expert(config: &MoEConfig) -> SharedExpert {
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

    SharedExpert { gate_proj, up_proj, down_proj }
}

fn make_model_config() -> ModelConfig {
    ModelConfig {
        vocab_size: 102400,
        hidden_size: HIDDEN_SIZE,
        num_layers: 1,
        num_attention_heads: NUM_HEADS,
        num_key_value_heads: NUM_KV_HEADS,
        qk_nope_head_dim: QK_NOPE_HEAD_DIM,
        qk_rope_head_dim: QK_ROPE_HEAD_DIM,
        v_head_dim: V_HEAD_DIM,
        kv_lora_rank: KV_LORA_RANK,
        moe: MoEConfig {
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            hidden_dim: HIDDEN_SIZE,
            intermediate_dim: INTERMEDIATE_DIM,
        },
        use_shared_experts: true,
    }
}

fn create_input(batch_size: usize, seq_len: usize) -> Tensor {
    let data: Vec<f32> = (0..batch_size * seq_len * HIDDEN_SIZE)
        .map(|i| ((i as f32) * 0.1).sin() + 0.5)
        .collect();
    Tensor::from_vec(data, (batch_size, seq_len, HIDDEN_SIZE), &Device::Cpu).unwrap()
}

fn create_attention(config: &ModelConfig) -> MlaAttention {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    MlaAttention::new(vb, config).unwrap()
}

fn create_decoder_layer_components(
    config: &ModelConfig,
) -> (MlaAttention, QMoELayer, candle_nn::LayerNorm, candle_nn::LayerNorm, Option<SharedExpert>) {
    let attn = create_attention(config);

    let gate_weight = generate_gate_weight(config.moe.num_experts, config.hidden_size);
    let experts = generate_experts(&config.moe);
    let moe = QMoELayer::new(config.moe.clone(), gate_weight, experts);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let input_layernorm = candle_nn::layer_norm(config.hidden_size, 1e-6, vb.pp("input_layernorm")).unwrap();
    let post_attention_layernorm = candle_nn::layer_norm(config.hidden_size, 1e-6, vb.pp("post_attention_layernorm")).unwrap();

    let shared_expert = if config.use_shared_experts {
        Some(generate_shared_expert(&config.moe))
    } else {
        None
    };

    (attn, moe, input_layernorm, post_attention_layernorm, shared_expert)
}

// ---------------------------------------------------------------------------
// Breakdown structs
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct AttentionBreakdown {
    q_proj_us: f64,
    kv_proj_norm_us: f64,
    kv_decompression_us: f64,
    rope_us: f64,
    attention_scores_us: f64,
    context_output_us: f64,
    kv_cache_ops_us: f64,
    total_us: f64,
}

#[derive(Clone, Debug)]
struct MoEBreakdown {
    gate_matmul_us: f64,
    topk_sorting_us: f64,
    softmax_binning_us: f64,
    expert_compute_us: f64,
    total_us: f64,
}

#[derive(Clone, Debug)]
struct DecoderLayerBreakdown {
    input_layernorm_us: f64,
    attention: AttentionBreakdown,
    residual_add_attn_us: f64,
    post_attention_layernorm_us: f64,
    moe: MoEBreakdown,
    shared_expert_us: f64,
    residual_add_final_us: f64,
    total_us: f64,
}

// ---------------------------------------------------------------------------
// Instrumented forward — replicates DecoderLayer::forward_prefill with
// timing at every sub-stage boundary.
// ---------------------------------------------------------------------------

fn decoder_layer_forward_with_breakdown(
    attn: &MlaAttention,
    moe: &QMoELayer,
    input_layernorm: &candle_nn::LayerNorm,
    post_attention_layernorm: &candle_nn::LayerNorm,
    shared_expert: &Option<SharedExpert>,
    _config: &ModelConfig,
    xs: &Tensor,
) -> (Tensor, KVCache, DecoderLayerBreakdown) {
    let t0 = Instant::now();
    let (b_sz, seq_len, _) = xs.dims3().unwrap();
    let device = xs.device();

    // =====================================================================
    // 1. input_layernorm
    // =====================================================================
    let residual = xs;
    let normed = input_layernorm.forward(xs).unwrap();
    let t_input_ln = Instant::now();

    // =====================================================================
    // 2. MlaAttention forward (with sub-stage breakdown)
    // =====================================================================
    let rope_freqs = precompute_rope_freqs(seq_len, attn.qk_rope_head_dim, attn.rope_mscale, device).unwrap();
    let t_rope_pre = Instant::now();

    // Q projection
    let q = attn.q_proj.forward(&normed).unwrap();
    let q = q.reshape((b_sz, seq_len, attn.num_heads, attn.qk_nope_head_dim + attn.qk_rope_head_dim)).unwrap();
    let q_nope = q.narrow(3, 0, attn.qk_nope_head_dim).unwrap().transpose(1, 2).unwrap();
    let q_rope = q.narrow(3, attn.qk_nope_head_dim, attn.qk_rope_head_dim).unwrap().transpose(1, 2).unwrap();
    let q_rope = apply_rope(&q_rope, &rope_freqs).unwrap();
    let t_q_proj = Instant::now();

    // KV projection + norm
    let kv_a = attn.kv_a_proj_with_mqa.forward(&normed).unwrap();
    let kv_a_nope = kv_a.narrow(2, 0, attn.kv_lora_rank).unwrap();
    let kv_a_rope = kv_a.narrow(2, attn.kv_lora_rank, attn.qk_rope_head_dim).unwrap();
    let kv_a_normed = attn.kv_a_layernorm.forward(&kv_a_nope).unwrap();
    let t_kv_proj = Instant::now();

    // KV decompression
    let kv_b = attn.kv_b_proj.forward(&kv_a_normed).unwrap();
    let (_b_sz_kv, _seq_len_kv, _) = kv_b.dims3().unwrap();
    let kv_out = kv_b.reshape((b_sz, seq_len, attn.num_kv_heads, attn.qk_nope_head_dim + attn.v_head_dim)).unwrap();
    let k = kv_out.narrow(3, 0, attn.qk_nope_head_dim).unwrap();
    let v_computed = kv_out.narrow(3, attn.qk_nope_head_dim, attn.v_head_dim).unwrap();
    let k = k.transpose(1, 2).unwrap();
    let v_computed = v_computed.transpose(1, 2).unwrap();
    let t_kv_decomp = Instant::now();

    // K RoPE
    let k_rope = kv_a_rope
        .unsqueeze(2).unwrap()
        .expand((b_sz, seq_len, attn.num_kv_heads, attn.qk_rope_head_dim)).unwrap()
        .transpose(1, 2).unwrap();
    let k_rope = apply_rope(&k_rope, &rope_freqs).unwrap();
    let t_k_rope = Instant::now();

    // Attention scores
    let q_nope = q_nope.contiguous().unwrap();
    let q_rope = q_rope.contiguous().unwrap();
    let k_t = k.transpose(2, 3).unwrap().contiguous().unwrap();
    let k_rope_t = k_rope.transpose(2, 3).unwrap().contiguous().unwrap();
    let scores = (q_nope.matmul(&k_t).unwrap() + q_rope.matmul(&k_rope_t).unwrap()).unwrap();
    let scores = (scores * attn.softmax_scale).unwrap();

    // Causal mask
    let causal_mask = {
        let r = Tensor::arange(0u32, seq_len as u32, device).unwrap();
        let row = r.unsqueeze(1).unwrap().expand((seq_len, seq_len)).unwrap();
        let col = r.unsqueeze(0).unwrap().expand((seq_len, seq_len)).unwrap();
        row.lt(&col).unwrap()
            .to_dtype(DType::F32).unwrap()
            .reshape((1, 1, seq_len, seq_len)).unwrap()
            .broadcast_as(scores.shape()).unwrap()
    };
    let scores = (scores + (causal_mask * (-1e18f64)).unwrap()).unwrap();

    let attn_weights = candle_nn::ops::softmax(&scores, 3).unwrap();
    let t_attn_scores = Instant::now();

    // Context + output projection
    let context = attn_weights.matmul(&v_computed.contiguous().unwrap()).unwrap();
    let context = context.transpose(1, 2).unwrap()
        .reshape((b_sz, seq_len, attn.num_heads * attn.v_head_dim)).unwrap();
    let attn_out = attn.o_proj.forward(&context).unwrap();
    let t_attn_out = Instant::now();

    // KV cache
    let cache = KVCache::new(k.contiguous().unwrap(), v_computed.contiguous().unwrap(), k_rope.contiguous().unwrap());
    let t_cache = Instant::now();

    let attention_bd = AttentionBreakdown {
        q_proj_us: (t_q_proj - t_rope_pre).as_secs_f64() * 1e6,
        kv_proj_norm_us: (t_kv_proj - t_q_proj).as_secs_f64() * 1e6,
        kv_decompression_us: (t_kv_decomp - t_kv_proj).as_secs_f64() * 1e6,
        rope_us: (t_rope_pre - t_input_ln).as_secs_f64() * 1e6
               + (t_k_rope - t_kv_decomp).as_secs_f64() * 1e6,
        attention_scores_us: (t_attn_scores - t_k_rope).as_secs_f64() * 1e6,
        context_output_us: (t_attn_out - t_attn_scores).as_secs_f64() * 1e6,
        kv_cache_ops_us: (t_cache - t_attn_out).as_secs_f64() * 1e6,
        total_us: (t_cache - t_input_ln).as_secs_f64() * 1e6,
    };

    // =====================================================================
    // 3. Residual add (post-attention)
    // =====================================================================
    let xs = (residual + attn_out).unwrap();
    let t_residual_attn = Instant::now();

    // =====================================================================
    // 4. post_attention_layernorm
    // =====================================================================
    let residual = &xs;
    let normed = post_attention_layernorm.forward(&xs).unwrap();
    let t_post_ln = Instant::now();

    // =====================================================================
    // 5. MoE forward (with sub-stage breakdown)
    // =====================================================================
    let (_b_flat, _s_flat, hidden_dim) = normed.dims3().unwrap();
    let flattened_xs = normed.reshape((b_sz * seq_len, hidden_dim)).unwrap();
    let (flat_seq_len, _) = flattened_xs.dims2().unwrap();

    // 5a. Gate matmul
    let gate_logits = flattened_xs.matmul(&moe.gate_weight.t().unwrap()).unwrap();
    let t_gate = Instant::now();

    // 5b. Gate data to vec for CPU routing
    let gate_data = gate_logits.to_vec2::<f32>().unwrap();
    let t_gate_vec = Instant::now();

    // 5c. Top-k routing + softmax + binning
    let mut expert_bins: Vec<Vec<(usize, f32)>> = vec![Vec::new(); moe.config.num_experts];

    for t in 0..flat_seq_len {
        let logits = &gate_data[t];
        let mut indexed_logits: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let max_val = indexed_logits.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
        let all_sum_exp: f32 = indexed_logits.iter().map(|x| (x.1 - max_val).exp()).sum();
        let top_k_elements = &indexed_logits[0..moe.config.top_k];

        for &(expert_idx, score) in top_k_elements {
            let weight = (score - max_val).exp() / all_sum_exp;
            expert_bins[expert_idx].push((t, weight));
        }
    }
    let t_binning = Instant::now();

    // 5d. Expert compute + weighted combine
    let mut output_buf = vec![0.0f32; flat_seq_len * hidden_dim];
    let xs_data = flattened_xs.to_vec2::<f32>().unwrap();

    for (expert_idx, bin) in expert_bins.iter().enumerate() {
        if bin.is_empty() {
            continue;
        }
        let expert = &moe.experts[expert_idx];

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

            let activated_tensor = Tensor::from_vec(activated, (1, moe.config.intermediate_dim), device).unwrap();
            let down_out = expert.down_proj.forward_simd(&activated_tensor).unwrap();
            let down_out_vec = down_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let out_offset = token_idx * hidden_dim;
            for h in 0..hidden_dim {
                output_buf[out_offset + h] += down_out_vec[h] * weight;
            }
        }
    }
    let moe_out = Tensor::from_vec(output_buf, (flat_seq_len, hidden_dim), device).unwrap();
    let moe_out = moe_out.reshape((b_sz, seq_len, hidden_dim)).unwrap();
    let t_moe = Instant::now();

    let moe_bd = MoEBreakdown {
        gate_matmul_us: (t_gate_vec - t_gate).as_secs_f64() * 1e6,
        topk_sorting_us: (t_binning - t_gate_vec).as_secs_f64() * 1e6,
        softmax_binning_us: 0.0, // folded into topk_sorting above
        expert_compute_us: (t_moe - t_binning).as_secs_f64() * 1e6,
        total_us: (t_moe - t_gate).as_secs_f64() * 1e6,
    };

    // =====================================================================
    // 6. Shared expert
    // =====================================================================
    let mut layer_out = moe_out;
    if let Some(ref shared) = *shared_expert {
        let shared_out = shared.forward(&normed).unwrap();
        layer_out = (layer_out + shared_out).unwrap();
    }
    let t_shared = Instant::now();

    // =====================================================================
    // 7. Residual add (final)
    // =====================================================================
    let t_residual_final = Instant::now();

    let breakdown = DecoderLayerBreakdown {
        input_layernorm_us: (t_input_ln - t0).as_secs_f64() * 1e6,
        attention: attention_bd,
        residual_add_attn_us: (t_residual_attn - t_cache).as_secs_f64() * 1e6,
        post_attention_layernorm_us: (t_post_ln - t_residual_attn).as_secs_f64() * 1e6,
        moe: moe_bd,
        shared_expert_us: (t_shared - t_moe).as_secs_f64() * 1e6,
        residual_add_final_us: (t_residual_final - t_shared).as_secs_f64() * 1e6,
        total_us: (t_residual_final - t0).as_secs_f64() * 1e6,
    };

    ((residual + layer_out).unwrap(), cache, breakdown)
}

// ---------------------------------------------------------------------------
// Criterion benchmark: prefill throughput — sweep seq_len
// ---------------------------------------------------------------------------

fn bench_prefill_throughput(c: &mut Criterion) {
    let config = make_model_config();
    let (attn, moe, input_ln, post_ln, shared) = create_decoder_layer_components(&config);
    let seq_lens: &[usize] = &[32, 64, 128, 256, 512, 1024, 2048];

    let mut group = c.benchmark_group("decoder_layer/prefill_throughput");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    for &seq_len in seq_lens {
        let input = create_input(1, seq_len);
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &(&attn, &moe, &input_ln, &post_ln, &shared, &config, &input),
            |b, (a, m, il, pl, s, cfg, xs)| {
                b.iter(|| {
                    decoder_layer_forward_with_breakdown(
                        black_box(a),
                        black_box(m),
                        black_box(il),
                        black_box(pl),
                        black_box(s),
                        black_box(cfg),
                        black_box(xs),
                    )
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion benchmark: prefill breakdown for a fixed seq_len
// ---------------------------------------------------------------------------

fn bench_prefill_breakdown(c: &mut Criterion) {
    let config = make_model_config();
    let (attn, moe, input_ln, post_ln, shared) = create_decoder_layer_components(&config);
    let seq_len = 512;
    let input = create_input(1, seq_len);

    let mut group = c.benchmark_group("decoder_layer/prefill_breakdown");
    group.sample_size(30);

    group.bench_function("seq_len_512", |b| {
        b.iter(|| {
            decoder_layer_forward_with_breakdown(
                black_box(&attn),
                black_box(&moe),
                black_box(&input_ln),
                black_box(&post_ln),
                black_box(&shared),
                black_box(&config),
                black_box(&input),
            )
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Manual analysis: wall-time pie chart for various seq_lens
// ---------------------------------------------------------------------------

fn print_decoder_layer_pie_chart() {
    println!("\n## Decoder Layer Wall-Time Pie Chart — Single-Shot Timing\n");
    println!("Config: hidden_size={}, num_heads={}, num_experts={}, top_k={}, intermediate_dim={}",
             HIDDEN_SIZE, NUM_HEADS, NUM_EXPERTS, TOP_K, INTERMEDIATE_DIM);
    println!();

    let config = make_model_config();
    let (attn, moe, input_ln, post_ln, shared) = create_decoder_layer_components(&config);
    let seq_lens: &[usize] = &[32, 128, 512, 2048];

    for &seq_len in seq_lens {
        let input = create_input(1, seq_len);

        const SAMPLES: usize = 10;
        let mut total_bd = DecoderLayerBreakdown {
            input_layernorm_us: 0.0,
            attention: AttentionBreakdown {
                q_proj_us: 0.0, kv_proj_norm_us: 0.0, kv_decompression_us: 0.0,
                rope_us: 0.0, attention_scores_us: 0.0, context_output_us: 0.0,
                kv_cache_ops_us: 0.0, total_us: 0.0,
            },
            residual_add_attn_us: 0.0,
            post_attention_layernorm_us: 0.0,
            moe: MoEBreakdown {
                gate_matmul_us: 0.0, topk_sorting_us: 0.0,
                softmax_binning_us: 0.0, expert_compute_us: 0.0, total_us: 0.0,
            },
            shared_expert_us: 0.0,
            residual_add_final_us: 0.0,
            total_us: 0.0,
        };

        for _ in 0..SAMPLES {
            let (_, _, bd) = decoder_layer_forward_with_breakdown(
                &attn, &moe, &input_ln, &post_ln, &shared, &config, &input,
            );
            total_bd.input_layernorm_us += bd.input_layernorm_us;
            total_bd.attention.q_proj_us += bd.attention.q_proj_us;
            total_bd.attention.kv_proj_norm_us += bd.attention.kv_proj_norm_us;
            total_bd.attention.kv_decompression_us += bd.attention.kv_decompression_us;
            total_bd.attention.rope_us += bd.attention.rope_us;
            total_bd.attention.attention_scores_us += bd.attention.attention_scores_us;
            total_bd.attention.context_output_us += bd.attention.context_output_us;
            total_bd.attention.kv_cache_ops_us += bd.attention.kv_cache_ops_us;
            total_bd.attention.total_us += bd.attention.total_us;
            total_bd.residual_add_attn_us += bd.residual_add_attn_us;
            total_bd.post_attention_layernorm_us += bd.post_attention_layernorm_us;
            total_bd.moe.gate_matmul_us += bd.moe.gate_matmul_us;
            total_bd.moe.topk_sorting_us += bd.moe.topk_sorting_us;
            total_bd.moe.softmax_binning_us += bd.moe.softmax_binning_us;
            total_bd.moe.expert_compute_us += bd.moe.expert_compute_us;
            total_bd.moe.total_us += bd.moe.total_us;
            total_bd.shared_expert_us += bd.shared_expert_us;
            total_bd.residual_add_final_us += bd.residual_add_final_us;
            total_bd.total_us += bd.total_us;
        }

        let n = SAMPLES as f64;
        let avg = DecoderLayerBreakdown {
            input_layernorm_us: total_bd.input_layernorm_us / n,
            attention: AttentionBreakdown {
                q_proj_us: total_bd.attention.q_proj_us / n,
                kv_proj_norm_us: total_bd.attention.kv_proj_norm_us / n,
                kv_decompression_us: total_bd.attention.kv_decompression_us / n,
                rope_us: total_bd.attention.rope_us / n,
                attention_scores_us: total_bd.attention.attention_scores_us / n,
                context_output_us: total_bd.attention.context_output_us / n,
                kv_cache_ops_us: total_bd.attention.kv_cache_ops_us / n,
                total_us: total_bd.attention.total_us / n,
            },
            residual_add_attn_us: total_bd.residual_add_attn_us / n,
            post_attention_layernorm_us: total_bd.post_attention_layernorm_us / n,
            moe: MoEBreakdown {
                gate_matmul_us: total_bd.moe.gate_matmul_us / n,
                topk_sorting_us: total_bd.moe.topk_sorting_us / n,
                softmax_binning_us: total_bd.moe.softmax_binning_us / n,
                expert_compute_us: total_bd.moe.expert_compute_us / n,
                total_us: total_bd.moe.total_us / n,
            },
            shared_expert_us: total_bd.shared_expert_us / n,
            residual_add_final_us: total_bd.residual_add_final_us / n,
            total_us: total_bd.total_us / n,
        };

        print_pie_chart(seq_len, &avg);
    }
    println!();
}

fn print_pie_chart(seq_len: usize, bd: &DecoderLayerBreakdown) {
    println!("### seq_len = {}\n", seq_len);
    if bd.total_us >= 1000.0 {
        println!("**Total layer time: {:.2} µs ({:.2} ms)**\n", bd.total_us, bd.total_us / 1000.0);
    } else {
        println!("**Total layer time: {:.2} µs**\n", bd.total_us);
    }

    // Top-level pie chart
    println!("| Layer sub-stage | μs | % of total |");
    println!("|----------------|----|-----------|");

    let top_stages: [(&str, f64); 7] = [
        ("input_layernorm", bd.input_layernorm_us),
        ("MlaAttention (total)", bd.attention.total_us),
        ("residual add (post-attn)", bd.residual_add_attn_us),
        ("post_attention_layernorm", bd.post_attention_layernorm_us),
        ("MoE Layer (total)", bd.moe.total_us),
        ("Shared Expert", bd.shared_expert_us),
        ("residual add (final)", bd.residual_add_final_us),
    ];

    for (name, us) in &top_stages {
        let pct = if bd.total_us > 0.0 { (us / bd.total_us) * 100.0 } else { 0.0 };
        if *us >= 1000.0 {
            println!("| {} | {:.2} µs ({:.2} ms) | {:.1}% |", name, us, us / 1000.0, pct);
        } else {
            println!("| {} | {:.2} µs | {:.1}% |", name, us, pct);
        }
    }
    println!("| **Total** | **{:.2} µs** | **100%** |", bd.total_us);

    // Attention sub-breakdown
    println!();
    println!("| Attention sub-stage | μs | % of attn | % of total |");
    println!("|--------------------|----|----------|-----------|");

    let attn_stages: [(&str, f64); 7] = [
        ("Q projection", bd.attention.q_proj_us),
        ("KV projection + norm", bd.attention.kv_proj_norm_us),
        ("KV decompression", bd.attention.kv_decompression_us),
        ("RoPE (precompute + apply)", bd.attention.rope_us),
        ("Attention scores", bd.attention.attention_scores_us),
        ("Context + output proj", bd.attention.context_output_us),
        ("KV cache ops", bd.attention.kv_cache_ops_us),
    ];

    for (name, us) in &attn_stages {
        let pct_attn = if bd.attention.total_us > 0.0 { (us / bd.attention.total_us) * 100.0 } else { 0.0 };
        let pct_total = if bd.total_us > 0.0 { (us / bd.total_us) * 100.0 } else { 0.0 };
        if *us >= 1000.0 {
            println!("| {} | {:.2} µs ({:.2} ms) | {:.1}% | {:.1}% |", name, us, us / 1000.0, pct_attn, pct_total);
        } else {
            println!("| {} | {:.2} µs | {:.1}% | {:.1}% |", name, us, pct_attn, pct_total);
        }
    }
    println!("| **Attention total** | **{:.2} µs** | **100%** | **{:.1}%** |", bd.attention.total_us, (bd.attention.total_us / bd.total_us) * 100.0);

    // MoE sub-breakdown
    println!();
    println!("| MoE sub-stage | μs | % of MoE | % of total |");
    println!("|--------------|----|---------|-----------|");

    let moe_stages: [(&str, f64); 4] = [
        ("Gate matmul", bd.moe.gate_matmul_us),
        ("Top-k sorting + softmax", bd.moe.topk_sorting_us),
        ("Expert compute (3× fwd_simd + SwiGLU)", bd.moe.expert_compute_us),
        ("Overhead & combine", bd.moe.total_us - bd.moe.gate_matmul_us - bd.moe.topk_sorting_us - bd.moe.expert_compute_us),
    ];

    for (name, us) in &moe_stages {
        let pct_moe = if bd.moe.total_us > 0.0 { (us / bd.moe.total_us) * 100.0 } else { 0.0 };
        let pct_total = if bd.total_us > 0.0 { (us / bd.total_us) * 100.0 } else { 0.0 };
        if *us >= 1000.0 {
            println!("| {} | {:.2} µs ({:.2} ms) | {:.1}% | {:.1}% |", name, us, us / 1000.0, pct_moe, pct_total);
        } else {
            println!("| {} | {:.2} µs | {:.1}% | {:.1}% |", name, us, pct_moe, pct_total);
        }
    }
    println!("| **MoE total** | **{:.2} µs** | **100%** | **{:.1}%** |", bd.moe.total_us, (bd.moe.total_us / bd.total_us) * 100.0);

    // Shared expert detail
    println!();
    println!("| Shared Expert | μs | % of total |");
    println!("|--------------|----|-----------|");
    let se_pct = if bd.total_us > 0.0 { (bd.shared_expert_us / bd.total_us) * 100.0 } else { 0.0 };
    if bd.shared_expert_us >= 1000.0 {
        println!("| Shared expert forward | {:.2} µs ({:.2} ms) | {:.1}% |", bd.shared_expert_us, bd.shared_expert_us / 1000.0, se_pct);
    } else {
        println!("| Shared expert forward | {:.2} µs | {:.1}% |", bd.shared_expert_us, se_pct);
    }

    println!();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group! {
    name = decoder_layer;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
        .sample_size(50);
    targets =
        bench_prefill_throughput,
        bench_prefill_breakdown
}

fn main() {
    // Run criterion benchmarks
    decoder_layer();

    // Print manual analysis tables
    print_decoder_layer_pie_chart();
}
