use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{VarBuilder, VarMap};
use qmoe_engine::model::{MlaAttention, ModelConfig, KVCache, precompute_rope_freqs, apply_rope};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Model config constants matching DeepSeek-style MLA
// ---------------------------------------------------------------------------
const HIDDEN_SIZE: usize = 2048;
const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 16;
const QK_NOPE_HEAD_DIM: usize = 128;
const QK_ROPE_HEAD_DIM: usize = 64;
const V_HEAD_DIM: usize = 128;
const KV_LORA_RANK: usize = 512;

// ---------------------------------------------------------------------------
// Test data generators
// ---------------------------------------------------------------------------

fn make_attention_config() -> ModelConfig {
    let mut config = ModelConfig::default();
    config.hidden_size = HIDDEN_SIZE;
    config.num_attention_heads = NUM_HEADS;
    config.num_key_value_heads = NUM_KV_HEADS;
    config.qk_nope_head_dim = QK_NOPE_HEAD_DIM;
    config.qk_rope_head_dim = QK_ROPE_HEAD_DIM;
    config.v_head_dim = V_HEAD_DIM;
    config.kv_lora_rank = KV_LORA_RANK;
    config.num_layers = 1;
    config
}

fn create_dummy_attention(config: &ModelConfig) -> MlaAttention {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    MlaAttention::new(vb, config).unwrap()
}

fn create_input(batch_size: usize, seq_len: usize) -> Tensor {
    let data: Vec<f32> = (0..batch_size * seq_len * HIDDEN_SIZE)
        .map(|i| ((i as f32) * 0.1).sin() + 0.5)
        .collect();
    Tensor::from_vec(data, (batch_size, seq_len, HIDDEN_SIZE), &Device::Cpu).unwrap()
}

/// Prefill a dummy KV cache to a given length by running forward_prefill.
fn create_kv_cache(attn: &MlaAttention, target_len: usize) -> KVCache {
    if target_len == 0 {
        let dummy_k = Tensor::zeros((1, NUM_KV_HEADS, 0, QK_NOPE_HEAD_DIM), DType::F32, &Device::Cpu).unwrap();
        let dummy_v = Tensor::zeros((1, NUM_KV_HEADS, 0, V_HEAD_DIM), DType::F32, &Device::Cpu).unwrap();
        let dummy_k_rope = Tensor::zeros((1, NUM_KV_HEADS, 0, QK_ROPE_HEAD_DIM), DType::F32, &Device::Cpu).unwrap();
        return KVCache::new(dummy_k, dummy_v, dummy_k_rope);
    }
    // Prefill in chunks of at most 256 to keep memory reasonable
    let chunk_size = 256usize;
    let mut cache = None;
    let mut processed = 0usize;
    while processed < target_len {
        let this_chunk = (target_len - processed).min(chunk_size);
        let xs = create_input(1, this_chunk);
        let (_, new_cache) = attn.forward_prefill(&xs).unwrap();
        match cache {
            None => cache = Some(new_cache),
            Some(ref mut c) => {
                // Append: accumulate K, V, and k_rope across chunks.
                c.k = Tensor::cat(&[&c.k, &new_cache.k], 2).unwrap().contiguous().unwrap();
                c.v = Tensor::cat(&[&c.v, &new_cache.v], 2).unwrap().contiguous().unwrap();
                c.k_rope = Tensor::cat(&[&c.k_rope, &new_cache.k_rope], 2).unwrap().contiguous().unwrap();
            }
        }
        processed += this_chunk;
    }
    cache.unwrap()
}

// ---------------------------------------------------------------------------
// Breakdown struct
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

// ---------------------------------------------------------------------------
// Prefill with breakdown — replicates MlaAttention::forward_prefill with
// timing instrumentation inserted at each sub-stage boundary.
// ---------------------------------------------------------------------------

fn forward_prefill_with_breakdown(
    attn: &MlaAttention,
    xs: &Tensor,
) -> (Tensor, KVCache, AttentionBreakdown) {
    let t0 = Instant::now();
    let (b_sz, seq_len, _) = xs.dims3().unwrap();
    let device = xs.device();

    // 1. RoPE precompute
    let rope_freqs =
        precompute_rope_freqs(seq_len, attn.qk_rope_head_dim, attn.rope_mscale, device).unwrap();
    let t1 = Instant::now();

    // 2. Q projection + reshape + split + apply RoPE
    let q = attn.q_proj.forward(xs).unwrap();
    let q = q
        .reshape((
            b_sz,
            seq_len,
            attn.num_heads,
            attn.qk_nope_head_dim + attn.qk_rope_head_dim,
        ))
        .unwrap();
    let q_nope = q
        .narrow(3, 0, attn.qk_nope_head_dim)
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let q_rope = q
        .narrow(3, attn.qk_nope_head_dim, attn.qk_rope_head_dim)
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let q_rope = apply_rope(&q_rope, &rope_freqs).unwrap();
    let t2 = Instant::now();

    // 3. KV projection + norm
    let kv_a = attn.kv_a_proj_with_mqa.forward(xs).unwrap();
    let kv_a_nope = kv_a.narrow(2, 0, attn.kv_lora_rank).unwrap();
    let kv_a_rope = kv_a
        .narrow(2, attn.kv_lora_rank, attn.qk_rope_head_dim)
        .unwrap();
    let kv_a_normed = attn.kv_a_layernorm.forward(&kv_a_nope).unwrap();
    let t3 = Instant::now();

    // 4. KV decompression (kv_b_proj)
    let kv_b = attn.kv_b_proj.forward(&kv_a_normed).unwrap();
    let (b_sz_kv, seq_len_kv, _) = kv_b.dims3().unwrap();
    let kv_out = kv_b
        .reshape((
            b_sz_kv,
            seq_len_kv,
            attn.num_kv_heads,
            attn.qk_nope_head_dim + attn.v_head_dim,
        ))
        .unwrap();
    let k = kv_out.narrow(3, 0, attn.qk_nope_head_dim).unwrap();
    let v = kv_out
        .narrow(3, attn.qk_nope_head_dim, attn.v_head_dim)
        .unwrap();
    let k = k.transpose(1, 2).unwrap();
    let v = v.transpose(1, 2).unwrap();
    let t4 = Instant::now();

    // 5. K RoPE (shared RoPE broadcast to all KV heads)
    let k_rope = kv_a_rope
        .unsqueeze(2)
        .unwrap()
        .expand((
            b_sz,
            seq_len,
            attn.num_kv_heads,
            attn.qk_rope_head_dim,
        ))
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let k_rope = apply_rope(&k_rope, &rope_freqs).unwrap();
    let t5 = Instant::now();

    // 6. Attention scores: Q_nope @ K_nope^T + Q_rope @ K_rope^T
    let q_nope = q_nope.contiguous().unwrap();
    let q_rope = q_rope.contiguous().unwrap();
    let k_t = k.transpose(2, 3).unwrap().contiguous().unwrap();
    let k_rope_t = k_rope.transpose(2, 3).unwrap().contiguous().unwrap();
    let scores = (q_nope.matmul(&k_t).unwrap()
        + q_rope.matmul(&k_rope_t).unwrap())
    .unwrap();
    let scores = (scores * attn.softmax_scale).unwrap();

    // Causal mask
    let causal_mask = {
        let r = Tensor::arange(0u32, seq_len as u32, device).unwrap();
        let row = r
            .unsqueeze(1)
            .unwrap()
            .expand((seq_len, seq_len))
            .unwrap();
        let col = r
            .unsqueeze(0)
            .unwrap()
            .expand((seq_len, seq_len))
            .unwrap();
        row.lt(&col)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((1, 1, seq_len, seq_len))
            .unwrap()
            .broadcast_as(scores.shape())
            .unwrap()
    };
    let scores = (scores + (causal_mask * (-1e18f64)).unwrap()).unwrap();

    let attn_weights = candle_nn::ops::softmax(&scores, 3).unwrap();
    let t6 = Instant::now();

    // 7. Context + output projection
    let context = attn_weights.matmul(&v.contiguous().unwrap()).unwrap();
    let context = context
        .transpose(1, 2)
        .unwrap()
        .reshape((b_sz, seq_len, attn.num_heads * attn.v_head_dim))
        .unwrap();
    let output = attn.o_proj.forward(&context).unwrap();
    let t7 = Instant::now();

    // 8. KV cache creation (includes k_rope)
    let cache = KVCache::new(k, v, k_rope);
    let t8 = Instant::now();

    let breakdown = AttentionBreakdown {
        q_proj_us: (t2 - t1).as_secs_f64() * 1e6,
        kv_proj_norm_us: (t3 - t2).as_secs_f64() * 1e6,
        kv_decompression_us: (t4 - t3).as_secs_f64() * 1e6,
        rope_us: (t1 - t0).as_secs_f64() * 1e6 + (t5 - t4).as_secs_f64() * 1e6,
        attention_scores_us: (t6 - t5).as_secs_f64() * 1e6,
        context_output_us: (t7 - t6).as_secs_f64() * 1e6,
        kv_cache_ops_us: (t8 - t7).as_secs_f64() * 1e6,
        total_us: (t8 - t0).as_secs_f64() * 1e6,
    };

    (output, cache, breakdown)
}

// ---------------------------------------------------------------------------
// Decode with breakdown — replicates MlaAttention::forward_with_cache
// ---------------------------------------------------------------------------

fn forward_decode_with_breakdown(
    attn: &MlaAttention,
    xs: &Tensor,
    cache: &mut KVCache,
) -> (Tensor, AttentionBreakdown) {
    let t0 = Instant::now();
    let (b_sz, seq_len, _) = xs.dims3().unwrap();
    let device = xs.device();
    let total_seq_len = cache.k.dim(2).unwrap() + seq_len;

    // RoPE freqs for total length
    let rope_freqs =
        precompute_rope_freqs(total_seq_len, attn.qk_rope_head_dim, attn.rope_mscale, device)
            .unwrap();
    let rope_slice = rope_freqs
        .narrow(0, (total_seq_len - seq_len) as usize, seq_len)
        .unwrap();
    let t1 = Instant::now();

    // Q projection
    let q = attn.q_proj.forward(xs).unwrap();
    let q = q
        .reshape((
            b_sz,
            seq_len,
            attn.num_heads,
            attn.qk_nope_head_dim + attn.qk_rope_head_dim,
        ))
        .unwrap();
    let q_nope = q
        .narrow(3, 0, attn.qk_nope_head_dim)
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let q_rope = q
        .narrow(3, attn.qk_nope_head_dim, attn.qk_rope_head_dim)
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let q_rope = apply_rope(&q_rope, &rope_slice).unwrap();
    let t2 = Instant::now();

    // KV projection + norm
    let kv_a = attn.kv_a_proj_with_mqa.forward(xs).unwrap();
    let kv_a_nope = kv_a.narrow(2, 0, attn.kv_lora_rank).unwrap();
    let kv_a_rope = kv_a
        .narrow(2, attn.kv_lora_rank, attn.qk_rope_head_dim)
        .unwrap();
    let kv_a_normed = attn.kv_a_layernorm.forward(&kv_a_nope).unwrap();
    let t3 = Instant::now();

    // KV decompression
    let kv_b = attn.kv_b_proj.forward(&kv_a_normed).unwrap();
    let (b_sz_kv, seq_len_kv, _) = kv_b.dims3().unwrap();
    let kv_out = kv_b
        .reshape((
            b_sz_kv,
            seq_len_kv,
            attn.num_kv_heads,
            attn.qk_nope_head_dim + attn.v_head_dim,
        ))
        .unwrap();
    let k_new = kv_out.narrow(3, 0, attn.qk_nope_head_dim).unwrap();
    let v_new = kv_out
        .narrow(3, attn.qk_nope_head_dim, attn.v_head_dim)
        .unwrap();
    let k_new = k_new.transpose(1, 2).unwrap();
    let v_new = v_new.transpose(1, 2).unwrap();
    let t4 = Instant::now();

    // K RoPE for new tokens (computed before cache append for timing clarity)
    let k_rope_new = kv_a_rope
        .unsqueeze(2)
        .unwrap()
        .expand((b_sz, seq_len, attn.num_kv_heads, attn.qk_rope_head_dim))
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let k_rope_new = apply_rope(&k_rope_new, &rope_slice).unwrap();
    let t5 = Instant::now();

    // KV cache append (includes k_rope)
    cache.append(&k_new, &v_new, &k_rope_new).unwrap();
    let t6 = Instant::now();

    // Attention scores against full cache (using cached k and k_rope)
    let k_rope_cached = cache.k_rope.narrow(2, 0, total_seq_len - seq_len).unwrap();
    let k_rope_full = Tensor::cat(&[&k_rope_cached, &k_rope_new], 2).unwrap().contiguous().unwrap();

    let q_nope = q_nope.contiguous().unwrap();
    let q_rope = q_rope.contiguous().unwrap();
    let scores_nope = q_nope.matmul(&cache.k.transpose(2, 3).unwrap()).unwrap();
    let scores_rope = q_rope.matmul(&k_rope_full.transpose(2, 3).unwrap()).unwrap();
    let scores = (scores_nope + scores_rope).unwrap();
    let scores = (scores * attn.softmax_scale).unwrap();

    let attn_weights = candle_nn::ops::softmax(&scores, 3).unwrap();
    let t7 = Instant::now();

    // Context + output
    let context = attn_weights.matmul(&cache.v).unwrap();
    let context = context
        .transpose(1, 2)
        .unwrap()
        .reshape((b_sz, seq_len, attn.num_heads * attn.v_head_dim))
        .unwrap();
    let output = attn.o_proj.forward(&context).unwrap();
    let t8 = Instant::now();

    let breakdown = AttentionBreakdown {
        q_proj_us: (t2 - t1).as_secs_f64() * 1e6,
        kv_proj_norm_us: (t3 - t2).as_secs_f64() * 1e6,
        kv_decompression_us: (t4 - t3).as_secs_f64() * 1e6,
        rope_us: (t1 - t0).as_secs_f64() * 1e6 + (t5 - t4).as_secs_f64() * 1e6,
        attention_scores_us: (t7 - t6).as_secs_f64() * 1e6,
        context_output_us: (t8 - t7).as_secs_f64() * 1e6,
        kv_cache_ops_us: (t6 - t5).as_secs_f64() * 1e6,
        total_us: (t8 - t0).as_secs_f64() * 1e6,
    };

    (output, breakdown)
}

// ---------------------------------------------------------------------------
// Benchmark: Prefill throughput — sweep seq_len
// ---------------------------------------------------------------------------

fn bench_prefill_throughput(c: &mut Criterion) {
    let config = make_attention_config();
    let attn = create_dummy_attention(&config);
    let seq_lens: &[usize] = &[32, 64, 128, 256, 512, 1024, 2048];

    let mut group = c.benchmark_group("attention/prefill_throughput");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    for &seq_len in seq_lens {
        let input = create_input(1, seq_len);
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            &(&attn, &input),
            |b, (a, xs)| b.iter(|| a.forward_prefill(black_box(xs)).unwrap()),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Decode throughput — sweep kv_cache_len
// ---------------------------------------------------------------------------

fn bench_decode_throughput(c: &mut Criterion) {
    let config = make_attention_config();
    let attn = create_dummy_attention(&config);
    let cache_lens: &[usize] = &[0, 64, 256, 1024, 4096];

    let mut group = c.benchmark_group("attention/decode_throughput");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    for &cache_len in cache_lens {
        let cache_template = create_kv_cache(&attn, cache_len);
        let input = create_input(1, 1);
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("kv_cache_len", cache_len),
            &(&attn, &input),
            |b, (a, xs)| {
                b.iter(|| {
                    let mut c = cache_template.clone();
                    a.forward_with_cache(black_box(xs), &mut c).unwrap()
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Prefill breakdown for a fixed moderate seq_len
// ---------------------------------------------------------------------------

fn bench_prefill_breakdown(c: &mut Criterion) {
    let config = make_attention_config();
    let attn = create_dummy_attention(&config);
    let seq_len = 512;
    let input = create_input(1, seq_len);

    let mut group = c.benchmark_group("attention/prefill_breakdown");
    group.sample_size(30);

    group.bench_function("seq_len_512", |b| {
        b.iter(|| forward_prefill_with_breakdown(black_box(&attn), black_box(&input)));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Decode breakdown for a fixed cache length
// ---------------------------------------------------------------------------

fn bench_decode_breakdown(c: &mut Criterion) {
    let config = make_attention_config();
    let attn = create_dummy_attention(&config);
    let cache_len = 1024;
    let cache = create_kv_cache(&attn, cache_len);
    let input = create_input(1, 1);

    let mut group = c.benchmark_group("attention/decode_breakdown");
    group.sample_size(30);

    group.bench_function("kv_cache_1024", |b| {
        b.iter(|| {
            let mut c = cache.clone();
            forward_decode_with_breakdown(black_box(&attn), black_box(&input), &mut c)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Manual single-shot analysis tables
// ---------------------------------------------------------------------------

fn print_prefill_breakdown_table() {
    println!("\n## Attention Prefill Breakdown — Single-Shot Timing\n");
    println!(
        "Config: hidden_size={}, num_heads={}, kv_lora_rank={}",
        HIDDEN_SIZE, NUM_HEADS, KV_LORA_RANK
    );
    println!();
    println!("| Sub-stage | μs | % of total |");
    println!("|-----------|----|-----------|");

    let config = make_attention_config();
    let attn = create_dummy_attention(&config);
    let seq_lens: &[usize] = &[32, 128, 512, 2048];

    for &seq_len in seq_lens {
        let input = create_input(1, seq_len);

        const SAMPLES: usize = 10;
        let mut total_bd = AttentionBreakdown {
            q_proj_us: 0.0,
            kv_proj_norm_us: 0.0,
            kv_decompression_us: 0.0,
            rope_us: 0.0,
            attention_scores_us: 0.0,
            context_output_us: 0.0,
            kv_cache_ops_us: 0.0,
            total_us: 0.0,
        };

        for _ in 0..SAMPLES {
            let (_, _, bd) = forward_prefill_with_breakdown(&attn, &input);
            total_bd.q_proj_us += bd.q_proj_us;
            total_bd.kv_proj_norm_us += bd.kv_proj_norm_us;
            total_bd.kv_decompression_us += bd.kv_decompression_us;
            total_bd.rope_us += bd.rope_us;
            total_bd.attention_scores_us += bd.attention_scores_us;
            total_bd.context_output_us += bd.context_output_us;
            total_bd.kv_cache_ops_us += bd.kv_cache_ops_us;
            total_bd.total_us += bd.total_us;
        }

        let avg = AttentionBreakdown {
            q_proj_us: total_bd.q_proj_us / SAMPLES as f64,
            kv_proj_norm_us: total_bd.kv_proj_norm_us / SAMPLES as f64,
            kv_decompression_us: total_bd.kv_decompression_us / SAMPLES as f64,
            rope_us: total_bd.rope_us / SAMPLES as f64,
            attention_scores_us: total_bd.attention_scores_us / SAMPLES as f64,
            context_output_us: total_bd.context_output_us / SAMPLES as f64,
            kv_cache_ops_us: total_bd.kv_cache_ops_us / SAMPLES as f64,
            total_us: total_bd.total_us / SAMPLES as f64,
        };

        println!("\n### seq_len = {}\n", seq_len);
        println!("| Sub-stage | μs | % of total |");
        println!("|-----------|----|-----------|");

        let stages: [(&str, f64); 7] = [
            ("Q projection", avg.q_proj_us),
            ("KV projection + norm", avg.kv_proj_norm_us),
            ("KV decompression", avg.kv_decompression_us),
            ("RoPE (precompute + apply)", avg.rope_us),
            ("Attention scores", avg.attention_scores_us),
            ("Context + output proj", avg.context_output_us),
            ("KV cache ops", avg.kv_cache_ops_us),
        ];

        for (name, us) in &stages {
            let pct = if avg.total_us > 0.0 {
                (us / avg.total_us) * 100.0
            } else {
                0.0
            };
            if *us >= 1000.0 {
                println!(
                    "| {} | {:.2} µs ({:.2} ms) | {:.1}% |",
                    name,
                    us,
                    us / 1000.0,
                    pct
                );
            } else {
                println!("| {} | {:.2} µs | {:.1}% |", name, us, pct);
            }
        }
        println!("| **Total** | **{:.2} µs** | **100%** |", avg.total_us);
    }
    println!();
}

fn print_decode_breakdown_table() {
    println!("\n## Attention Decode Breakdown — Single-Shot Timing\n");
    println!(
        "Config: hidden_size={}, num_heads={}, kv_lora_rank={}",
        HIDDEN_SIZE, NUM_HEADS, KV_LORA_RANK
    );
    println!();

    let config = make_attention_config();
    let attn = create_dummy_attention(&config);
    let cache_lens: &[usize] = &[0, 64, 256, 1024, 4096];

    for &cache_len in cache_lens {
        let cache = create_kv_cache(&attn, cache_len);
        let input = create_input(1, 1);

        const SAMPLES: usize = 10;
        let mut total_bd = AttentionBreakdown {
            q_proj_us: 0.0,
            kv_proj_norm_us: 0.0,
            kv_decompression_us: 0.0,
            rope_us: 0.0,
            attention_scores_us: 0.0,
            context_output_us: 0.0,
            kv_cache_ops_us: 0.0,
            total_us: 0.0,
        };

        for _ in 0..SAMPLES {
            let mut c = cache.clone();
            let (_, bd) = forward_decode_with_breakdown(&attn, &input, &mut c);
            total_bd.q_proj_us += bd.q_proj_us;
            total_bd.kv_proj_norm_us += bd.kv_proj_norm_us;
            total_bd.kv_decompression_us += bd.kv_decompression_us;
            total_bd.rope_us += bd.rope_us;
            total_bd.attention_scores_us += bd.attention_scores_us;
            total_bd.context_output_us += bd.context_output_us;
            total_bd.kv_cache_ops_us += bd.kv_cache_ops_us;
            total_bd.total_us += bd.total_us;
        }

        let avg = AttentionBreakdown {
            q_proj_us: total_bd.q_proj_us / SAMPLES as f64,
            kv_proj_norm_us: total_bd.kv_proj_norm_us / SAMPLES as f64,
            kv_decompression_us: total_bd.kv_decompression_us / SAMPLES as f64,
            rope_us: total_bd.rope_us / SAMPLES as f64,
            attention_scores_us: total_bd.attention_scores_us / SAMPLES as f64,
            context_output_us: total_bd.context_output_us / SAMPLES as f64,
            kv_cache_ops_us: total_bd.kv_cache_ops_us / SAMPLES as f64,
            total_us: total_bd.total_us / SAMPLES as f64,
        };

        println!("\n### kv_cache_len = {}\n", cache_len);
        println!("| Sub-stage | μs | % of total |");
        println!("|-----------|----|-----------|");

        let stages: [(&str, f64); 7] = [
            ("Q projection", avg.q_proj_us),
            ("KV projection + norm", avg.kv_proj_norm_us),
            ("KV decompression", avg.kv_decompression_us),
            ("RoPE (precompute + apply)", avg.rope_us),
            ("Attention scores", avg.attention_scores_us),
            ("Context + output proj", avg.context_output_us),
            ("KV cache append", avg.kv_cache_ops_us),
        ];

        for (name, us) in &stages {
            let pct = if avg.total_us > 0.0 {
                (us / avg.total_us) * 100.0
            } else {
                0.0
            };
            if *us >= 1000.0 {
                println!(
                    "| {} | {:.2} µs ({:.2} ms) | {:.1}% |",
                    name,
                    us,
                    us / 1000.0,
                    pct
                );
            } else {
                println!("| {} | {:.2} µs | {:.1}% |", name, us, pct);
            }
        }
        println!("| **Total** | **{:.2} µs** | **100%** |", avg.total_us);
    }
    println!();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group! {
    name = attention;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
        .sample_size(50);
    targets =
        bench_prefill_throughput,
        bench_decode_throughput,
        bench_prefill_breakdown,
        bench_decode_breakdown
}

fn main() {
    // Run criterion benchmarks
    attention();

    // Print analysis tables
    print_prefill_breakdown_table();
    print_decode_breakdown_table();
}
