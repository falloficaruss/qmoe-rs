use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, VarBuilder};
use crate::moe::{MoEConfig, PackedExpert, QMoELayer};
use crate::tensor::PackedQMoETensor;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub moe: MoEConfig,
    pub use_shared_experts: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 102400,
            hidden_size: 2048,
            num_layers: 27,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            kv_lora_rank: 512,
            moe: MoEConfig {
                num_experts: 64,
                top_k: 6,
                hidden_dim: 2048,
                intermediate_dim: 1408,
            },
            use_shared_experts: true,
        }
    }
}

/// YaRN mscale computation.
fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// Precompute RoPE frequencies for a given sequence length.
pub fn precompute_rope_freqs(seq_len: usize, head_dim: usize, rope_mscale: f64, device: &Device) -> Result<Tensor> {
    let inv_freq: Vec<f32> = (0..head_dim / 2)
        .map(|i| 1.0 / 10000.0_f64.powf(2.0 * i as f64 / head_dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, head_dim / 2, device)?;
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, seq_len, device)?;
    let angles = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = (angles.cos()? * rope_mscale)?;
    let sin = (angles.sin()? * rope_mscale)?;
    Ok(Tensor::cat(&[&cos, &sin], 1)?)
}

/// Apply RoPE to a tensor of shape [b, heads (or 1), seq, head_dim].
/// Uses even-odd pairing (2i, 2i+1) matching the HuggingFace implementation.
pub fn apply_rope(x: &Tensor, cos_sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(3)?;
    let half = head_dim / 2;
    // cos/sin shape: [seq, half], expand to [1, 1, seq, half]
    let cos = cos_sin.narrow(1, 0, half)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = cos_sin.narrow(1, half, half)?.unsqueeze(0)?.unsqueeze(0)?;
    // Reshape to separate even-odd pairs: [b, h, seq, half, 2]
    let x_pairs = x.reshape((x.dim(0)?, x.dim(1)?, x.dim(2)?, half, 2))?;
    let x_even = x_pairs.narrow(4, 0, 1)?.squeeze(4)?;
    let x_odd = x_pairs.narrow(4, 1, 1)?.squeeze(4)?;
    let rotated = (x_even.broadcast_mul(&cos)? - x_odd.broadcast_mul(&sin)?)?;
    let passed = (x_odd.broadcast_mul(&cos)? + x_even.broadcast_mul(&sin)?)?;
    let result = Tensor::stack(&[&rotated, &passed], 4)?.reshape(x.shape())?;
    Ok(result)
}

/// KV cache for MLA: stores the decompressed K, V, and RoPE'd K accumulated so far.
#[derive(Clone)]
pub struct KVCache {
    pub k: Tensor,
    pub v: Tensor,
    pub k_rope: Tensor,
}

impl KVCache {
    pub fn new(k: Tensor, v: Tensor, k_rope: Tensor) -> Self {
        Self { k, v, k_rope }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor, k_rope: &Tensor) -> Result<()> {
        self.k = Tensor::cat(&[&self.k, k], 2)?.contiguous()?;
        self.v = Tensor::cat(&[&self.v, v], 2)?.contiguous()?;
        self.k_rope = Tensor::cat(&[&self.k_rope, k_rope], 2)?.contiguous()?;
        Ok(())
    }
}

/// DeepSeek-style Multi-head Latent Attention (MLA).
pub struct MlaAttention {
    pub q_proj: candle_nn::Linear,
    pub kv_a_proj_with_mqa: candle_nn::Linear,
    pub kv_a_layernorm: candle_nn::LayerNorm,
    pub kv_b_proj: candle_nn::Linear,
    pub o_proj: candle_nn::Linear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub rope_mscale: f64,
    pub softmax_scale: f64,
}

impl MlaAttention {
    pub fn new(vb: VarBuilder, config: &ModelConfig) -> Result<Self> {
        let q_proj = candle_nn::linear(
            config.hidden_size,
            config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
            vb.pp("q_proj"),
        )?;
        let kv_a_proj_with_mqa = candle_nn::linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            vb.pp("kv_a_proj_with_mqa"),
        )?;
        let kv_a_layernorm = candle_nn::layer_norm(config.kv_lora_rank, 1e-6, vb.pp("kv_a_layernorm"))?;
        let kv_b_proj = candle_nn::linear(
            config.kv_lora_rank,
            config.num_key_value_heads * (config.qk_nope_head_dim + config.v_head_dim),
            vb.pp("kv_b_proj"),
        )?;
        let o_proj = candle_nn::linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            qk_nope_head_dim: config.qk_nope_head_dim,
            qk_rope_head_dim: config.qk_rope_head_dim,
            v_head_dim: config.v_head_dim,
            kv_lora_rank: config.kv_lora_rank,
            rope_mscale: {
                let factor = 40.0;
                let mscale = 1.0;
                let mscale_all_dim = 0.707;
                yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
            },
            softmax_scale: {
                let q_head_dim = (config.qk_nope_head_dim + config.qk_rope_head_dim) as f64;
                let factor = 40.0;
                let mscale_all_dim = 0.707;
                let base_scale = q_head_dim.powf(-0.5);
                let mscale_yarn = yarn_get_mscale(factor, mscale_all_dim);
                base_scale * mscale_yarn * mscale_yarn
            },
        })
    }

    /// Compute K and V from the compressed KV latent.
    fn compute_kv(&self, kv_a_normed: &Tensor) -> Result<(Tensor, Tensor)> {
        let kv_b = self.kv_b_proj.forward(kv_a_normed)?;
        let (b_sz, seq_len, _) = kv_b.dims3()?;
        let kv_out = kv_b.reshape((
            b_sz,
            seq_len,
            self.num_kv_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ))?;
        let k = kv_out.narrow(3, 0, self.qk_nope_head_dim)?;
        let v = kv_out.narrow(3, self.qk_nope_head_dim, self.v_head_dim)?;
        Ok((k, v))
    }

    /// Full prefill: process entire sequence, return output + KV cache.
    pub fn forward_prefill(&self, xs: &Tensor) -> Result<(Tensor, KVCache)> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let device = xs.device();
        let rope_freqs = precompute_rope_freqs(seq_len, self.qk_rope_head_dim, self.rope_mscale, device)?;

        // Q projection: [b, seq, 3072]
        let q = self.q_proj.forward(xs)?;
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim))?;
        let q_nope = q.narrow(3, 0, self.qk_nope_head_dim)?.transpose(1, 2)?;
        let q_rope = q.narrow(3, self.qk_nope_head_dim, self.qk_rope_head_dim)?.transpose(1, 2)?;
        let q_rope = apply_rope(&q_rope, &rope_freqs)?;

        // KV compressed latent: [b, seq, 576]
        let kv_a = self.kv_a_proj_with_mqa.forward(xs)?;
        let kv_a_nope = kv_a.narrow(2, 0, self.kv_lora_rank)?;
        let kv_a_rope = kv_a.narrow(2, self.kv_lora_rank, self.qk_rope_head_dim)?;
        let kv_a_normed = self.kv_a_layernorm.forward(&kv_a_nope)?;

        // Decompress K and V
        let (k, v) = self.compute_kv(&kv_a_normed)?;
        // k, v: [b, seq, kv_heads, dim] → [b, kv_heads, seq, dim]
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply shared RoPE to K (broadcast to all KV heads)
        let k_rope = kv_a_rope.unsqueeze(2)?.expand((b_sz, seq_len, self.num_kv_heads, self.qk_rope_head_dim))?.transpose(1, 2)?;
        let k_rope = apply_rope(&k_rope, &rope_freqs)?;

        #[cfg(feature = "fused_attn")]
        let context = crate::attention_kernel::fused_prefill_attn(
            &q_nope, &q_rope, &k, &k_rope, &v,
            self.softmax_scale, true,
        )?;

        #[cfg(not(feature = "fused_attn"))]
        let context = {
            // Attention scores: Q_nope @ K_nope^T + Q_rope @ K_rope^T
            let scores = (q_nope.matmul(&k.transpose(2, 3)?)? + q_rope.matmul(&k_rope.transpose(2, 3)?)?)?;
            let scores = (scores * self.softmax_scale)?;

            // Causal mask
            let causal_mask = {
                let r = Tensor::arange(0u32, seq_len as u32, device)?;
                let row = r.unsqueeze(1)?.expand((seq_len, seq_len))?;
                let col = r.unsqueeze(0)?.expand((seq_len, seq_len))?;
                row.lt(&col)?
                    .to_dtype(DType::F32)?
                    .reshape((1, 1, seq_len, seq_len))?
                    .broadcast_as(scores.shape())?
            };
            let scores = (scores + (causal_mask * (-1e18f64))?)?;

            let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
            attn_weights.matmul(&v)?
        };
        let context = context.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.v_head_dim))?;
        let output = self.o_proj.forward(&context)?;
        let cache = KVCache::new(k.contiguous()?, v.contiguous()?, k_rope.contiguous()?);

        Ok((output, cache))
    }

    /// Incremental decode with existing KV cache.
    pub fn forward_with_cache(&self, xs: &Tensor, cache: &mut KVCache) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let device = xs.device();
        let total_seq_len = cache.k.dim(2)? + seq_len;
        let rope_freqs = precompute_rope_freqs(total_seq_len, self.qk_rope_head_dim, self.rope_mscale, device)?;

        // Q projection: transpose to [b, heads, seq, dim]
        let q = self.q_proj.forward(xs)?;
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim))?;
        let q_nope = q.narrow(3, 0, self.qk_nope_head_dim)?.transpose(1, 2)?;
        let q_rope = q.narrow(3, self.qk_nope_head_dim, self.qk_rope_head_dim)?.transpose(1, 2)?;
        let q_rope = apply_rope(&q_rope, &rope_freqs.narrow(0, (total_seq_len - seq_len) as usize, seq_len)?)?;

        // KV compressed latent
        let kv_a = self.kv_a_proj_with_mqa.forward(xs)?;
        let kv_a_nope = kv_a.narrow(2, 0, self.kv_lora_rank)?;
        let kv_a_rope = kv_a.narrow(2, self.kv_lora_rank, self.qk_rope_head_dim)?;
        let kv_a_normed = self.kv_a_layernorm.forward(&kv_a_nope)?;

        // Decompress K and V, transpose to [b, kv_heads, seq, dim]
        let (k_new, v_new) = self.compute_kv(&kv_a_normed)?;
        let k_new = k_new.transpose(1, 2)?;
        let v_new = v_new.transpose(1, 2)?;

        // Apply shared RoPE to K for the new tokens
        let k_rope_new = if seq_len > 0 {
            apply_rope(
                &kv_a_rope.unsqueeze(2)?.expand((b_sz, seq_len, self.num_kv_heads, self.qk_rope_head_dim))?.transpose(1, 2)?,
                &rope_freqs.narrow(0, (total_seq_len - seq_len) as usize, seq_len)?,
            )?
        } else {
            Tensor::zeros((b_sz, self.num_kv_heads, 0, self.qk_rope_head_dim), DType::F32, device)?
        };

        // Append to cache (includes k_rope)
        cache.append(&k_new, &v_new, &k_rope_new)?;

        // Full attention against cached history
        let k_rope_cached = cache.k_rope.narrow(2, 0, total_seq_len - seq_len)?;
        let k_rope_full = Tensor::cat(&[&k_rope_cached, &k_rope_new], 2)?.contiguous()?;

        let scores_nope = q_nope.matmul(&cache.k.transpose(2, 3)?)?;
        let scores_rope = q_rope.matmul(&k_rope_full.transpose(2, 3)?)?;
        let scores = (scores_nope + scores_rope)?;
        let scores = (scores * self.softmax_scale)?;

        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
        let context = attn_weights.matmul(&cache.v)?;
        let context = context.transpose(1, 2)?.reshape((b_sz, seq_len, self.num_heads * self.v_head_dim))?;

        Ok(self.o_proj.forward(&context)?)
    }
}

pub struct DecoderLayer {
    pub attn: MlaAttention,
    pub moe: QMoELayer,
    pub input_layernorm: candle_nn::LayerNorm,
    pub post_attention_layernorm: candle_nn::LayerNorm,
    pub shared_expert: Option<SharedExpert>,
}

pub struct SharedExpert {
    pub gate_proj: PackedQMoETensor,
    pub up_proj: PackedQMoETensor,
    pub down_proj: PackedQMoETensor,
}

impl SharedExpert {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
        let flattened = xs.reshape((b_sz * seq_len, hidden_dim))?;
        let gate_out = self.gate_proj.forward_simd(&flattened)?;
        let up_out = self.up_proj.forward_simd(&flattened)?;
        let gate_vec = gate_out.flatten_all()?.to_vec1::<f32>()?;
        let up_vec = up_out.flatten_all()?.to_vec1::<f32>()?;
        let mut activated = Vec::with_capacity(up_vec.len());
        for i in 0..up_vec.len() {
            let g = gate_vec[i];
            let u = up_vec[i];
            let swish = g * (1.0 / (1.0 + (-g).exp()));
            activated.push(swish * u);
        }
        let intermediate_size = up_vec.len() / (b_sz * seq_len);
        let activated_tensor = Tensor::from_vec(
            activated,
            (b_sz * seq_len, intermediate_size),
            xs.device(),
        )?;
        let down_out = self.down_proj.forward_simd(&activated_tensor)?;
        Ok(down_out.reshape((b_sz, seq_len, hidden_dim))?)
    }
}

impl DecoderLayer {
    pub fn new(
        vb: VarBuilder,
        config: &ModelConfig,
        experts: Vec<PackedExpert>,
        shared_expert: Option<SharedExpert>,
    ) -> Result<Self> {
        let attn = MlaAttention::new(vb.pp("self_attn"), config)?;

        let gate_weight = vb.get(
            (config.moe.num_experts, config.hidden_size),
            "moe.gate.weight",
        )?;

        let moe = QMoELayer::new(config.moe.clone(), gate_weight, experts);

        let input_layernorm = candle_nn::layer_norm(config.hidden_size, 1e-6, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::layer_norm(config.hidden_size, 1e-6, vb.pp("post_attention_layernorm"))?;

        Ok(Self { attn, moe, input_layernorm, post_attention_layernorm, shared_expert })
    }

    pub fn forward_prefill(&self, xs: &Tensor) -> Result<(Tensor, KVCache)> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let (attn_out, cache) = self.attn.forward_prefill(&xs)?;
        let xs = (residual + attn_out)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
        let flattened_xs = xs.reshape((b_sz * seq_len, hidden_dim))?;
        let mut moe_out = self.moe.forward(&flattened_xs)?;
        moe_out = moe_out.reshape((b_sz, seq_len, hidden_dim))?;

        // Add shared expert output if present
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(&xs)?;
            moe_out = (moe_out + shared_out)?;
        }

        Ok(((residual + moe_out)?, cache))
    }

    pub fn forward_with_cache(&self, xs: &Tensor, cache: &mut KVCache) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let attn_out = self.attn.forward_with_cache(&xs, cache)?;
        let xs = (residual + attn_out)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
        let flattened_xs = xs.reshape((b_sz * seq_len, hidden_dim))?;
        let mut moe_out = self.moe.forward(&flattened_xs)?;
        moe_out = moe_out.reshape((b_sz, seq_len, hidden_dim))?;

        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(&xs)?;
            moe_out = (moe_out + shared_out)?;
        }

        Ok((residual + moe_out)?)
    }
}

pub struct DeepSeekCoderV2 {
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: candle_nn::LayerNorm,
    pub lm_head: candle_nn::Linear,
}

impl DeepSeekCoderV2 {
    pub fn new(
        vb: VarBuilder,
        config: &ModelConfig,
        mut all_layers_experts: Vec<Vec<PackedExpert>>,
        mut all_layers_shared: Vec<Option<SharedExpert>>,
    ) -> Result<Self> {
        let embed = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed"))?;
        let norm = candle_nn::layer_norm(config.hidden_size, 1e-6, vb.pp("norm"))?;
        let lm_head = candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer_vb = vb.pp(format!("layers.{}", i));
            let layer_experts = all_layers_experts.remove(0);
            let layer_shared = all_layers_shared.remove(0);
            layers.push(DecoderLayer::new(layer_vb, config, layer_experts, layer_shared)?);
        }

        Ok(Self { embed, layers, norm, lm_head })
    }

    pub fn forward_prefill(&self, input_ids: &Tensor) -> Result<(Tensor, Vec<KVCache>)> {
        let mut xs = self.embed.forward(input_ids)?;
        let mut caches = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (new_xs, cache) = layer.forward_prefill(&xs)?;
            xs = new_xs;
            caches.push(cache);
        }

        let xs = self.norm.forward(&xs)?;
        let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
        let last_token = xs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&last_token.reshape((b_sz, hidden_dim))?)?;

        Ok((logits, caches))
    }

    pub fn forward_prefill_all_logits(&self, input_ids: &Tensor) -> Result<(Tensor, Vec<KVCache>)> {
        let mut xs = self.embed.forward(input_ids)?;
        let mut caches = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (new_xs, cache) = layer.forward_prefill(&xs)?;
            xs = new_xs;
            caches.push(cache);
        }

        let xs = self.norm.forward(&xs)?;
        let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
        let logits = self.lm_head.forward(&xs.reshape((b_sz * seq_len, hidden_dim))?)?;
        let logits = logits.reshape((b_sz, seq_len, logits.dim(1)?))?;

        Ok((logits, caches))
    }

    pub fn forward_next(&self, input_id: &Tensor, caches: &mut [KVCache]) -> Result<Tensor> {
        let mut xs = self.embed.forward(input_id)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_with_cache(&xs, &mut caches[i])?;
        }
        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs.reshape((1, xs.dim(2)?))?)?;
        Ok(logits)
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (logits, _) = self.forward_prefill(input_ids)?;
        Ok(logits)
    }
}
