use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Embedding, VarBuilder};
use crate::moe::{QMoELayer, MoEConfig, PackedExpert};

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub moe: MoEConfig,
}

/// KV cache for a single attention layer.
/// Stores the full key and value tensors accumulated so far.
pub struct KVCache {
    pub k: Tensor,
    pub v: Tensor,
}

impl KVCache {
    pub fn new(k: Tensor, v: Tensor) -> Self {
        Self { k, v }
    }

    /// Appends new key/value projections (from a single new token) along the
    /// sequence dimension (dim=2).
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        self.k = Tensor::cat(&[&self.k, k], 2)?;
        self.v = Tensor::cat(&[&self.v, v], 2)?;
        Ok(())
    }
}

pub struct AttentionScaffold {
    pub q_proj: candle_nn::Linear,
    pub k_proj: candle_nn::Linear,
    pub v_proj: candle_nn::Linear,
    pub o_proj: candle_nn::Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl AttentionScaffold {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let q_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    /// Full prefill forward: processes an entire sequence and returns both the
    /// output and a fully-populated KV cache for incremental decoding.
    pub fn forward_prefill(&self, xs: &Tensor) -> Result<(Tensor, KVCache)> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let hidden_size = self.num_heads * self.head_dim;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (scores * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;

        let context = attn_weights.matmul(&v)?;
        let context = context.transpose(1, 2)?.reshape((b_sz, seq_len, hidden_size))?;

        let output = self.o_proj.forward(&context)?;
        let cache = KVCache::new(k, v);

        Ok((output, cache))
    }

    /// Incremental forward with an existing KV cache.  Only the *new* token(s)
    /// in `xs` are projected; the cache is appended and full attention is
    /// computed against the cached history.
    pub fn forward_with_cache(&self, xs: &Tensor, cache: &mut KVCache) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let hidden_size = self.num_heads * self.head_dim;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        // Append the new keys/values to the running cache.
        cache.append(&k, &v)?;

        // Full attention against cached history.
        let scores = q.matmul(&cache.k.transpose(2, 3)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (scores * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, 3)?;
        let context = attn_weights.matmul(&cache.v)?;
        let context = context.transpose(1, 2)?.reshape((b_sz, seq_len, hidden_size))?;

        Ok(self.o_proj.forward(&context)?)
    }
}

pub struct DecoderLayer {
    pub attn: AttentionScaffold,
    pub moe: QMoELayer,
}

impl DecoderLayer {
    pub fn new(vb: VarBuilder, config: &ModelConfig, experts: Vec<PackedExpert>) -> Result<Self> {
        let attn = AttentionScaffold::new(vb.pp("self_attn"), config.hidden_size, 16)?;

        let gate_weight = vb.get(
            (config.moe.num_experts, config.hidden_size),
            "moe.gate.weight",
        )?;

        let moe = QMoELayer::new(config.moe.clone(), gate_weight, experts);

        Ok(Self { attn, moe })
    }

    /// Prefill mode – processes the full token sequence and returns a KV cache.
    pub fn forward_prefill(&self, xs: &Tensor) -> Result<(Tensor, KVCache)> {
        let (attn_out, cache) = self.attn.forward_prefill(xs)?;
        let xs = (xs + attn_out)?;

        let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
        let flattened_xs = xs.reshape((b_sz * seq_len, hidden_dim))?;
        let moe_out = self.moe.forward(&flattened_xs)?;
        let moe_out = moe_out.reshape((b_sz, seq_len, hidden_dim))?;

        Ok(((xs + moe_out)?, cache))
    }

    /// Incremental decode mode – processes one new token using an existing KV cache.
    pub fn forward_with_cache(&self, xs: &Tensor, cache: &mut KVCache) -> Result<Tensor> {
        let attn_out = self.attn.forward_with_cache(xs, cache)?;
        let xs = (xs + attn_out)?;

        let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
        let flattened_xs = xs.reshape((b_sz * seq_len, hidden_dim))?;
        let moe_out = self.moe.forward(&flattened_xs)?;
        let moe_out = moe_out.reshape((b_sz, seq_len, hidden_dim))?;

        Ok((xs + moe_out)?)
    }
}

pub struct DeepSeekCoderV2 {
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: candle_nn::LayerNorm,
    pub lm_head: candle_nn::Linear,
}

impl DeepSeekCoderV2 {
    pub fn new(vb: VarBuilder, config: &ModelConfig, mut all_layers_experts: Vec<Vec<PackedExpert>>) -> Result<Self> {
        let embed = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed"))?;
        let norm = candle_nn::layer_norm(config.hidden_size, 1e-5, vb.pp("norm"))?;
        let lm_head = candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer_vb = vb.pp(format!("layers.{}", i));
            let layer_experts = all_layers_experts.remove(0);
            layers.push(DecoderLayer::new(layer_vb, config, layer_experts)?);
        }

        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    /// Full prefill forward: processes the entire prompt, returns logits for
    /// the last position and a vector of per-layer KV caches.
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

    /// Incremental decode: processes a single token using the existing KV caches.
    pub fn forward_next(&self, input_id: &Tensor, caches: &mut [KVCache]) -> Result<Tensor> {
        let mut xs = self.embed.forward(input_id)?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_with_cache(&xs, &mut caches[i])?;
        }
        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs.reshape((1, xs.dim(2)?))?)?;
        Ok(logits)
    }

    /// Convenience forward for backward compatibility (no KV cache returned).
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (logits, _) = self.forward_prefill(input_ids)?;
        Ok(logits)
    }
}
