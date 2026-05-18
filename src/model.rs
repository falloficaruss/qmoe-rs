use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Embedding, VarBuilder};
use crate::moe::{QMoELayer, MoEConfig, PackedExpert};

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub moe: MoEConfig,
}

pub struct AttentionScaffold {
    pub q_proj: candle_nn::Linear,
    pub k_proj: candle_nn::Linear,
    pub v_proj: candle_nn::Linear,
    pub o_proj: candle_nn::Linear,
    pub num_heads: usize,
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
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Simplified self-attention logic for inference scaffold
        let head_dim = q.dim(2)? / self.num_heads;
        let q = q.reshape((b_sz, seq_len, self.num_heads, head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b_sz, seq_len, self.num_heads, head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b_sz, seq_len, self.num_heads, head_dim))?.transpose(1, 2)?;

        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (scores * scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, candle_core::sym::Last)?;
        
        let context = attn_weights.matmul(&v)?;
        let context = context.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;
        
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
        
        // Gate weight loaded from standard weights (usually FP16/BF16)
        let gate_weight = vb.get(
            (config.moe.num_experts, config.hidden_size),
            "moe.gate.weight",
        )?;
        
        let moe = QMoELayer::new(config.moe.clone(), gate_weight, experts);
        
        Ok(Self { attn, moe })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Residual Attention
        let attn_out = self.attn.forward(xs)?;
        let xs = (xs + attn_out)?;

        // Residual MoE (flatten batch dimension to seq_len for routing)
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
    pub fn new(vb: VarBuilder, config: &ModelConfig, all_layers_experts: Vec<Vec<PackedExpert>>) -> Result<Self> {
        let embed = candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed"))?;
        let norm = candle_nn::layer_norm(config.hidden_size, 1e-5, vb.pp("norm"))?;
        let lm_head = candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
        
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer_vb = vb.pp(format!("layers.{}", i));
            let layer_experts = all_layers_experts[i].clone();
            layers.push(DecoderLayer::new(layer_vb, config, layer_experts)?);
        }

        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut xs = self.embed.forward(input_ids)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        let xs = self.norm.forward(&xs)?;
        // Output logits for the last token position
        let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
        let last_token = xs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&last_token.reshape((b_sz, hidden_dim))?)?;
        Ok(logits)
    }
}
