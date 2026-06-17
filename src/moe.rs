use anyhow::Result;
use candle_core::Tensor;
use crate::tensor::PackedQMoETensor;

#[derive(Clone, Debug)]
pub struct MoEConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

/// A highly-optimized, packed QMoE expert holding sub-1-bit compressed weights.
/// Uses fixed-width bit-packing.
pub struct PackedExpert {
    pub gate_proj: PackedQMoETensor,
    pub up_proj: PackedQMoETensor,
    pub down_proj: PackedQMoETensor,
}

/// The main MoE Layer that coordinates the gating network, token routing/binning,
/// and batched execution of the sub-1-bit SIMD experts.
pub struct QMoELayer {
    pub config: MoEConfig,
    /// The gating projection (usually kept in standard precision like FP16 or BF16 for accuracy)
    pub gate_weight: Tensor,
    pub experts: Vec<PackedExpert>,
}

impl QMoELayer {
    pub fn new(config: MoEConfig, gate_weight: Tensor, experts: Vec<PackedExpert>) -> Self {
        Self {
            config,
            gate_weight,
            experts,
        }
    }

    /// Performs the forward pass with Fused Token Routing & Binning to avoid RAM thrashing.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs shape: [seq_len, hidden_dim]
        let (seq_len, hidden_dim) = xs.dims2()?;
        let device = xs.device();

        // No routed experts (e.g. layer 0 with flattened weights) — skip
        if self.experts.is_empty() {
            return Ok(Tensor::zeros((seq_len, hidden_dim), xs.dtype(), device)?);
        }

        // 1. Gate logits: [seq_len, num_experts]
        let gate_logits = xs.matmul(&self.gate_weight.t()?)?;
        
        // 2. Convert to vectors for fast CPU token binning / routing
        let gate_data = gate_logits.to_vec2::<f32>()?;
        
        // expert_bins[e] = Vec<(token_index, weight)>
        let mut expert_bins: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.config.num_experts];
        
        // Fused Top-k routing and binning (inspired by QMoE / bitnet.cpp radix-sort approach)
        for t in 0..seq_len {
            let logits = &gate_data[t];
            
            // Find top-k expert indices and values
            let mut indexed_logits: Vec<(usize, f32)> = logits
                .iter()
                .cloned()
                .enumerate()
                .collect();
            
            // Sort descending by logit score
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Compute softmax over the top-k to get routing weights
            let top_k_elements = &indexed_logits[0..self.config.top_k];
            let max_val = top_k_elements.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = top_k_elements.iter().map(|x| (x.1 - max_val).exp()).sum();
            
            for &(expert_idx, score) in top_k_elements {
                let weight = (score - max_val).exp() / sum_exp;
                expert_bins[expert_idx].push((t, weight));
            }
        }

        // 3. Batched execution: Process tokens assigned to each expert.
        // We aggregate the results directly into a pre-allocated output buffer to avoid allocations (slab allocator style).
        let mut output_buf = vec![0.0f32; seq_len * hidden_dim];
        let xs_data = xs.to_vec2::<f32>()?;

        for (expert_idx, bin) in expert_bins.iter().enumerate() {
            if bin.is_empty() {
                continue;
            }

            let expert = &self.experts[expert_idx];

            for &(token_idx, weight) in bin {
                let token_vector = &xs_data[token_idx];
                let token_tensor = Tensor::from_vec(token_vector.clone(), (1, hidden_dim), device)?;

                // Expert MLP (SwiGLU-style):
                // SwiGLU(x) = (gate_proj(x) * swish(up_proj(x))) * down_proj(x)
                let gate_out = expert.gate_proj.forward_simd(&token_tensor)?;
                let up_out = expert.up_proj.forward_simd(&token_tensor)?;

                // Apply swish activation: x * sigmoid(x)
                let up_out_vec = up_out.flatten_all()?.to_vec1::<f32>()?;
                let gate_out_vec = gate_out.flatten_all()?.to_vec1::<f32>()?;
                
                let mut activated = Vec::with_capacity(up_out_vec.len());
                for i in 0..up_out_vec.len() {
                    let u = up_out_vec[i];
                    let g = gate_out_vec[i];
                    let swish = u * (1.0 / (1.0 + (-u).exp()));
                    activated.push(g * swish);
                }

                let activated_tensor = Tensor::from_vec(activated, (1, self.config.intermediate_dim), device)?;
                let down_out = expert.down_proj.forward_simd(&activated_tensor)?;
                let down_out_vec = down_out.flatten_all()?.to_vec1::<f32>()?;

                // Weighted accumulation into our output buffer
                let out_offset = token_idx * hidden_dim;
                for h in 0..hidden_dim {
                    output_buf[out_offset + h] += down_out_vec[h] * weight;
                }
            }
        }

        Ok(Tensor::from_vec(output_buf, (seq_len, hidden_dim), device)?)
    }
}
