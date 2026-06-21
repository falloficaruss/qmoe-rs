use candle_core::{DType, Device, Result, Tensor};

fn tile_size(seq_len: usize) -> usize {
    if seq_len > 2048 {
        16
    } else if seq_len > 1024 {
        32
    } else {
        64
    }
}

#[allow(unused)]
fn make_causal_mask(
    t_start: usize,
    t_end: usize,
    s: usize,
    h: usize,
    device: &Device,
) -> Result<Tensor> {
    let t_len = t_end - t_start;
    let row = Tensor::arange(t_start as u32, t_end as u32, device)?
        .unsqueeze(1)?.expand((t_len, s))?;
    let col = Tensor::arange(0u32, s as u32, device)?
        .unsqueeze(0)?.expand((t_len, s))?;
    let mask = row
        .lt(&col)?
        .to_dtype(DType::F32)?
        .reshape((1, 1, t_len, s))?;
    (mask * (-1e18f64))?.broadcast_as((1, h, t_len, s))
}

/// Eager prefill attention (reference version for testing)
pub fn eager_prefill_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
) -> Result<Tensor> {
    let (_, _, s, _) = q.dims4()?;
    let device = q.device();
    let k_t = k.transpose(2, 3)?;
    let mut scores = q.matmul(&k_t)?;
    scores = (scores * scale)?;

    if causal {
        let r = Tensor::arange(0u32, s as u32, device)?;
        let row = r.unsqueeze(1)?.expand((s, s))?;
        let col = r.unsqueeze(0)?.expand((s, s))?;
        let mask = row.lt(&col)?.to_dtype(DType::F32)?;
        scores = (scores + (mask * (-1e18f64))?)?;
    }

    let attn = candle_nn::ops::softmax(&scores, 3)?;
    attn.matmul(&v)
}

/// Fused prefill attention: compute `softmax(Q @ K^T) @ V`
/// in tiles along the sequence dimension.
///
/// Instead of materializing the full `[batch, heads, S, S]` attention scores matrix,
/// we produce `[batch, heads, tile, S]` per tile and immediately reduce with V.
pub fn fused_prefill_attn(
    q_nope: &Tensor,
    q_rope: &Tensor,
    k: &Tensor,
    k_rope: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
) -> Result<Tensor> {
    let (_, h, s, _) = q_nope.dims4()?;
    let device = q_nope.device();

    let k_t = k.transpose(2, 3)?.contiguous()?;
    let k_rope_t = k_rope.transpose(2, 3)?.contiguous()?;

    let tile = tile_size(s);
    let mut context_tiles: Vec<Tensor> = Vec::new();

    for t_start in (0..s).step_by(tile) {
        let t_end = (t_start + tile).min(s);
        let t_len = t_end - t_start;

        let qt_nope = q_nope.narrow(2, t_start, t_len)?.contiguous()?;
        let qt_rope = q_rope.narrow(2, t_start, t_len)?.contiguous()?;

        let scores_nope = qt_nope.matmul(&k_t)?;
        let scores_rope = qt_rope.matmul(&k_rope_t)?;
        let mut scores = (scores_nope + scores_rope)?;
        scores = (scores * scale)?;

        if causal {
            let mask = make_causal_mask(t_start, t_end, s, h, device)?
                .broadcast_as(scores.shape())?;
            scores = (scores + mask)?;
        }

        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let ctx = attn.matmul(&v.contiguous()?)?;

        context_tiles.push(ctx);
    }

    Tensor::cat(&context_tiles, 2)
}

/// Run the standard eager softmax attention (for verification).
pub fn reference_eager_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
) -> Result<Tensor> {
    let (_, _, s, _) = q.dims4()?;
    let device = q.device();
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let mut scores = q.matmul(&k_t)?;
    scores = (scores * scale)?;

    if causal {
        let r = Tensor::arange(0u32, s as u32, device)?;
        let row = r.unsqueeze(1)?.expand((s, s))?;
        let col = r.unsqueeze(0)?.expand((s, s))?;
        let mask = row.lt(&col)?.to_dtype(DType::F32)?;
        let mask = mask.reshape((1, 1, s, s))?.broadcast_as(scores.shape())?;
        scores = (scores + (mask * (-1e18f64))?)?;
    }

    let attn = candle_nn::ops::softmax(&scores, 3)?;
    attn.matmul(&v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(b: usize, h: usize, s: usize, d: usize, seed: f32) -> Result<Tensor> {
        let data: Vec<f32> = (0..b * h * s * d)
            .map(|i| ((i as f32) * seed).sin() + 0.5)
            .collect();
        Tensor::from_vec(data, (b, h, s, d), &Device::Cpu)
    }

    /// Test with nope_dim = rope_dim (i.e., fused and eager have same score dims)
    fn run_comparison(b: usize, h: usize, s: usize, d: usize, causal: bool) -> Result<f32> {
        let scale = (d as f64).powf(-0.5);
        let q = make_tensor(b, h, s, d, 0.1)?;
        let k = make_tensor(b, h, s, d, 0.2)?;
        let v = make_tensor(b, h, s, d, 0.3)?;

        // Use fused with q_nope=q_rope=half-of-q, k=k_tiled, k_rope=k for same dims
        let half_d = d / 2;
        let q_nope = q.narrow(3, 0, half_d)?;
        let q_rope = q.narrow(3, half_d, half_d)?;
        let k_nope = k.narrow(3, 0, half_d)?;
        let k_rope = k.narrow(3, half_d, half_d)?;

        let eager = reference_eager_attn(&q, &k, &v, scale, causal)?;
        let fused = fused_prefill_attn(&q_nope, &q_rope, &k_nope, &k_rope, &v, scale, causal)?;

        let e = eager.flatten_all()?.to_vec1::<f32>()?;
        let f = fused.flatten_all()?.to_vec1::<f32>()?;
        Ok(e.iter().zip(f.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max))
    }

    #[test]
    fn test_no_causal_small() -> Result<()> {
        let diff = run_comparison(1, 4, 64, 32, false)?;
        assert!(diff < 1e-3, "no_causal_small diff={}", diff);
        Ok(())
    }

    #[test]
    fn test_causal_small() -> Result<()> {
        let diff = run_comparison(1, 4, 64, 32, true)?;
        assert!(diff < 1e-3, "causal_small diff={}", diff);
        Ok(())
    }

    #[test]
    fn test_causal_medium() -> Result<()> {
        // s=256 triggers 4 tiles of 64 each
        let diff = run_comparison(1, 2, 256, 32, true)?;
        assert!(diff < 1e-3, "causal_medium diff={}", diff);
        Ok(())
    }

    #[test]
    fn test_causal_large() -> Result<()> {
        // s=2048 triggers tile size 32
        let diff = run_comparison(1, 2, 256, 32, true)?;
        assert!(diff < 1e-3, "causal_large diff={}", diff);
        Ok(())
    }

}
