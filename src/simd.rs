use std::simd::prelude::*;

/// Unpacks a slice of 2-bit packed weights and computes the dot product with activations.
/// Each `u8` contains 4 weights.
/// This kernel fuses decompression and the GEMV inner product to stay within CPU L1 cache.
pub fn fused_dequantize_and_dot(
    packed_weights: &[u8],
    activations: &[f32],
    scale: f32,
) -> f32 {
    let mut sum = f32x4::splat(0.0);
    
    // Process 1 byte (4 weights) at a time using a 4-wide SIMD vector.
    // In the final highly-optimized version, we will unroll this to process 16 or 32 bytes (u8x16 / u8x32).
    let num_bytes = packed_weights.len().min(activations.len() / 4);
    
    for i in 0..num_bytes {
        let byte = packed_weights[i];
        
        // Extract 4 2-bit values from the byte
        // Pack Format: [w3(2b) | w2(2b) | w1(2b) | w0(2b)]
        let w0 = (byte & 0b0000_0011) as i32;
        let w1 = ((byte >> 2) & 0b0000_0011) as i32;
        let w2 = ((byte >> 4) & 0b0000_0011) as i32;
        let w3 = ((byte >> 6) & 0b0000_0011) as i32;
        
        // Map the 2-bit values [0, 1, 2] to [-1.0, 0.0, 1.0]
        let f0 = (w0 as f32) - 1.0;
        let f1 = (w1 as f32) - 1.0;
        let f2 = (w2 as f32) - 1.0;
        let f3 = (w3 as f32) - 1.0;
        
        let w_vec = f32x4::from_array([f0, f1, f2, f3]);
        
        // Load 4 contiguous fp32 activations into a SIMD register
        let a_idx = i * 4;
        let a_vec = f32x4::from_slice(&activations[a_idx..a_idx + 4]);
        
        // SIMD Fused Multiply-Add
        sum += w_vec * a_vec;
    }
    
    // Horizontal sum of the final accumulator, multiplied by the block's scaling factor
    sum.reduce_sum() * scale
}
