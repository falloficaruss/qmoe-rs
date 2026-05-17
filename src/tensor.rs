use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;

/// Represents a deeply compressed, sub-1-bit QMoE tensor using fixed-width packing.
/// By default, we might pack 3, 4, or 8 weights per byte depending on the
/// optimal scheme found during quantization.
pub struct PackedQMoETensor {
    /// The raw bit-packed weights memory mapped directly from disk.
    /// This prevents loading massive trillion-parameter models entirely into RAM.
    pub data: memmap2::Mmap,
    /// Shape of the original FP16 tensor (Out_Features x In_Features)
    pub shape: (usize, usize),
    /// Group-wise scaling factors to restore magnitude after unpacking.
    pub scales: Tensor,
}

impl PackedQMoETensor {
    /// Memory maps a packed expert tensor from disk without loading it into RAM.
    pub fn mmap_from_file<P: AsRef<Path>>(
        path: P,
        shape: (usize, usize),
        scales: Tensor,
    ) -> Result<Self> {
        let file = File::open(&path)
            .with_context(|| format!("Failed to open packed tensor file at {:?}", path.as_ref()))?;
        
        // Safety: We assume the file on disk is not being modified while we map it.
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(Self {
            data: mmap,
            shape,
            scales,
        })
    }

    /// Performs the fused decompression + Matrix-Vector multiplication.
    /// This is where the core SIMD unpacking logic will reside.
    pub fn forward_simd(&self, x: &Tensor) -> Result<Tensor> {
        let (out_features, in_features) = self.shape;
        
        // We assume `x` is a 1D tensor for this basic GEMV
        let x_vec = x.flatten_all()?.to_vec1::<f32>()?;
        if x_vec.len() != in_features {
            anyhow::bail!("Input tensor size {} does not match in_features {}", x_vec.len(), in_features);
        }
        
        let scales_vec = self.scales.flatten_all()?.to_vec1::<f32>()?;
        
        let mut out_data = Vec::with_capacity(out_features);
        
        // The packed data is continuous. Each output feature row corresponds to a chunk of packed weights.
        // Assuming 4 weights per byte, each row is in_features / 4 bytes long.
        let bytes_per_row = in_features / 4;
        
        for i in 0..out_features {
            let start = i * bytes_per_row;
            let end = start + bytes_per_row;
            let row_packed = &self.data[start..end];
            let scale = scales_vec[i];
            
            // Call the highly optimized SIMD kernel
            let dot = crate::simd::fused_dequantize_and_dot(row_packed, &x_vec, scale);
            out_data.push(dot);
        }
        
        // Create the final output tensor
        let out_tensor = Tensor::from_vec(out_data, out_features, &Device::Cpu)?;
        Ok(out_tensor)
    }
}
