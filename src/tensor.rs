use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;

/// Internal storage for packed tensor data — either mmap'd from disk or owned in memory.
enum PackedData {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

/// Represents a deeply compressed, sub-1-bit QMoE tensor using fixed-width packing.
/// By default, we might pack 3, 4, or 8 weights per byte depending on the
/// optimal scheme found during quantization.
pub struct PackedQMoETensor {
    /// The raw bit-packed weights (memory mapped or owned).
    data: PackedData,
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
            data: PackedData::Mmap(mmap),
            shape,
            scales,
        })
    }

    /// Construct a packed tensor from an already-loaded byte buffer.
    /// This is used when loading packed weights from a safetensors file.
    pub fn from_bytes(bytes: Vec<u8>, shape: (usize, usize), scales: Tensor) -> Self {
        Self {
            data: PackedData::Owned(bytes),
            shape,
            scales,
        }
    }

    fn raw_data(&self) -> &[u8] {
        match &self.data {
            PackedData::Mmap(m) => &m[..],
            PackedData::Owned(v) => v.as_slice(),
        }
    }

    /// Performs the fused decompression + Matrix-Vector multiplication.
    /// This is where the core SIMD unpacking logic will reside.
    pub fn forward_simd(&self, x: &Tensor) -> Result<Tensor> {
        let (out_features, in_features) = self.shape;
        let x_dims = x.dims();
        let batch_size: usize = x_dims.iter().product::<usize>() / in_features;
        if batch_size * in_features != x_dims.iter().product::<usize>() {
            anyhow::bail!("Input tensor size {} does not match batch_size * in_features = {} * {}", x_dims.iter().product::<usize>(), batch_size, in_features);
        }
        
        let x_flat = x.flatten_all()?.to_vec1::<f32>()?;
        let scales_vec = self.scales.flatten_all()?.to_vec1::<f32>()?;
        let bytes_per_row = in_features / 4;
        let raw = self.raw_data();
        
        let mut out_data = Vec::with_capacity(out_features * batch_size);
        
        for b in 0..batch_size {
            let x_vec = &x_flat[b * in_features..(b + 1) * in_features];
            for i in 0..out_features {
                let start = i * bytes_per_row;
                let end = start + bytes_per_row;
                let row_packed = &raw[start..end];
                let scale = scales_vec[i];
                let dot = crate::simd::fused_dequantize_and_dot(row_packed, x_vec, scale);
                out_data.push(dot);
            }
        }
        
        let mut out_shape: Vec<usize> = x_dims.to_vec();
        *out_shape.last_mut().unwrap() = out_features;
        let out_tensor = Tensor::from_vec(out_data, out_shape, &Device::Cpu)?;
        Ok(out_tensor)
    }

    /// Raw &[f32] path — bypasses Tensor construction overhead.
    /// Used for benchmarking the input-source overhead.
    pub fn forward_simd_raw(&self, x: &[f32]) -> Vec<f32> {
        let (out_features, in_features) = self.shape;
        let batch_size = x.len() / in_features;
        let scales_vec = self.scales.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bytes_per_row = in_features / 4;
        let raw = self.raw_data();

        let mut out_data = Vec::with_capacity(out_features * batch_size);

        for b in 0..batch_size {
            let x_vec = &x[b * in_features..(b + 1) * in_features];
            for i in 0..out_features {
                let start = i * bytes_per_row;
                let end = start + bytes_per_row;
                let row_packed = &raw[start..end];
                let scale = scales_vec[i];
                let dot = crate::simd::fused_dequantize_and_dot(row_packed, x_vec, scale);
                out_data.push(dot);
            }
        }

        out_data
    }
}
