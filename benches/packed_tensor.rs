use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use candle_core::{Device, Tensor, DType};
use qmoe_engine::tensor::PackedQMoETensor;
use std::io::Write;

// ---------------------------------------------------------------------------
// Deterministic test-data generators
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

fn generate_input_tensor(in_features: usize, batch_size: usize) -> Tensor {
    let data: Vec<f32> = (0..in_features * batch_size)
        .map(|i| ((i as f32) * 0.1).sin() + 0.5)
        .collect();
    Tensor::from_vec(data, (batch_size, in_features), &Device::Cpu).unwrap()
}

fn generate_input_vec(in_features: usize, batch_size: usize) -> Vec<f32> {
    (0..in_features * batch_size)
        .map(|i| ((i as f32) * 0.1).sin() + 0.5)
        .collect()
}

/// Generate a column-vector input for FP16 matmul (shape [in_features, 1]).
fn generate_fp16_activation(in_features: usize) -> Tensor {
    let data: Vec<f32> = (0..in_features)
        .map(|i| ((i as f32) * 0.1).sin() + 0.5)
        .collect();
    Tensor::from_vec(data, (in_features, 1), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
}

fn generate_fp16_weight(out_features: usize, in_features: usize) -> Tensor {
    let data: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i as f32) * 0.07).cos())
        .collect();
    Tensor::from_vec(data, (out_features, in_features), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap()
}

// ---------------------------------------------------------------------------
// Temp-file helper for mmap benchmarks
// ---------------------------------------------------------------------------

fn create_temp_mmap_tensor(
    data: &[u8],
    shape: (usize, usize),
    scales: &Tensor,
) -> (PackedQMoETensor, std::path::PathBuf) {
    let mut path = std::env::temp_dir();
    path.push(format!("qmoe_bench_{:020x}.bin", rand::random::<u64>()));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(data).unwrap();
    f.sync_all().unwrap();
    let tensor = PackedQMoETensor::mmap_from_file(&path, shape, scales.clone()).unwrap();
    (tensor, path)
}

// ---------------------------------------------------------------------------
// Benchmark group 1: Owned storage, Tensor input
// Sweeps: out_features ∈ {1, 64, 256, 1408, 2048}
//         in_features ∈ {1408, 2048}
// ---------------------------------------------------------------------------

fn bench_forward_owned(c: &mut Criterion) {
    let in_features_list = [1408, 2048];
    let out_features_list = [1, 64, 256, 1408, 2048];

    let mut group = c.benchmark_group("packed_tensor/forward_owned");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    for &in_features in &in_features_list {
        for &out_features in &out_features_list {
            let (packed_bytes, scales_vec) = generate_packed_data(out_features, in_features);
            let scales = Tensor::from_vec(scales_vec, out_features, &Device::Cpu).unwrap();
            let tensor =
                PackedQMoETensor::from_bytes(packed_bytes, (out_features, in_features), scales);
            let input = generate_input_tensor(in_features, 1);

            let param_str = format!("{}x{}", out_features, in_features);
            group.throughput(Throughput::Bytes(
                (out_features * in_features / 4) as u64,
            ));

            group.bench_with_input(
                BenchmarkId::new("owned", &param_str),
                &(&tensor, &input),
                |b, (t, i)| b.iter(|| t.forward_simd(black_box(i)).unwrap()),
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 2: mmap storage, Tensor input
// Sweeps: same dimensions as owned
// ---------------------------------------------------------------------------

fn bench_forward_mmap(c: &mut Criterion) {
    let in_features_list = [1408, 2048];
    let out_features_list = [1, 64, 256, 1408, 2048];

    let mut handles = Vec::new();
    let mut tensors = Vec::new();
    let mut inputs = Vec::new();
    let mut labels = Vec::new();

    for &in_features in &in_features_list {
        for &out_features in &out_features_list {
            let (packed_bytes, scales_vec) = generate_packed_data(out_features, in_features);
            let scales = Tensor::from_vec(scales_vec, out_features, &Device::Cpu).unwrap();
            let (tensor, path) =
                create_temp_mmap_tensor(&packed_bytes, (out_features, in_features), &scales);
            let input = generate_input_tensor(in_features, 1);
            handles.push(path);
            tensors.push(tensor);
            inputs.push(input);
            labels.push(format!("{}x{}", out_features, in_features));
        }
    }

    let mut group = c.benchmark_group("packed_tensor/forward_mmap");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    for ((tensor, input), param_str) in tensors.iter().zip(&inputs).zip(&labels) {
        let (out_features, in_features) = tensor.shape;
        group.throughput(Throughput::Bytes((out_features * in_features / 4) as u64));

        group.bench_with_input(
            BenchmarkId::new("mmap", param_str),
            &(tensor, input),
            |b, (t, i)| b.iter(|| t.forward_simd(black_box(i)).unwrap()),
        );
    }

    group.finish();

    for p in handles {
        let _ = std::fs::remove_file(&p);
    }
}

// ---------------------------------------------------------------------------
// Benchmark group 3: Input source comparison — Vec<f32> vs Tensor
// Measures the Tensor → vec1 copy overhead incurred by forward_simd.
// ---------------------------------------------------------------------------

fn bench_input_source(c: &mut Criterion) {
    let in_features_list = [1408, 2048];
    let out_features_list = [64, 256, 1408];

    let mut group = c.benchmark_group("packed_tensor/input_source");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    for &in_features in &in_features_list {
        for &out_features in &out_features_list {
            let (packed_bytes, scales_vec) = generate_packed_data(out_features, in_features);
            let scales = Tensor::from_vec(scales_vec, out_features, &Device::Cpu).unwrap();
            let tensor =
                PackedQMoETensor::from_bytes(packed_bytes, (out_features, in_features), scales);
            let input_tensor = generate_input_tensor(in_features, 1);
            let input_vec = generate_input_vec(in_features, 1);

            let param_str = format!("{}x{}", out_features, in_features);
            group.throughput(Throughput::Bytes(
                (out_features * in_features / 4) as u64,
            ));

            // Tensor input (standard path — incurs flatten + to_vec1 overhead)
            group.bench_with_input(
                BenchmarkId::new("tensor_input", &param_str),
                &(&tensor, &input_tensor),
                |b, (t, i)| b.iter(|| t.forward_simd(black_box(i)).unwrap()),
            );

            // Raw &[f32] input (bypasses Tensor conversion)
            group.bench_with_input(
                BenchmarkId::new("raw_slice_input", &param_str),
                &(&tensor, &input_vec),
                |b, (t, v)| b.iter(|| t.forward_simd_raw(black_box(v))),
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 4: vs FP16 Candle linear()
// Compares packed 2-bit forward against FP16 matrix-vector multiply.
// ---------------------------------------------------------------------------

fn bench_vs_fp16(c: &mut Criterion) {
    let in_features_list = [1408, 2048];
    let out_features_list = [64, 256, 1408, 2048];

    let mut group = c.benchmark_group("packed_tensor/vs_fp16");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    for &in_features in &in_features_list {
        for &out_features in &out_features_list {
            let param_str = format!("{}x{}", out_features, in_features);
            group.throughput(Throughput::Bytes(
                (out_features * in_features / 4) as u64,
            ));

            // Packed 2-bit forward (Owned)
            let (packed_bytes, scales_vec) = generate_packed_data(out_features, in_features);
            let scales = Tensor::from_vec(scales_vec, out_features, &Device::Cpu).unwrap();
            let tensor =
                PackedQMoETensor::from_bytes(packed_bytes, (out_features, in_features), scales);
            let input = generate_input_tensor(in_features, 1);

            group.bench_with_input(
                BenchmarkId::new("packed_2bit", &param_str),
                &(&tensor, &input),
                |b, (t, i)| b.iter(|| t.forward_simd(black_box(i)).unwrap()),
            );

            // FP16 matmul (weight: [out, in], activation: [in, 1])
            let weight = generate_fp16_weight(out_features, in_features);
            let act = generate_fp16_activation(in_features);

            group.bench_with_input(
                BenchmarkId::new("fp16_matmul", &param_str),
                &(&weight, &act),
                |b, (w, a)| b.iter(|| w.matmul(black_box(a)).unwrap()),
            );
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 5: Batch-size scaling (bonus — not in spec but useful)
// Shows how forward_simd scales with batch_size for a fixed expert shape.
// ---------------------------------------------------------------------------

fn bench_batch_scaling(c: &mut Criterion) {
    let dim = 1408usize;
    let batch_sizes = [1, 2, 4, 8];

    let mut group = c.benchmark_group("packed_tensor/batch_scaling");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_secs(2));
    group.sample_size(50);

    let (packed_bytes, scales_vec) = generate_packed_data(dim, dim);
    let scales = Tensor::from_vec(scales_vec, dim, &Device::Cpu).unwrap();
    let tensor = PackedQMoETensor::from_bytes(packed_bytes, (dim, dim), scales);

    for &batch_size in &batch_sizes {
        let input = generate_input_tensor(dim, batch_size);
        group.throughput(Throughput::Elements((batch_size * dim) as u64));

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &(&tensor, &input),
            |b, (t, i)| b.iter(|| t.forward_simd(black_box(i)).unwrap()),
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group! {
    name = packed_tensor;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2))
        .sample_size(50);
    targets = bench_forward_owned, bench_forward_mmap, bench_input_source, bench_vs_fp16, bench_batch_scaling
}

criterion_main!(packed_tensor);
