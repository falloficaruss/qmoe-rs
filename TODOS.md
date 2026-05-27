# QMoE.rs — Execution Plan

## Legend
- **P0**: Blocking — nothing works without this
- **P1**: Critical — needed for real evaluation
- **P2**: Performance — throughput & scale
- **P3**: Stretch — production readiness

---

## P0 — Core Pipeline (Blocking)

- [ ] **Wire up real safetensors loading** — connect `quantize.py` output to Rust inference
  - [x] Add `PackedQMoETensor::from_bytes` constructor
  - [x] Create `loader.rs` — parse tensor names, discover config, build experts
  - [x] Update `main.rs` — accept `--model` path, use loader instead of dummy data
  - [x] Integrate a tokenizer (`tokenizers` crate) for text-in/text-out
  - [ ] Build real config pipeline — parse model hyperparams from tensor shapes or config file

## P1 — Correctness & Evaluation

- [ ] **Fix attention bug** — `model.rs:57` has invalid `reshape((b_sz, seq_len, ()))`
- [ ] **Integration tests & perplexity eval** — load real quantized model, run on WikiText-2
- [ ] **Benchmark tokens/sec** — profile SIMD kernel, routing overhead, memory bandwidth

## P2 — Performance & Scaling

- [ ] **GPU/CUDA fused dequant kernel** — rewrite `fused_dequantize_and_dot` in Triton/raw CUDA
- [ ] **Multi-layer & real dimensions** — scale from toy 64-dim to real DeepSeek sizes
- [ ] **SIMD kernel widening** — process 16-32 bytes at once (u8x16/u8x32), AVX-512

## P3 — Production Readiness

- [ ] **Streaming/overlapping I/O** — leverage mmap to overlap disk I/O with compute
- [ ] **Sharded expert loading** — support distributing experts across devices
- [x] **KVCache integration** — full autoregressive generation with caching
