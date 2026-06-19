# Regenerating Results

Run:

```bash
cargo bench --bench simd_kernel 2>&1 | tee results/simd_kernel_raw.txt
```

To save a criterion baseline snapshot for regression tracking:

```bash
cargo bench --bench simd_kernel -- --save-baseline simd_kernel
```

To compare against a saved baseline later:

```bash
cargo bench --bench simd_kernel -- --baseline simd_kernel
```
