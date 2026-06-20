#!/usr/bin/env bash
# Run the MoE layer micro-benchmarks and capture results.
#
# Usage: ./scripts/run_moe_bench.sh [--quick]
#
#   --quick   Run with reduced sample count for fast validation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RAW_FILE="$RESULTS_DIR/moe_layer_raw.txt"
MD_FILE="$RESULTS_DIR/moe_layer_results.md"

cd "$PROJECT_DIR"

QUICK_FLAG=""
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_FLAG="-- --quick"
    echo "⚠️  Quick mode — reduced samples, results not publishable."
fi

echo "🚀 Running MoE Layer benchmarks..."
echo "   Raw output  -> $RAW_FILE"
echo "   Summary     -> $MD_FILE"
echo ""

if [[ -n "$QUICK_FLAG" ]]; then
    # Quick validation: uses the sample_size/warmup configured in the benchmark code
    # (groups override to smaller values for fast checks)
    cargo bench --bench moe_layer 2>&1 | tee "$RAW_FILE"
else
    # Full run: production-quality measurements
    cargo bench --bench moe_layer 2>&1 | tee "$RAW_FILE"
fi

echo ""
echo "✅ Benchmarks complete."
echo "   Raw output saved to $RAW_FILE"

# Extract key information for summary markdown
cat > "$MD_FILE" << 'MDFOOTER'

## Analysis

The criterion output above contains throughput tables, breakdown measurements,
and comparison baselines. Key insights:

1. **Routing overhead:** Compare oracle vs standard to see what fraction of time
   goes to top-k routing vs expert compute.
2. **Binning efficiency:** Compare naive loop vs standard to see cache benefits
   of batching tokens by expert.
3. **Packed vs FP16:** Compare packed_2bit vs fp16_matmul to quantify the
   compute cost of fused dequantization vs dense matmul.
MDFOOTER

# Extract the analysis tables from the raw output (section after "breakdown analysis")
if grep -q "MoE Layer Breakdown" "$RAW_FILE"; then
    echo "" >> "$MD_FILE"
    echo "---" >> "$MD_FILE"
    echo "" >> "$MD_FILE"
    echo "### Breakdown Tables (from benchmark output)" >> "$MD_FILE"
    echo "" >> "$MD_FILE"
    echo '```' >> "$MD_FILE"
    # Extract lines from "MoE Layer Breakdown" to end of file
    awk '/MoE Layer Breakdown/,0' "$RAW_FILE" >> "$MD_FILE"
    echo '```' >> "$MD_FILE"
fi

echo "   Summary saved to $MD_FILE"
