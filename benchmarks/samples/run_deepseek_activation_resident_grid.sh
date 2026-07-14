#!/usr/bin/env bash
set -euo pipefail

sample_dir="${1:?Usage: $0 SAMPLE_DIR [OUTPUT_DIR] [WARMUP] [PAIRS] [ROWS2_ACTIVE_CTAS_PER_SM] [ROWS4_ACTIVE_CTAS_PER_SM]}"
output_dir="${2:-/tmp/flashinfer-activation-resident-grid}"
warmup="${3:-20}"
pairs="${4:-200}"
rows2_active_ctas_per_sm="${5:-}"
rows4_active_ctas_per_sm="${6:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
flashinfer_data=/usr/local/lib/python3.11/dist-packages/flashinfer/data
binary="$output_dir/activation-replay-resident-grid"

mkdir -p "$output_dir"
/usr/local/cuda/bin/nvcc \
  -std=c++17 \
  --expt-relaxed-constexpr \
  -O3 \
  -use_fast_math \
  -lineinfo \
  -Xfatbin=-compress-all \
  -gencode=arch=compute_100a,code=sm_100a \
  -I"$flashinfer_data/cutlass/include" \
  -I"$flashinfer_data/cccl/cub" \
  -I"$flashinfer_data/cccl/libcudacxx/include" \
  -I"$flashinfer_data/cccl/thrust" \
  "$script_dir/activation-replay-resident-grid.cu" \
  -o "$binary"

sha256sum "$binary"
rows2_occupancy_args=()
if [[ -n "$rows2_active_ctas_per_sm" ]]; then
  rows2_occupancy_args=(--grid-active-ctas-per-sm "$rows2_active_ctas_per_sm")
fi
rows4_occupancy_args=()
if [[ -n "$rows4_active_ctas_per_sm" ]]; then
  rows4_occupancy_args=(--grid-active-ctas-per-sm "$rows4_active_ctas_per_sm")
fi

CUDA_VISIBLE_DEVICES=0 "$binary" \
  --num-tokens 8 \
  --warmup "$warmup" \
  --pairs "$pairs" \
  "${rows2_occupancy_args[@]}"
for num_tokens in 64 1024; do
  CUDA_VISIBLE_DEVICES=0 "$binary" \
    --num-tokens "$num_tokens" \
    --warmup "$warmup" \
    --pairs "$pairs" \
    "${rows4_occupancy_args[@]}"
done
CUDA_VISIBLE_DEVICES=0 "$binary" \
  --sample-dir "$sample_dir" \
  --warmup "$warmup" \
  --pairs "$pairs" \
  "${rows4_occupancy_args[@]}"
