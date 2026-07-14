#!/usr/bin/env bash
set -euo pipefail

sample_dir="${1:?Usage: $0 SAMPLE_DIR [OUTPUT_DIR] [WARMUP] [PAIRS]}"
output_dir="${2:-/tmp/flashinfer-activation-resident-grid}"
warmup="${3:-20}"
pairs="${4:-200}"
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
for num_tokens in 8 64 1024; do
  CUDA_VISIBLE_DEVICES=0 "$binary" \
    --num-tokens "$num_tokens" \
    --warmup "$warmup" \
    --pairs "$pairs"
done
CUDA_VISIBLE_DEVICES=0 "$binary" \
  --sample-dir "$sample_dir" \
  --warmup "$warmup" \
  --pairs "$pairs"
