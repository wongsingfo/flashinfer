// Standalone resident-grid sweep for FlashInfer activationDeepSeekKernel on SM100a.
//
// Synthetic token points exercise the production 1/2/4-row dispatch heuristic.
// The captured production point additionally checks the default four-resident-grid
// launch against every valid FP8 output byte and FP32 output-scale value.

#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cub/cub.cuh>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Fp8 = cutlass::float_e4m3_t;
constexpr int kThreads = 128;
constexpr int kDefaultResidentGridRounds = 4;
constexpr int kNumGlobalExperts = 256;
constexpr int kNumLocalExperts = 32;
constexpr int kTopK = 8;
constexpr int kInnerDim = 4096;
constexpr int kTileTokens = 64;
constexpr float kE4m3Max = 448.0F;
constexpr std::array<int, 5> kResidentGridRounds{1, 2, 4, 6, 8};

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t const status_ = (call);                                                  \
    if (status_ != cudaSuccess) {                                                        \
      throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(status_)); \
    }                                                                                    \
  } while (false)

struct Params {
  Fp8 const* input;
  Fp8* output;
  float const* input_scales;
  float* output_scales;
  int32_t inner_dim;
  int32_t const* total_padded;
  int32_t const* cta_mn_limits;
  int32_t const* num_non_exiting_ctas;
  int32_t tile_tokens;
};

struct Workload {
  std::string source;
  int num_tokens;
  int top_k;
  int inner_dim;
  int tile_tokens;
  int padded_rows;
  int active_ctas;
  std::vector<uint8_t> input;
  std::vector<float> input_scales;
  std::vector<int32_t> mn_limits;
  std::vector<int> valid_rows;
  std::vector<uint8_t> reference_output;
  std::vector<float> reference_scales;
};

__device__ __forceinline__ float silu(float value) { return value / (1.0F + expf(-value)); }

struct Float4Max {
  __device__ __forceinline__ float4 operator()(float4 const& lhs, float4 const& rhs) const {
    return make_float4(fmaxf(lhs.x, rhs.x), fmaxf(lhs.y, rhs.y), fmaxf(lhs.z, rhs.z),
                       fmaxf(lhs.w, rhs.w));
  }
};

struct Float2Max {
  __device__ __forceinline__ float2 operator()(float2 const& lhs, float2 const& rhs) const {
    return make_float2(fmaxf(lhs.x, rhs.x), fmaxf(lhs.y, rhs.y));
  }
};

struct FloatMax {
  __device__ __forceinline__ float operator()(float lhs, float rhs) const {
    return fmaxf(lhs, rhs);
  }
};

template <int RowsPerGroup>
struct KernelTraits;

template <>
struct KernelTraits<4> {
  using Packed = float4;
  using Max = Float4Max;
};

template <>
struct KernelTraits<2> {
  using Packed = float2;
  using Max = Float2Max;
};

template <>
struct KernelTraits<1> {
  using Packed = float;
  using Max = FloatMax;
};

template <typename Packed, int RowsPerGroup>
__device__ __forceinline__ Packed pack(float const values[RowsPerGroup]);

template <>
__device__ __forceinline__ float4 pack<float4, 4>(float const values[4]) {
  return make_float4(values[0], values[1], values[2], values[3]);
}

template <>
__device__ __forceinline__ float2 pack<float2, 2>(float const values[2]) {
  return make_float2(values[0], values[1]);
}

template <>
__device__ __forceinline__ float pack<float, 1>(float const values[1]) {
  return values[0];
}

template <typename Packed, int RowsPerGroup>
__device__ __forceinline__ cutlass::Array<float, RowsPerGroup> unpack(Packed value);

template <>
__device__ __forceinline__ cutlass::Array<float, 4> unpack<float4, 4>(float4 value) {
  return {value.x, value.y, value.z, value.w};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 2> unpack<float2, 2>(float2 value) {
  return {value.x, value.y};
}

template <>
__device__ __forceinline__ cutlass::Array<float, 1> unpack<float, 1>(float value) {
  return {value};
}

template <int RowsPerGroup>
__global__ void activation_compact(Params params) {
  using Traits = KernelTraits<RowsPerGroup>;
  using Packed = typename Traits::Packed;
  using Max = typename Traits::Max;
  using BlockReduce = cub::BlockReduce<Packed, kThreads>;

  __shared__ float shared_output_scales[RowsPerGroup];
  __shared__ typename BlockReduce::TempStorage reduction_storage;

  int const total_padded = params.total_padded[0];
  int const active_ctas = params.num_non_exiting_ctas[0];
  int const groups_per_tile = params.tile_tokens / RowsPerGroup;
  int const total_groups = active_ctas * groups_per_tile;
  int const hidden = threadIdx.x + blockDim.x * blockIdx.x;
  int const output_dim = params.inner_dim / 2;

  for (int group = blockIdx.y; group < total_groups; group += gridDim.y) {
    int const cta = group / groups_per_tile;
    int const group_in_tile = group - cta * groups_per_tile;
    int const row_begin = cta * params.tile_tokens + group_in_tile * RowsPerGroup;
    int const row_limit = params.cta_mn_limits[cta];

    float scale1[RowsPerGroup];
    float scale2[RowsPerGroup];
    float data1[RowsPerGroup];
    float data2[RowsPerGroup];
    float outputs[RowsPerGroup];
    float absolute_outputs[RowsPerGroup];
    int rows[RowsPerGroup];

#pragma unroll
    for (int slot = 0; slot < RowsPerGroup; ++slot) {
      int const row = row_begin + slot;
      rows[slot] = row < row_limit ? row : -1;
      scale1[slot] = 0.0F;
      scale2[slot] = 0.0F;
      data1[slot] = 0.0F;
      data2[slot] = 0.0F;
      outputs[slot] = 0.0F;
      absolute_outputs[slot] = 0.0F;
    }

    if (hidden < output_dim) {
#pragma unroll
      for (int slot = 0; slot < RowsPerGroup; ++slot) {
        int const row = rows[slot];
        if (row == -1) {
          continue;
        }
        int64_t const base = static_cast<int64_t>(row) * params.inner_dim + hidden;
        int64_t const scale1_index =
            static_cast<int64_t>(row) + static_cast<int64_t>(total_padded) * (hidden / 128);
        int64_t const scale2_index =
            static_cast<int64_t>(row) +
            static_cast<int64_t>(total_padded) * ((hidden / 128) + output_dim / 128);
        scale1[slot] = params.input_scales[scale1_index];
        scale2[slot] = params.input_scales[scale2_index];
        data1[slot] = static_cast<float>(params.input[base]);
        data2[slot] = static_cast<float>(params.input[base + output_dim]);
      }
    }

#pragma unroll
    for (int slot = 0; slot < RowsPerGroup; ++slot) {
      float const x1 = scale1[slot] * data1[slot];
      float const x2 = scale2[slot] * data2[slot];
      outputs[slot] = silu(x2) * x1;
      absolute_outputs[slot] = fabsf(outputs[slot]);
    }

    auto const maxima = unpack<Packed, RowsPerGroup>(
        BlockReduce(reduction_storage).Reduce(pack<Packed, RowsPerGroup>(absolute_outputs), Max{}));

#pragma unroll
    for (int slot = 0; slot < RowsPerGroup; ++slot) {
      if (threadIdx.x == 0) {
        int const row = rows[slot];
        if (row == -1) {
          continue;
        }
        float const output_scale =
            fmaxf(maxima[slot] / kE4m3Max, std::numeric_limits<float>::min());
        shared_output_scales[slot] = output_scale;
        int64_t const scale_index =
            static_cast<int64_t>(row) + static_cast<int64_t>(total_padded) * blockIdx.x;
        params.output_scales[scale_index] = output_scale;
      }
    }
    __syncthreads();

    if (hidden < output_dim) {
#pragma unroll
      for (int slot = 0; slot < RowsPerGroup; ++slot) {
        int const row = rows[slot];
        if (row == -1) {
          continue;
        }
        int64_t const output_index = static_cast<int64_t>(row) * output_dim + hidden;
        params.output[output_index] = static_cast<Fp8>(outputs[slot] / shared_output_scales[slot]);
      }
    }
    if (group + gridDim.y < total_groups) {
      __syncthreads();
    }
  }
}

std::string read_text(std::string const& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Cannot open " + path);
  }
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

int parse_integer(std::string const& json, std::string const& key) {
  std::string const needle = "\"" + key + "\"";
  std::size_t position = json.find(needle);
  if (position == std::string::npos) {
    throw std::runtime_error("Missing manifest key " + key);
  }
  position = json.find(':', position + needle.size());
  if (position == std::string::npos) {
    throw std::runtime_error("Malformed manifest key " + key);
  }
  char* end = nullptr;
  long const value = std::strtol(json.c_str() + position + 1, &end, 10);
  if (end == json.c_str() + position + 1) {
    throw std::runtime_error("Manifest key is not an integer: " + key);
  }
  return static_cast<int>(value);
}

template <typename T>
std::vector<T> read_binary(std::string const& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("Cannot open " + path);
  }
  std::streamsize const bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
    throw std::runtime_error("Invalid byte size for " + path);
  }
  stream.seekg(0);
  std::vector<T> values(static_cast<std::size_t>(bytes) / sizeof(T));
  if (bytes > 0 && !stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
    throw std::runtime_error("Short read from " + path);
  }
  return values;
}

Workload load_capture(std::string const& sample_dir) {
  std::string const manifest = read_text(sample_dir + "/manifest.json");
  Workload workload;
  workload.source = "capture";
  workload.num_tokens = parse_integer(manifest, "num_tokens");
  workload.top_k = parse_integer(manifest, "top_k");
  workload.inner_dim = parse_integer(manifest, "inner_dim");
  workload.tile_tokens = parse_integer(manifest, "tile_tokens_dim");
  workload.padded_rows = parse_integer(manifest, "total_num_padded_tokens");
  workload.active_ctas = parse_integer(manifest, "num_non_exiting_ctas");
  workload.input = read_binary<uint8_t>(sample_dir + "/input_fp8.bin");
  workload.input_scales = read_binary<float>(sample_dir + "/input_scale_f32.bin");
  workload.mn_limits = read_binary<int32_t>(sample_dir + "/cta_mn_limit_i32.bin");
  workload.reference_output = read_binary<uint8_t>(sample_dir + "/reference_output_fp8.bin");
  workload.reference_scales = read_binary<float>(sample_dir + "/reference_output_scale_f32.bin");

  for (int cta = 0; cta < workload.active_ctas; ++cta) {
    int const begin = cta * workload.tile_tokens;
    int const limit = workload.mn_limits.at(cta);
    if (limit < begin || limit > begin + workload.tile_tokens) {
      throw std::runtime_error("Invalid capture CTA MN limit");
    }
    for (int row = begin; row < limit; ++row) {
      workload.valid_rows.push_back(row);
    }
  }
  return workload;
}

Workload make_synthetic(int num_tokens) {
  if (num_tokens <= 0) {
    throw std::runtime_error("Synthetic num_tokens must be positive");
  }

  Workload workload;
  workload.source = "synthetic";
  workload.num_tokens = num_tokens;
  workload.top_k = kTopK;
  workload.inner_dim = kInnerDim;
  workload.tile_tokens = kTileTokens;

  std::array<int, kNumLocalExperts> local_counts{};
  std::vector<int> experts(kNumGlobalExperts);
  std::iota(experts.begin(), experts.end(), 0);
  std::mt19937 random(0x4F50454EU + static_cast<uint32_t>(num_tokens));
  for (int token = 0; token < num_tokens; ++token) {
    std::shuffle(experts.begin(), experts.end(), random);
    for (int slot = 0; slot < kTopK; ++slot) {
      int const expert = experts[slot];
      if (expert < kNumLocalExperts) {
        ++local_counts[expert];
      }
    }
  }

  int cta = 0;
  for (int count : local_counts) {
    for (int consumed = 0; consumed < count; consumed += workload.tile_tokens) {
      int const begin = cta * workload.tile_tokens;
      int const valid = std::min(workload.tile_tokens, count - consumed);
      workload.mn_limits.push_back(begin + valid);
      for (int row = begin; row < begin + valid; ++row) {
        workload.valid_rows.push_back(row);
      }
      ++cta;
    }
  }
  if (cta == 0) {
    throw std::runtime_error("Synthetic routing produced no rank-local work");
  }

  workload.active_ctas = cta;
  workload.padded_rows = cta * workload.tile_tokens;
  std::size_t const input_elements =
      static_cast<std::size_t>(workload.padded_rows) * workload.inner_dim;
  workload.input.resize(input_elements);
  static_assert(sizeof(Fp8) == sizeof(uint8_t));
  for (std::size_t index = 0; index < input_elements; ++index) {
    int const centered =
        static_cast<int>((index * 17 + index / workload.inner_dim * 13) % 113) - 56;
    Fp8 const value(static_cast<float>(centered) / 8.0F);
    std::memcpy(&workload.input[index], &value, sizeof(value));
  }

  std::size_t const input_scale_values =
      static_cast<std::size_t>(workload.padded_rows) * (workload.inner_dim / 128);
  workload.input_scales.resize(input_scale_values);
  for (std::size_t index = 0; index < input_scale_values; ++index) {
    workload.input_scales[index] = 0.001F + static_cast<float>((index * 19) % 101) * 0.00001F;
  }
  return workload;
}

void validate_workload(Workload const& workload) {
  int const output_dim = workload.inner_dim / 2;
  int const scale_blocks = output_dim / 128;
  if (workload.top_k <= 0 || workload.inner_dim <= 0 || workload.inner_dim % 256 != 0 ||
      workload.tile_tokens <= 0 || workload.active_ctas <= 0 ||
      workload.padded_rows != workload.active_ctas * workload.tile_tokens ||
      workload.mn_limits.size() != static_cast<std::size_t>(workload.active_ctas) ||
      workload.input.size() !=
          static_cast<std::size_t>(workload.padded_rows) * workload.inner_dim ||
      workload.input_scales.size() !=
          static_cast<std::size_t>(workload.padded_rows) * (workload.inner_dim / 128)) {
    throw std::runtime_error("Workload dimensions or file sizes are inconsistent");
  }
  if (!workload.reference_output.empty() &&
      (workload.reference_output.size() !=
           static_cast<std::size_t>(workload.padded_rows) * output_dim ||
       workload.reference_scales.size() !=
           static_cast<std::size_t>(workload.padded_rows) * scale_blocks)) {
    throw std::runtime_error("Reference file sizes are inconsistent");
  }
}

struct Summary {
  double mean;
  double p50;
  double p90;
  double minimum;
  double maximum;
};

Summary summarize(std::vector<float> values) {
  if (values.empty()) {
    throw std::runtime_error("No timings collected");
  }
  std::sort(values.begin(), values.end());
  double const mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  std::size_t const middle = values.size() / 2;
  double const median = values.size() % 2 == 0
                            ? (static_cast<double>(values[middle - 1]) + values[middle]) / 2.0
                            : values[middle];
  std::size_t const p90_index = static_cast<std::size_t>(std::ceil(values.size() * 0.9)) - 1;
  return {mean, median, values[p90_index], values.front(), values.back()};
}

template <typename Launch>
float measure_once(cudaEvent_t start, cudaEvent_t stop, Launch const& launch) {
  CUDA_CHECK(cudaEventRecord(start));
  launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  return milliseconds * 1000.0F;
}

struct Comparison {
  std::size_t output_mismatch_bytes{0};
  std::size_t scale_mismatch_values{0};
  int first_output_row{-1};
  int first_scale_row{-1};
};

Comparison compare_valid(std::vector<uint8_t> const& output, std::vector<float> const& scales,
                         Workload const& workload) {
  Comparison result;
  int const output_dim = workload.inner_dim / 2;
  int const scale_blocks = output_dim / 128;
  for (int row : workload.valid_rows) {
    std::size_t const base = static_cast<std::size_t>(row) * output_dim;
    for (int column = 0; column < output_dim; ++column) {
      if (output[base + column] != workload.reference_output[base + column]) {
        ++result.output_mismatch_bytes;
        if (result.first_output_row < 0) {
          result.first_output_row = row;
        }
      }
    }
    for (int block = 0; block < scale_blocks; ++block) {
      std::size_t const index = static_cast<std::size_t>(block) * workload.padded_rows + row;
      if (std::memcmp(&scales[index], &workload.reference_scales[index], sizeof(float)) != 0) {
        ++result.scale_mismatch_values;
        if (result.first_scale_row < 0) {
          result.first_scale_row = row;
        }
      }
    }
  }
  return result;
}

int compute_grid_y(int grid_x, int logical_grid_y, int num_sms, int active_ctas_per_sm,
                   int resident_grid_rounds) {
  int64_t const logical_ctas = static_cast<int64_t>(grid_x) * logical_grid_y;
  int64_t const resident_ctas = static_cast<int64_t>(num_sms) * active_ctas_per_sm;
  int64_t const launch_ctas = std::min(logical_ctas, resident_ctas * resident_grid_rounds);
  return static_cast<int>(std::min<int64_t>(logical_grid_y, (launch_ctas + grid_x - 1) / grid_x));
}

struct Variant {
  std::string label;
  int resident_grid_rounds;
  int grid_y;
  std::vector<float> timings;
  Summary summary{};
};

template <int RowsPerGroup>
void run_workload(Workload const& workload, cudaDeviceProp const& properties, int warmup,
                  int pairs) {
  int const output_dim = workload.inner_dim / 2;
  int const grid_x = output_dim / 128;
  int const logical_grid_y =
      std::min(8192, (workload.num_tokens + RowsPerGroup - 1) / RowsPerGroup * workload.top_k);

  int active_ctas_per_sm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_ctas_per_sm, activation_compact<RowsPerGroup>, kThreads, 0));
  if (active_ctas_per_sm <= 0) {
    throw std::runtime_error("Kernel occupancy query returned no resident CTAs");
  }

  Fp8* device_input = nullptr;
  Fp8* device_output = nullptr;
  float* device_input_scales = nullptr;
  float* device_output_scales = nullptr;
  int32_t* device_total_padded = nullptr;
  int32_t* device_mn_limits = nullptr;
  int32_t* device_active_ctas = nullptr;
  std::size_t const output_bytes =
      static_cast<std::size_t>(workload.padded_rows) * output_dim * sizeof(Fp8);
  std::size_t const output_scale_bytes =
      static_cast<std::size_t>(workload.padded_rows) * (output_dim / 128) * sizeof(float);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), workload.input.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input_scales),
                        workload.input_scales.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output_scales), output_scale_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_total_padded), sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_mn_limits),
                        workload.mn_limits.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_active_ctas), sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(device_input, workload.input.data(), workload.input.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_input_scales, workload.input_scales.data(),
                        workload.input_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_total_padded, &workload.padded_rows, sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_mn_limits, workload.mn_limits.data(),
                        workload.mn_limits.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_active_ctas, &workload.active_ctas, sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  Params const params{device_input,         device_output,      device_input_scales,
                      device_output_scales, workload.inner_dim, device_total_padded,
                      device_mn_limits,     device_active_ctas, workload.tile_tokens};

  std::vector<Variant> variants;
  variants.push_back({"logical", 0, logical_grid_y});
  for (int resident_grid_rounds : kResidentGridRounds) {
    variants.push_back({"rounds-" + std::to_string(resident_grid_rounds), resident_grid_rounds,
                        compute_grid_y(grid_x, logical_grid_y, properties.multiProcessorCount,
                                       active_ctas_per_sm, resident_grid_rounds)});
  }

  auto launch = [&](int grid_y) {
    activation_compact<RowsPerGroup><<<dim3(grid_x, grid_y, 1), kThreads>>>(params);
    CUDA_CHECK(cudaGetLastError());
  };

  if (!workload.reference_output.empty()) {
    auto const default_variant =
        std::find_if(variants.begin(), variants.end(), [](Variant const& variant) {
          return variant.resident_grid_rounds == kDefaultResidentGridRounds;
        });
    if (default_variant == variants.end()) {
      throw std::runtime_error("Default resident-grid configuration is missing");
    }
    CUDA_CHECK(cudaMemset(device_output, 0xA5, output_bytes));
    CUDA_CHECK(cudaMemset(device_output_scales, 0xA5, output_scale_bytes));
    launch(default_variant->grid_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> output(output_bytes);
    std::vector<float> output_scales(output_scale_bytes / sizeof(float));
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_scales.data(), device_output_scales, output_scale_bytes,
                          cudaMemcpyDeviceToHost));
    Comparison const comparison = compare_valid(output, output_scales, workload);
    bool const pass =
        comparison.output_mismatch_bytes == 0 && comparison.scale_mismatch_values == 0;
    std::cout << "CORRECTNESS label=rounds-" << kDefaultResidentGridRounds
              << " valid_rows=" << workload.valid_rows.size()
              << " output_mismatch_bytes=" << comparison.output_mismatch_bytes
              << " scale_mismatch_values=" << comparison.scale_mismatch_values
              << " first_output_row=" << comparison.first_output_row
              << " first_scale_row=" << comparison.first_scale_row
              << " bitwise=" << (pass ? "PASS" : "FAIL") << '\n';
    if (!pass) {
      throw std::runtime_error("Default resident-grid configuration failed bitwise validation");
    }
  }

  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, activation_compact<RowsPerGroup>));
  int const total_groups = workload.active_ctas * workload.tile_tokens / RowsPerGroup;
  std::cout << "SHAPE source=" << workload.source << " num_tokens=" << workload.num_tokens
            << " top_k=" << workload.top_k << " inner_dim=" << workload.inner_dim
            << " output_dim=" << output_dim << " padded_rows=" << workload.padded_rows
            << " valid_rows=" << workload.valid_rows.size()
            << " active_routing_ctas=" << workload.active_ctas << " total_groups=" << total_groups
            << " tile_tokens=" << workload.tile_tokens << '\n';
  std::cout << "KERNEL rows_per_group=" << RowsPerGroup << " threads=" << kThreads
            << " regs=" << attributes.numRegs << " static_smem=" << attributes.sharedSizeBytes
            << " active_ctas_per_sm=" << active_ctas_per_sm
            << " resident_ctas=" << properties.multiProcessorCount * active_ctas_per_sm << '\n';
  for (Variant const& variant : variants) {
    double const cta_waves =
        static_cast<double>(grid_x) * variant.grid_y / properties.multiProcessorCount;
    std::cout << std::fixed << std::setprecision(3) << "CONFIG label=" << variant.label
              << " resident_grid_rounds=" << variant.resident_grid_rounds << " grid=" << grid_x
              << 'x' << variant.grid_y << "x1 cta_waves=" << cta_waves << '\n';
  }

  for (int iteration = 0; iteration < warmup; ++iteration) {
    for (Variant const& variant : variants) {
      launch(variant.grid_y);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t event_start{};
  cudaEvent_t event_stop{};
  CUDA_CHECK(cudaEventCreate(&event_start));
  CUDA_CHECK(cudaEventCreate(&event_stop));
  for (Variant& variant : variants) {
    variant.timings.reserve(pairs * 2);
  }
  for (int pair = 0; pair < pairs; ++pair) {
    for (Variant& variant : variants) {
      variant.timings.push_back(
          measure_once(event_start, event_stop, [&]() { launch(variant.grid_y); }));
    }
    for (auto variant = variants.rbegin(); variant != variants.rend(); ++variant) {
      variant->timings.push_back(
          measure_once(event_start, event_stop, [&]() { launch(variant->grid_y); }));
    }
  }
  for (Variant& variant : variants) {
    variant.summary = summarize(std::move(variant.timings));
  }

  Summary const& logical = variants.front().summary;
  for (Variant const& variant : variants) {
    std::cout << std::fixed << std::setprecision(3) << "RESULT source=" << workload.source
              << " num_tokens=" << workload.num_tokens << " label=" << variant.label
              << " rows_per_group=" << RowsPerGroup << " grid=" << grid_x << 'x' << variant.grid_y
              << "x1 samples=" << pairs * 2 << " mean_us=" << variant.summary.mean
              << " p50_us=" << variant.summary.p50 << " p90_us=" << variant.summary.p90
              << " min_us=" << variant.summary.minimum << " max_us=" << variant.summary.maximum
              << " mean_speedup_vs_logical=" << logical.mean / variant.summary.mean
              << " p50_speedup_vs_logical=" << logical.p50 / variant.summary.p50 << '\n';
  }

  CUDA_CHECK(cudaEventDestroy(event_start));
  CUDA_CHECK(cudaEventDestroy(event_stop));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_input_scales));
  CUDA_CHECK(cudaFree(device_output_scales));
  CUDA_CHECK(cudaFree(device_total_padded));
  CUDA_CHECK(cudaFree(device_mn_limits));
  CUDA_CHECK(cudaFree(device_active_ctas));
}

}  // namespace

int main(int argc, char** argv) try {
  std::string sample_dir;
  int num_tokens = 0;
  int warmup = 20;
  int pairs = 200;
  for (int index = 1; index < argc; ++index) {
    std::string const argument = argv[index];
    if (argument == "--sample-dir" && index + 1 < argc) {
      sample_dir = argv[++index];
    } else if (argument == "--num-tokens" && index + 1 < argc) {
      num_tokens = std::stoi(argv[++index]);
    } else if (argument == "--warmup" && index + 1 < argc) {
      warmup = std::stoi(argv[++index]);
    } else if (argument == "--pairs" && index + 1 < argc) {
      pairs = std::stoi(argv[++index]);
    } else {
      throw std::runtime_error(
          "Usage: activation-replay-resident-grid "
          "(--sample-dir DIR | --num-tokens N) "
          "[--warmup N] [--pairs N]");
    }
  }
  if ((sample_dir.empty() == (num_tokens == 0)) || warmup < 0 || pairs <= 0) {
    throw std::runtime_error("Specify exactly one workload source and valid timing arguments");
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 10) {
    throw std::runtime_error("Replay expects an SM100-class GPU");
  }
  std::cout << "DEVICE name=\"" << properties.name
            << "\" sm_count=" << properties.multiProcessorCount
            << " compute_capability=" << properties.major << '.' << properties.minor << '\n';

  Workload const workload =
      sample_dir.empty() ? make_synthetic(num_tokens) : load_capture(sample_dir);
  validate_workload(workload);
  int const grid_x = workload.inner_dim / 2 / 128;
  int64_t const heuristic_ctas =
      static_cast<int64_t>(grid_x) * workload.num_tokens * workload.top_k;
  int rows_per_group = 1;
  if (heuristic_ctas > static_cast<int64_t>(properties.multiProcessorCount) * 32) {
    rows_per_group = 4;
  } else if (heuristic_ctas > static_cast<int64_t>(properties.multiProcessorCount) * 4) {
    rows_per_group = 2;
  }
  if (workload.tile_tokens < rows_per_group || workload.tile_tokens % rows_per_group != 0) {
    throw std::runtime_error("tile_tokens is incompatible with selected rows_per_group");
  }

  if (rows_per_group == 4) {
    run_workload<4>(workload, properties, warmup, pairs);
  } else if (rows_per_group == 2) {
    run_workload<2>(workload, properties, warmup, pairs);
  } else {
    run_workload<1>(workload, properties, warmup, pairs);
  }
  return 0;
} catch (std::exception const& error) {
  std::cerr << "ERROR " << error.what() << '\n';
  return 2;
}
