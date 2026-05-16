#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "mma.hpp"

namespace {

__device__ __forceinline__ uint32_t pack_bf16_pair(const __nv_bfloat16& hi,
                                                      const __nv_bfloat16& lo) {
  uint32_t hi_bits = *reinterpret_cast<const uint16_t*>(&hi);
  uint32_t lo_bits = *reinterpret_cast<const uint16_t*>(&lo);
  return (hi_bits << 16) | lo_bits;
}

__device__ __forceinline__ __nv_bfloat16 load_or_zero(
    const __nv_bfloat16* ptr,
    int row,
    int col,
    int ld,
    int rows,
    int cols) {
  return (row < rows && col < cols) ? ptr[row * ld + col]
                                    : __float2bfloat16(0.0f);
}

__device__ __forceinline__ void load_a_fragment(
    const __nv_bfloat16* A,
    int m0,
    int k0,
    int M,
    int K,
    int lane,
    uint32_t frag_a[4]) {
  const int group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int k_pair = lane_in_group * 2;

  const __nv_bfloat16 a0 = load_or_zero(A, m0 + group, k0 + k_pair, K, M, K);
  const __nv_bfloat16 a1 = load_or_zero(A, m0 + group, k0 + k_pair + 1, K, M, K);
  const __nv_bfloat16 a2 = load_or_zero(A, m0 + group + 8, k0 + k_pair, K, M, K);
  const __nv_bfloat16 a3 = load_or_zero(A, m0 + group + 8, k0 + k_pair + 1, K, M, K);
  const __nv_bfloat16 a4 = load_or_zero(A, m0 + group, k0 + k_pair + 8, K, M, K);
  const __nv_bfloat16 a5 = load_or_zero(A, m0 + group, k0 + k_pair + 9, K, M, K);
  const __nv_bfloat16 a6 = load_or_zero(A, m0 + group + 8, k0 + k_pair + 8, K, M, K);
  const __nv_bfloat16 a7 = load_or_zero(A, m0 + group + 8, k0 + k_pair + 9, K, M, K);

  frag_a[0] = pack_bf16_pair(a1, a0);
  frag_a[1] = pack_bf16_pair(a3, a2);
  frag_a[2] = pack_bf16_pair(a5, a4);
  frag_a[3] = pack_bf16_pair(a7, a6);
}

__device__ __forceinline__ void load_b_fragment(
    const __nv_bfloat16* B,
    int n0,
    int k0,
    int N,
    int K,
    int lane,
    uint32_t frag_b[2]) {
  const int group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int k_pair = lane_in_group * 2;

  const __nv_bfloat16 b0 = load_or_zero(B, k0 + k_pair, n0 + group, N, K, N);
  const __nv_bfloat16 b1 = load_or_zero(B, k0 + k_pair + 1, n0 + group, N, K, N);
  const __nv_bfloat16 b2 = load_or_zero(B, k0 + k_pair + 8, n0 + group, N, K, N);
  const __nv_bfloat16 b3 = load_or_zero(B, k0 + k_pair + 9, n0 + group, N, K, N);

  frag_b[0] = pack_bf16_pair(b1, b0);
  frag_b[1] = pack_bf16_pair(b3, b2);
}

__device__ __forceinline__ void store_accumulator(
    float* C,
    int m0,
    int n0,
    int M,
    int N,
    int lane,
    const float accum[4]) {
  const int group = lane >> 2;
  const int lane_in_group = lane & 3;
  const int col = n0 + lane_in_group * 2;

  const int row0 = m0 + group;
  const int row1 = m0 + group + 8;

  if (row0 < M && col < N) {
    C[row0 * N + col] = accum[0];
  }
  if (row0 < M && col + 1 < N) {
    C[row0 * N + col + 1] = accum[1];
  }
  if (row1 < M && col < N) {
    C[row1 * N + col] = accum[2];
  }
  if (row1 < M && col + 1 < N) {
    C[row1 * N + col + 1] = accum[3];
  }
}

}  // namespace

__global__ __launch_bounds__(32) void bfgemm_mma_gemm(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M,
    int N,
    int K) {
  constexpr int TILE_M = 16;
  constexpr int TILE_N = 8;

  int m0 = blockIdx.y * TILE_M;
  int n0 = blockIdx.x * TILE_N;
  if (m0 >= M || n0 >= N) {
    return;
  }

  const int lane = threadIdx.x & 31;
  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t frag_a[4];
    uint32_t frag_b[2];
    load_a_fragment(A, m0, k0, M, K, lane, frag_a);
    load_b_fragment(B, n0, k0, N, K, lane, frag_b);
    bfgemm::mma::mma_sync(frag_a, frag_b, accum);
  }

  store_accumulator(C, m0, n0, M, N, lane, accum);
}

void bfgemm_mma_launch(const void* A, const void* B, void* C, int M, int N, int K) {
  dim3 grid((N + 7) / 8, (M + 15) / 16);
  bfgemm_mma_gemm<<<grid, 32>>>((const __nv_bfloat16*)A,
                                (const __nv_bfloat16*)B,
                                (float*)C,
                                M,
                                N,
                                K);
}
