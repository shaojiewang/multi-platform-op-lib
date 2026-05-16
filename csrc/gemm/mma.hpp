#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace bfgemm {
namespace mma {

enum class MmaOpcode {
  kM16N8K16RowColF32Bf16Bf16F32,
};

template <MmaOpcode Opcode>
struct Mma;

// RTX 3070 is sm_86 (Ampere). BF16 Tensor Core MMA uses the SM80+ PTX opcode.
template <>
struct Mma<MmaOpcode::kM16N8K16RowColF32Bf16Bf16F32> {
  static constexpr int kMinComputeCapability = 80;
  static constexpr int kM = 16;
  static constexpr int kN = 8;
  static constexpr int kK = 16;
  static constexpr int kARegisters = 4;
  static constexpr int kBRegisters = 2;
  static constexpr int kAccumulatorRegisters = 4;

  using ARegisters = uint32_t[kARegisters];
  using BRegisters = uint32_t[kBRegisters];
  using AccumulatorRegisters = float[kAccumulatorRegisters];

  __device__ __forceinline__ static void fma(
      const uint32_t (&a)[kARegisters],
      const uint32_t (&b)[kBRegisters],
      float (&acc)[kAccumulatorRegisters]) {
    fma(&a[0], &b[0], &acc[0]);
  }

  __device__ __forceinline__ static void fma(
      const uint32_t* a,
      const uint32_t* b,
      float* acc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
#else
    asm volatile("trap;\n");
#endif
  }
};

using Rtx3070Bf16Mma =
    Mma<MmaOpcode::kM16N8K16RowColF32Bf16Bf16F32>;

__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f32_bf16_bf16(
    const uint32_t* frag_a,
    const uint32_t* frag_b,
    float* accum) {
  Rtx3070Bf16Mma::fma(frag_a, frag_b, accum);
}

__device__ __forceinline__ void mma_sync(
    const uint32_t* frag_a,
    const uint32_t* frag_b,
    float* accum) {
  mma_sync_m16n8k16_row_col_f32_bf16_bf16(frag_a, frag_b, accum);
}

}  // namespace mma
}  // namespace bfgemm
