#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>
#include <stdio.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using half_t = half;

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

#define HOST __forceinline__ __host__
#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__

int get_sm_count() {
  int device_id;
  cudaError_t result = cudaGetDevice(&device_id);
  if (result != cudaSuccess) {
    std::cerr << "cudaGetDevice() returned error " << cudaGetErrorString(result)
              << std::endl;
    return 1;
  }
  int multiprocessor_count;
  result = cudaDeviceGetAttribute(&multiprocessor_count,
                                  cudaDevAttrMultiProcessorCount, device_id);
  if (result != cudaSuccess) {
    std::cerr << "cudaDeviceGetAttribute() returned error "
              << cudaGetErrorString(result) << std::endl;
    return 1;
  }
  return multiprocessor_count;
}

static constexpr int MAX_CLUSTER_SIZE = 16;
static constexpr int WARP_GROUP_SIZE = 128;
static constexpr int WARP_NUMBER_IN_WARP_GROUP = 4;
static constexpr int WARP_SIZE = 32;

template <int ScaleA, int ScaleB, int ScaleD, int TransA, int TransB>
struct SM90_64x128x16_F32BF16BF16_SS {
  DEVICE static void wgmma(
      uint64_t const& desc_a, uint64_t const& desc_b, float& d00, float& d01,
      float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
      float& d08, float& d09, float& d10, float& d11, float& d12, float& d13,
      float& d14, float& d15, float& d16, float& d17, float& d18, float& d19,
      float& d20, float& d21, float& d22, float& d23, float& d24, float& d25,
      float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
      float& d32, float& d33, float& d34, float& d35, float& d36, float& d37,
      float& d38, float& d39, float& d40, float& d41, float& d42, float& d43,
      float& d44, float& d45, float& d46, float& d47, float& d48, float& d49,
      float& d50, float& d51, float& d52, float& d53, float& d54, float& d55,
      float& d56, float& d57, float& d58, float& d59, float& d60, float& d61,
      float& d62, float& d63) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
        " %64,"
        " %65,"
        " p,    %67,  %68,  %69,  %70;\n"
        "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05),
          "+f"(d06), "+f"(d07), "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
          "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15), "+f"(d16), "+f"(d17),
          "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
          "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29),
          "+f"(d30), "+f"(d31), "+f"(d32), "+f"(d33), "+f"(d34), "+f"(d35),
          "+f"(d36), "+f"(d37), "+f"(d38), "+f"(d39), "+f"(d40), "+f"(d41),
          "+f"(d42), "+f"(d43), "+f"(d44), "+f"(d45), "+f"(d46), "+f"(d47),
          "+f"(d48), "+f"(d49), "+f"(d50), "+f"(d51), "+f"(d52), "+f"(d53),
          "+f"(d54), "+f"(d55), "+f"(d56), "+f"(d57), "+f"(d58), "+f"(d59),
          "+f"(d60), "+f"(d61), "+f"(d62), "+f"(d63)
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
  }
};


