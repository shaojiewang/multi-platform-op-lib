#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <torch/extension.h>
#include <torch/types.h>

struct WgmmaDescirptor {
  uint64_t desc;
  __device__ static WgmmaDescirptor make(const void* ptr, int leading, int stride, int layout) {
    WgmmaDescirptor d;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d.desc = 

};

template <int BLOCK_M = 64,
  int BLOCK_N = 64,
  int BLOCK_K = 64,
  int WGMMA_K = 16,
  int WGMMA_STRIDE = 1024>
__global__ __launch_bounds__(128)
void wgmma_bf16_gemm(
  const __nv_bfloat16* A,
  const __nv_bfloat16* B,
  float* C,
  int M,
  int N,
  int K)
{
  __shared__ __align__(1024) __nv_bfloat16 sA[BLOCK_M * BLOCK_K];
  __shared__ __align__(1024) __nv_bfloat16 sB[BLOCK_N * BLOCK_K];

  int tid = threadIdx.x;
  int bm = blockIdx.y * BLOCK_M, bn = blockIdx.x * BLOCK_N;
  if (bm >= M || bn >= N) return;

  float acc[32];
#pragma unroll
  for (int i = 0; i < 32; i++)
  {
    acc[i] = 0.f;
  }

}


template <int BLOCK_M = 64,
  int BLOCK_N = 64,
  int BLOCK_K = 64,
  int WGMMA_K = 16,
  int WGMMA_STRIDE = 1024>
void bfgemm_launch(
  const void* A, 
  const void* B,
  void* C,
  const int M,
  const int N,
  const int K)
{
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  wgmma_bf16_gemm<BLOCK_M, BLOCK_N, BLOCK_K, WGMMA_K, WGMMA_STRIDE><<<grid, 128>>>((const __nv_bfloat16*)A, (const __nv_bfloat16*)B, (float*)C, M, N, K);
}

void bfgemm_torch(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
  const int M = A.size(0);
  const int N = B.size(1);
  const int K = B.size(0);

  constexpr int BLOCK_M = 64;
  constexpr int BLOCK_N = 64;
  constexpr int BLOCK_K = 64;
  constexpr int WGMMA_K = 16;
  constexpr int WGMMA_STRIDE = 1024;

  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  wgmma_bf16_gemm<BLOCK_M, BLOCK_N, BLOCK_K, WGMMA_K, WGMMA_STRIDE><<<grid, 128>>>(
    (const __nv_bfloat16*)(A.data_ptr()), 
    (const __nv_bfloat16*)(B.data_ptr()), 
    (float*)(C.data_ptr()), 
    M, 
    N, 
    K);

}


#define STRINGFY(str) #str
#define TORCH_BINDING_EXTENSION(func) \ 
  m.def(STRINGFY(func), &func, STRINGFY(func));

void bfgemm_launch(const void* A, const void* B, void* C, int M, int N, int K);
void bfgemm_torch(torch::Tensor A, torch::Tensor B, torch::Tensor C);

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  TORCH_BINDING_EXTENSION(bfgemm_torch)
//}




