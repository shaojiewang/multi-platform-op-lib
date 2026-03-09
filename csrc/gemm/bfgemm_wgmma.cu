#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <torch/extension.h>
#include <torch/types.h>

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

  printf("tidx=%d\n", tid);
  
}

void bfgemm_launch(
  const void* A, 
  const void* B,
  void* C,
  int M,
  int N,
  int K)
{
  
}

#define STRINGFY(str) #str
#define TORCH_BINGDING_EXTENSION (func) \ 
  m.def(STRINGFY(func), &func, STRINGFY(func));



