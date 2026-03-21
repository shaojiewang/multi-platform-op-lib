#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <torch/extension.h>
#include <torch/types.h>

struct GmmaDescriptor 
{
  uint64_t desc;
  __device__ static GmmaDescriptor make(const void* ptr, int leading, int stride, int layout) 
  {
    GmmaDescriptor d;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d.desc = ((uint64_t)((addr >> 4) & 0x3FFF)) | 
             ((uint64_t)((leading >> 4) & 0x3FFF) << 16) | 
             ((uint64_t)((stride >> 4) & 0x3FFF) << 32) |
             ((uint64_t)(layout & 0x3) << 62);
    return d;
  }
};

__device__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_fence_operand(float& reg) {
  asm volatile("" : "+f"(reg) :: "memory");
}

__device__ void wgmma_m64n64k16_bf16(float* acc, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
    "{\n"
    ".reg.pred p;\n"
    "setp.ne.b32 p, %34, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10,%11,%12,%13,%14,%15,"
    " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
    "%32,"
    "%33,"
    "p,1,1,0,0;\n}"
    : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),
      "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),
      "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),
      "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31])
    : "l"(da),"l"(db),"r"(scale_d)
  );
}

__device__ __forceinline__ int idx_swizzle(int row, int col) {
  int col_swizzled = col ^ ((row & 7) * 8);
  return col_swizzled;
}

__device__ void get_coord(int tid, int reg, int& row, int& col) {
  int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
  int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
  int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
  row = lin % 64;
  col = lin / 64;
}

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

  for (int k_base = 0; k_base < K; k_base += BLOCK_K) 
  {
    for (int i = tid; i < BLOCK_M * BLOCK_K; i += 128)
    {
      int m = i / BLOCK_K, k = i % BLOCK_K;
      int gm = bm + m, gk = k_base + k;
      __nv_bfloat16 val = (gk < K && gm < M) ? A[gm * K + gk] : __float2bfloat16(0.0f);
      sA[idx_swizzle(m, k)] = val;
    }
    for (int i = tid; i < BLOCK_K * BLOCK_N; i += 128)
    {
      int n = i % BLOCK_N, k = i / BLOCK_N;
      int gn = bn + n, gk = k_base + k;
      __nv_bfloat16 val = (gk < k && gn < N) ? B[gk * N + gn] : __float2bfloat16(0.0f);
      sB[idx_swizzle(n, k)] = val;
    }

    __syncthreads();
    asm volatile("fence.proxy.async;\n"::: "memory");
    __syncwarp();

#pragma unroll
    for (int ki = 0; ki < BLOCK_K; ki += WGMMA_K)
    {
#pragma unroll
      for (int i = 0; i < 32; i++)
      {
        wgmma_fence_operand(acc[i]);
      }

      wgmma_fence();
      uint64_t da = GmmaDescriptor::make(sA + ki, 0, WGMMA_STRIDE, 1).desc;
      uint64_t db = GmmaDescriptor::make(sB + ki, 0, WGMMA_STRIDE, 1).desc;

      wgmma_m64n64k16_bf16(acc, da, db, 1);
      wgmma_commit();
      
#pragma unroll
      for (int i = 0; i < 32; i++)
      {
        wgmma_fence_operand(acc[i]);
      }
      wgmma_wait();
    }

  }

#pragma unroll
  for (int r = 0; r < 32; r++)
  {
    int lm, ln;
    get_coord(tid, r, lm, ln);
    int gm = bm + lm, gn = bn + ln;
    if (gm < M && gn < N)
    {
      C[gm * N + gn] = acc[r];
    }
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




