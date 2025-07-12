#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include "cublas_v2.h"

static cublasHandle_t g_handle = nullptr;

void init_cublas_handle() {
  if (g_handle == nullptr) {
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to create cuBLAS handle: %d", status);
      exit(EXIT_FAILURE);
    }
    status = cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to set cuBLAS Math Mode: %d", status);
      exit(EXIT_FAILURE);
    }
  }
}

void destroy_cublas_handle() {
  if (g_handle != nullptr) {
    cublasStatus_t status = cublasDestroy(g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to destroy cuBLAS handle: %d", status);
    }
    g_handle = nullptr;
  }
}

// NN: A/B/C All row major
void cublas_tensor_op_nn(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C,  size_t M, size_t N, size_t K) {

  static __nv_bfloat16 alpha = 1.0;
  static __nv_bfloat16 beta = 0.0;

  if (g_handle == nullptr) {
    init_cublas_handle();
  }

  cublasGemmEx(g_handle, 
               CUBLAS_OP_N, 
               CUBLAS_OP_N, 
               N, M, K, 
               &alpha, 
               B, CUDA_R_16BF, N, 
               A, CUDA_R_16BF, K, 
               &beta,  
               C, CUDA_R_16BF, N, 
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
if(((T).options().dtype() != (th_type))) {                   \
  std::cout << "Tensor Info:" << (T).options() << std::endl; \
  throw std::runtime_error("values must be "#th_type);       \
}

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)           \
if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) { \
  throw std::runtime_error("Tensor size mismatch!");  \
}

// NN: A/B/C All row major
void hgemm_cublas_tensor_op_nn(
  torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kBFloat16)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kBFloat16)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kBFloat16)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  cublas_tensor_op_nn(
    reinterpret_cast<__nv_bfloat16*>(a.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(b.data_ptr()),
    reinterpret_cast<__nv_bfloat16*>(c.data_ptr()),
    M, N, K
  );
}
