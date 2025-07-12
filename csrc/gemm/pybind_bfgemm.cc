#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(STRINGFY(func), &func, STRINGFY(func));

// from bfgemm_cublas.cu
void init_cublas_handle();
void destroy_cublas_handle();
void hgemm_cublas_tensor_op_nn(torch::Tensor a, torch::Tensor b, torch::Tensor c); 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // cuBLAS Tensor Cores
  TORCH_BINDING_COMMON_EXTENSION(init_cublas_handle)
  TORCH_BINDING_COMMON_EXTENSION(destroy_cublas_handle)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublas_tensor_op_nn)
}