import torch
import random
import bfgemm_test as bfgemm
from utils import cuda_timer

def construct(m: int, n: int, k: int):
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    out_fp32 = torch.empty((m, n), device='cuda', dtype=torch.float)
    ref_out = x @ y.t()
    return x, y, out, ref_out, out_fp32

@cuda_timer(sync=False, repetitions=10, warmup=10)
def call_wgmma_bfgemm_torch(x, y, out):
    bfgemm.bfgemm_torch(x, y, out)

@cuda_timer(sync=False, repetitions=10, warmup=10)
def call_cublas_bfgemm(x, y, out):
    bfgemm.hgemm_cublas_tensor_op_nn(x, y, out)

@cuda_timer(sync=False, repetitions=10, warmup=10)
def call_pytorch_bfgemm(x, y, out):
    out = x @ y

def test_gemm():
    for m in (4096,):
        for k, n in [(7168, 2112)]:
            x, y, out, ref_out, out_fp32 = construct(m, n, k)
            y_t = y.t()
            print(f"{m=}, {n=}, {k=}")
            call_cublas_bfgemm(x, y_t, out)
            call_pytorch_bfgemm(x, y_t, out)
            call_wgmma_bfgemm_torch(x, y_t, out_fp32)

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')

    bfgemm.init_cublas_handle()

    test_gemm()

    bfgemm.destroy_cublas_handle()
    
