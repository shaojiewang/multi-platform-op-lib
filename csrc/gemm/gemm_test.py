import torch
import random
import ampere_bfgemm as bfgemm
from utils import cuda_timer

def construct(m: int, n: int, k: int):
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()
    return x, y, out, ref_out

@cuda_timer(repetitions=10, warmup=3)
def call_cublas_bfgemm(x, y, out):
    bfgemm.hgemm_cublas_tensor_op_nn(x, y, out)

@cuda_timer(repetitions=10, warmup=3)
def call_pytorch_bfgemm(x, y, out):
    out = x @ y

def test_gemm():
    for m in (64, 128, 2048):
        for k, n in [(576, 7168), (7168, 2112)]:
            x, y, out, ref_out = construct(m, n, k)
            y_t = y.t()
            print(f"{m=}, {n=}, {k=}")
            call_cublas_bfgemm(x, y_t, out)
            call_pytorch_bfgemm(x, y_t, out)

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')

    bfgemm.init_cublas_handle()

    test_gemm()

    bfgemm.destroy_cublas_handle()
    