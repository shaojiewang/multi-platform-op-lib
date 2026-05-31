import torch
import random
import bfgemm_test as bfgemm
from utils import cuda_timer

def construct(m: int, n: int, k: int):
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    out_fp32 = torch.empty((m, n), device='cuda', dtype=torch.float)
    ref_out = x @ y
    return x, y, out, ref_out, out_fp32

@torch.no_grad()
def check_correctness(name, out, ref_out, rtol=1e-2, atol=1e-1):
    torch.cuda.synchronize()

    out_f32 = out.float()
    ref_f32 = ref_out.float()
    diff = torch.nan_to_num((out_f32 - ref_f32).abs(), nan=float("inf"))
    rel_diff = diff / ref_f32.abs().clamp_min(1e-6)

    max_abs = diff.max()
    max_rel = torch.nan_to_num(rel_diff, nan=float("inf")).max()
    ok = torch.allclose(out_f32, ref_f32, rtol=rtol, atol=atol)

    print(
        f"Correctness [{name}]: {ok}, "
        f"max_abs={max_abs.item():.6f}, max_rel={max_rel.item():.6f}, "
        f"rtol={rtol}, atol={atol}"
    )

    if not ok:
        is_close = torch.isclose(out_f32, ref_f32, rtol=rtol, atol=atol)
        mismatch_count = (~is_close).sum().item()
        mismatch_idx = diff.argmax().item()
        row = mismatch_idx // ref_out.size(1)
        col = mismatch_idx % ref_out.size(1)
        out_val = out_f32[row, col].item()
        ref_val = ref_f32[row, col].item()
        raise AssertionError(
            f"{name} mismatch_count={mismatch_count}, "
            f"first_max_diff_at=({row}, {col}), "
            f"out={out_val:.6f}, ref={ref_val:.6f}, "
            f"abs_diff={max_abs.item():.6f}, rel_diff={max_rel.item():.6f}"
        )

@cuda_timer(sync=False, repetitions=10, warmup=10)
def call_wgmma_bfgemm_torch(x, y, out):
    bfgemm.bfgemm_torch(x, y, out)

@cuda_timer(sync=False, repetitions=10, warmup=10)
def call_cublas_bfgemm(x, y, out):
    bfgemm.hgemm_cublas_tensor_op_nn(x, y, out)

@cuda_timer(sync=False, repetitions=10, warmup=10)
def call_pytorch_bfgemm(x, y, out):
    return x @ y

def test_gemm():
    for m in (4096,):
        for k, n in [(7168, 2048)]:
            x, y, out, ref_out, out_fp32 = construct(m, n, k)
            #y_t = y.t().contiguous()
            print(f"{m=}, {n=}, {k=}")
            #call_cublas_bfgemm(x, y_t, out)
            #check_correctness("cuBLAS", out, ref_out)
            pytorch_out, _ = call_pytorch_bfgemm(x, y, out)
            check_correctness("PyTorch", pytorch_out, ref_out)
            call_wgmma_bfgemm_torch(x, y, out_fp32)
            check_correctness("WGMMA", out_fp32, ref_out)

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')

    bfgemm.init_cublas_handle()

    test_gemm()

    bfgemm.destroy_cublas_handle()
    
