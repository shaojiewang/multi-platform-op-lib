import torch

def construct(m: int, n: int, k: int):
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()
    return x, y, out, ref_out

def test_gemm():
    for m in (64, 128, 2048):
        for k, n in [(576, 7168), (7168, 2112)]:
            x, y, out, ref_out = construct(m, n, k)
            

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')

    test_gemm()