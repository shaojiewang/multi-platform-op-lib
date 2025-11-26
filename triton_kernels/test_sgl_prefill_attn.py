import torch

from sgl_prefill_attn import context_attention_fwd
from typing import Optional
# from sglang.srt.utils import get_device

def get_device(device_id: Optional[int] = None):
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if device_id is None:
            return "cuda"
        return "cuda:{}".format(device_id)

class TestTritonAttention:
    def __init__(self):
        pass

    def _test_context_attention_once(self, head_dim, is_causal):
        # Set up a simple test case
        device = get_device()
        num_heads = 16
        seq_lens = [8160]
        max_seq_len = max(seq_lens)

        total_times = 100
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Create random input tensors
        q = torch.randn(sum(seq_lens), num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(sum(seq_lens), num_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(sum(seq_lens), num_heads, head_dim, dtype=torch.bfloat16, device=device)
        o = torch.zeros(sum(seq_lens), num_heads, head_dim, dtype=torch.bfloat16, device=device)

        # Create b_start_loc and b_seq_len tensors
        b_start_loc = torch.tensor([0, seq_lens[0]], device=device)
        b_seq_len = torch.tensor(seq_lens, device=device)

        for i in range(total_times):
            context_attention_fwd(
                q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
            )

        start_event.record()
        for i in range(total_times):
            context_attention_fwd(
                q, k, v, o, b_start_loc, b_seq_len, max_seq_len, is_causal=is_causal
            )

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event) / total_times
        print(f'Inference time: {elapsed_time:.2f} ms')
        cu_seq_lens = [0] * (len(seq_lens) + 1)
        for i, seq_len in enumerate(seq_lens):
            cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

        for i in range(len(seq_lens)):
            start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
            o_torch = torch.nn.functional.scaled_dot_product_attention(
                q[start:end].permute(1, 0, 2),
                k[start:end].permute(1, 0, 2),
                v[start:end].permute(1, 0, 2),
                is_causal=is_causal,
            ).permute(1, 0, 2)

            cos_sim = torch.nn.functional.cosine_similarity(
                o[start:end].flatten(), o_torch.flatten(), dim=0
            )
            #self.assertTrue(cos_sim.item() > 1 - (1e-5))
            assert(torch.allclose(o[start:end], o_torch, atol=1e-2))

if __name__ == "__main__":
    test = TestTritonAttention()
    hd = 128
    is_causal = False
    
    test._test_context_attention_once(hd, is_causal)


