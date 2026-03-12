import os
import sys
import torch

# 1. Force use of local repository and the CUTLASS DSL path
REPO_PATH = os.path.expanduser("~/haocizhang/flash-attention")
DSL_PATH = "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/python_packages"

sys.path.insert(0, REPO_PATH)
sys.path.insert(0, DSL_PATH)

import flash_attn.cute.interface as fa_cute

def run_test():
    print(f"--- Running FlashAttention-4 BF16 + Dummy Scales Test (B200) ---")
    
    # Configuration
    batch, seqlen, nheads, d = 1, 512, 16, 128
    device = "cuda"
    dtype = torch.float32
    
    # 2. Setup Inputs: BF16 3D for varlen
    q = torch.randn(batch * seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch * seqlen, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch * seqlen, nheads, d, device=device, dtype=dtype)

    # 3. Setup Output: BF16 (Pre-allocated)
    out_bf16 = torch.empty(batch * seqlen, nheads, d, device=device, dtype=dtype)

    # 4. Dummy Scales (Must be Float32 CUDA Tensors)
    # Even in BF16 mode, we pass these to verify the interface/TVM argument order
    scale_args = {
        "q_scale": torch.ones(1, device=device, dtype=torch.float32),
        "k_scale": torch.ones(1, device=device, dtype=torch.float32),
        "v_scale": torch.ones(1, device=device, dtype=torch.float32),
        "o_scale": torch.ones(1, device=device, dtype=torch.float32),
    }

    # 5. Varlen Metadata
    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    print(f"Dispatching BF16 with Scales to B200...")

    try:
        fa_cute.flash_attn_varlen_func(
            q=q, 
            k=k, 
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            out=out_bf16,
            causal=True,
            **scale_args
        )

        print("\n✅ SUCCESS: BF16 Kernel with scale arguments executed.")
        print(f"Output Mean: {out_bf16.mean().item():.6f}")

    except Exception as e:
        print(f"\n❌ FAILED")
        print(f"Error Message: {e}")

if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    run_test()

