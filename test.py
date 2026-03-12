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
    print(f"--- Running FlashAttention-4 FP8 -> BF16 Test (B200) ---")
    print(f"Loading module from: {fa_cute.__file__}")
    
    # Configuration: Batch=1
    batch = 1
    seqlen = 512
    nheads = 16
    d = 128
    device = "cuda"
    
    # 2. Setup Inputs: FP8 (e4m3fn)
    # Shape for varlen: (Total_Tokens, nheads, d) -> (1 * 512, 16, 128)
    q = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(-1, nheads, d)
    k = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(-1, nheads, d)
    v = torch.randn(batch, seqlen, nheads, d, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(-1, nheads, d)

    # 3. Setup Output: BF16
    out_bf16 = torch.empty(batch * seqlen, nheads, d, device=device, dtype=torch.bfloat16)

    # 4. Scale Factors (B200 requirement for FP8)
    q_scale = torch.ones(1, device=device, dtype=torch.float32)
    k_scale = torch.ones(1, device=device, dtype=torch.float32)
    v_scale = torch.ones(1, device=device, dtype=torch.float32)
    o_scale = torch.ones(1, device=device, dtype=torch.float32)
    # 5. Varlen Metadata (Cumulative lengths for 1 sequence)
    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)

    print(f"Shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Output Target Dtype: {out_bf16.dtype}")

    #try:
    # Dispatch to Blackwell kernels
    fa_cute.flash_attn_varlen_func(
        q=q, 
        k=k, 
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_scale=o_scale,
        out=out_bf16,
        causal=False
    )

    print("\n✅ SUCCESS: Blackwell Kernel Executed.")
    # Reshape to (Batch, Seq, Head, Dim) for verification
    final_out = out_bf16.view(batch, seqlen, nheads, d)
    print(f"Final Shape: {final_out.shape}")
    print(f"Sample Mean: {final_out.mean().item():.6f}")

    """
    except Exception as e:
        print(f"\n❌ FAILED")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        
        # Verify if the signature we are calling matches your local edits
        import inspect
        print(f"\nDetected Function Signature:\n{inspect.signature(fa_cute.flash_attn_varlen_func)}")
    """
if __name__ == "__main__":
    # Use blocking to catch asynchronous CUDA errors immediately
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    run_test()

