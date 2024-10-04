# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math
import triton
import triton.language as tl

import os
os.environ["TRITON_DEJAVU_STORAGE"] = "/workspace/.cache/triton_dejavu"
import triton_dejavu
autotune = triton_dejavu.autotune
autotune = triton.autotune

# code based https://github.com/fpgaminer/GPTQ-triton
def kernel_config_pruner(configs, nargs, **kwargs):
    m = max(2 ** int(math.ceil(math.log2(nargs['M']))), 16) #Need at least 16 here for tl.dot
    n = max(2 ** int(math.ceil(math.log2(nargs['N']))), 16)
    k = max(2 ** int(math.ceil(math.log2(nargs['K']))), 16)
    g = nargs['group_size']

    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = min(m, config.kwargs['BLOCK_SIZE_M'])
        block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
        block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
        block_size_k = min(block_size_k, g) #Makes BLOCK_SIZE_K compatible with the group_size
        
        if (block_size_m, block_size_n, block_size_k, group_size_m, config.num_stages, config.num_warps) in used:
            continue

        used.add((block_size_m, block_size_n, block_size_k, group_size_m, config.num_stages, config.num_warps))
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
                'BLOCK_SIZE_K': block_size_k,
                'GROUP_SIZE_M': group_size_m
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps
        )

def get_gemm_config():
    #Tuned on 4090 RTX
    _configs = []
    for _M in [16, 32, 64, 128, 256, 512]: #might need higher values for larger batch-sizes
        for _N in [32, 64, 128]: 
            for _K in [32, 64, 128]: #[32, 64, 128], 32 <= block_size
                for _w in [2, 4]: 
                    for _s in [2, 4]:
                        _configs.append(
                                triton.Config(
                                    {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 'GROUP_SIZE_M': 8}, 
                                    num_stages=_s, num_warps=_w)
                                )
    return _configs

def dummy_config():
    return [triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2)]

# @triton.heuristics(values={'CLOSEST_M': lambda args: 2 ** int(math.ceil(math.log2(args['M'])))})
# @autotune(
#     configs = get_gemm_config(),
#     key=['CLOSEST_M', 'N', 'K', 'group_size', 'W_nbits'],
#     prune_configs_by={
#         'early_config_prune': kernel_config_pruner,
#     },
#     warmup=200, 
#     rep=50, #20 for faster tuning 
# )

@triton.jit
def gemm_A16fWnO16f_int32packing_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K, 
    W_nbits: tl.constexpr, group_size: tl.constexpr, unpack_mask: tl.constexpr, elements_per_sample: tl.constexpr, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta, 
    acc_dtype: tl.constexpr,
    CLOSEST_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    num_stages: tl.constexpr
):
    """
    Based on https://github.com/fpgaminer/GPTQ-triton
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//8, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K <= group_size
    """

    pid       = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_SIZE_M
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m            = first_pid_m + (pid % group_size_m)
    pid_n            = (pid % num_pid_in_group) // group_size_m

    #Offsets
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k  = tl.arange(0, BLOCK_SIZE_K)

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  
    a_mask  = (offs_am[:, None] < M)
    b_ptrs  = b_ptr + ((offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 

    #Output
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]

    #Meta data stuff
    q_shifts    = ((offs_k  % elements_per_sample) * W_nbits).to(tl.int32)[:, None]
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :]
    stride_mul  = BLOCK_SIZE_K / group_size 

    ####################################################################################
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype) 

    for k in tl.range(0, num_pid_k, 1, num_stages=1):
        b = tl.load(b_ptrs, eviction_policy='evict_first') #(BLOCK_SIZE_K, BLOCK_SIZE_N) - repeated over K dim

        k_m    = (k * stride_mul).to(tl.int32)
        scales = tl.load(scales_ptrs + k_m * stride_meta)
        zeros  = tl.load(zeros_ptrs  + k_m * stride_meta)

        a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') #(BLOCK_SIZE_M, BLOCK_SIZE_K)

        # Unpack and dequantize
        b = ((b >> q_shifts) & unpack_mask).to(a.dtype)
        b = ((b - zeros) * scales)
        
        #Dot
        acc = tl.dot(a, b, acc=acc, out_dtype=acc_dtype, input_precision="ieee") #(BLOCK_SIZE_M, BLOCK_SIZE_N)

        #Advance
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // elements_per_sample) * stride_bk

    tl.store(c_ptrs, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

# Llama-70B tp=2
LLAMA_3_1_70B_TP2 = {
    (4096, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4096, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4096, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4096, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4096, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4096, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1024, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1024, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1024, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1024, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1024, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1024, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2048, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2048, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2048, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2048, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2048, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2048, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (8, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (8, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (8, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (8, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (8, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (8, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (16, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (16, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (16, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (16, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (16, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (16, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (32, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (32, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (32, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (32, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (32, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (32, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (64, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (64, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (128, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (128, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (128, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (128, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (128, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (128, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (256, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (256, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (256, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (256, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (256, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (256, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (512, 5120, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (512, 8192, 4096, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (512, 28672, 8192, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (512, 8192, 14336, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (512, 28672, 8192, 32, 2): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (512, 8192, 14336, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)]
}

# Qwen32b tp=1
QWEN_2_5_32B_TP1 = {
    (4096, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4096, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4096, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4096, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4096, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4096, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (4, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (4, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1024, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1024, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1024, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1024, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (1024, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (1024, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (2048, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2048, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2048, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2048, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2048, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (2048, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (32, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (32, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (32, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (32, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (32, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (32, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (8, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (8, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (8, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (8, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (8, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (8, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (16, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (16, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (16, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (16, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (16, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (16, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (64, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (64, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (128, 7168, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (128, 5120, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (128, 55296, 5120, 128, 4): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=4, maxnreg=None)],
    (128, 5120, 27648, 128, 4): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (128, 55296, 5120, 32, 2): [triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)],
    (128, 5120, 27648, 32, 2): [triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_ctas=1, num_stages=2, maxnreg=None)]
}

OPT_CONFIGS = {**LLAMA_3_1_70B_TP2, **QWEN_2_5_32B_TP1}

def gemm_A16fWnO16f_int32packing_forward(x, W_q, scales, zeros, W_nbits, group_size, unpack_mask, elements_per_sample, acc_dtype=tl.float16):
    output = torch.empty((x.shape[0], W_q.shape[1]), device=W_q.device, dtype=scales.dtype)

    #assert x.shape[1] == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"

    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    grid = lambda META: (
        triton.cdiv(x.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(W_q.shape[1], META['BLOCK_SIZE_N']),
    )
    
    CLOSEST_M = 2 ** int(math.ceil(math.log2(M)))
    config = OPT_CONFIGS.get((CLOSEST_M, N, K, group_size, W_nbits), None)
    if config is None:
        autotune_fn = triton_dejavu.autotune(
            configs=get_gemm_config(),
            key=['CLOSEST_M', 'N', 'K', 'group_size', 'W_nbits'],
            warmup=200,
            rep=100,
            prune_configs_by={
                'early_config_prune': kernel_config_pruner,
            },
        )
    else:
        autotune_fn = triton.autotune(
            configs=config,
            key=['CLOSEST_M', 'N', 'K', 'group_size', 'W_nbits'],
            warmup=200,
            rep=100,
            prune_configs_by={
                'early_config_prune': kernel_config_pruner,
            },
        )
    
    heuristic_fn = triton.heuristics(values={'CLOSEST_M': lambda args: 2 ** int(math.ceil(math.log2(args['M'])))})
    kernel = heuristic_fn(autotune_fn(gemm_A16fWnO16f_int32packing_kernel))
    kernel[grid](
        x, W_q, output,
        scales, zeros, 
        x.shape[0], W_q.shape[1], x.shape[1], 
        W_nbits, group_size, unpack_mask, elements_per_sample,  
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0),
        acc_dtype,
    )

    return output

class gemm_A16fWnO16f_int32packing:
    kernel = gemm_A16fWnO16f_int32packing_kernel
    forward = gemm_A16fWnO16f_int32packing_forward
    matmul_type = "GEMM"

__all__ = ["gemm_A16fWnO16f_int32packing"]