# Written by Dr. Hicham Badri @Mobius Labs GmbH - 2024
#********************************************************
import torch, math, random, copy
from torch import Tensor
import triton
import triton.language as tl

from .config import AUTOTUNE_ENABLE
from .utils import *

KEYS        = ['M', 'N', 'K', 'group_size', 'elements_per_sample']
MATMUL_TYPE = "GEMM_SPLITK"

def kernel_config_pruner(configs, nargs, **kwargs):
    global KEYS
    from ..core import GEMLITE_TRITON_CONFIG_CACHE

    m = 2 ** int(math.ceil(math.log2(nargs['M'])))
    n = nargs['N'] 
    k = nargs['K'] 
    g = nargs['group_size']
    e = nargs['elements_per_sample']

    #Check cache
    if(MATMUL_TYPE in GEMLITE_TRITON_CONFIG_CACHE):
        _signature = str(tuple([m, n, k, g, e]))
        if(_signature in GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE]):
            _config     = copy.deepcopy(GEMLITE_TRITON_CONFIG_CACHE[MATMUL_TYPE][_signature])
            _num_stages = _config.pop('num_stages')
            _num_warps  = _config.pop('num_warps')
            _num_ctas   = _config.pop('num_ctas')

            yield triton.Config(_config,
                num_stages=_num_stages,
                num_warps=_num_warps,
                pre_hook=init_to_zero("c_ptr") if (_config['SPLIT_K'] > 1) else None,
            )

            return

    used = set()
    for config in configs:
        group_size_m = config.kwargs['GROUP_SIZE_M']
        block_size_m = config.kwargs['BLOCK_SIZE_M']
        block_size_n = config.kwargs['BLOCK_SIZE_N']
        block_size_k = config.kwargs['BLOCK_SIZE_K']
        split_k      = config.kwargs['SPLIT_K']

        #Makes autotuning faster: mainly for batch-size 1 .. 64
        if(m <= 16): block_size_m = 16
        if(m >= 32): block_size_m = min(max(block_size_m, 16), 32) #[16, 32]
        if(m >= 64): block_size_m = min(max(block_size_m, 32), 64) #[32, 64]

        #Only use higher split_k values for smaller m
        if(m >= 32): split_k = min(split_k, 8)
        #Only use lower split_k values for larger m
        if(m <= 16): split_k = max(split_k, 2)

        #Filter 
        block_area = block_size_k * block_size_n
        if(block_area > 4096 * 8): #Limit area for faster autotuning. Use for more 4096 * 8
            continue

        #Constraints
        #BLOCK_SIZE_K >= group_size
        block_size_k = min(block_size_k, g)

        #K needs to be devisible by BLOCK_SIZE_K * SPLIT_K 
        if(not is_divisible(k, block_size_k * split_k)):
            continue

        num_warps  = config.num_warps
        num_stages = config.num_stages

        #if(e > 1): num_stages = 1 #TODO: Remove this after fix
        if(e == 1 and num_stages == 1): continue #skip num_stages=1 for non-packed weights

        A_load_order      = config.kwargs['A_load_order']
        meta_evict_policy = config.kwargs['meta_evict_policy']
        atomic_mode       = config.kwargs['atomic_mode']

        _key = (block_size_m, block_size_n, block_size_k, group_size_m, split_k, 
                A_load_order, meta_evict_policy, atomic_mode,
                config.num_stages, config.num_warps,
                )
        
        if _key in used:
            continue

        used.add(_key)
        yield triton.Config(
            {
                'BLOCK_SIZE_M': block_size_m,
                'BLOCK_SIZE_N': block_size_n,
                'BLOCK_SIZE_K': block_size_k,
                'GROUP_SIZE_M': group_size_m,
                'SPLIT_K'     : split_k,

                'A_load_order'      : A_load_order,
                'meta_evict_policy' : meta_evict_policy,
                'atomic_mode'       : atomic_mode,
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps,
            pre_hook=init_to_zero("c_ptr") if (split_k > 1) else None, 
        )

#These autotunes are optimized for batch-size 1 to 64 (!)
def get_autotune_config():
    _stages  = [1, 2, 4, 5] if gpu_has_more_shared_memory() else [1, 2, 4]
    _configs = []
    for _M in [16, 32, 64]: #for better performance at batch-sizes [4-64]
        for _N in [32, 64, 128, 256]:
            for _K in [32, 64, 128, 256]:
                for _w in [4, 8]:
                    for _s in _stages:
                        for _sK in [1, 2, 4, 8, 16]: 
                            for _a_load_order in [0, 2]: #[0, 2], [0, 1, 2, 3] - [2] for 4090, [0]: for A100/H100
                                for _meta_evict_policy in ['']: #[', 'evict_last']
                                    for _atomic_mode in ['relaxed']: #['release', 'relaxed']:
                                        _configs.append(
                                                triton.Config(
                                                    {'BLOCK_SIZE_M': _M, 'BLOCK_SIZE_N': _N, 'BLOCK_SIZE_K': _K, 'GROUP_SIZE_M': 8, 'SPLIT_K': _sK,
                                                    'A_load_order': _a_load_order, 'meta_evict_policy': _meta_evict_policy, 'atomic_mode': _atomic_mode,
                                                    }, 
                                                    num_stages=_s, num_warps=_w,
                                                    pre_hook=init_to_zero("c_ptr") if (_sK > 1) else None,
                                                    )
                                                )
    return _configs


compute_capability = torch.cuda.get_device_capability(0)

#Optimized for low-batch size decoding: K needs to be divisible by BLOCK_SIZE_K * SPLIT_K = 256 !!!
def get_default_config():
    #4090: default
    config = triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':32, 'BLOCK_SIZE_K':32, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 
                           'A_load_order':2, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                            num_warps=4, num_stages=1, pre_hook=init_to_zero("c_ptr"))

    if(compute_capability == (8, 0)): #A100
        config = triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 
                             'A_load_order':0, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                             num_warps=4, num_stages=1, pre_hook=init_to_zero("c_ptr"))

    if(compute_capability == (9, 0)): #H100
        config = triton.Config({'BLOCK_SIZE_M':16, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32, 'SPLIT_K':2, 'GROUP_SIZE_M':8, 
                             'A_load_order':0, 'meta_evict_policy':'', 'atomic_mode':'relaxed'}, 
                             num_warps=4, num_stages=1, pre_hook=init_to_zero("c_ptr"))

    return [config]

ENABLE_AUTOTUNE = AUTOTUNE_ENABLE.GEMM_SPLITK

# @autotune(
#     configs=get_autotune_config() if ENABLE_AUTOTUNE else get_default_config(),
#     key = ['M', 'N', 'K', 'group_size', 'elements_per_sample'],
#     prune_configs_by = {'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
#     warmup = 50, 
#     rep = 50,
#     use_cuda_graph = AUTOTUNE_ENABLE.USE_CUDA_GRAPH,
# )


@triton.jit
def gemm_splitK_A16fWnO16f_int32packing_kernel(
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr, scales_a_ptr,
    M, N, K, 
    ######### Quant parms #########
    W_nbits: tl.constexpr, 
    group_size: tl.constexpr, 
    unpack_mask: tl.constexpr, 
    elements_per_sample: tl.constexpr, 
    ######### Strides #########
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_meta_g, stride_meta_n,
    ######### Dtypes #########
    input_dtype: tl.constexpr,
    output_dtype: tl.constexpr,
    acc_dtype: tl.constexpr,
    meta_dtype: tl.constexpr,
    ######### Meta-data mode #########
    channel_scale_mode: tl.constexpr,
    W_group_mode: tl.constexpr,
    zero_is_scalar: tl.constexpr,
    ######### tuning params #########
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, SPLIT_K: tl.constexpr,
    A_load_order: tl.constexpr, meta_evict_policy: tl.constexpr, atomic_mode: tl.constexpr,
    data_contiguous: tl.constexpr,
):
    """
    Based on https://github.com/foundation-model-stack/foundation-model-stack/blob/triton/triton/kernels/gptq/splitk_dequant_gemm.py
    GEMM for C = matmul(A, dequantize(B, scales, zeros))
    A is of shape (M, K): float16 or bfloat16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16 or bfloat16 depending on the input A
    scales and zeros is of shape (group_size, N): float16 or bfloat16

    BLOCK_SIZE_M >=16
    BLOCK_SIZE_K * SPLIT_K <= group_size for imp1
    BLOCK_SIZE_K == SPLIT_K for imp2 (similar to original)
    """

    pid   = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    #Swizzle?
    #pid_m, pid_n = linear_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, None)
    pid_m, pid_n = swizzle_tile(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    #Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) 

    #Vectorized coalesced load
    ##############################
    offs_am = offs_m
    offs_ak = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)

    if(data_contiguous):
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N) 
        offs_bk = offs_k
    else:
        offs_bn = offs_n
        offs_bk = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    ###############################

    b_ptrs  = b_ptr + ((offs_bk[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn) 
    q_shift = ((offs_bk % elements_per_sample) * W_nbits).to(tl.int32)[:, None] 

    #Inputs
    a_ptrs  = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)  
    a_mask  = offs_am[:, None] < M
    
    #Meta data stuff
    scales_ptrs = scales_ptr + offs_bn[None, :] * stride_meta_n
    zeros_ptrs  = zeros_ptr  + offs_bn[None, :] * stride_meta_n

    stride_mul: tl.constexpr     = BLOCK_SIZE_K / group_size
    BLOCK_SIZE_K_U: tl.constexpr = BLOCK_SIZE_K   * SPLIT_K
    BLOCK_SIZE_K_P: tl.constexpr = (BLOCK_SIZE_K // elements_per_sample) * SPLIT_K

    if(zero_is_scalar):
        zero_scalar = tl.load(zeros_ptr, eviction_policy='evict_last')
    ####################################################################################
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(num_pid_k):

        if(A_load_order == 0): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') 

        b = tl.load(b_ptrs, eviction_policy='evict_first')

        if(A_load_order == 1): #Early load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last') 
        
        #Meta-data loading policy
        if(W_group_mode > 0):
            k_m = ((k * SPLIT_K + pid_k) * stride_mul).to(tl.int32) 

        if(W_group_mode >= 2): #[2, 3, 4]
            scales = tl.load(scales_ptrs + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            scales = None

        if(W_group_mode == 1 or W_group_mode >= 3): #[1, 3, 4]
            if(zero_is_scalar):
                zeros = zero_scalar
            else:
                zeros = tl.load(zeros_ptrs  + k_m * stride_meta_g, eviction_policy=meta_evict_policy) 
        else:
            zeros = None
        
        if(A_load_order == 2): #Mid load
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')

        # Unpack and dequantize
        b = dequantize(b, scales, zeros, q_shift, meta_dtype, unpack_mask, elements_per_sample, W_group_mode, zero_is_scalar)
        #if(elements_per_sample > 1): b = b.to(tl.float32) #hack to enable pipelining with for-loop on triton==3.1.0

        if(A_load_order == 3): #Late load 
            a = tl.load(a_ptrs, mask=a_mask, other=0., eviction_policy='evict_last')
        
        #Dot
        acc = tl.dot(a, b.to(input_dtype), acc=acc, out_dtype=acc_dtype, input_precision="tf32") 
        
        #Advance
        a_ptrs += BLOCK_SIZE_K_U * stride_ak
        b_ptrs += BLOCK_SIZE_K_P * stride_bk

    ##################################################################
    #Channel-wise scaling
    if(channel_scale_mode == 1): #weight-only
        scales_b = tl.load(scales_ptr + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * scales_b[None, :]

    if(channel_scale_mode == 2): #activation-only
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.full((BLOCK_SIZE_N,), value=1, dtype=meta_dtype)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    if(channel_scale_mode == 3): #weight + activation
        scales_a = tl.load(scales_a_ptr + offs_am, mask=offs_am < M, other=1, eviction_policy=meta_evict_policy)
        scales_b = tl.load(scales_ptr   + offs_bn, mask=offs_bn < N, other=1, eviction_policy=meta_evict_policy)
        acc      = acc.to(meta_dtype) * (scales_a[:, None] * scales_b[None, :])

    acc = acc.to(output_dtype)
    ##################################################################

    #Output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs  = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    if(SPLIT_K > 1):
        tl.atomic_add(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N), sem=atomic_mode) #release / relaxed
    else:
        tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N)) 


_costum_op_id = '_' + str(int(random.random()*10000))


QWEN_2_5_32B_TP1 = {
    (4, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (4, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (4, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (4, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (4, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (4, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (2, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (2, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (2, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (2, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (2, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (2, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (27, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (29, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (3, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (3, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (3, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (3, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (3, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (3, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (5, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (6, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (6, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (6, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (6, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (6, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (6, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (7, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (7, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (7, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (7, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (7, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (7, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (8, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (8, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (8, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (8, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (8, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (8, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (9, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (9, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (9, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (9, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (9, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (9, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (10, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (10, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (10, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (10, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (10, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (10, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (11, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (12, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (13, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (14, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (15, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (16, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (17, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (17, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (17, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (17, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (17, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (17, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (18, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (18, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (18, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (18, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (18, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (18, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (19, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (20, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (20, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (20, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (20, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (20, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (20, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (21, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (21, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (21, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (21, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (21, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (21, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (22, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (22, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (22, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (22, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (22, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (22, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (23, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (23, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (23, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (23, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (23, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (23, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (24, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (24, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (24, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (24, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (24, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (24, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (25, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (25, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (25, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (25, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (25, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (25, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (26, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (26, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (26, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (26, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (26, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (26, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (28, 7168, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (28, 5120, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (28, 55296, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (28, 5120, 27648, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (28, 55296, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (28, 5120, 27648, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))]
}


QWEN_2_5_32B_TP2 = {
    (4, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (4, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (4, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (4, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (4, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (4, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (2, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (2, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (2, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (2, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (2, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (2, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (27, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (27, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (29, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (29, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (30, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (31, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (32, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (3, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (3, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (3, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (3, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (3, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (3, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (5, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (5, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (6, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (6, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (6, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (6, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (6, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (6, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (7, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 2, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (7, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (7, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (7, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (7, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (7, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (8, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (8, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (8, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (8, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (8, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (8, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (9, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (9, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (9, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (9, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (9, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (9, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (10, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (10, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (10, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (10, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (10, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (10, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (11, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (11, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (12, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (12, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (13, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (13, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (14, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (14, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (15, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (15, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (16, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (16, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (17, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (17, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (17, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (17, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (17, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (17, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (18, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (18, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (18, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (18, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (18, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (18, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (19, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (19, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (20, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (20, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (20, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (20, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (20, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (20, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (21, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (21, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (21, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (21, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (21, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (21, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (22, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (22, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (22, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (22, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (22, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (22, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (23, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (23, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (23, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (23, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (23, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (23, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (24, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (24, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (24, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (24, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (24, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (24, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (25, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (25, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (25, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (25, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (25, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (25, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (26, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (26, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (26, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (26, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (26, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (26, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (28, 3584, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=2, pre_hook=init_to_zero("c_ptr"))],
    (28, 5120, 2560, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (28, 27648, 5120, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 4, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (28, 5120, 13824, 128, 8) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (28, 27648, 5120, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=2, num_warps=4, pre_hook=init_to_zero("c_ptr"))],
    (28, 5120, 13824, 32, 16) : [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 8, 'A_load_order': 2, 'meta_evict_policy': '', 'atomic_mode': 'relaxed'}, num_stages=1, num_warps=2, pre_hook=init_to_zero("c_ptr"))]
}

OPT_CONFIGS = {**QWEN_2_5_32B_TP1, **QWEN_2_5_32B_TP2}

@torch.library.custom_op("gemlite::gemm_splitK_A16fWnO16f_int32packing_forward" + _costum_op_id, mutates_args=())
def gemm_splitK_A16fWnO16f_int32packing_forward(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                                W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int,
                                                input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                                                channel_scale_mode: int, W_group_mode: int, data_contiguous: bool,
                                                ) -> Tensor: 
    
    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]

    #assert K == W_q.shape[0] * elements_per_sample, "Invalid Input Shapes"
    output = torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), META['SPLIT_K'])

    config = OPT_CONFIGS.get((M, N, K, group_size, elements_per_sample), None)
    if config is None:
        autotune_fn = triton.autotune(
            configs = get_autotune_config() if ENABLE_AUTOTUNE else get_default_config(),
            key = ['M', 'N', 'K', 'group_size', 'elements_per_sample'],
            prune_configs_by = {'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
            warmup = 50, 
            rep = 50,
            use_cuda_graph = AUTOTUNE_ENABLE.USE_CUDA_GRAPH)
    else:
        autotune_fn = triton.autotune(
            configs = config,
            key = ['M', 'N', 'K', 'group_size', 'elements_per_sample'],
            prune_configs_by = {'early_config_prune': kernel_config_pruner} if ENABLE_AUTOTUNE else None,
            warmup = 50, 
            rep = 50,
            use_cuda_graph = AUTOTUNE_ENABLE.USE_CUDA_GRAPH)

    kernel = autotune_fn(gemm_splitK_A16fWnO16f_int32packing_kernel)
    kernel[grid](
        x, W_q, output,
        scales, zeros, scales_x,
        M, N, K,
        W_nbits, group_size, unpack_mask, elements_per_sample,  
        x.stride(0), x.stride(1),
        W_q.stride(0), W_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        ########################
        input_dtype  = DTYPE_TO_TRITON[input_dtype],
        output_dtype = DTYPE_TO_TRITON[output_dtype],
        acc_dtype    = DTYPE_TO_TRITON[acc_dtype],
        meta_dtype   = DTYPE_TO_TRITON[meta_dtype],
        ########################
        channel_scale_mode = channel_scale_mode,
        W_group_mode       = W_group_mode,
        zero_is_scalar     = zeros.numel() == 1,
        data_contiguous    = data_contiguous,
    )

    return output

@torch.library.register_fake("gemlite::gemm_splitK_A16fWnO16f_int32packing_forward" + _costum_op_id)
def gemm_splitK_A16fWnO16f_int32packing_forward_fake(x: Tensor, W_q: Tensor, scales: Tensor, zeros: Tensor, scales_x: Tensor,
                                              W_nbits: int, group_size: int, unpack_mask: int, elements_per_sample: int, 
                                              input_dtype: int, output_dtype: int, acc_dtype: int, meta_dtype:int, 
                                              channel_scale_mode: int, W_group_mode: int, data_contiguous: bool,
                                              ) -> Tensor:
    
    M, K, N = x.shape[0], x.shape[1], W_q.shape[1]
    return torch.empty((M, N), device=W_q.device, dtype=DTYPE_TO_TORCH[output_dtype])


class gemm_splitK_A16fWnO16f:
    kernel = gemm_splitK_A16fWnO16f_int32packing_kernel
    forward = gemm_splitK_A16fWnO16f_int32packing_forward
    matmul_type = MATMUL_TYPE

__all__ = ["gemm_splitK_A16fWnO16f"]
