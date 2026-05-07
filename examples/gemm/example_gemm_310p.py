import argparse
from pathlib import Path

import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

parser = argparse.ArgumentParser(description="310P GEMM kernel source generation")
parser.add_argument("--m", type=int, default=128, help="Matrix M dimension")
parser.add_argument("--n", type=int, default=128, help="Matrix N dimension")
parser.add_argument("--k", type=int, default=64, help="Matrix K dimension")
parser.add_argument("--block-m", type=int, default=128, help="GEMM block M dimension")
parser.add_argument("--block-n", type=int, default=128, help="GEMM block N dimension")
parser.add_argument("--block-k", type=int, default=64, help="GEMM block K dimension")
parser.add_argument(
    "--backend",
    choices=["cube", "scalar"],
    default="cube",
    help="Use cube for the normal GEMM path, or scalar for small CAModel correctness checks.",
)
parser.add_argument("--debug-root", type=str, default="debug_310p_gemm", help="Directory for generated source")
parser.add_argument(
    "--mode",
    choices=["lower", "jit"],
    default="lower",
    help="Use lower for offline source generation, or jit for runtime adapter creation.",
)
args = parser.parse_args()

M = args.m
N = args.n
K = args.k
BLOCK_M = args.block_m
BLOCK_N = args.block_n
BLOCK_K = args.block_k
BACKEND = args.backend

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def matmul_scalar(M, N, K, block_M, block_N, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_UB = T.alloc_ub((block_M, K), dtype)
            B_UB = T.alloc_ub((K, block_N), dtype)
            C_UB = T.alloc_ub((block_M, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M, 0], A_UB)
                T.copy(B[0, by * block_N], B_UB)

                for local_m in T.serial(block_M):
                    for local_n in T.serial(block_N):
                        acc = T.alloc_var(accum_dtype, init=0.0)
                        acc = T.float32(0.0)
                        for kk in T.serial(K):
                            a_value = T.float32(A_UB[local_m, kk])
                            b_value = T.float32(B_UB[kk, local_n])
                            acc = acc + a_value * b_value
                        C_UB[local_m, local_n] = acc

                T.copy(C_UB, C[bx * block_M, by * block_N])

    return main


if BACKEND == "cube":
    program = matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
else:
    program = matmul_scalar(M, N, K, BLOCK_M, BLOCK_N)

if args.mode == "lower":
    debug_root = Path(args.debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    with tilelang.tvm.transform.PassContext(opt_level=3, config=pass_configs):
        artifact = tilelang.lower(program, target="ascendc", platform="310P")
    source_path = debug_root / "tilelang_gemm_310p.cpp"
    source_path.write_text(artifact.kernel_source)
    print(f"lower ok: {source_path}")
    print(artifact.kernel_source)
else:
    jit_matmul = tilelang.jit(
        out_idx=[-1],
        target="ascendc",
        platform="310P",
        pass_configs=pass_configs,
        debug_root_path=args.debug_root,
    )(matmul if BACKEND == "cube" else matmul_scalar)
    if BACKEND == "cube":
        func = jit_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    else:
        func = jit_matmul(M, N, K, BLOCK_M, BLOCK_N)
    print(func.get_kernel_source())
