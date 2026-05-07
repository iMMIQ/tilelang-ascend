#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse
import importlib.util
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.tools import ascend310p_gemm_camodel as camodel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_stage8_camodel"

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"


@dataclass(frozen=True)
class Stage8Case:
    name: str
    program_factory: Callable[[], object]


def _moe_token_unpermute_shape_program():
    num_tokens, topk, hidden_size, experts = 4, 2, 16, 8

    @T.prim_func
    def main(
        PermTokens: T.Tensor((experts, hidden_size), "float16"),
        SortedIdx: T.Tensor((1, num_tokens * topk), "int32"),
        Probs: T.Tensor((1, num_tokens * topk), "float16"),
        Out: T.Tensor((num_tokens, hidden_size), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            idx_ub = T.alloc_ub((1, topk), "int32")
            probs_ub = T.alloc_ub((1, topk), "float16")
            probs_f32 = T.alloc_ub((1, topk), "float32")
            row_buf = T.alloc_ub((topk, hidden_size), "float16")
            row_tmp = T.alloc_ub((1, hidden_size), "float16")
            row_f32 = T.alloc_ub((1, hidden_size), "float32")
            acc_buf = T.alloc_ub((1, hidden_size), "float32")
            out_buf = T.alloc_ub((1, hidden_size), "float16")

            with T.Scope("V"):
                for token in T.serial(num_tokens):
                    T.copy(SortedIdx[0, token * topk], idx_ub)
                    T.copy(Probs[0, token * topk], probs_ub)
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    T.tile.cast(probs_f32, probs_ub, CAST_LOW2HIGH, topk)
                    T.tile.fill(acc_buf, 0.0)

                    for lane in T.serial(topk):
                        src = idx_ub[0, lane]
                        T.copy(PermTokens[src, 0], row_buf[lane, :])
                    T.set_flag("mte2", "v", 1)
                    T.wait_flag("mte2", "v", 1)

                    for lane in T.serial(topk):
                        T.copy(row_buf[lane, :], row_tmp)
                        T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, hidden_size)
                        T.tile.axpy(acc_buf, row_f32, probs_f32[0, lane])

                    T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, hidden_size)
                    T.pipe_barrier("v")
                    T.copy(out_buf, Out[token, 0])
                    T.pipe_barrier("mte3")

    return main


def _moe_token_permute_grad_shape_program():
    num_tokens, topk, hidden_size = 4, 2, 16

    @T.prim_func
    def main(
        PermGrad: T.Tensor((num_tokens * topk, hidden_size), "float16"),
        SortedIdx: T.Tensor((1, num_tokens * topk), "int32"),
        InputGrad: T.Tensor((num_tokens, hidden_size), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            idx_ub = T.alloc_ub((1, topk), "int32")
            row_buf = T.alloc_ub((topk, hidden_size), "float16")
            row_tmp = T.alloc_ub((1, hidden_size), "float16")
            row_f32 = T.alloc_ub((1, hidden_size), "float32")
            acc_buf = T.alloc_ub((1, hidden_size), "float32")
            out_buf = T.alloc_ub((1, hidden_size), "float16")

            with T.Scope("V"):
                for token in T.serial(num_tokens):
                    T.copy(SortedIdx[0, token * topk], idx_ub)
                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)
                    T.tile.fill(acc_buf, 0.0)

                    for lane in T.serial(topk):
                        src = idx_ub[0, lane]
                        T.copy(PermGrad[src, 0], row_buf[lane, :])
                    T.set_flag("mte2", "v", 1)
                    T.wait_flag("mte2", "v", 1)

                    for lane in T.serial(topk):
                        T.copy(row_buf[lane, :], row_tmp)
                        T.tile.cast(row_f32, row_tmp, CAST_LOW2HIGH, hidden_size)
                        T.tile.add(acc_buf, acc_buf, row_f32)

                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.tile.cast(out_buf, acc_buf, CAST_HIGH2LOW, hidden_size)
                    T.copy(out_buf, InputGrad[token, 0])
                    T.pipe_barrier("mte3")

    return main


def _aclgraph_rms_rope_shape_program():
    rows, head_dim, rope_dim = 2, 16, 16
    eps = 1.0e-5

    @T.prim_func
    def main(
        X: T.Tensor((rows, head_dim), "float16"),
        Sin: T.Tensor((1, rope_dim), "float16"),
        Cos: T.Tensor((1, rope_dim), "float16"),
        Out: T.Tensor((rows, head_dim), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            x_half = T.alloc_shared((rows, head_dim), "float16")
            x_f32 = T.alloc_shared((rows, head_dim), "float32")
            square = T.alloc_shared((rows, head_dim), "float32")
            rms = T.alloc_shared((rows,), "float32")

            T.copy(X[0, 0], x_half)
            T.copy(x_half, x_f32)
            T.tile.mul(square, x_f32, x_f32)
            T.reduce_sum(square, rms, dim=-1)
            T.tile.div(rms, rms, head_dim)
            T.tile.add(rms, rms, eps)
            T.tile.sqrt(rms, rms)
            for row in T.serial(rows):
                T.tile.div(x_f32[row, :], x_f32[row, :], rms[row])

            sin_half = T.alloc_shared((1, rope_dim), "float16")
            cos_half = T.alloc_shared((1, rope_dim), "float16")
            sin = T.alloc_shared((1, rope_dim), "float32")
            cos = T.alloc_shared((1, rope_dim), "float32")
            T.copy(Sin[0, 0], sin_half)
            T.copy(Cos[0, 0], cos_half)
            T.copy(sin_half, sin)
            T.copy(cos_half, cos)

            idx_i32 = T.alloc_shared((rows, rope_dim), "int32")
            idx_i16 = T.alloc_shared((rows, rope_dim), "int16")
            ones_i16 = T.alloc_shared((rows, rope_dim), "int16")
            mask_i16 = T.alloc_shared((rows, rope_dim), "int16")
            mask_f32 = T.alloc_shared((rows, rope_dim), "float32")
            mask_i32 = T.alloc_shared((rows, rope_dim), "int32")
            mask_u32 = T.alloc_shared((rows, rope_dim), "uint32")
            T.tile.createvecindex(idx_i32, 0)
            T.copy(idx_i32, idx_i16)
            T.tile.fill(ones_i16, 1)
            T.tile.bitwise_xor(mask_i16, idx_i16, ones_i16)
            T.copy(mask_i16, mask_f32)
            T.copy(mask_f32, mask_i32)
            T.tile.mul(mask_i32, mask_i32, 4)
            T.reinterpretcast(mask_u32, mask_i32, "uint32_t")

            sin_block = T.alloc_shared((rows, rope_dim), "float32")
            cos_block = T.alloc_shared((rows, rope_dim), "float32")
            rotated = T.alloc_shared((rows, rope_dim), "float32")
            out_f32 = T.alloc_shared((rows, head_dim), "float32")
            T.tile.broadcast(sin_block, sin)
            T.tile.broadcast(cos_block, cos)
            T.tile.gather(rotated, x_f32, mask_u32, 0)
            T.tile.mul(x_f32, x_f32, cos_block)
            T.tile.mul(rotated, rotated, sin_block)
            T.tile.add(out_f32, x_f32, rotated)
            T.copy(out_f32, x_half)
            T.copy(x_half, Out[0, 0])

    return main


CASES = {
    "aclgraph_rms_rope_shape": Stage8Case(
        name="aclgraph_rms_rope_shape",
        program_factory=_aclgraph_rms_rope_shape_program,
    ),
    "moe_token_permute_grad_shape": Stage8Case(
        name="moe_token_permute_grad_shape",
        program_factory=_moe_token_permute_grad_shape_program,
    ),
    "moe_token_unpermute_shape": Stage8Case(
        name="moe_token_unpermute_shape",
        program_factory=_moe_token_unpermute_shape_program,
    ),
}


def _lower_case(case: Stage8Case, work_dir: Path) -> Path:
    source_path = work_dir / "tilelang_stage8.cpp"
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(case.program_factory(), target="ascendc", platform="310P")
    source_path.write_text(artifact.kernel_source, encoding="utf-8")
    return source_path


def run_case(case: Stage8Case, root: Path, compile_core: str) -> None:
    work_dir = root / case.name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    source_path = _lower_case(case, work_dir)
    kernel_cpp = work_dir / "main_kernel.cpp"
    camodel._copy_kernel(source_path, kernel_cpp, compile_core)

    ascend_home = camodel._find_ascend_home()
    env = camodel._prepare_env(ascend_home)
    kernel_o = camodel._compile_kernel(env, ascend_home, work_dir, kernel_cpp, "-O3", compile_core)
    print(f"{case.name}: 310P kernel object: {kernel_o}")


def dependency_report() -> list[str]:
    report = []
    if importlib.util.find_spec("torch_npu") is None:
        report.append("torch_npu: missing; torch_tl_ascend runtime integration is skipped")
    else:
        report.append("torch_npu: available")

    if importlib.util.find_spec("shmem") is None:
        report.append("shmem: missing; shmem and dispatch_combine runtime examples are skipped")
    else:
        report.append("shmem: available, but TL_ASCEND_310P excludes shmem helpers in common.h")

    report.append("TL_ASCEND_310P: shmem helpers are intentionally excluded in common.h")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lower and compile Stage 8 310P MoE/aclgraph integration cases."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--case", choices=sorted(CASES), action="append", help="Run one case. Repeatable."
    )
    parser.add_argument("--compile-core", choices=["AiCore", "VectorCore"], default="AiCore")
    parser.add_argument(
        "--dependency-report",
        action="store_true",
        help="Print local runtime dependency availability for integration examples.",
    )
    args = parser.parse_args()

    if args.dependency_report:
        for line in dependency_report():
            print(line)

    root = args.work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    selected = args.case or sorted(CASES)
    for name in selected:
        print(f"=== {name} ===")
        run_case(CASES[name], root, args.compile_core)


if __name__ == "__main__":
    main()
