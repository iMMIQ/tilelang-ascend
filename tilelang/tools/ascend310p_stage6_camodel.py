#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.tools import ascend310p_gemm_camodel as camodel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_stage6_camodel"

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@dataclass(frozen=True)
class Stage6Case:
    name: str
    program_factory: Callable[[], object]


def _cross_scope_fusion_program():
    m, n, k = 1, 1, 16

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        D: T.Tensor((m, n), "float16"),
        C: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_shared((m, k), "float16")
            b_ub = T.alloc_shared((k, n), "float16")
            c_ub = T.alloc_shared((m, n), "float16")
            d_ub = T.alloc_shared((m, n), "float16")

            with T.Scope("C"):
                T.copy(A[0, 0], a_ub)
                T.copy(B[0, 0], b_ub)
                acc = T.alloc_var("float32", init=0.0)
                acc = T.float32(0.0)
                for kk in T.serial(k):
                    acc = acc + T.float32(a_ub[0, kk]) * T.float32(b_ub[kk, 0])
                c_ub[0, 0] = acc
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)
                T.copy(C[0, 0], c_ub)
                T.copy(D[0, 0], d_ub)
                T.barrier_all()
                c_ub[0, 0] = T.float32(c_ub[0, 0]) + T.float32(d_ub[0, 0])
                T.barrier_all()
                T.copy(c_ub, C[0, 0])

    return main


def _pipelined_vector_program():
    n = 64
    block = 16
    stages = 2

    @T.prim_func
    def main(A: T.Tensor((n,), "float16"), B: T.Tensor((n,), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_shared((stages, block), "float16")
            b_ub = T.alloc_shared((stages, block), "float16")
            for i in T.Pipelined(n // block, num_stages=2):
                cur = i % stages
                T.copy(A[i * block : (i + 1) * block], a_ub[cur, :])
                T.pipe_barrier("v")
                for j in T.Parallel(block):
                    b_ub[cur, j] = a_ub[cur, j] + T.float16(1.0)
                T.pipe_barrier("v")
                T.copy(b_ub[cur, :], B[i * block : (i + 1) * block])

    return main


CASES = {
    "cross_scope_fusion": Stage6Case(
        name="cross_scope_fusion",
        program_factory=_cross_scope_fusion_program,
    ),
    "pipelined_vector": Stage6Case(
        name="pipelined_vector",
        program_factory=_pipelined_vector_program,
    ),
}


def _lower_case(case: Stage6Case, work_dir: Path) -> Path:
    source_path = work_dir / "tilelang_stage6.cpp"
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(case.program_factory(), target="ascendc", platform="310P")
    source_path.write_text(artifact.kernel_source, encoding="utf-8")
    return source_path


def run_case(case: Stage6Case, root: Path, compile_core: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lower and compile Stage 6 310P cross-scope/pipeline cases."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--case", choices=sorted(CASES), action="append", help="Run one case. Repeatable."
    )
    parser.add_argument("--compile-core", choices=["AiCore", "VectorCore"], default="AiCore")
    args = parser.parse_args()

    root = args.work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    selected = args.case or sorted(CASES)
    for name in selected:
        print(f"=== {name} ===")
        run_case(CASES[name], root, args.compile_core)


if __name__ == "__main__":
    main()
