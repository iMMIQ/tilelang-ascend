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
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_stage7_camodel"

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@dataclass(frozen=True)
class Stage7Case:
    name: str
    program_factory: Callable[[], object]


def _flash_attention_shape_program():
    block_m, block_n, dim = 16, 16, 16
    sm_scale = 0.25

    @T.prim_func
    def main(
        Q: T.Tensor((block_m, dim), "float16"),
        K: T.Tensor((block_n, dim), "float16"),
        V: T.Tensor((block_n, dim), "float16"),
        O: T.Tensor((block_m, dim), "float16"),
        WorkspaceScores: T.Tensor((block_m, block_n), "float16"),
        WorkspaceOut: T.Tensor((block_m, dim), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_shared((block_m, dim), "float16")
            k_l1 = T.alloc_shared((block_n, dim), "float16")
            v_l1 = T.alloc_shared((block_n, dim), "float16")
            scores = T.alloc_shared((block_m, block_n), "float16")
            scores_norm = T.alloc_shared((block_m, block_n), "float16")
            row_max = T.alloc_shared((block_m,), "float16")
            row_sum = T.alloc_shared((block_m,), "float16")
            out = T.alloc_shared((block_m, dim), "float16")

            T.copy(Q[0, 0], q_l1)
            T.copy(K[0, 0], k_l1)
            T.copy(V[0, 0], v_l1)

            for i in T.serial(block_m):
                for j in T.serial(block_n):
                    acc = T.alloc_var("float32", init=0.0)
                    acc = T.float32(0.0)
                    for kk in T.serial(dim):
                        acc = acc + T.float32(q_l1[i, kk]) * T.float32(k_l1[j, kk])
                    scores[i, j] = acc * sm_scale

            T.reduce_max(scores, row_max, dim=-1)
            for i in T.serial(block_m):
                for j in T.serial(block_n):
                    scores_norm[i, j] = scores[i, j] - row_max[i]
            T.tile.exp(scores_norm, scores_norm)
            T.reduce_sum(scores_norm, row_sum, dim=-1)
            for i in T.serial(block_m):
                for j in T.serial(block_n):
                    scores_norm[i, j] = scores_norm[i, j] / row_sum[i]

            for i in T.serial(block_m):
                for d in T.serial(dim):
                    acc = T.alloc_var("float32", init=0.0)
                    acc = T.float32(0.0)
                    for j in T.serial(block_n):
                        acc = acc + T.float32(scores_norm[i, j]) * T.float32(v_l1[j, d])
                    out[i, d] = acc

            T.copy(scores_norm, WorkspaceScores[0, 0])
            T.copy(out, WorkspaceOut[0, 0])
            T.copy(out, O[0, 0])

    return main


def _sparse_attention_index_program():
    topk, dim = 16, 16

    @T.prim_func
    def main(
        Q: T.Tensor((dim,), "float16"),
        KV: T.Tensor((topk, dim), "float16"),
        Indices: T.Tensor((topk,), "uint32"),
        O: T.Tensor((dim,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_ub = T.alloc_shared((dim,), "float16")
            kv_flat = T.alloc_shared((topk * dim,), "float16")
            idx_ub = T.alloc_shared((topk,), "uint32")
            gathered = T.alloc_shared((topk,), "float16")
            sum_ub = T.alloc_shared((1,), "float16")
            out_ub = T.alloc_shared((dim,), "float16")

            T.copy(Q[0], q_ub)
            T.copy(KV[0, 0], kv_flat)
            T.copy(Indices[0], idx_ub)
            T.tile.gather(gathered, kv_flat, idx_ub, 0)
            T.reduce_sum(gathered, sum_ub, dim=-1)
            for d in T.Parallel(dim):
                out_ub[d] = sum_ub[0] + q_ub[d]
            T.copy(out_ub, O[0])

    return main


CASES = {
    "flash_attention_shape": Stage7Case(
        name="flash_attention_shape",
        program_factory=_flash_attention_shape_program,
    ),
    "sparse_attention_index": Stage7Case(
        name="sparse_attention_index",
        program_factory=_sparse_attention_index_program,
    ),
}


def _lower_case(case: Stage7Case, work_dir: Path) -> Path:
    source_path = work_dir / "tilelang_stage7.cpp"
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(case.program_factory(), target="ascendc", platform="310P")
    source_path.write_text(artifact.kernel_source, encoding="utf-8")
    return source_path


def run_case(case: Stage7Case, root: Path, compile_core: str) -> None:
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
        description="Lower and compile Stage 7 310P attention-shape cases."
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
