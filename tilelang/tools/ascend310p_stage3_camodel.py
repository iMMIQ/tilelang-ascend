#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tilelang
import tilelang.language as T
from tilelang.tools import ascend310p_gemm_camodel as camodel
from tilelang.tools import ascend310p_stage2_camodel as stage2


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_stage3_camodel"

DTYPE_TO_NUMPY = {
    "float16": np.float16,
    "float32": np.float32,
    "float": np.float32,
    "int32": np.int32,
    "uint8": np.uint8,
    "uint32": np.uint32,
}

DTYPE_TO_ASCENDEBUG = {
    "float16": "float16",
    "float32": "float32",
    "float": "float32",
    "int32": "int32",
    "uint8": "uint8",
    "uint32": "uint32",
}

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Stage3Case:
    name: str
    program_factory: Callable[[], object]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    make_data: Callable[
        [Path],
        tuple[dict[str, Path], dict[str, Path], dict[str, Path]],
    ]
    rtol: float = 1e-4
    atol: float = 1e-4
    default_run: bool = True


def _write_array(path: Path, array: np.ndarray) -> Path:
    array.tofile(path)
    return path


def _reduce_sum_row_program():
    m = 8
    n = 8

    @T.prim_func
    def main(A: T.Tensor((m, n), "float16"), B: T.Tensor((m,), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((m, n), "float16")
            b_ub = T.alloc_ub((m,), "float16")
            with T.Scope("C"):
                T.copy(A, a_ub)
                T.barrier_all()
                T.reduce_sum(a_ub, b_ub, dim=-1)
                T.barrier_all()
                T.copy(b_ub, B)

    return main


def _reduce_max_row_program():
    m = 8
    n = 8

    @T.prim_func
    def main(A: T.Tensor((m, n), "float16"), B: T.Tensor((m,), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((m, n), "float16")
            b_ub = T.alloc_ub((m,), "float16")
            with T.Scope("C"):
                T.copy(A, a_ub)
                T.barrier_all()
                T.reduce_max(a_ub, b_ub, dim=-1)
                T.barrier_all()
                T.copy(b_ub, B)

    return main


def _reduce_min_col_program():
    m = 8
    n = 8

    @T.prim_func
    def main(A: T.Tensor((m, n), "float16"), B: T.Tensor((n,), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((m, n), "float16")
            b_ub = T.alloc_ub((n,), "float16")
            with T.Scope("C"):
                T.copy(A, a_ub)
                T.barrier_all()
                T.reduce_min(a_ub, b_ub, dim=0)
                T.barrier_all()
                T.copy(b_ub, B)

    return main


def _compare_select_program():
    n = 128

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float16"),
        B: T.Tensor((n,), "float16"),
        C: T.Tensor((n,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((n,), "float16")
            b_ub = T.alloc_ub((n,), "float16")
            c_ub = T.alloc_ub((n,), "float16")
            mask_ub = T.alloc_ub((n // 8,), "uint8")
            with T.Scope("C"):
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.barrier_all()
                T.tile.compare(mask_ub, a_ub, b_ub, "GT")
                T.barrier_all()
                T.tile.select(c_ub, mask_ub, a_ub, b_ub, "VSEL_TENSOR_TENSOR_MODE")
                T.barrier_all()
                T.copy(c_ub, C)

    return main


def _gather_program():
    n = 16

    @T.prim_func
    def main(
        A: T.Tensor((n,), "int32"),
        Index: T.Tensor((n,), "uint32"),
        B: T.Tensor((n,), "int32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((n,), "int32")
            index_ub = T.alloc_ub((n,), "uint32")
            b_ub = T.alloc_ub((n,), "int32")
            with T.Scope("C"):
                T.copy(A, a_ub)
                T.copy(Index, index_ub)
                T.barrier_all()
                T.tile.gather(b_ub, a_ub, index_ub, 0)
                T.barrier_all()
                T.copy(b_ub, B)

    return main


def _gather_mask_program():
    n = 16

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n // 2,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((n,), "float32")
            b_ub = T.alloc_ub((n // 2,), "float32")
            with T.Scope("C"):
                T.copy(A, a_ub)
                T.barrier_all()
                T.tile.gather_mask(b_ub, a_ub, "P0101")
                T.barrier_all()
                T.copy(b_ub, B)

    return main


def _topk_program():
    n = 32
    k = 4

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((2 * k,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((n,), "float32")
            b_ub = T.alloc_ub((2 * k,), "float32")
            with T.Scope("C"):
                T.copy(A, a_ub)
                T.barrier_all()
                T.tile.topk(b_ub, a_ub, k, n)
                T.barrier_all()
                T.copy(b_ub, B)

    return main


def _merge_sort_2way_program():
    n = 16
    elems = 2 * n

    @T.prim_func
    def main(
        A: T.Tensor((elems,), "float32"),
        B: T.Tensor((elems,), "float32"),
        C: T.Tensor((2 * elems,), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_shared((elems,), "float32")
            b_ub = T.alloc_shared((elems,), "float32")
            c_ub = T.alloc_shared((2 * elems,), "float32")
            with T.Scope("C"):
                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.barrier_all()
                T.tile.merge_sort(c_ub, a_ub, b_ub)
                T.barrier_all()
                T.copy(c_ub, C)

    return main


def _make_reduce_sum_row_data(work_dir: Path):
    rng = np.random.default_rng(10)
    a = rng.uniform(-2.0, 2.0, size=(8, 8)).astype(np.float16)
    b = np.sum(a.astype(np.float32), axis=1).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_reduce_max_row_data(work_dir: Path):
    rng = np.random.default_rng(11)
    a = rng.uniform(-2.0, 2.0, size=(8, 8)).astype(np.float16)
    b = np.max(a.astype(np.float32), axis=1).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_reduce_min_col_data(work_dir: Path):
    rng = np.random.default_rng(12)
    a = rng.uniform(-2.0, 2.0, size=(8, 8)).astype(np.float16)
    b = np.min(a.astype(np.float32), axis=0).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_compare_select_data(work_dir: Path):
    rng = np.random.default_rng(13)
    a = rng.uniform(-4.0, 4.0, size=(128,)).astype(np.float16)
    b = rng.uniform(-4.0, 4.0, size=(128,)).astype(np.float16)
    c = np.where(a > b, a, b).astype(np.float16)
    return (
        {
            "A": _write_array(work_dir / "A.bin", a),
            "B": _write_array(work_dir / "B.bin", b),
        },
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", c)},
    )


def _make_gather_data(work_dir: Path):
    a = np.arange(16, dtype=np.int32) * np.int32(3)
    element_index = np.array(
        [0, 7, 2, 9, 4, 1, 12, 5, 15, 6, 8, 3, 10, 13, 11, 14],
        dtype=np.uint32,
    )
    index = element_index * np.uint32(a.itemsize)
    b = a[element_index.astype(np.int64)].astype(np.int32)
    return (
        {
            "A": _write_array(work_dir / "A.bin", a),
            "Index": _write_array(work_dir / "Index.bin", index),
        },
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_gather_mask_data(work_dir: Path):
    a = np.arange(1, 17, dtype=np.float32)
    b = a[::2].astype(np.float32)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_topk_data(work_dir: Path):
    rng = np.random.default_rng(14)
    a = rng.uniform(-3.0, 3.0, size=(32,)).astype(np.float32)
    order = np.argsort(-a, kind="stable")[:4]
    pairs = np.empty((8,), dtype=np.float32)
    pairs[0::2] = a[order]
    pairs[1::2] = order.astype(np.float32)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", pairs)},
    )


def _make_sorted_pairs(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    pairs = np.empty((values.size * 2,), dtype=np.float32)
    pairs[0::2] = values[order]
    pairs[1::2] = order.astype(np.float32)
    return pairs


def _make_merge_sort_data(work_dir: Path):
    rng = np.random.default_rng(15)
    block0 = _make_sorted_pairs(rng.uniform(-3.0, 3.0, size=(16,)).astype(np.float32))
    block1_values = rng.uniform(-3.0, 3.0, size=(16,)).astype(np.float32)
    block1 = _make_sorted_pairs(block1_values)
    tuples = [(block0[i], block0[i + 1]) for i in range(0, block0.size, 2)]
    tuples.extend((block1[i], block1[i + 1]) for i in range(0, block1.size, 2))
    tuples.sort(key=lambda item: item[0], reverse=True)
    merged = np.empty((64,), dtype=np.float32)
    for i, (value, index) in enumerate(tuples):
        merged[2 * i] = value
        merged[2 * i + 1] = index
    return (
        {
            "A": _write_array(work_dir / "A.bin", block0),
            "B": _write_array(work_dir / "B.bin", block1),
        },
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", merged)},
    )


CASES = {
    "reduce_sum_row": Stage3Case(
        name="reduce_sum_row",
        program_factory=_reduce_sum_row_program,
        inputs=(TensorSpec("A", "float16", (8, 8)),),
        outputs=(TensorSpec("B", "float16", (8,)),),
        make_data=_make_reduce_sum_row_data,
        default_run=False,
    ),
    "reduce_max_row": Stage3Case(
        name="reduce_max_row",
        program_factory=_reduce_max_row_program,
        inputs=(TensorSpec("A", "float16", (8, 8)),),
        outputs=(TensorSpec("B", "float16", (8,)),),
        make_data=_make_reduce_max_row_data,
        default_run=False,
    ),
    "reduce_min_col": Stage3Case(
        name="reduce_min_col",
        program_factory=_reduce_min_col_program,
        inputs=(TensorSpec("A", "float16", (8, 8)),),
        outputs=(TensorSpec("B", "float16", (8,)),),
        make_data=_make_reduce_min_col_data,
        default_run=False,
    ),
    "compare_select": Stage3Case(
        name="compare_select",
        program_factory=_compare_select_program,
        inputs=(
            TensorSpec("A", "float16", (128,)),
            TensorSpec("B", "float16", (128,)),
        ),
        outputs=(TensorSpec("C", "float16", (128,)),),
        make_data=_make_compare_select_data,
    ),
    "gather": Stage3Case(
        name="gather",
        program_factory=_gather_program,
        inputs=(
            TensorSpec("A", "int32", (16,)),
            TensorSpec("Index", "uint32", (16,)),
        ),
        outputs=(TensorSpec("B", "int32", (16,)),),
        make_data=_make_gather_data,
        rtol=0.0,
        atol=0.0,
    ),
    "gather_mask": Stage3Case(
        name="gather_mask",
        program_factory=_gather_mask_program,
        inputs=(TensorSpec("A", "float32", (16,)),),
        outputs=(TensorSpec("B", "float32", (8,)),),
        make_data=_make_gather_mask_data,
        default_run=False,
    ),
    "topk": Stage3Case(
        name="topk",
        program_factory=_topk_program,
        inputs=(TensorSpec("A", "float32", (32,)),),
        outputs=(TensorSpec("B", "float32", (8,)),),
        make_data=_make_topk_data,
        default_run=False,
    ),
    "merge_sort_2way": Stage3Case(
        name="merge_sort_2way",
        program_factory=_merge_sort_2way_program,
        inputs=(
            TensorSpec("A", "float32", (32,)),
            TensorSpec("B", "float32", (32,)),
        ),
        outputs=(TensorSpec("C", "float32", (64,)),),
        make_data=_make_merge_sort_data,
        default_run=False,
    ),
}


def _lower_case(case: Stage3Case, work_dir: Path) -> Path:
    source_path = work_dir / "tilelang_stage3.cpp"
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(case.program_factory(), target="ascendc", platform="310P")
    source_path.write_text(artifact.kernel_source, encoding="utf-8")
    return source_path


def _write_camodel_script(
    case: Stage3Case,
    work_dir: Path,
    kernel_o: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    core_type: str,
    timeout: int,
) -> Path:
    script = work_dir / "run_camodel_api.py"
    op_type = f"Tilelang310PStage3{case.name}"
    workspace = work_dir / "ascendebug_workspace"
    ascend_toolkit_root = str(camodel._find_ascend_home().parent)
    lines = [
        "from ascendebug import DebugOp, OpExecutor, RunSimuOptions, TilingInfo, NpuCompileInfo",
        "",
        "def main():",
        (
            f'    debug_op = DebugOp("{op_type}", core_type="{core_type}", '
            'chip_version="Ascend310P1")'
        ),
    ]
    for spec in case.inputs:
        lines.append(
            "    debug_op.custom_input("
            f'"{spec.name}", "{DTYPE_TO_ASCENDEBUG[spec.dtype]}", {list(spec.shape)}, '
            f'r"{input_paths[spec.name]}")'
        )
    for spec in case.outputs:
        output_path = output_paths[spec.name]
        np.zeros(spec.shape, dtype=DTYPE_TO_NUMPY[spec.dtype]).tofile(output_path)
        lines.append(
            "    debug_op.custom_output("
            f'"{spec.name}", "{DTYPE_TO_ASCENDEBUG[spec.dtype]}", {list(spec.shape)}, '
            f'r"{output_path}")'
        )
    lines.extend(
        [
            "",
            f'    executor = OpExecutor(debug_op, r"{workspace}", r"{ascend_toolkit_root}")',
            "    executor.run_camodel(",
            f'        r"{kernel_o}",',
            f"        RunSimuOptions(block_num=1, timeout={timeout}),",
            "        NpuCompileInfo(syncall=False),",
            "        TilingInfo(\"\", 0, 1, 1),",
            "    )",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    )
    script.write_text("\n".join(lines), encoding="utf-8")
    return script


def _compare_case(
    case: Stage3Case,
    work_dir: Path,
    golden_paths: dict[str, Path],
    output_paths: dict[str, Path],
) -> None:
    messages: list[str] = []
    for spec in case.outputs:
        camodel_output = (
            work_dir
            / "ascendebug_workspace"
            / f"Tilelang310PStage3{case.name}"
            / "simulator"
            / "output"
            / output_paths[spec.name].name
        )
        if not camodel_output.exists():
            raise FileNotFoundError(f"CAModel output was not found: {camodel_output}")
        actual = np.fromfile(camodel_output, dtype=DTYPE_TO_NUMPY[spec.dtype]).reshape(
            spec.shape
        )
        expect = np.fromfile(golden_paths[spec.name], dtype=DTYPE_TO_NUMPY[spec.dtype]).reshape(
            spec.shape
        )
        np.testing.assert_allclose(actual, expect, rtol=case.rtol, atol=case.atol)
        shutil.copy2(camodel_output, work_dir / f"{spec.name}_actual.bin")
        messages.append(f"{spec.name}: matches golden")
    message = "\n".join(messages)
    (work_dir / "logs" / "compare.log").write_text(message + "\n", encoding="utf-8")
    print(message)


def run_case(
    case: Stage3Case,
    root: Path,
    compile_core: str,
    run_core: str,
    timeout: int,
    skip_run: bool,
) -> None:
    work_dir = root / case.name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    source_path = _lower_case(case, work_dir)
    kernel_cpp = work_dir / "main_kernel.cpp"
    stage2._copy_kernel_for_case(source_path, kernel_cpp, compile_core)
    input_paths, output_paths, golden_paths = case.make_data(work_dir)

    ascend_home = camodel._find_ascend_home()
    env = camodel._prepare_env(ascend_home)
    kernel_o = camodel._compile_kernel(env, ascend_home, work_dir, kernel_cpp, "-O3", compile_core)
    print(f"{case.name}: 310P kernel object: {kernel_o}")

    if skip_run:
        return

    script = _write_camodel_script(
        case, work_dir, kernel_o, input_paths, output_paths, run_core, timeout
    )
    camodel._run(
        [sys.executable, str(script)],
        env=env,
        cwd=work_dir,
        log_path=work_dir / "logs" / "camodel.log",
    )
    _compare_case(case, work_dir, golden_paths, output_paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lower, compile, and run Stage 3 310P reduce/indexing CAModel cases."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--case", choices=sorted(CASES), action="append", help="Run one case. Repeatable."
    )
    parser.add_argument("--compile-core", choices=["AiCore", "VectorCore"], default="AiCore")
    parser.add_argument("--run-core", choices=["AiCore", "VectorCore"], default="AiCore")
    parser.add_argument(
        "--skip-run", action="store_true", help="Only lower and compile the generated kernels."
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="CAModel launch timeout per case in seconds."
    )
    args = parser.parse_args()

    root = args.work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    selected = args.case or sorted(name for name, case in CASES.items() if case.default_run)
    for name in selected:
        print(f"=== {name} ===")
        run_case(CASES[name], root, args.compile_core, args.run_core, args.timeout, args.skip_run)


if __name__ == "__main__":
    main()
