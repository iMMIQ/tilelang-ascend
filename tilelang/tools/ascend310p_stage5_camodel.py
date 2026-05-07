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
from tilelang.tools import ascend310p_stage4_camodel as stage4


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_stage5_camodel"

DTYPE_TO_NUMPY = {
    "float16": np.float16,
    "float32": np.float32,
    "float": np.float32,
    "int8": np.int8,
}

DTYPE_TO_ASCENDEBUG = {
    "float16": "float16",
    "float32": "float32",
    "float": "float32",
    "int8": "int8",
}

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Stage5Case:
    name: str
    program_factory: Callable[[], object]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    make_data: Callable[
        [Path],
        tuple[dict[str, Path], dict[str, Path], dict[str, Path]],
    ]
    rtol: float = 1e-2
    atol: float = 1e-2
    default_run: bool = True
    default_compile: bool = True


def _write_array(path: Path, array: np.ndarray) -> Path:
    array.tofile(path)
    return path


def _developer_gelu_mul_program():
    m, n = 1, 32
    block_m, block_n = 1, 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), B: T.Tensor((m, n // 2), "float32")):
        with T.Kernel(1, threads=2, is_npu=True) as (cid):
            a1_ub = T.alloc_shared((block_m, block_n), "float32")
            a2_ub = T.alloc_shared((block_m, block_n), "float32")
            b_ub = T.alloc_shared((block_m, block_n), "float32")
            temp_ub = T.alloc_shared((block_m, block_n), "float32")

            T.copy(A[0, 0], a1_ub)
            T.copy(A[0, n // 2], a2_ub)
            T.tile.mul(temp_ub, a1_ub, a1_ub)
            T.tile.mul(temp_ub, a1_ub, temp_ub)
            T.tile.mul(temp_ub, temp_ub, 0.044715)
            T.tile.add(temp_ub, a1_ub, temp_ub)
            T.tile.mul(temp_ub, temp_ub, -1.5957691)
            T.tile.exp(temp_ub, temp_ub)
            T.tile.add(temp_ub, temp_ub, 1.0)
            T.tile.div(temp_ub, a1_ub, temp_ub)
            T.tile.mul(b_ub, temp_ub, a2_ub)
            T.copy(b_ub, B[0, 0])

    return main


def _developer_gemm_program():
    m, n, k = 128, 128, 64
    block_m, block_n, block_k = 128, 128, 64

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        C: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_shared = T.alloc_shared((block_m, block_k), "float16")
            b_shared = T.alloc_shared((block_k, block_n), "float16")
            c_local = T.alloc_fragment((block_m, block_n), "float")

            T.copy(A[0, 0], a_shared)
            T.copy(B[0, 0], b_shared)
            T.gemm_v0(a_shared, b_shared, c_local, init=True)
            T.copy(c_local, C[0, 0])

    return main


def _blocksparse_program():
    m, n, k = 32, 32, 32
    block_m, block_n, block_k = 32, 32, 32

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        BlockMask: T.Tensor((1, 1, 1), "int8"),
        C: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_shared = T.alloc_shared((block_m, block_k), "float16")
            b_shared = T.alloc_shared((block_k, block_n), "float16")
            c_local = T.alloc_fragment((block_m, block_n), "float")

            if BlockMask[0, 0, 0]:
                T.copy(A[0, 0], a_shared)
                T.copy(B[0, 0], b_shared)
                T.gemm_v0(a_shared, b_shared, c_local, init=True)
            T.copy(c_local, C[0, 0])

    return main


def _make_gelu_mul_data(work_dir: Path):
    rng = np.random.default_rng(50)
    a = rng.standard_normal((1, 32)).astype(np.float32)
    a1 = a[:, :16]
    a2 = a[:, 16:]
    gelu = a1 / (1.0 + np.exp(-1.5957691 * (a1 + 0.044715 * np.power(a1, 3))))
    b = (gelu * a2).astype(np.float32)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_gemm_data(work_dir: Path):
    rng = np.random.default_rng(51)
    a = rng.standard_normal((128, 64)).astype(np.float16)
    b = rng.standard_normal((64, 128)).astype(np.float16)
    c = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a), "B": _write_array(work_dir / "B.bin", b)},
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", c)},
    )


def _make_blocksparse_data(work_dir: Path):
    rng = np.random.default_rng(52)
    a = rng.standard_normal((32, 32)).astype(np.float16)
    b = rng.standard_normal((32, 32)).astype(np.float16)
    mask = np.ones((1, 1, 1), dtype=np.int8)
    c = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    return (
        {
            "A": _write_array(work_dir / "A.bin", a),
            "B": _write_array(work_dir / "B.bin", b),
            "BlockMask": _write_array(work_dir / "BlockMask.bin", mask),
        },
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", c)},
    )


CASES = {
    "developer_gelu_mul": Stage5Case(
        name="developer_gelu_mul",
        program_factory=_developer_gelu_mul_program,
        inputs=(TensorSpec("A", "float32", (1, 32)),),
        outputs=(TensorSpec("B", "float32", (1, 16)),),
        make_data=_make_gelu_mul_data,
        rtol=1e-4,
        atol=1e-4,
    ),
    "developer_gemm": Stage5Case(
        name="developer_gemm",
        program_factory=_developer_gemm_program,
        inputs=(
            TensorSpec("A", "float16", (128, 64)),
            TensorSpec("B", "float16", (64, 128)),
        ),
        outputs=(TensorSpec("C", "float16", (128, 128)),),
        make_data=_make_gemm_data,
        default_run=False,
    ),
    "blocksparse_gemm": Stage5Case(
        name="blocksparse_gemm",
        program_factory=_blocksparse_program,
        inputs=(
            TensorSpec("A", "float16", (32, 32)),
            TensorSpec("B", "float16", (32, 32)),
            TensorSpec("BlockMask", "int8", (1, 1, 1)),
        ),
        outputs=(TensorSpec("C", "float16", (32, 32)),),
        make_data=_make_blocksparse_data,
        default_run=False,
    ),
}


def _lower_case(case: Stage5Case, work_dir: Path) -> Path:
    source_path = work_dir / "tilelang_stage5.cpp"
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(case.program_factory(), target="ascendc", platform="310P")
    source_path.write_text(artifact.kernel_source, encoding="utf-8")
    return source_path


def _write_camodel_script(
    case: Stage5Case,
    work_dir: Path,
    kernel_o: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    core_type: str,
    timeout: int,
) -> Path:
    script = work_dir / "run_camodel_api.py"
    op_type = f"Tilelang310PStage5{case.name}"
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
    case: Stage5Case,
    work_dir: Path,
    golden_paths: dict[str, Path],
    output_paths: dict[str, Path],
) -> None:
    messages: list[str] = []
    for spec in case.outputs:
        camodel_output = (
            work_dir
            / "ascendebug_workspace"
            / f"Tilelang310PStage5{case.name}"
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


def _check_carver_310p(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        from tilelang import carver
        from tilelang.carver.arch.ascend import Ascend
    except ModuleNotFoundError as err:
        message = f"carver_310p skipped: missing dependency {err.name}"
        (work_dir / "carver_310p.log").write_text(message + "\n", encoding="utf-8")
        print(message)
        return

    arch = Ascend(chip_name="Ascend310P")
    hints = carver.MatmulTemplate(
        M=128,
        N=128,
        K=64,
        in_dtype="float16",
        accum_dtype="float16",
        out_dtype="float16",
    ).with_arch(arch).recommend_hints(topk=8)
    lines = [
        f"chip_name={arch.chip_name}",
        f"ub_cap={arch.ub_cap}",
        f"l1_cap={arch.l1_cap}",
        f"l0a_cap={arch.l0a_cap}",
        f"l0b_cap={arch.l0b_cap}",
        f"l0c_cap={arch.l0c_cap}",
    ]
    for hint in hints:
        block_m, block_n = hint.block
        block_k = hint.rstep[0]
        l1_bytes = (block_m * block_k + block_k * block_n) * 2
        l0c_bytes = block_m * block_n * 4
        if l1_bytes > arch.l1_cap or l0c_bytes > arch.l0c_cap:
            raise AssertionError(
                f"carver hint exceeds 310P memory: block=({block_m}, {block_n}), "
                f"k={block_k}, l1={l1_bytes}, l0c={l0c_bytes}"
            )
        lines.append(
            f"block=({block_m}, {block_n}) rstep=({block_k}) "
            f"l1_bytes={l1_bytes} l0c_bytes={l0c_bytes}"
        )
    (work_dir / "carver_310p.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def run_case(
    case: Stage5Case,
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
    camodel._copy_kernel(source_path, kernel_cpp, compile_core)
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
        description="Lower, compile, and run Stage 5 310P developer-mode CAModel cases."
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
    parser.add_argument("--skip-carver", action="store_true")
    parser.add_argument(
        "--compile-all-default",
        action="store_true",
        help="Compile all Stage 5 cases that are enabled for default compile coverage.",
    )
    args = parser.parse_args()

    root = args.work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.case:
        selected = args.case
    elif args.compile_all_default:
        selected = [name for name in sorted(CASES) if CASES[name].default_compile]
        args.skip_run = True
    else:
        selected = [name for name in sorted(CASES) if CASES[name].default_run]
    for name in selected:
        print(f"=== {name} ===")
        run_case(CASES[name], root, args.compile_core, args.run_core, args.timeout, args.skip_run)

    if not args.skip_carver:
        print("=== carver_310p ===")
        _check_carver_310p(root / "carver_310p")


if __name__ == "__main__":
    main()
