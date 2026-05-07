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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_stage4_camodel"

DTYPE_TO_NUMPY = {
    "float16": np.float16,
    "float32": np.float32,
    "float": np.float32,
}

DTYPE_TO_ASCENDEBUG = {
    "float16": "float16",
    "float32": "float",
    "float": "float",
}

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Stage4Case:
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


def _write_array(path: Path, array: np.ndarray) -> Path:
    array.tofile(path)
    return path


def _gemm_tail_program():
    m, n, k = 3, 5, 7

    @T.prim_func
    def main(A: T.Tensor((m, k), "float16"), B: T.Tensor((k, n), "float16"), C: T.Tensor((m, n), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((m, k), "float16")
            b_ub = T.alloc_ub((k, n), "float16")
            c_ub = T.alloc_ub((m, n), "float16")
            T.copy(A[0, 0], a_ub)
            T.copy(B[0, 0], b_ub)
            with T.Scope("C"):
                for i in T.serial(m):
                    for j in T.serial(n):
                        acc = T.alloc_var("float32", init=0.0)
                        acc = T.float32(0.0)
                        for kk in T.serial(k):
                            acc = acc + T.float32(a_ub[i, kk]) * T.float32(b_ub[kk, j])
                        c_ub[i, j] = acc
                T.copy(c_ub, C[0, 0])

    return main


def _batch_gemm_program():
    b, m, n, k = 1, 1, 1, 16

    @T.prim_func
    def main(
        A: T.Tensor((b, m, k), "float16"),
        B: T.Tensor((b, k, n), "float16"),
        C: T.Tensor((b, m, n), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((b, m, k), "float16")
            b_ub = T.alloc_ub((b, k, n), "float16")
            c_ub = T.alloc_ub((b, m, n), "float16")
            T.copy(A[0, 0, 0], a_ub)
            T.copy(B[0, 0, 0], b_ub)
            with T.Scope("C"):
                for bid in T.serial(b):
                    for i in T.serial(m):
                        for j in T.serial(n):
                            acc = T.alloc_var("float32", init=0.0)
                            acc = T.float32(0.0)
                            for kk in T.serial(k):
                                acc = acc + T.float32(a_ub[bid, i, kk]) * T.float32(b_ub[bid, kk, j])
                            c_ub[bid, i, j] = acc
                T.copy(c_ub, C[0, 0, 0])

    return main


def _gemv_program():
    n, k = 8, 8

    @T.prim_func
    def main(
        X: T.Tensor((k,), "float16"),
        A: T.Tensor((n, k), "float16"),
        Y: T.Tensor((n,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((k,), "float16")
            a_ub = T.alloc_ub((n, k), "float16")
            y_ub = T.alloc_ub((n,), "float16")
            T.copy(X[0], x_ub)
            T.copy(A[0, 0], a_ub)
            with T.Scope("C"):
                for i in T.serial(n):
                    acc = T.alloc_var("float32", init=0.0)
                    acc = T.float32(0.0)
                    for kk in T.serial(k):
                        acc = acc + T.float32(a_ub[i, kk]) * T.float32(x_ub[kk])
                    y_ub[i] = acc
                T.copy(y_ub, Y[0])

    return main


def _simple_fusion_program():
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
                for i in T.serial(m):
                    for j in T.serial(n):
                        acc = T.alloc_var("float32", init=0.0)
                        acc = T.float32(0.0)
                        for kk in T.serial(k):
                            acc = acc + T.float32(a_ub[i, kk]) * T.float32(b_ub[kk, j])
                        c_ub[i, j] = acc
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)
                T.copy(C[0, 0], c_ub)
                T.copy(D[0, 0], d_ub)
                T.barrier_all()
                for i in T.serial(m):
                    for j in T.serial(n):
                        c_ub[i, j] = T.float32(c_ub[i, j]) + T.float32(d_ub[i, j])
                T.barrier_all()
                T.copy(c_ub, C[0, 0])

    return main


def _convolution_program():
    # Direct 1x1x4x4 -> 1x1x2x2 convolution with 3x3 kernel.
    h, w, kh, kw = 4, 4, 3, 3
    oh, ow = 2, 2

    @T.prim_func
    def main(
        X: T.Tensor((h * w,), "float16"),
        W: T.Tensor((kh * kw,), "float16"),
        Y: T.Tensor((oh * ow,), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            x_ub = T.alloc_ub((h * w,), "float16")
            w_ub = T.alloc_ub((kh * kw,), "float16")
            y_ub = T.alloc_ub((oh * ow,), "float16")
            T.copy(X[0], x_ub)
            T.copy(W[0], w_ub)
            with T.Scope("C"):
                for oy in T.serial(oh):
                    for ox in T.serial(ow):
                        acc = T.alloc_var("float32", init=0.0)
                        acc = T.float32(0.0)
                        for ky in T.serial(kh):
                            for kx in T.serial(kw):
                                iy = oy + ky
                                ix = ox + kx
                                acc = acc + T.float32(x_ub[iy * w + ix]) * T.float32(w_ub[ky * kw + kx])
                        y_ub[oy * ow + ox] = acc
                T.copy(y_ub, Y[0])

    return main


def _make_gemm_tail_data(work_dir: Path):
    rng = np.random.default_rng(21)
    a = rng.standard_normal((3, 7)).astype(np.float16)
    b = rng.standard_normal((7, 5)).astype(np.float16)
    c = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a), "B": _write_array(work_dir / "B.bin", b)},
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", c)},
    )


def _make_batch_gemm_data(work_dir: Path):
    rng = np.random.default_rng(22)
    a = rng.standard_normal((1, 1, 16)).astype(np.float16)
    b = rng.standard_normal((1, 16, 1)).astype(np.float16)
    c = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a), "B": _write_array(work_dir / "B.bin", b)},
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", c)},
    )


def _make_gemv_data(work_dir: Path):
    rng = np.random.default_rng(23)
    x = rng.standard_normal((8,)).astype(np.float16)
    a = rng.standard_normal((8, 8)).astype(np.float16)
    y = (a.astype(np.float32) @ x.astype(np.float32)).astype(np.float16)
    return (
        {"X": _write_array(work_dir / "X.bin", x), "A": _write_array(work_dir / "A.bin", a)},
        {"Y": work_dir / "Y_out.bin"},
        {"Y": _write_array(work_dir / "Y_golden.bin", y)},
    )


def _make_simple_fusion_data(work_dir: Path):
    rng = np.random.default_rng(24)
    a = rng.standard_normal((1, 16)).astype(np.float16)
    b = rng.standard_normal((16, 1)).astype(np.float16)
    d = rng.standard_normal((1, 1)).astype(np.float16)
    c = (a.astype(np.float32) @ b.astype(np.float32) + d.astype(np.float32)).astype(np.float16)
    return (
        {
            "A": _write_array(work_dir / "A.bin", a),
            "B": _write_array(work_dir / "B.bin", b),
            "D": _write_array(work_dir / "D.bin", d),
        },
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", c)},
    )


def _make_convolution_data(work_dir: Path):
    rng = np.random.default_rng(25)
    x = rng.standard_normal((16,)).astype(np.float16)
    w = rng.standard_normal((9,)).astype(np.float16)
    x_2d = x.reshape(4, 4).astype(np.float32)
    w_2d = w.reshape(3, 3).astype(np.float32)
    y = np.zeros((2, 2), dtype=np.float32)
    for oy in range(2):
        for ox in range(2):
            acc = 0.0
            for ky in range(3):
                for kx in range(3):
                    acc += x_2d[oy + ky, ox + kx] * w_2d[ky, kx]
            y[oy, ox] = acc
    return (
        {"X": _write_array(work_dir / "X.bin", x), "W": _write_array(work_dir / "W.bin", w)},
        {"Y": work_dir / "Y_out.bin"},
        {"Y": _write_array(work_dir / "Y_golden.bin", y.astype(np.float16).reshape(4))},
    )


CASES = {
    "gemm_tail": Stage4Case(
        name="gemm_tail",
        program_factory=_gemm_tail_program,
        inputs=(
            TensorSpec("A", "float16", (3, 7)),
            TensorSpec("B", "float16", (7, 5)),
        ),
        outputs=(TensorSpec("C", "float16", (3, 5)),),
        make_data=_make_gemm_tail_data,
    ),
    "batch_gemm": Stage4Case(
        name="batch_gemm",
        program_factory=_batch_gemm_program,
        inputs=(
            TensorSpec("A", "float16", (1, 1, 16)),
            TensorSpec("B", "float16", (1, 16, 1)),
        ),
        outputs=(TensorSpec("C", "float16", (1, 1, 1)),),
        make_data=_make_batch_gemm_data,
    ),
    "gemv": Stage4Case(
        name="gemv",
        program_factory=_gemv_program,
        inputs=(
            TensorSpec("X", "float16", (8,)),
            TensorSpec("A", "float16", (8, 8)),
        ),
        outputs=(TensorSpec("Y", "float16", (8,)),),
        make_data=_make_gemv_data,
    ),
    "simple_fusion": Stage4Case(
        name="simple_fusion",
        program_factory=_simple_fusion_program,
        inputs=(
            TensorSpec("A", "float16", (1, 16)),
            TensorSpec("B", "float16", (16, 1)),
            TensorSpec("D", "float16", (1, 1)),
        ),
        outputs=(TensorSpec("C", "float16", (1, 1)),),
        make_data=_make_simple_fusion_data,
        default_run=False,
    ),
    "convolution": Stage4Case(
        name="convolution",
        program_factory=_convolution_program,
        inputs=(
            TensorSpec("X", "float16", (16,)),
            TensorSpec("W", "float16", (9,)),
        ),
        outputs=(TensorSpec("Y", "float16", (4,)),),
        make_data=_make_convolution_data,
    ),
}


def _lower_case(case: Stage4Case, work_dir: Path) -> Path:
    source_path = work_dir / "tilelang_stage4.cpp"
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(case.program_factory(), target="ascendc", platform="310P")
    source_path.write_text(artifact.kernel_source, encoding="utf-8")
    return source_path


def _write_camodel_script(
    case: Stage4Case,
    work_dir: Path,
    kernel_o: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    core_type: str,
    timeout: int,
) -> Path:
    script = work_dir / "run_camodel_api.py"
    op_type = f"Tilelang310PStage4{case.name}"
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
    case: Stage4Case,
    work_dir: Path,
    golden_paths: dict[str, Path],
    output_paths: dict[str, Path],
) -> None:
    messages: list[str] = []
    for spec in case.outputs:
        camodel_output = (
            work_dir
            / "ascendebug_workspace"
            / f"Tilelang310PStage4{case.name}"
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
    case: Stage4Case,
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
        description="Lower, compile, and run Stage 4 310P GEMM/Cube CAModel cases."
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

    selected = args.case or [name for name in sorted(CASES) if CASES[name].default_run]
    for name in selected:
        print(f"=== {name} ===")
        run_case(CASES[name], root, args.compile_core, args.run_core, args.timeout, args.skip_run)


if __name__ == "__main__":
    main()
