#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import numpy as np

import tilelang
import tilelang.language as T
from tilelang.tools import ascend310p_gemm_camodel as camodel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_stage2_camodel"

DTYPE_TO_NUMPY = {
    "float16": np.float16,
    "float32": np.float32,
    "float": np.float32,
    "int32": np.int32,
}

DTYPE_TO_ASCENDEBUG = {
    "float16": "float16",
    "float32": "float32",
    "float": "float32",
    "int32": "int32",
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
class Stage2Case:
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


def _write_array(path: Path, array: np.ndarray) -> Path:
    array.tofile(path)
    return path


def _elementwise_add_program():
    m = 1
    n = 16

    @T.prim_func
    def main(
        A: T.Tensor((m, n), "float16"),
        B: T.Tensor((m, n), "float16"),
        C: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1, n), "float16")
            b_ub = T.alloc_ub((1, n), "float16")
            c_ub = T.alloc_ub((1, n), "float16")
            with T.Scope("C"):
                T.copy(A[0, 0], a_ub)
                T.copy(B[0, 0], b_ub)
                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()
                T.copy(c_ub, C[0, 0])

    return main


def _activation_program():
    m = 1
    n = 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), B: T.Tensor((m, n), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1, n), "float32")
            tmp_ub = T.alloc_ub((1, n), "float32")
            zero_ub = T.alloc_ub((1, n), "float32")
            one_ub = T.alloc_ub((1, n), "float32")
            out_ub = T.alloc_ub((1, n), "float32")
            with T.Scope("C"):
                T.copy(A[0, 0], a_ub)
                T.tile.fill(zero_ub, 0.0)
                T.tile.fill(one_ub, 1.0)
                T.tile.sub(tmp_ub, zero_ub, a_ub)
                T.tile.exp(tmp_ub, tmp_ub)
                T.tile.add(tmp_ub, tmp_ub, one_ub)
                T.tile.div(out_ub, a_ub, tmp_ub)
                T.copy(out_ub, B[0, 0])

    return main


def _cast_program():
    m = 1
    n = 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "float16"), B: T.Tensor((m, n), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_shared((1, n), "float16")
            a_f32 = T.alloc_shared((1, n), "float32")
            b_ub = T.alloc_shared((1, n), "float16")
            with T.Scope("C"):
                T.copy(A[0, 0], a_ub)
                T.tile.cast(a_f32, a_ub, "CAST_NONE", n)
                T.tile.cast(b_ub, a_f32, "CAST_RINT", n)
                T.copy(b_ub, B[0, 0])

    return main


def _broadcast_program():
    m = 1
    n = 16

    @T.prim_func
    def main(A: T.Tensor((1, n), "float16"), B: T.Tensor((m, n), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1, n), "float16")
            b_ub = T.alloc_ub((1, n), "float16")
            with T.Scope("C"):
                T.copy(A[0, 0], a_ub)
                T.tile.broadcast(b_ub, a_ub)
                T.copy(b_ub, B[0, 0])

    return main


def _rms_norm_program():
    m = 1
    n = 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), B: T.Tensor((m, n), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((1, n), "float32")
            sq_ub = T.alloc_ub((1, n), "float32")
            sum_ub = T.alloc_ub((1,), "float32")
            inv_ub = T.alloc_ub((1,), "float32")
            inv_tile = T.alloc_ub((1, n), "float32")
            out_ub = T.alloc_ub((1, n), "float32")
            with T.Scope("C"):
                T.copy(A[0, 0], a_ub)
                T.tile.mul(sq_ub, a_ub, a_ub)
                T.reduce_sum(sq_ub, sum_ub, dim=-1)
                T.tile.div(sum_ub, sum_ub, float(n))
                T.tile.add(sum_ub, sum_ub, 1.0e-5)
                T.tile.sqrt(inv_ub, sum_ub)
                T.tile.rsqrt(inv_ub, sum_ub)
                T.tile.broadcast(inv_tile, inv_ub)
                T.tile.mul(out_ub, a_ub, inv_tile)
                T.copy(out_ub, B[0, 0])

    return main


def _random_1d_program():
    elems = 16
    block_size = 16
    seed = 42
    lcg_a = 1103515245
    lcg_c = 12345

    @T.prim_func
    def main(output: T.Tensor((elems,), "int32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            idx_ub = T.alloc_ub((block_size,), "int32")
            state_ub = T.alloc_ub((block_size,), "int32")
            temp_ub = T.alloc_ub((block_size,), "int32")
            a_ub = T.alloc_ub((block_size,), "int32")
            c_ub = T.alloc_ub((block_size,), "int32")
            seed_ub = T.alloc_ub((block_size,), "int32")
            with T.Scope("C"):
                T.tile.fill(a_ub, lcg_a)
                T.tile.fill(c_ub, lcg_c)
                T.tile.fill(seed_ub, seed)
                T.tile.arith_progression(idx_ub, 0, 1, block_size)
                T.tile.add(state_ub, idx_ub, seed_ub)
                for _ in T.serial(3):
                    T.tile.mul(temp_ub, state_ub, a_ub)
                    T.tile.add(state_ub, temp_ub, c_ub)
                T.copy(state_ub, output[0:block_size])

    return main


def _make_add_data(work_dir: Path):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1, 16)).astype(np.float16)
    b = rng.standard_normal((1, 16)).astype(np.float16)
    c = (a + b).astype(np.float16)
    return (
        {
            "A": _write_array(work_dir / "A.bin", a),
            "B": _write_array(work_dir / "B.bin", b),
        },
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", c)},
    )


def _make_activation_data(work_dir: Path):
    rng = np.random.default_rng(1)
    a = rng.uniform(-3.0, 3.0, size=(1, 16)).astype(np.float32)
    b = (a / (1.0 + np.exp(-a))).astype(np.float32)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_broadcast_data(work_dir: Path):
    rng = np.random.default_rng(2)
    a = rng.standard_normal((1, 16)).astype(np.float16)
    b = np.broadcast_to(a, (1, 16)).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_rms_norm_data(work_dir: Path):
    rng = np.random.default_rng(3)
    a = rng.uniform(-2.0, 2.0, size=(1, 16)).astype(np.float32)
    inv_rms = 1.0 / np.sqrt(np.mean(a * a, axis=1, keepdims=True) + 1.0e-5)
    b = (a * inv_rms).astype(np.float32)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_random_1d_data(work_dir: Path):
    seed = 42
    lcg_a = 1103515245
    lcg_c = 12345
    values = np.zeros((16,), dtype=np.int32)
    for i in range(values.size):
        state = np.int32(seed + i)
        for _ in range(3):
            state = np.int32(state * lcg_a + lcg_c)
        values[i] = state
    return (
        {},
        {"output": work_dir / "output_out.bin"},
        {"output": _write_array(work_dir / "output_golden.bin", values)},
    )


def _make_cast_data(work_dir: Path):
    rng = np.random.default_rng(4)
    a = rng.standard_normal((1, 16)).astype(np.float16)
    b = a.astype(np.float32).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


CASES = {
    "elementwise_add": Stage2Case(
        name="elementwise_add",
        program_factory=_elementwise_add_program,
        inputs=(
            TensorSpec("A", "float16", (1, 16)),
            TensorSpec("B", "float16", (1, 16)),
        ),
        outputs=(TensorSpec("C", "float16", (1, 16)),),
        make_data=_make_add_data,
    ),
    "activation_silu": Stage2Case(
        name="activation_silu",
        program_factory=_activation_program,
        inputs=(TensorSpec("A", "float32", (1, 16)),),
        outputs=(TensorSpec("B", "float32", (1, 16)),),
        make_data=_make_activation_data,
        rtol=1e-4,
        atol=1e-4,
    ),
    "cast_roundtrip": Stage2Case(
        name="cast_roundtrip",
        program_factory=_cast_program,
        inputs=(TensorSpec("A", "float16", (1, 16)),),
        outputs=(TensorSpec("B", "float16", (1, 16)),),
        make_data=_make_cast_data,
    ),
    "pad_broadcast": Stage2Case(
        name="pad_broadcast",
        program_factory=_broadcast_program,
        inputs=(TensorSpec("A", "float16", (1, 16)),),
        outputs=(TensorSpec("B", "float16", (1, 16)),),
        make_data=_make_broadcast_data,
    ),
    "normalization_rms": Stage2Case(
        name="normalization_rms",
        program_factory=_rms_norm_program,
        inputs=(TensorSpec("A", "float32", (1, 16)),),
        outputs=(TensorSpec("B", "float32", (1, 16)),),
        make_data=_make_rms_norm_data,
        rtol=1e-4,
        atol=1e-4,
    ),
    "random_1d": Stage2Case(
        name="random_1d",
        program_factory=_random_1d_program,
        inputs=(),
        outputs=(TensorSpec("output", "int32", (16,)),),
        make_data=_make_random_1d_data,
        rtol=0.0,
        atol=0.0,
    ),
}


def _lower_case(case: Stage2Case, work_dir: Path) -> Path:
    source_path = work_dir / "tilelang_stage2.cpp"
    with tilelang.tvm.transform.PassContext(opt_level=3, config=PASS_CONFIGS):
        artifact = tilelang.lower(case.program_factory(), target="ascendc", platform="310P")
    source_path.write_text(artifact.kernel_source, encoding="utf-8")
    return source_path


def _copy_kernel_for_case(source_path: Path, kernel_cpp: Path, compile_core: str) -> None:
    if compile_core == "AiCore":
        source = source_path.read_text(encoding="utf-8")
        if "TL_ASCEND_310P" not in source:
            source = "#define TL_ASCEND_310P 1\n" + source
        source = source.replace('#include "acl/acl.h"\n', "")
        source = source.replace("#include <runtime/rt_ffts.h>\n", "")
        source = source.replace(", uint64_t fftsAddr) {", ") {")
        source = source.replace("  pipe.Destroy();\n", "")
        source = source.replace(
            "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);",
            "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);",
        )
        source = source.replace(
            "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);",
            "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);",
        )
        source = source.replace(
            "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);",
            "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);",
        )
        source = _unwrap_all_core_guards(source)
        source = re.sub(r"\nvoid\s+\w+_tiling\([^{}]*\)\s*\{\s*\}\n", "\n", source)
        source = re.sub(
            r'\nextern\s+"C"\s+void\s+call\([^{}]*\)\s*\{.*?\n\}\s*$',
            "\n",
            source,
            flags=re.DOTALL,
        )
        kernel_cpp.write_text(source, encoding="utf-8")
        return
    camodel._copy_kernel(source_path, kernel_cpp, compile_core)


def _unwrap_all_core_guards(source: str) -> str:
    source = re.sub(r"\bif ASCEND_IS_AIV \{", "{", source)
    source = re.sub(r"\bif ASCEND_IS_AIC \{", "{", source)
    return source


def _write_camodel_script(
    case: Stage2Case,
    work_dir: Path,
    kernel_o: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    core_type: str,
    timeout: int,
) -> Path:
    script = work_dir / "run_camodel_api.py"
    op_type = f"Tilelang310PStage2{case.name}"
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
    case: Stage2Case,
    work_dir: Path,
    golden_paths: dict[str, Path],
    output_paths: dict[str, Path],
) -> None:
    messages: list[str] = []
    for spec in case.outputs:
        camodel_output = (
            work_dir
            / "ascendebug_workspace"
            / f"Tilelang310PStage2{case.name}"
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
    case: Stage2Case,
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
    _copy_kernel_for_case(source_path, kernel_cpp, compile_core)
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
        description="Lower, compile, and run Stage 2 310P UB/vector CAModel cases."
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

    selected = args.case or sorted(CASES)
    for name in selected:
        print(f"=== {name} ===")
        run_case(CASES[name], root, args.compile_core, args.run_core, args.timeout, args.skip_run)


if __name__ == "__main__":
    main()
