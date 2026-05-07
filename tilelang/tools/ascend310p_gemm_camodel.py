#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_gemm_camodel"


KERNEL_TEMPLATE = Template(r'''
#define TL_ASCEND_310P 1
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void main_kernel(GM_ADDR A_handle, GM_ADDR B_handle, GM_ADDR C_handle) {
  AscendC::GlobalTensor<half> A;
  AscendC::GlobalTensor<half> B;
  AscendC::GlobalTensor<half> C;
  A.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(A_handle), $a_elems);
  B.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(B_handle), $b_elems);
  C.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(C_handle), $c_elems);

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ub;
  pipe.InitBuffer(ub, ($a_elems + $b_elems + $c_elems) * sizeof(half));
  AscendC::LocalTensor<half> A_local = ub.GetWithOffset<half>($a_elems, 0);
  AscendC::LocalTensor<half> B_local = ub.GetWithOffset<half>($b_elems, $a_bytes);
  AscendC::LocalTensor<half> C_local = ub.GetWithOffset<half>($c_elems, $ab_bytes);

  if (AscendC::GetBlockIdx() == 0) {
    AscendC::DataCopy(A_local, A, $a_elems);
    AscendC::DataCopy(B_local, B, $b_elems);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int m = 0; m < $m; ++m) {
      for (int n = 0; n < $n; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < $k; ++k) {
          acc += static_cast<float>(A_local.GetValue(m * $k + k)) *
                 static_cast<float>(B_local.GetValue(k * $n + n));
        }
        C_local.SetValue(m * $n + n, static_cast<half>(acc));
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(C, C_local, $c_elems);
  }
}
''')


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path, log_path: Path | None = None) -> None:
    print("+ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("+ " + " ".join(cmd) + "\n")
            log_file.write(f"# cwd: {cwd}\n")
            log_file.write(proc.stdout or "")
            if proc.stdout and not proc.stdout.endswith("\n"):
                log_file.write("\n")
            log_file.write(f"# exit_code: {proc.returncode}\n\n")
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)


def _find_ascend_home() -> Path:
    for name in ("ASCEND_HOME_PATH", "ASCEND_HOME"):
        value = os.environ.get(name)
        if value:
            return Path(value).resolve()
    default = Path.home() / "Ascend" / "ascend-toolkit" / "latest"
    if default.exists():
        return default.resolve()
    raise RuntimeError("ASCEND_HOME_PATH is not set and ~/Ascend/ascend-toolkit/latest was not found")


def _prepare_env(ascend_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = str(ascend_home)
    env.setdefault("ASCEND_HOME", str(ascend_home))
    env["TOOLCHAIN_HOME"] = str(ascend_home / "toolkit")
    extra_paths = [
        ascend_home / "tools" / "ccec_compiler" / "bin",
        ascend_home / "tools" / "ascendc_tools",
        ascend_home / "tools" / "ascendc_tools" / "npu_kernel_launch",
        ascend_home / "toolkit" / "tools" / "ccec_compiler" / "bin",
        ascend_home / "toolkit" / "tools" / "ascendc_tools",
        ascend_home / "toolkit" / "tools" / "ascendc_tools" / "npu_kernel_launch",
    ]
    env["PATH"] = ":".join(str(path) for path in extra_paths if path.exists()) + ":" + env.get("PATH", "")
    python_paths = [
        ascend_home / "python" / "site-packages",
        ascend_home / "toolkit" / "python" / "site-packages",
    ]
    env["PYTHONPATH"] = ":".join(str(path) for path in python_paths if path.exists()) + ":" + env.get("PYTHONPATH", "")
    lib_paths = [
        ascend_home / "lib64",
        ascend_home / "tools" / "ascendc_tools" / "npu_kernel_launch",
        ascend_home / "toolkit" / "tools" / "ascendc_tools" / "npu_kernel_launch",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in lib_paths if path.exists()) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["TILELANG_ASCEND_HOME"] = str(REPO_ROOT)
    return env


def _write_kernel(path: Path, m: int, n: int, k: int) -> None:
    a_elems = m * k
    b_elems = k * n
    c_elems = m * n
    source = KERNEL_TEMPLATE.substitute(
        m=m,
        n=n,
        k=k,
        a_elems=a_elems,
        b_elems=b_elems,
        c_elems=c_elems,
        a_bytes=a_elems * 2,
        ab_bytes=(a_elems + b_elems) * 2,
    )
    path.write_text(source, encoding="utf-8")


def _copy_kernel(kernel_src: Path, dst: Path, core_type: str) -> None:
    source = kernel_src.read_text(encoding="utf-8")
    if "TL_ASCEND_310P" not in source:
        source = "#define TL_ASCEND_310P 1\n" + source
    source = _normalize_external_kernel_source(source, core_type)
    dst.write_text(source, encoding="utf-8")


def _normalize_external_kernel_source(source: str, core_type: str) -> str:
    """Keep generated TileLang source focused on one CAModel core type."""
    source = source.replace('#include "acl/acl.h"\n', "")
    source = source.replace("#include <runtime/rt_ffts.h>\n", "")
    source = source.replace(", uint64_t fftsAddr) {", ") {")
    source = source.replace("  pipe.Destroy();\n", "")
    if core_type == "VectorCore":
        source = source.replace("KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);",
                                "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);")
        source = source.replace("KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);",
                                "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);")
        source = _select_core_branch(source, "ASCEND_IS_AIV")
    else:
        source = source.replace("KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);",
                                "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);")
        source = source.replace("KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);",
                                "KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);")
        source = _select_core_branch(source, "ASCEND_IS_AIC")
    source = re.sub(r"\nvoid\s+\w+_tiling\([^{}]*\)\s*\{\s*\}\n", "\n", source)
    source = re.sub(
        r'\nextern\s+"C"\s+void\s+call\([^{}]*\)\s*\{.*?\n\}\s*$',
        "\n",
        source,
        flags=re.DOTALL,
    )
    return source


def _select_core_branch(source: str, keep_condition: str) -> str:
    for condition in ("ASCEND_IS_AIV", "ASCEND_IS_AIC"):
        source = _rewrite_condition_blocks(
            source, condition, keep=(condition == keep_condition)
        )
    return source


def _rewrite_condition_blocks(source: str, condition: str, keep: bool) -> str:
    pattern = f"if {condition} {{"
    cursor = 0
    result: list[str] = []
    while True:
        start = source.find(pattern, cursor)
        if start < 0:
            result.append(source[cursor:])
            break
        line_start = source.rfind("\n", cursor, start) + 1
        indent = source[line_start:start]
        result.append(source[cursor:line_start])
        body_start = start + len(pattern)
        depth = 1
        pos = body_start
        while pos < len(source) and depth:
            if source[pos] == "{":
                depth += 1
            elif source[pos] == "}":
                depth -= 1
            pos += 1
        if depth:
            result.append(source[line_start:])
            return "".join(result)
        body = source[body_start:pos - 1]
        if keep:
            result.append(indent + "{")
            result.append(body)
            result.append("\n" + indent + "}")
        cursor = pos
        if cursor < len(source) and source[cursor] == "\n":
            cursor += 1
    return "".join(result)


def _write_data(work_dir: Path, m: int, n: int, k: int) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    c = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    a_path = work_dir / "A.bin"
    b_path = work_dir / "B.bin"
    c_path = work_dir / "C_golden.bin"
    c_out_path = work_dir / "C_out.bin"
    a.tofile(a_path)
    b.tofile(b_path)
    c.tofile(c_path)
    np.zeros((m, n), dtype=np.float16).tofile(c_out_path)
    return a_path, b_path, c_path


def _compile_kernel(
    env: dict[str, str],
    ascend_home: Path,
    work_dir: Path,
    kernel_cpp: Path,
    opt_level: str,
    core_type: str = "AiCore",
) -> Path:
    kernel_o = work_dir / "main_kernel.o"
    logs_dir = work_dir / "logs"
    includes = [
        ascend_home / "x86_64-linux" / "include",
        ascend_home / "x86_64-linux" / "include" / "ascendc",
        ascend_home / "x86_64-linux" / "include" / "ascendc" / "basic_api",
        ascend_home / "x86_64-linux" / "include" / "ascendc" / "basic_api" / "impl",
        ascend_home / "x86_64-linux" / "include" / "ascendc" / "basic_api" / "interface",
        ascend_home / "x86_64-linux" / "include" / "ascendc" / "basic_api" / "inner_interface",
        ascend_home / "x86_64-linux" / "ascendc" / "include",
        ascend_home / "x86_64-linux" / "ascendc" / "include" / "basic_api",
        ascend_home / "x86_64-linux" / "ascendc" / "include" / "basic_api" / "impl",
        ascend_home / "x86_64-linux" / "ascendc" / "include" / "basic_api" / "interface",
        ascend_home / "x86_64-linux" / "ascendc" / "include" / "basic_api" / "inner_interface",
        ascend_home / "x86_64-linux" / "tikcpp",
        ascend_home / "x86_64-linux" / "tikcpp" / "tikcfw",
        ascend_home / "x86_64-linux" / "tikcpp" / "tikcfw" / "impl",
        ascend_home / "x86_64-linux" / "tikcpp" / "tikcfw" / "interface",
        ascend_home / "x86_64-linux" / "tikcpp" / "tikcfw" / "inner_interface",
        ascend_home / "include",
        ascend_home / "include" / "experiment" / "runtime",
        ascend_home / "include" / "experiment" / "msprof",
        ascend_home / "compiler" / "tikcpp",
        ascend_home / "compiler" / "tikcpp" / "tikcfw",
        ascend_home / "compiler" / "tikcpp" / "tikcfw" / "impl",
        ascend_home / "compiler" / "tikcpp" / "tikcfw" / "interface",
    ]
    tmp_o = work_dir / "main_kernel_tmp.o"
    compile_cmd = [
        "ccec",
        "-c",
        "-xcce",
        str(kernel_cpp),
        "--cce-aicore-arch=dav-m200",
        "-Dmain_kernel=main_kernel_1",
        "-D__NPU_TILING__",
        opt_level,
        "-std=c++17",
        "-mllvm",
        "-cce-aicore-function-stack-size=16000",
        "-mllvm",
        "-cce-aicore-fp-ceiling=0",
        "-mllvm",
        "-cce-aicore-record-overflow=false",
        "-DTL_ASCEND_310P=1",
        f"-I{REPO_ROOT / 'src'}",
        f"-I{REPO_ROOT}",
        f"-I{REPO_ROOT / '3rdparty' / 'catlass' / 'include'}",
        "-o",
        str(tmp_o),
    ]
    if core_type == "VectorCore":
        compile_cmd.insert(compile_cmd.index("-std=c++17"), "--cce-aiv")
    else:
        compile_cmd.insert(compile_cmd.index("-std=c++17"), "--cce-aicore-only")
    for include in includes:
        compile_cmd.insert(-2, f"-I{include}")
    _run(compile_cmd, env=env, cwd=work_dir, log_path=logs_dir / "compile.log")
    if core_type == "VectorCore":
        shutil.copy2(tmp_o, kernel_o)
        return kernel_o
    _run(
        ["ld.lld", "-m", "aicorelinux", "-Ttext=0", str(tmp_o), "-static", "-n", "-o", str(kernel_o)],
        env=env,
        cwd=work_dir,
        log_path=logs_dir / "link.log",
    )
    return kernel_o


def _run_camodel_api(
    env: dict[str, str],
    work_dir: Path,
    kernel_o: Path,
    core_type: str,
    timeout: int,
    m: int,
    n: int,
    k: int,
) -> None:
    script = work_dir / "run_camodel_api.py"
    script.write_text(
        f"""
from ascendebug import DebugOp, OpExecutor, RunSimuOptions, TilingInfo, NpuCompileInfo

def main():
    debug_op = DebugOp("Tilelang310PGemm", core_type="{core_type}", chip_version="Ascend310P1")
    debug_op.custom_input("A", "float16", [{m}, {k}], r"{work_dir / 'A.bin'}")
    debug_op.custom_input("B", "float16", [{k}, {n}], r"{work_dir / 'B.bin'}")
    debug_op.custom_output("C", "float16", [{m}, {n}], r"{work_dir / 'C_out.bin'}")

    executor = OpExecutor(debug_op, r"{work_dir / 'ascendebug_workspace'}", r"{str(_find_ascend_home().parent)}")
    executor.run_camodel(
        r"{kernel_o}",
        RunSimuOptions(block_num=1, timeout={timeout}),
        NpuCompileInfo(syncall=False),
        TilingInfo("", 0, 1, 1),
    )

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )
    _run([sys.executable, str(script)], env=env, cwd=work_dir, log_path=work_dir / "logs" / "camodel.log")


def _compare_outputs(work_dir: Path, golden_path: Path, m: int, n: int) -> None:
    output_path = (
        work_dir
        / "ascendebug_workspace"
        / "Tilelang310PGemm"
        / "simulator"
        / "output"
        / "C_out.bin"
    )
    if not output_path.exists():
        raise FileNotFoundError(f"CAModel output was not found: {output_path}")
    actual = np.fromfile(output_path, dtype=np.float16).reshape(m, n)
    expect = np.fromfile(golden_path, dtype=np.float16).reshape(m, n)
    np.testing.assert_allclose(actual, expect, rtol=1e-2, atol=1e-2)
    shutil.copy2(output_path, work_dir / "C_actual.bin")
    message = f"CAModel output matches golden: {output_path}"
    (work_dir / "logs" / "compare.log").write_text(message + "\n", encoding="utf-8")
    print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and run a 310P GEMM CAModel smoke test.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--kernel-cpp", type=Path, default=None,
                        help="Compile and run this generated kernel source instead of the built-in smoke kernel.")
    parser.add_argument("--skip-run", action="store_true", help="Only generate data/source and compile the .o file.")
    parser.add_argument("--compile-opt-level", default=None, choices=["-O0", "-O1", "-O2", "-O3"],
                        help="Override the ccec optimization level. External generated kernels default to -O0.")
    parser.add_argument("--core-type", choices=["AiCore", "VectorCore"], default="AiCore",
                        help="CAModel core type for external generated kernels.")
    parser.add_argument("--timeout", type=int, default=600, help="CAModel launch timeout in seconds.")
    args = parser.parse_args()

    if args.kernel_cpp is None and (args.m % 16 or args.n % 16 or args.k % 16):
        raise ValueError("M, N, and K must be multiples of 16 for the built-in smoke kernel.")

    ascend_home = _find_ascend_home()
    env = _prepare_env(ascend_home)

    work_dir = args.work_dir.expanduser().resolve()
    kernel_src = args.kernel_cpp.expanduser().resolve() if args.kernel_cpp is not None else None

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    kernel_cpp = work_dir / "main_kernel.cpp"
    if kernel_src is None:
        _write_kernel(kernel_cpp, args.m, args.n, args.k)
    else:
        _copy_kernel(kernel_src, kernel_cpp, args.core_type)
    _, _, golden_path = _write_data(work_dir, args.m, args.n, args.k)
    opt_level = args.compile_opt_level or ("-O3" if kernel_src is None else "-O0")
    kernel_o = _compile_kernel(env, ascend_home, work_dir, kernel_cpp, opt_level, args.core_type)
    print(f"310P kernel object: {kernel_o}")

    if not args.skip_run:
        _run_camodel_api(env, work_dir, kernel_o, args.core_type, args.timeout, args.m, args.n, args.k)
        _compare_outputs(work_dir, golden_path, args.m, args.n)


if __name__ == "__main__":
    main()
