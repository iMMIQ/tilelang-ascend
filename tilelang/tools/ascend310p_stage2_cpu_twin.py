#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np

from ascendebug import CpuOptions, DebugOp, OpExecutor, OpKernelInfo

import tilelang
import tilelang.language as T
from tilelang.tools import ascend310p_gemm_camodel as camodel
from tilelang.tools.ascend310p_stage2_camodel import (
    CASES,
    DTYPE_TO_ASCENDEBUG,
    DTYPE_TO_NUMPY,
    Stage2Case,
    TensorSpec,
    _copy_kernel_for_case,
    _lower_case,
    _write_array,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "debug_310p_stage2_cpu_twin"
DEFAULT_CASE = "activation_silu"
NO_AUTO_SYNC_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
NO_AUTO_SYNC_CASES = {
    "example_flash_attention_example_style",
    "example_gemm_310p_small",
    "example_shmem_get_nbi",
    "example_shmem_put_nbi",
    "example_shmem_ub_get_nbi",
    "example_shmem_ub_put_nbi",
}


def _example_online_softmax_program():
    m = 1
    n = 32
    block_m = 1
    block_n = 16
    vec_num = 1
    sub_block_m = block_m // vec_num
    n_num = T.ceildiv(n, block_n)

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), B: T.Tensor((m, n), "float32")):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(1, is_npu=True) as (cid, vid):
            row_start = 0
            a = T.alloc_ub((sub_block_m, block_n), "float32")
            tile_max = T.alloc_ub((sub_block_m, 1), "float32")
            tile_max_2d = T.alloc_ub((sub_block_m, block_n), "float32")
            prev_max = T.alloc_ub((sub_block_m, 1), "float32")
            prev_max_2d = T.alloc_ub((sub_block_m, block_n), "float32")
            tile_sum = T.alloc_ub((sub_block_m, 1), "float32")
            prev_sum = T.alloc_ub((sub_block_m, 1), "float32")
            prev_sum_2d = T.alloc_ub((sub_block_m, block_n), "float32")
            tmp_exp = T.alloc_ub((sub_block_m, 1), "float32")

            T.tile.fill(prev_max, -T.infinity("float32"))
            T.tile.fill(prev_sum, 0.0)
            for by in T.serial(n_num):
                col_start = by * block_n
                T.copy(A[row_start : row_start + sub_block_m, col_start : col_start + block_n], a)
                T.reduce_max(a, tile_max, dim=-1)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)
                T.tile.mul(tmp_exp, prev_sum, tmp_exp)
                T.tile.broadcast(tile_max_2d, tile_max)
                T.tile.sub(a, a, tile_max_2d)
                T.tile.exp(a, a)
                T.reduce_sum(a, tile_sum, dim=-1)
                T.tile.add(prev_sum, tile_sum, tmp_exp)
                T.tile.add(prev_max, tile_max, 0.0)

            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)
            for by in T.serial(n_num):
                col_start = by * block_n
                T.copy(A[row_start : row_start + sub_block_m, col_start : col_start + block_n], a)
                T.tile.sub(a, a, prev_max_2d)
                T.tile.exp(a, a)
                T.tile.div(a, a, prev_sum_2d)
                T.copy(a, B[row_start : row_start + sub_block_m, col_start : col_start + block_n])

    return main


def _example_rms_norm_program():
    m = 1
    n = 32
    block_n = 16
    rows = 1
    n_num = 2

    @T.prim_func
    def main(A: T.Tensor((m, n), "float32"), B: T.Tensor((m, n), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            row_start = 0
            a_ub_0 = T.alloc_ub((rows, block_n), "float32")
            a_ub_1 = T.alloc_ub((rows, block_n), "float32")
            sum_sq_acc = T.alloc_ub((rows, block_n), "float32")
            sum_sq_row = T.alloc_ub((rows, 1), "float32")
            inv_rms_ub = T.alloc_ub((rows, 1), "float32")
            inv_rms_tile = T.alloc_ub((rows, block_n), "float32")

            T.tile.fill(sum_sq_acc, 0.0)
            for by in T.serial(n_num // 2):
                col_start_0 = (by * 2) * block_n
                T.copy(A[row_start : row_start + rows, col_start_0 : col_start_0 + block_n], a_ub_0)
                T.tile.mul(a_ub_0, a_ub_0, a_ub_0)
                T.tile.add(sum_sq_acc, sum_sq_acc, a_ub_0)

                col_start_1 = (by * 2 + 1) * block_n
                T.copy(A[row_start : row_start + rows, col_start_1 : col_start_1 + block_n], a_ub_1)
                T.tile.mul(a_ub_1, a_ub_1, a_ub_1)
                T.tile.add(sum_sq_acc, sum_sq_acc, a_ub_1)

            T.reduce_sum(sum_sq_acc, sum_sq_row, dim=-1)
            T.tile.div(sum_sq_row, sum_sq_row, float(n))
            T.tile.add(sum_sq_row, sum_sq_row, 1.0e-5)
            T.tile.rsqrt(inv_rms_ub, sum_sq_row)
            T.tile.broadcast(inv_rms_tile, inv_rms_ub)

            for by in T.serial(n_num // 2):
                col_start_0 = (by * 2) * block_n
                T.copy(A[row_start : row_start + rows, col_start_0 : col_start_0 + block_n], a_ub_0)
                T.tile.mul(a_ub_0, a_ub_0, inv_rms_tile)
                T.copy(a_ub_0, B[row_start : row_start + rows, col_start_0 : col_start_0 + block_n])

                col_start_1 = (by * 2 + 1) * block_n
                T.copy(A[row_start : row_start + rows, col_start_1 : col_start_1 + block_n], a_ub_1)
                T.tile.mul(a_ub_1, a_ub_1, inv_rms_tile)
                T.copy(a_ub_1, B[row_start : row_start + rows, col_start_1 : col_start_1 + block_n])

    return main


def _example_gemv_program():
    n = 8
    k = 16
    block_n = 8
    block_k = 16
    vec_num = 2
    n_num = T.ceildiv(n, block_n)
    k_num = T.ceildiv(k, block_k)
    kernel_num = T.ceildiv(n_num, vec_num)

    @T.prim_func
    def main(
        x: T.Tensor((k,), "float32"),
        A: T.Tensor((n, k), "float32"),
        y: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(kernel_num, is_npu=True) as (cid, vid):
            bn = (cid * vec_num + vid) % n_num
            x_ub = T.alloc_ub((1, block_k), "float32")
            a_ub = T.alloc_ub((block_n, block_k), "float32")
            y_single_ub = T.alloc_ub((block_n,), "float32")
            y_total_ub = T.alloc_ub((block_n,), "float32")

            T.tile.fill(y_total_ub, 0.0)
            for bk in T.serial(k_num):
                T.copy(x[bk * block_k], x_ub)
                T.copy(A[bn * block_n, bk * block_k], a_ub)
                for i in T.serial(block_n):
                    T.tile.mul(a_ub[i, :], a_ub[i, :], x_ub)
                T.reduce_sum(a_ub, y_single_ub, dim=-1)
                T.tile.add(y_total_ub, y_total_ub, y_single_ub)

            T.copy(y_total_ub, y[bn * block_n])

    return main


def _example_flash_attention_program():
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
            q_ub = T.alloc_ub((block_m, dim), "float16")
            k_ub = T.alloc_ub((block_n, dim), "float16")
            v_ub = T.alloc_ub((block_n, dim), "float16")
            scores = T.alloc_ub((block_m, block_n), "float16")
            scores_half = T.alloc_ub((block_m, block_n), "float16")
            row_sum = T.alloc_ub((block_m,), "float16")
            out = T.alloc_ub((block_m, dim), "float16")
            out_half = T.alloc_ub((block_m, dim), "float16")

            T.copy(Q[0, 0], q_ub)
            T.copy(K[0, 0], k_ub)
            T.copy(V[0, 0], v_ub)

            T.tile.fill(scores, 0.0)
            for i in T.serial(block_m):
                for j in T.serial(block_n):
                    for kk in T.serial(dim):
                        scores[i, j] = (
                            scores[i, j]
                            + q_ub[i, kk] * k_ub[j, kk]
                        )
                    scores[i, j] = scores[i, j] * sm_scale

            T.tile.exp(scores, scores)
            T.tile.fill(row_sum, 0.0)
            for i in T.serial(block_m):
                for j in T.serial(block_n):
                    row_sum[i] = row_sum[i] + scores[i, j]
            for i in T.serial(block_m):
                for j in T.serial(block_n):
                    scores[i, j] = scores[i, j] / row_sum[i]
            T.copy(scores, WorkspaceScores[0, 0])
            T.copy(scores, scores_half)

            T.tile.fill(out, 0.0)
            for i in T.serial(block_m):
                for d in T.serial(dim):
                    for j in T.serial(block_n):
                        out[i, d] = (
                            out[i, d]
                            + scores_half[i, j] * v_ub[j, d]
                        )

            T.copy(out, WorkspaceOut[0, 0])
            T.copy(out, out_half)
            T.copy(out_half, O[0, 0])

    return main


def _example_flash_attention_example_style_program():
    block_m, block_n, dim = 8, 8, 8
    sm_scale = (1.0 / dim) ** 0.5

    @T.prim_func
    def main(
        Q: T.Tensor((1, 1, block_m, dim), "float16"),
        K: T.Tensor((1, 1, block_n, dim), "float16"),
        V: T.Tensor((1, 1, block_n, dim), "float16"),
        O: T.Tensor((1, 1, block_m, dim), "float16"),
        WorkspaceScores: T.Tensor((1, 1, block_m, block_n), "float32"),
        WorkspaceProbs: T.Tensor((1, 1, block_m, block_n), "float16"),
        WorkspaceOut: T.Tensor((1, 1, block_m, dim), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1((block_m, dim), "float16")
            k_l1 = T.alloc_L1((block_n, dim), "float16")
            v_l1 = T.alloc_L1((block_n, dim), "float16")
            probs_l1 = T.alloc_L1((block_m, block_n), "float16")
            scores_l0c = T.alloc_L0C((block_m, block_n), "float32")
            scores_ub = T.alloc_ub((block_m, block_n), "float32")
            scores_half = T.alloc_ub((block_m, block_n), "float16")
            max_ub = T.alloc_ub((block_m,), "float32")
            prev_max = T.alloc_ub((block_m,), "float32")
            sum_ub = T.alloc_ub((block_m,), "float32")
            out_l0c = T.alloc_L0C((block_m, dim), "float32")
            out_ub = T.alloc_ub((block_m, dim), "float32")
            out_half = T.alloc_ub((block_m, dim), "float16")

            with T.Scope("C"):
                T.copy(Q[0, 0, 0, 0], q_l1)
                T.copy(K[0, 0, 0, 0], k_l1)
                T.copy(V[0, 0, 0, 0], v_l1)
                T.gemm_v0(q_l1, k_l1, scores_l0c, transpose_B=True, init=True)
                T.copy(scores_l0c, WorkspaceScores[0, 0, 0, 0])
                T.set_cross_flag("FIX", 0)

            with T.Scope("V"):
                T.wait_cross_flag(0)
                T.copy(WorkspaceScores[0, 0, 0, 0], scores_ub)
                T.tile.fill(prev_max, -T.infinity("float32"))
                T.tile.mul(scores_ub, scores_ub, sm_scale)
                T.reduce_max(scores_ub, max_ub, dim=-1)
                T.tile.max(max_ub, max_ub, prev_max)
                for row in range(block_m):
                    T.tile.sub(scores_ub[row, :], scores_ub[row, :], max_ub[row])
                T.tile.exp(scores_ub, scores_ub)
                T.tile.fill(sum_ub, 0.0)
                T.reduce_sum(scores_ub, sum_ub, dim=-1)
                for row in range(block_m):
                    T.tile.div(scores_ub[row, :], scores_ub[row, :], sum_ub[row])
                T.tile.cast(scores_half, scores_ub, "CAST_NONE", block_m * block_n)
                T.copy(scores_half, WorkspaceProbs[0, 0, 0, 0])
                T.set_cross_flag("V", 1)

            with T.Scope("C"):
                T.wait_cross_flag(1)
                T.copy(WorkspaceProbs[0, 0, 0, 0], probs_l1)
                T.gemm_v0(probs_l1, v_l1, out_l0c, init=True)
                T.copy(out_l0c, WorkspaceOut[0, 0, 0, 0])
                T.set_cross_flag("FIX", 2)

            with T.Scope("V"):
                T.wait_cross_flag(2)
                T.copy(WorkspaceOut[0, 0, 0, 0], out_ub)
                T.tile.cast(out_half, out_ub, "CAST_NONE", block_m * dim)
                T.copy(out_half, O[0, 0, 0, 0])

    return main


def _example_gemm_program():
    m = 16
    n = 16
    k = 16
    block_m = 16
    block_n = 16
    block_k = 16

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        C: T.Tensor((m, n), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1((block_m, block_k), "float16")
            b_l1 = T.alloc_L1((block_k, block_n), "float16")
            c_l0 = T.alloc_L0C((block_m, block_n), "float")

            T.copy(A[0, 0], a_l1)
            T.copy(B[0, 0], b_l1)
            T.gemm_v0(a_l1, b_l1, c_l0, init=True)
            T.copy(c_l0, C[0, 0])

    return main


def _example_copy_roundtrip_program():
    m = 16
    n = 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "float16"), B: T.Tensor((m, n), "float16")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_l1 = T.alloc_L1((m, n), "float16")
            T.copy(A[0, 0], a_l1)
            T.copy(a_l1, B[0, 0])

    return main


def _example_shmem_get_nbi_program():
    m = 1
    n = 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "int8"), B: T.Tensor((m, n), "int8")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            T.shmem_get_nbi(B, A, m * n, 0)

    return main


def _example_shmem_put_nbi_program():
    m = 1
    n = 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "int8"), B: T.Tensor((m, n), "int8")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            T.shmem_put_nbi(B, A, m * n, 0)

    return main


def _example_shmem_ub_get_nbi_program():
    m = 1
    n = 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "int8"), B: T.Tensor((m, n), "int8")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((m, n), "int8")
            T.shmem_ub_get_nbi(a_ub, A, m * n, 0)
            T.copy(a_ub, B)

    return main


def _example_shmem_ub_put_nbi_program():
    m = 1
    n = 16

    @T.prim_func
    def main(A: T.Tensor((m, n), "int8"), B: T.Tensor((m, n), "int8")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((m, n), "int8")
            T.copy(A, a_ub)
            T.shmem_ub_put_nbi(a_ub, B, m * n, 0, 0)

    return main


def _example_dispatch_combine_experiment_program():
    n = 8

    @T.prim_func
    def main(
        A: T.Tensor((1, n), "float32"),
        B: T.Tensor((1, n), "float32"),
        C: T.Tensor((1, n), "float32"),
        D: T.Tensor((1,), "float32"),
        E: T.Tensor((1, n), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                a_ub = T.alloc_ub((1, n), "float32")
                b_ub = T.alloc_ub((1, n), "float32")
                sub_ub = T.alloc_ub((1, n), "float32")
                abs_ub = T.alloc_ub((1, n), "float32")
                min_ub = T.alloc_ub((1, n), "float32")
                sum_ub = T.alloc_ub((1,), "float32")

                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.tile.sub_experiment(sub_ub, a_ub, b_ub, n)
                T.tile.abs_experiment(abs_ub, sub_ub, n)
                T.tile.mins_experiment(min_ub, abs_ub, 2.5, n)
                T.tile.reduce_sum_experiment(sum_ub, min_ub, n)
                T.copy(abs_ub, C)
                T.copy(sum_ub, D)
                T.copy(min_ub, E)

    return main


def _example_reduce_sum_mask_experiment_program():
    n = 8

    @T.prim_func
    def main(A: T.Tensor((1, n), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                a_ub = T.alloc_ub((1, n), "float32")
                b_ub = T.alloc_ub((1,), "float32")
                T.copy(A, a_ub)
                T.tile.reduce_sum_mask_experiment(b_ub, a_ub, 1, 1, 1)
                T.copy(b_ub, B)

    return main


def _example_gathermask_sum_experiment_program():
    n = 8

    @T.prim_func
    def main(
        A: T.Tensor((1, n), "float32"),
        P: T.Tensor((1,), "uint32"),
        C: T.Tensor((1, n), "float32"),
        D: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                a_ub = T.alloc_ub((1, n), "float32")
                p_ub = T.alloc_ub((1,), "uint32")
                c_ub = T.alloc_ub((1, n), "float32")
                d_ub = T.alloc_ub((1,), "float32")

                T.copy(A, a_ub)
                T.copy(P, p_ub)
                T.tile.gathermask_experiment(c_ub, a_ub, p_ub, True, 2, [1, 1, 1, 0], 0)
                T.tile.sum_experiment(d_ub, c_ub, [1, n, n])
                T.copy(c_ub, C)
                T.copy(d_ub, D)

    return main


def _make_online_softmax_data(work_dir: Path):
    rng = np.random.default_rng(10)
    a = rng.uniform(-4.0, 4.0, size=(1, 32)).astype(np.float32)
    shifted = a - np.max(a, axis=1, keepdims=True)
    exp = np.exp(shifted)
    b = (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_rms_norm_example_data(work_dir: Path):
    rng = np.random.default_rng(11)
    a = rng.uniform(-2.0, 2.0, size=(1, 32)).astype(np.float32)
    inv_rms = 1.0 / np.sqrt(np.mean(a * a, axis=1, keepdims=True) + 1.0e-5)
    b = (a * inv_rms).astype(np.float32)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_gemv_example_data(work_dir: Path):
    rng = np.random.default_rng(12)
    x = rng.standard_normal((16,)).astype(np.float32)
    a = rng.standard_normal((8, 16)).astype(np.float32)
    y = (a @ x).astype(np.float32)
    return (
        {
            "x": _write_array(work_dir / "x.bin", x),
            "A": _write_array(work_dir / "A.bin", a),
        },
        {"y": work_dir / "y_out.bin"},
        {"y": _write_array(work_dir / "y_golden.bin", y)},
    )


def _make_flash_attention_data(work_dir: Path):
    rng = np.random.default_rng(13)
    q = rng.standard_normal((16, 16)).astype(np.float16)
    k = rng.standard_normal((16, 16)).astype(np.float16)
    v = rng.standard_normal((16, 16)).astype(np.float16)
    scores = (q.astype(np.float32) @ k.astype(np.float32).T) * np.float32(0.25)
    scores_norm = scores - np.max(scores, axis=1, keepdims=True)
    scores_norm = np.exp(scores_norm).astype(np.float32)
    row_sum = np.sum(scores_norm, axis=1, keepdims=True)
    scores_norm = (scores_norm / row_sum).astype(np.float16)
    out = (scores_norm.astype(np.float32) @ v.astype(np.float32)).astype(np.float16)
    return (
        {
            "Q": _write_array(work_dir / "Q.bin", q),
            "K": _write_array(work_dir / "K.bin", k),
            "V": _write_array(work_dir / "V.bin", v),
        },
        {
            "O": work_dir / "O_out.bin",
            "WorkspaceScores": work_dir / "WorkspaceScores_out.bin",
            "WorkspaceOut": work_dir / "WorkspaceOut_out.bin",
        },
        {
            "O": _write_array(work_dir / "O_golden.bin", out),
            "WorkspaceScores": _write_array(work_dir / "WorkspaceScores_golden.bin", scores_norm),
            "WorkspaceOut": _write_array(work_dir / "WorkspaceOut_golden.bin", out),
        },
    )


def _make_flash_attention_example_style_data(work_dir: Path):
    rng = np.random.default_rng(14)
    q = rng.standard_normal((1, 1, 8, 8)).astype(np.float16)
    k = rng.standard_normal((1, 1, 8, 8)).astype(np.float16)
    v = rng.standard_normal((1, 1, 8, 8)).astype(np.float16)
    raw_scores = q.astype(np.float32) @ k.astype(np.float32).transpose(0, 1, 3, 2)
    scores = raw_scores * np.float32((1.0 / 8) ** 0.5)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    probs_half = probs.astype(np.float16)
    out_accum = probs_half.astype(np.float32) @ v.astype(np.float32)
    out = out_accum.astype(np.float16)
    return (
        {
            "Q": _write_array(work_dir / "Q.bin", q),
            "K": _write_array(work_dir / "K.bin", k),
            "V": _write_array(work_dir / "V.bin", v),
        },
        {
            "O": work_dir / "O_out.bin",
            "WorkspaceScores": work_dir / "WorkspaceScores_out.bin",
            "WorkspaceProbs": work_dir / "WorkspaceProbs_out.bin",
            "WorkspaceOut": work_dir / "WorkspaceOut_out.bin",
        },
        {
            "O": _write_array(work_dir / "O_golden.bin", out),
            "WorkspaceScores": _write_array(
                work_dir / "WorkspaceScores_golden.bin", raw_scores.astype(np.float32)
            ),
            "WorkspaceProbs": _write_array(
                work_dir / "WorkspaceProbs_golden.bin", probs_half
            ),
            "WorkspaceOut": _write_array(
                work_dir / "WorkspaceOut_golden.bin", out_accum.astype(np.float32)
            ),
        },
    )


def _make_gemm_example_data(work_dir: Path):
    rng = np.random.default_rng(15)
    a = rng.standard_normal((16, 16)).astype(np.float16)
    b = rng.standard_normal((16, 16)).astype(np.float16)
    c = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    return (
        {
            "A": _write_array(work_dir / "A.bin", a),
            "B": _write_array(work_dir / "B.bin", b),
        },
        {"C": work_dir / "C_out.bin"},
        {"C": _write_array(work_dir / "C_golden.bin", c)},
    )


def _make_copy_roundtrip_data(work_dir: Path):
    rng = np.random.default_rng(16)
    a = rng.standard_normal((16, 16)).astype(np.float16)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", a)},
    )


def _make_shmem_get_put_data(work_dir: Path):
    rng = np.random.default_rng(17)
    a = rng.integers(-128, 127, size=(1, 16), dtype=np.int8)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", a)},
    )


def _make_dispatch_combine_experiment_data(work_dir: Path):
    rng = np.random.default_rng(23)
    a = rng.uniform(-4.0, 4.0, size=(1, 8)).astype(np.float32)
    b = rng.uniform(-4.0, 4.0, size=(1, 8)).astype(np.float32)
    sub = a - b
    abs_out = np.abs(sub)
    min_out = np.minimum(abs_out, np.float32(2.5))
    sum_out = np.array([min_out.sum()], dtype=np.float32)
    return (
        {
            "A": _write_array(work_dir / "A.bin", a),
            "B": _write_array(work_dir / "B.bin", b),
        },
        {
            "C": work_dir / "C_out.bin",
            "D": work_dir / "D_out.bin",
            "E": work_dir / "E_out.bin",
        },
        {
            "C": _write_array(work_dir / "C_golden.bin", abs_out),
            "D": _write_array(work_dir / "D_golden.bin", sum_out),
            "E": _write_array(work_dir / "E_golden.bin", min_out),
        },
    )


def _make_reduce_sum_mask_experiment_data(work_dir: Path):
    rng = np.random.default_rng(24)
    a = np.zeros((1, 8), dtype=np.float32)
    a[0, 0] = rng.uniform(-4.0, 4.0)
    b = np.array([a[0, 0]], dtype=np.float32)
    return (
        {"A": _write_array(work_dir / "A.bin", a)},
        {"B": work_dir / "B_out.bin"},
        {"B": _write_array(work_dir / "B_golden.bin", b)},
    )


def _make_gathermask_sum_experiment_data(work_dir: Path):
    rng = np.random.default_rng(25)
    a = np.zeros((1, 8), dtype=np.float32)
    a[0, 1] = rng.uniform(-4.0, 4.0)
    c = np.zeros((1, 8), dtype=np.float32)
    c[0, 0] = a[0, 1]
    d = np.array([a[0, 1]], dtype=np.float32)
    p = np.array([2], dtype=np.uint32)
    return (
        {
            "A": _write_array(work_dir / "A.bin", a),
            "P": _write_array(work_dir / "P.bin", p),
        },
        {
            "C": work_dir / "C_out.bin",
            "D": work_dir / "D_out.bin",
        },
        {
            "C": _write_array(work_dir / "C_golden.bin", c),
            "D": _write_array(work_dir / "D_golden.bin", d),
        },
    )


EXAMPLE_CASES = {
    "example_online_softmax": Stage2Case(
        name="example_online_softmax",
        program_factory=_example_online_softmax_program,
        inputs=(TensorSpec("A", "float32", (1, 32)),),
        outputs=(TensorSpec("B", "float32", (1, 32)),),
        make_data=_make_online_softmax_data,
        rtol=1e-4,
        atol=1e-4,
    ),
    "example_rms_norm_streaming": Stage2Case(
        name="example_rms_norm_streaming",
        program_factory=_example_rms_norm_program,
        inputs=(TensorSpec("A", "float32", (1, 32)),),
        outputs=(TensorSpec("B", "float32", (1, 32)),),
        make_data=_make_rms_norm_example_data,
        rtol=2e-3,
        atol=2e-3,
    ),
    "example_gemv_vector": Stage2Case(
        name="example_gemv_vector",
        program_factory=_example_gemv_program,
        inputs=(
            TensorSpec("x", "float32", (16,)),
            TensorSpec("A", "float32", (8, 16)),
        ),
        outputs=(TensorSpec("y", "float32", (8,)),),
        make_data=_make_gemv_example_data,
        rtol=1e-4,
        atol=1e-4,
    ),
    "example_flash_attention_shape": Stage2Case(
        name="example_flash_attention_shape",
        program_factory=_example_flash_attention_program,
        inputs=(
            TensorSpec("Q", "float16", (16, 16)),
            TensorSpec("K", "float16", (16, 16)),
            TensorSpec("V", "float16", (16, 16)),
        ),
        outputs=(
            TensorSpec("O", "float16", (16, 16)),
            TensorSpec("WorkspaceScores", "float16", (16, 16)),
            TensorSpec("WorkspaceOut", "float16", (16, 16)),
        ),
        make_data=_make_flash_attention_data,
        rtol=1e-2,
        atol=1e-2,
    ),
    "example_flash_attention_example_style": Stage2Case(
        name="example_flash_attention_example_style",
        program_factory=_example_flash_attention_example_style_program,
        inputs=(
            TensorSpec("Q", "float16", (1, 1, 8, 8)),
            TensorSpec("K", "float16", (1, 1, 8, 8)),
            TensorSpec("V", "float16", (1, 1, 8, 8)),
        ),
        outputs=(
            TensorSpec("O", "float16", (1, 1, 8, 8)),
            TensorSpec("WorkspaceScores", "float32", (1, 1, 8, 8)),
            TensorSpec("WorkspaceProbs", "float16", (1, 1, 8, 8)),
            TensorSpec("WorkspaceOut", "float32", (1, 1, 8, 8)),
        ),
        make_data=_make_flash_attention_example_style_data,
        rtol=2e-2,
        atol=2e-2,
    ),
    "example_gemm_310p_small": Stage2Case(
        name="example_gemm_310p_small",
        program_factory=_example_gemm_program,
        inputs=(
            TensorSpec("A", "float16", (16, 16)),
            TensorSpec("B", "float16", (16, 16)),
        ),
        outputs=(TensorSpec("C", "float16", (16, 16)),),
        make_data=_make_gemm_example_data,
        rtol=1e-2,
        atol=1e-2,
    ),
    "example_copy_roundtrip": Stage2Case(
        name="example_copy_roundtrip",
        program_factory=_example_copy_roundtrip_program,
        inputs=(TensorSpec("A", "float16", (16, 16)),),
        outputs=(TensorSpec("B", "float16", (16, 16)),),
        make_data=_make_copy_roundtrip_data,
        rtol=1e-6,
        atol=1e-6,
    ),
    "example_shmem_get_nbi": Stage2Case(
        name="example_shmem_get_nbi",
        program_factory=_example_shmem_get_nbi_program,
        inputs=(TensorSpec("A", "int8", (1, 16)),),
        outputs=(TensorSpec("B", "int8", (1, 16)),),
        make_data=_make_shmem_get_put_data,
        rtol=1e-6,
        atol=1e-6,
    ),
    "example_shmem_put_nbi": Stage2Case(
        name="example_shmem_put_nbi",
        program_factory=_example_shmem_put_nbi_program,
        inputs=(TensorSpec("A", "int8", (1, 16)),),
        outputs=(TensorSpec("B", "int8", (1, 16)),),
        make_data=_make_shmem_get_put_data,
        rtol=1e-6,
        atol=1e-6,
    ),
    "example_shmem_ub_get_nbi": Stage2Case(
        name="example_shmem_ub_get_nbi",
        program_factory=_example_shmem_ub_get_nbi_program,
        inputs=(TensorSpec("A", "int8", (1, 16)),),
        outputs=(TensorSpec("B", "int8", (1, 16)),),
        make_data=_make_shmem_get_put_data,
        rtol=1e-6,
        atol=1e-6,
    ),
    "example_shmem_ub_put_nbi": Stage2Case(
        name="example_shmem_ub_put_nbi",
        program_factory=_example_shmem_ub_put_nbi_program,
        inputs=(TensorSpec("A", "int8", (1, 16)),),
        outputs=(TensorSpec("B", "int8", (1, 16)),),
        make_data=_make_shmem_get_put_data,
        rtol=1e-6,
        atol=1e-6,
    ),
    "example_dispatch_combine_experiments": Stage2Case(
        name="example_dispatch_combine_experiments",
        program_factory=_example_dispatch_combine_experiment_program,
        inputs=(
            TensorSpec("A", "float32", (1, 8)),
            TensorSpec("B", "float32", (1, 8)),
        ),
        outputs=(
            TensorSpec("C", "float32", (1, 8)),
            TensorSpec("D", "float32", (1,)),
            TensorSpec("E", "float32", (1, 8)),
        ),
        make_data=_make_dispatch_combine_experiment_data,
        rtol=1e-6,
        atol=1e-6,
    ),
    "example_reduce_sum_mask_experiment": Stage2Case(
        name="example_reduce_sum_mask_experiment",
        program_factory=_example_reduce_sum_mask_experiment_program,
        inputs=(TensorSpec("A", "float32", (1, 8)),),
        outputs=(TensorSpec("B", "float32", (1,)),),
        make_data=_make_reduce_sum_mask_experiment_data,
        rtol=1e-6,
        atol=1e-6,
    ),
    "example_gathermask_sum_experiment": Stage2Case(
        name="example_gathermask_sum_experiment",
        program_factory=_example_gathermask_sum_experiment_program,
        inputs=(
            TensorSpec("A", "float32", (1, 8)),
            TensorSpec("P", "uint32", (1,)),
        ),
        outputs=(
            TensorSpec("C", "float32", (1, 8)),
            TensorSpec("D", "float32", (1,)),
        ),
        make_data=_make_gathermask_sum_experiment_data,
        rtol=1e-6,
        atol=1e-6,
    ),
}

ALL_CASES = {**CASES, **EXAMPLE_CASES}


def _install_env() -> dict[str, str]:
    ascend_home = camodel._find_ascend_home()
    env = camodel._prepare_env(ascend_home)
    os.environ.update(env)
    return env


def _ascend_toolkit_root() -> Path:
    ascend_home = camodel._find_ascend_home()
    candidates = [
        ascend_home,
        ascend_home / "toolkit",
        ascend_home.parent / "8.0.RC3",
    ]
    for candidate in candidates:
        if (candidate / "x86_64-linux" / "ascendc" / "include").exists():
            return candidate
    raise RuntimeError(f"Ascend C include root was not found under {ascend_home}")


def _source_include_roots() -> list[Path]:
    toolkit_root = _ascend_toolkit_root()

    return [
        REPO_ROOT / "src",
        REPO_ROOT / "3rdparty/catlass/include",
        toolkit_root / "x86_64-linux/ascendc/include",
        toolkit_root / "x86_64-linux/ascendc/include/basic_api",
        toolkit_root / "x86_64-linux/ascendc/include/basic_api/interface",
        toolkit_root / "x86_64-linux/ascendc/include/basic_api/impl",
        toolkit_root / "x86_64-linux/ascendc/include/highlevel_api",
        toolkit_root / "runtime/include",
        toolkit_root / "x86_64-linux/include",
    ]


def _build_include_root(work_dir: Path) -> Path:
    include_root = work_dir / "cpu_twin_include"
    if include_root.exists():
        shutil.rmtree(include_root)
    include_root.mkdir()

    for source_root in _source_include_roots():
        if not source_root.exists():
            continue
        for child in source_root.iterdir():
            target = include_root / child.name
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(child, target_is_directory=child.is_dir())
    return include_root


def _prepare_case(
    case_name: str,
    root: Path,
) -> tuple[Path, Path, dict[str, Path], dict[str, Path], dict[str, Path]]:
    case = ALL_CASES[case_name]
    work_dir = root / case.name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    if case_name in NO_AUTO_SYNC_CASES:
        source_path = work_dir / f"tilelang_{case_name}.cpp"
        with tilelang.tvm.transform.PassContext(opt_level=3, config=NO_AUTO_SYNC_PASS_CONFIGS):
            artifact = tilelang.lower(case.program_factory(), target="ascendc", platform="310P")
        source_path.write_text(artifact.kernel_source, encoding="utf-8")
    else:
        source_path = _lower_case(case, work_dir)
    kernel_cpp = work_dir / "main_kernel.cpp"
    _copy_kernel_for_case(source_path, kernel_cpp, "AiCore")
    if case_name == "example_flash_attention_example_style":
        text = kernel_cpp.read_text(encoding="utf-8")
        text = re.sub(r"^\s*AscendC::CrossCoreSetFlag<[^>]+>\([^;]+;\n", "", text, flags=re.M)
        text = re.sub(r"^\s*AscendC::CrossCoreWaitFlag\([^;]+;\n", "", text, flags=re.M)
        kernel_cpp.write_text(text, encoding="utf-8")

    input_paths, output_paths, golden_paths = case.make_data(work_dir)
    for spec in case.outputs:
        shutil.copy2(golden_paths[spec.name], output_paths[spec.name])
    include_root = _build_include_root(work_dir)
    return kernel_cpp, include_root, input_paths, output_paths, golden_paths


def _build_op_executor(case_name: str, root: Path) -> tuple[DebugOp, OpExecutor, Path]:
    ascend_home = camodel._find_ascend_home()
    work_dir = root / case_name / "ascendebug_workspace"
    debug_op = DebugOp(
        f"Tilelang310PStage2CpuTwin{case_name}",
        core_type="AiCore",
        chip_version="Ascend310P1",
    )
    executor = OpExecutor(debug_op, str(work_dir), str(ascend_home.parent))
    return debug_op, executor, work_dir


def run_case(case_name: str, root: Path, block_num: int, npucheck: bool) -> None:
    case = ALL_CASES[case_name]
    kernel_cpp, include_root, input_paths, output_paths, golden_paths = _prepare_case(case_name, root)
    debug_op, executor, work_dir = _build_op_executor(case_name, root)

    for spec in case.inputs:
        debug_op.custom_input(
            spec.name,
            DTYPE_TO_ASCENDEBUG[spec.dtype],
            list(spec.shape),
            str(input_paths[spec.name]),
        )
    for spec in case.outputs:
        debug_op.custom_output(
            spec.name,
            DTYPE_TO_ASCENDEBUG[spec.dtype],
            list(spec.shape),
            str(output_paths[spec.name]),
        )

    kernel_info = OpKernelInfo(
        source_file=str(kernel_cpp),
        kernel_name="main_kernel",
        header_files=[str(include_root)],
    )
    cpu_option = CpuOptions(
        npucheck=npucheck,
        dump_mode="",
        rel_err_thd=case.rtol,
        abs_err_thd=case.atol,
    )
    executor.run_call_kernel_cpu(kernel_info, block_num, cpu_option)

    actual_dir = work_dir / debug_op.op_type / "cpu" / "output"
    for spec in case.outputs:
        actual = actual_dir / output_paths[spec.name].name
        if not actual.exists():
            raise FileNotFoundError(f"CPU twin output not found: {actual}")
        actual_arr = np.fromfile(actual, dtype=DTYPE_TO_NUMPY[spec.dtype]).reshape(spec.shape)
        expect_arr = np.fromfile(golden_paths[spec.name], dtype=DTYPE_TO_NUMPY[spec.dtype]).reshape(
            spec.shape
        )
        np.testing.assert_allclose(actual_arr, expect_arr, rtol=case.rtol, atol=case.atol)
        print(f"{case.name}:{spec.name} matches golden")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 310P ascendebug CPU twin precision checks.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--case", choices=sorted(ALL_CASES), action="append")
    parser.add_argument("--block-num", type=int, default=1)
    parser.add_argument("--npucheck", action="store_true")
    args = parser.parse_args()

    _install_env()

    root = args.work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = args.case or [DEFAULT_CASE]
    for case_name in selected:
        print(f"=== {case_name} ===")
        run_case(case_name, root, args.block_num, args.npucheck)


if __name__ == "__main__":
    main()
