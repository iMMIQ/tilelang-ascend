# Ascend 310P Support Plan

This document records the current 310P support status and the recommended
implementation order for making the examples work on the Ascend C backend with
310P CAModel validation.

## Current Status

### Environment and CAModel

- `ASCEND_HOME_PATH` is set to `/home/ayd/Ascend/ascend-toolkit/latest`.
- `ccec` and `ld.lld` are available from the CANN toolchain.
- `ascendebug` is importable.
- `torch` is installed, but `torch_npu` is missing.
- The repository has existing build outputs, including `build/libtilelang.so`,
  `build/libtilelang_module.so`, and TVM libraries.
- Some submodules are not initialized:
  - `3rdparty/composable_kernel`
  - `3rdparty/cutlass`

The machine can run a built-in 310P CAModel smoke test through:

```bash
python3 tilelang/tools/ascend310p_gemm_camodel.py \
  --work-dir debug_310p_stage0_builtin_rel \
  --m 16 --n 16 --k 16
```

The smoke test successfully compiles, links, launches CAModel, and compares the
output against golden data.

Stage 0 is complete. `tilelang/tools/ascend310p_gemm_camodel.py` now normalizes
relative and absolute `--work-dir` and `--kernel-cpp` paths, writes deterministic
debug artifacts, and captures compile/link/CAModel/compare logs under `logs/`.

### TileLang 310P Progress

The following pieces already exist:

- 310P platform normalization and hardware spec in
  `tilelang/jit/adapter/ascend_platform.py`.
- 310P memory sizes and `TL_ASCEND_310P` macro emission in
  `src/target/codegen_ascend.cc`.
- A 310P GEMM source generation example in
  `examples/gemm/example_gemm_310p.py`.

The GEMM example can lower to Ascend C source:

```bash
python3 examples/gemm/example_gemm_310p.py \
  --mode lower \
  --debug-root debug_310p_gemm_lower \
  --m 128 --n 128 --k 64
```

However, the generated TileLang 310P GEMM source does not yet compile through
the current CAModel compile path.

## Known Blockers

1. `src/tl_templates/ascend/common.h` has incomplete conditional compilation
   under `TL_ASCEND_310P`; it currently exposes an unmatched `#endif`.
2. Some helper implementations use Ascend C APIs that are not available in the
   current 310P CANN headers, including `AscendC::Pattern::Reduce`.
3. Some helper implementations use APIs whose namespace or signature differs on
   this toolchain, such as `AscendC::Broadcast`.
4. Generated source includes `<runtime/rt_ffts.h>`, which is not found by the
   current CAModel compile include path.
5. Resolved in Stage 0: `tilelang/tools/ascend310p_gemm_camodel.py` now works
   with both relative and absolute `--work-dir` values.
6. `torch_npu` is not installed, so runtime JIT paths that require real NPU
   integration cannot be validated on this machine yet. CAModel-only validation
   is still usable.

## Support Strategy

The goal is to support all examples on 310P Ascend C. The work should proceed by
backend capability closure rather than by directory order, because later examples
depend heavily on the primitive coverage established by earlier stages.

### Stage 0: Stabilize CAModel Harness

Status: complete.

Completed changes:

- Fixed `tilelang/tools/ascend310p_gemm_camodel.py` so relative `--work-dir`
  values are resolved before the compile command runs inside the work directory.
- Resolved `--kernel-cpp` to an absolute path before copying external generated
  kernel source into the debug directory.
- Kept the built-in smoke GEMM kernel as the toolchain health check.
- Added deterministic logging for compile, link, CAModel execution, and output
  comparison: `logs/compile.log`, `logs/link.log`, `logs/camodel.log`, and
  `logs/compare.log`.
- Kept the debug directory self-contained with `main_kernel.cpp`,
  `main_kernel.o`, `A.bin`, `B.bin`, `C_golden.bin`, `C_actual.bin`, and the
  `ascendebug_workspace`.

Verification commands run:

```bash
python3 -m py_compile tilelang/tools/ascend310p_gemm_camodel.py

python3 tilelang/tools/ascend310p_gemm_camodel.py \
  --work-dir debug_310p_stage0_builtin_rel \
  --m 16 --n 16 --k 16

python3 tilelang/tools/ascend310p_gemm_camodel.py \
  --work-dir /home/ayd/code/tilelang-ascend/debug_310p_stage0_builtin_abs \
  --m 16 --n 16 --k 16

python3 tilelang/tools/ascend310p_gemm_camodel.py \
  --work-dir debug_310p_stage0_external_rel \
  --kernel-cpp debug_310p_stage0_builtin_rel/main_kernel.cpp \
  --m 16 --n 16 --k 16

python3 tilelang/tools/ascend310p_gemm_camodel.py \
  --work-dir /home/ayd/code/tilelang-ascend/debug_310p_stage0_external_abs \
  --kernel-cpp debug_310p_stage0_builtin_rel/main_kernel.cpp \
  --m 16 --n 16 --k 16
```

All four CAModel runs compiled, linked, launched `npu_kernel_launch --camodel`,
and matched `C_actual.bin` against `C_golden.bin`.

Original objectives:

- Fix `tilelang/tools/ascend310p_gemm_camodel.py` so relative `--work-dir` works.
- Add a reusable CAModel runner for generated TileLang kernel source.
- Keep a small built-in smoke kernel as a toolchain health check.
- Record kernel source, object file, input data, output data, and CAModel logs in
  a deterministic debug directory.

Exit criteria:

- Built-in smoke test passes.
- The harness can compile and run a simple generated source file with absolute
  and relative work directories.

### Stage 1: 310P Template and GEMM Compile

Objectives:

- Fix `src/tl_templates/ascend/common.h` under `TL_ASCEND_310P`.
- Guard or replace helper code that depends on unavailable APIs.
- Resolve the generated `<runtime/rt_ffts.h>` dependency for CAModel compile.
- Compile `examples/gemm/example_gemm_310p.py` generated source with `ccec`.
- Run that generated GEMM through CAModel and compare against golden output.

Exit criteria:

- `example_gemm_310p.py --mode lower` output compiles with `ccec`.
- Generated GEMM runs in CAModel and matches golden data.

### Stage 2: Pure UB and Vector Examples

Directories:

- `examples/elementwise`
- `examples/pad`
- `examples/activation`
- `examples/normalization`
- `examples/pos_embedding`
- `examples/softmax`
- `examples/cross_entropy_loss`
- `examples/random_1d`

Required primitive coverage:

- `T.copy`
- `T.alloc_ub`
- `T.alloc_shared` mapped to UB where appropriate
- `T.tile.add`
- `T.tile.sub`
- `T.tile.mul`
- `T.tile.div`
- `T.tile.fill`
- `T.tile.cast`
- `T.tile.exp`
- `T.tile.sqrt`
- `T.tile.rsqrt`
- Basic barriers

Exit criteria:

- Each example lowers to 310P Ascend C.
- Each generated kernel compiles through the CAModel compile path.
- Representative examples run in CAModel and match golden data.

### Stage 3: Reduce and Indexing

Directories:

- `examples/reduce`
- `examples/topk_selector`
- `examples/sort`
- `examples/hadamard_transform`
- `examples/lightning_indexer`

Required primitive coverage:

- `T.reduce_sum`
- `T.reduce_max`
- `T.reduce_min`
- `T.tile.broadcast`
- `T.tile.compare`
- `T.tile.select`
- `T.tile.gather`
- `T.tile.gather_mask`
- `T.tile.topk`
- `T.tile.merge_sort`

Exit criteria:

- Reduce helpers have 310P-compatible implementations.
- Indexing examples compile; dynamic or workspace-heavy examples can initially
  be marked as partial if the static path is validated.

### Stage 4: Basic Cube and GEMM Closure

Directories:

- `examples/gemm`
- `examples/gemm_aot`
- `examples/batch_gemm`
- `examples/gemv`
- `examples/convolution`
- `examples/simple_fusion`

Required primitive coverage:

- `T.alloc_L1`
- `T.alloc_L0A`
- `T.alloc_L0B`
- `T.alloc_L0C`
- `T.gemm_v0`
- `T.mma`
- GM/L1/L0 copy helpers
- Tail block handling
- Basic C/V scope split

Exit criteria:

- GEMM, batch GEMM, GEMV, and simple fusion compile and run in CAModel.
- Tail-block examples have correctness coverage.

### Stage 5: Developer Mode and Automatic Passes

Directories:

- `examples/developer_mode`
- `examples/autotune`
- `examples/grouped_gemm`
- `examples/blocksparse_gemm`
- `examples/quant_batch_matmul`

Required pass coverage:

- `TL_ASCEND_AUTO_SYNC`
- `TL_ASCEND_MEMORY_PLANNING`
- buffer scope inference
- parallel lowering to vector instructions
- automatic copy lowering

Exit criteria:

- Developer-mode generated Ascend C is legal for 310P.
- Autotune/carver can select or restrict configs that fit 310P memory limits.

### Stage 6: Fusion, Pipeline, and Cross-Scope Sync

Directories:

- `examples/pipeline`
- `examples/causal_conv1d`
- `examples/linear_attention_and_rnn`
- `examples/chunk_gated_delta_rule`
- `examples/fused_sigmoid_gating_delta_rule`

Required primitive coverage:

- `T.Scope("C")`
- `T.Scope("V")`
- `T.set_cross_flag`
- `T.wait_cross_flag`
- `T.set_flag`
- `T.wait_flag`
- `T.Pipelined`
- `T.pipe_barrier`
- workspace arguments

Exit criteria:

- Cross-scope synchronization is correct on 310P.
- Pipeline examples compile and pass representative CAModel correctness tests.

### Stage 7: FlashAttention and Sparse Attention

Directories:

- `examples/flash_attention`
- `examples/flash_attention/fa_opt`
- `examples/sparse_flash_attention`
- `examples/sparse_flash_attention/bench_sfa`
- `examples/deepseek_v4`

Required coverage:

- Large workspace handling
- attention reduction patterns
- C/V synchronization
- optional dynamic shape handling
- sparse/paged mask paths
- performance-oriented tiling variants

Exit criteria:

- Main FlashAttention examples compile and run.
- Sparse attention examples compile and pass at least one representative
  correctness case.
- Optimization variants can be separated into "functional" and "performance"
  milestones.

### Stage 8: MoE, Shared Memory, and Integration Examples

Directories:

- `examples/moe_token_permute`
- `examples/dispatch_combine`
- `examples/shmem`
- `examples/aclgraph`
- `examples/torch_tl_ascend`

Risks:

- `shmem` is currently excluded under `TL_ASCEND_310P` in `common.h`.
- `dispatch_combine` depends on shmem and experimental tile APIs.
- `torch_tl_ascend` requires broader runtime integration and likely needs
  `torch_npu`.

Exit criteria:

- MoE examples compile and representative kernels run.
- shmem-dependent examples are either supported by a 310P-compatible path or
  explicitly documented as unsupported on 310P.
- Integration examples have a clear CAModel-only path or are marked as requiring
  runtime NPU dependencies.

## Recommended Immediate Work

1. Fix the CAModel runner relative path issue.
2. Fix `common.h` conditional compilation for `TL_ASCEND_310P`.
3. Make generated GEMM source compile with `ccec`.
4. Run generated GEMM through CAModel and compare output.
5. Add a small 310P CAModel regression script for the generated GEMM path.
6. Start Stage 2 with `examples/elementwise` and `examples/activation`.
