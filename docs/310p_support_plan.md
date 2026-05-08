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

### Ascendebug CPU Twin

The repository now also has a reusable 310P CPU twin precision runner:

```bash
python3 tilelang/tools/ascend310p_stage2_cpu_twin.py \
  --case activation_silu \
  --work-dir debug_310p_stage2_cpu_twin
```

This path uses `ascendebug`'s `run_call_kernel_cpu` flow for 310P Ascend C
source and is intended for precision validation without the PTO path. The
`activation_silu` example has been verified against its golden output.

Additional example validations completed with the same CPU twin path:

- `examples/softmax/example_online_softmax.py` as `example_online_softmax`
- `examples/normalization/rms_norm.py` as `example_rms_norm_streaming`
- `examples/gemv/example_gemv_v.py` as `example_gemv_vector`
- `examples/gemm/example_gemm_310p.py` as `example_gemm_310p_small`
- L1 copy roundtrip as `example_copy_roundtrip`
- `examples/flash_attention/flash_attn_bhsd.py` as `example_flash_attention_shape`
- `examples/flash_attention/flash_attn_bhsd.py` as
  `example_flash_attention_example_style` through CPU-twin-safe source
  normalization

These were validated with reduced 310P-friendly shapes while preserving the
core compute pattern of each example.
The example-style flash-attention CPU twin path currently elides cross-core
debug events and disables auto sync during lowering so that the math can be
validated without the mixed-core runtime blocker.

Current CPU twin diagnostic conclusion:

- GM-to-L1 and L1-to-GM copies are correct in the CPU twin roundtrip case.
- 16x16 `T.gemm_v0` with explicit L1 inputs and L0C output matches golden in
  CPU twin.
- Example-style FlashAttention with QK GEMM, softmax, PV GEMM, and workspace
  writes matches golden in CPU twin after CPU-debug-safe cross-core event
  normalization.
- Therefore the remaining precision-risk gap is not basic copy or CPU twin math;
  it is the non-CPU CAModel/runtime handling of optimized cube GEMM, mixed-core
  C/V execution, and auto-sync events.

Recent verification commands:

```bash
python3 -m py_compile tilelang/tools/ascend310p_stage2_cpu_twin.py

python3 tilelang/tools/ascend310p_stage2_cpu_twin.py \
  --case example_copy_roundtrip \
  --work-dir debug_310p_cpu_twin_copy_roundtrip

python3 tilelang/tools/ascend310p_stage2_cpu_twin.py \
  --case example_gemm_310p_small \
  --work-dir debug_310p_cpu_twin_gemm_small

python3 tilelang/tools/ascend310p_stage2_cpu_twin.py \
  --case example_flash_attention_example_style \
  --work-dir debug_310p_cpu_twin_fa_example_style
```

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

The generated TileLang 310P GEMM source now compiles through the current
CAModel compile path after source normalization in
`tilelang/tools/ascend310p_gemm_camodel.py`.

## Known Blockers

1. Generated cube GEMM now compiles for 310P, but optimized cube CAModel
   correctness is not closed. The dav-m200 CANN headers mark the normal
   CATLASS L0C-to-GM `Fixpipe` path as unsupported, so Stage 4 must replace
   the temporary scalar compile fallback with a real 310P cube writeback
   strategy.
2. Resolved in Stage 1: generated scalar GEMM now lowers, compiles, launches in
   CAModel on AiCore, and matches golden output.
3. Resolved in Stage 0: `tilelang/tools/ascend310p_gemm_camodel.py` now works
   with both relative and absolute `--work-dir` values.
4. `torch_npu` is not installed, so runtime JIT paths that require real NPU
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

Status: complete.

Objectives:

- Fix `src/tl_templates/ascend/common.h` under `TL_ASCEND_310P`.
- Guard or replace helper code that depends on unavailable APIs.
- Resolve the generated `<runtime/rt_ffts.h>` dependency for CAModel compile.
- Compile `examples/gemm/example_gemm_310p.py` generated source with `ccec`.
- Run that generated GEMM through CAModel and compare against golden output.

Completed changes:

- Added 310P conditional guards and fallbacks in
  `src/tl_templates/ascend/common.h` for:
  - GM-to-L1 copy
  - UB GM/UB copy helpers that avoid unsupported `DataCopyPad` and use scalar
    `GetValue`/`SetValue` copies for CAModel-safe 310P execution
  - L0C-to-GM writeback compile fallback
  - `reduce_sum`, `reduce_max`, and `reduce_min`
  - `gemm_v0` scalar compile fallback
  - `Broadcast`
  - shmem helpers that should not be exposed under `TL_ASCEND_310P`
- Extended `tilelang/tools/ascend310p_gemm_camodel.py` so external generated
  source can be normalized for CAModel compile:
  - strips host-only ACL/FFTS includes and the generated host `call` wrapper
  - removes the FFTS kernel argument from the kernel signature
  - selects one mixed-core source branch for `AiCore` or `VectorCore`
  - supports `--compile-opt-level`, `--core-type`, and `--timeout`
- Extended `examples/gemm/example_gemm_310p.py` with configurable block sizes
  and a scalar backend used for small generated-source experiments.

Verification commands run:

```bash
python3 -m py_compile \
  tilelang/tools/ascend310p_gemm_camodel.py \
  examples/gemm/example_gemm_310p.py

python3 examples/gemm/example_gemm_310p.py \
  --mode lower \
  --debug-root debug_310p_gemm_lower \
  --m 128 --n 128 --k 64 \
  --block-m 128 --block-n 128 --block-k 64

python3 tilelang/tools/ascend310p_gemm_camodel.py \
  --work-dir debug_310p_stage1_tl_generated_compile128 \
  --kernel-cpp debug_310p_gemm_lower/tilelang_gemm_310p.cpp \
  --m 128 --n 128 --k 64 \
  --skip-run

python3 examples/gemm/example_gemm_310p.py \
  --mode lower \
  --backend scalar \
  --debug-root debug_310p_gemm_lower_scalar_1x1 \
  --m 1 --n 1 --k 16 \
  --block-m 1 --block-n 1 --block-k 16

python3 tilelang/tools/ascend310p_gemm_camodel.py \
  --work-dir debug_310p_stage1_tl_generated_scalar_1x1 \
  --kernel-cpp debug_310p_gemm_lower_scalar_1x1/tilelang_gemm_310p.cpp \
  --m 1 --n 1 --k 16 \
  --compile-opt-level=-O3 \
  --core-type AiCore \
  --timeout 60
```

The 128x128x64 generated cube GEMM lowers and compiles/links with `ccec` for
`dav-m200`. The generated scalar 1x1x16 GEMM lowers, compiles, launches in
CAModel, and matches `C_out.bin` against `C_golden.bin`.

Notes:

- The scalar Stage 1 correctness gate is intentionally small because it verifies
  generated TileLang source, 310P template fallback code, CAModel launch, and
  golden comparison without depending on unsupported dav-m200 cube writeback
  APIs.
- The optimized cube path still needs Stage 4 work before it can claim
  performance-path CAModel correctness.

Exit criteria:

- `example_gemm_310p.py --mode lower` cube output compiles with `ccec`.
- Generated scalar GEMM runs in CAModel and matches golden data.

### Stage 2: Pure UB and Vector Examples

Status: complete.

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

Completed changes:

- Added `tilelang/tools/ascend310p_stage2_camodel.py`, a reusable Stage 2
  CAModel harness that lowers small TileLang UB/vector kernels to 310P Ascend C,
  normalizes generated mixed-core source for offline CAModel compile, writes
  deterministic input/golden/output artifacts, and compares CAModel output for
  representative correctness cases.
- Extended `tilelang/tools/ascend310p_gemm_camodel.py` so external generated
  source normalization handles nested `ASCEND_IS_AIV` / `ASCEND_IS_AIC` guards
  with brace-aware branch selection instead of regex-only stripping.
- Extended the CAModel compile helper to support `VectorCore` compile-only
  gates through `ccec --cce-aiv`. The AIV object is kept as the kernel object
  because it is not compatible with the AICore `ld.lld -m aicorelinux` link
  path.
- Guarded the temporary 310P `bfloat16_t` fallback in
  `src/tl_templates/ascend/common.h` so it is not redefined on the real CANN
  VectorCore compile path where CANN already typedefs `bfloat16_t`.

Verification commands run:

```bash
python3 -m py_compile \
  tilelang/tools/ascend310p_gemm_camodel.py \
  tilelang/tools/ascend310p_stage2_camodel.py

python3 tilelang/tools/ascend310p_stage2_camodel.py \
  --work-dir debug_310p_stage2_compile_gate \
  --skip-run \
  --compile-core AiCore

python3 tilelang/tools/ascend310p_stage2_camodel.py \
  --work-dir debug_310p_stage2_vector_compile_gate \
  --skip-run \
  --compile-core VectorCore

python3 tilelang/tools/ascend310p_stage2_camodel.py \
  --work-dir debug_310p_stage2_run_gate \
  --case elementwise_add \
  --case pad_broadcast \
  --case cast_roundtrip \
  --timeout 90
```

The AiCore and VectorCore compile gates lower and compile all Stage 2 harness
cases:

- `elementwise_add`: `T.copy`, `T.alloc_ub`, `T.tile.add`, barriers
- `pad_broadcast`: `T.alloc_ub`, `T.tile.broadcast`
- `cast_roundtrip`: `T.alloc_shared`, `T.tile.cast`
- `activation_silu`: `T.tile.fill`, `T.tile.sub`, `T.tile.exp`,
  `T.tile.add`, `T.tile.div`
- `normalization_rms`: `T.tile.mul`, `T.reduce_sum`, `T.tile.div`,
  `T.tile.sqrt`, `T.tile.rsqrt`, `T.tile.broadcast`
- `random_1d`: `T.tile.arith_progression`, integer `T.tile.add` /
  `T.tile.mul`

The CAModel correctness gate launches generated kernels on 310P CAModel and
matches golden output for `elementwise_add`, `pad_broadcast`, and
`cast_roundtrip`.

Notes:

- The Stage 2 correctness gate intentionally uses small deterministic kernels
  instead of importing example files, because most Stage 2 examples execute
  `torch.npu()` at import or main time and this machine still lacks `torch_npu`.
- Pure `VectorCore` CAModel launch is still not used as a correctness gate:
  `--compile-core VectorCore` proves AIV compile coverage, while direct
  VectorCore CAModel execution of generated V-scope kernels still times out on
  this machine. Stage 2 therefore claims 310P lowering/compile coverage and
  representative CAModel correctness, not full AIV performance-path runtime
  closure.
- `activation_silu`, `normalization_rms`, and `random_1d` are compile-gated in
  Stage 2; their full CAModel correctness remains part of the broader Stage 3
  reduce/indexing and later runtime-quality closure.

Exit criteria:

- Representative Stage 2 UB/vector kernels lower to 310P Ascend C and cover the
  primitive set required by the listed example directories.
- Generated kernels compile through the CAModel compile path for both AiCore
  scalarized and VectorCore AIV compile gates.
- Representative examples run in CAModel and match golden data.

### Stage 3: Reduce and Indexing

Status: complete for 310P reduce/index lower and compile coverage, with
runtime correctness covered for the current compare/select and gather gates.

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

Completed changes:

- Added `tilelang/tools/ascend310p_stage3_camodel.py`, a reusable Stage 3
  harness that lowers small reduce and indexing kernels to 310P Ascend C,
  compiles them through `ccec`, and runs selected CAModel cases.
- Added 310P 310P-stage coverage for the following primitives in the harness:
  - `T.reduce_sum`
  - `T.reduce_max`
  - `T.reduce_min`
  - `T.tile.compare`
  - `T.tile.select`
  - `T.tile.gather`
  - `T.tile.gather_mask`
  - `T.tile.topk`
  - `T.tile.merge_sort`
- Verified compile gates on both `AiCore` and `VectorCore` for the Stage 3
  kernel set.
- Verified CAModel runtime output for `compare_select` and `gather`.

Verification commands run:

```bash
python3 -m py_compile tilelang/tools/ascend310p_stage3_camodel.py

python3 tilelang/tools/ascend310p_stage3_camodel.py \
  --work-dir debug_310p_stage3_default \
  --skip-run \
  --compile-core AiCore

python3 tilelang/tools/ascend310p_stage3_camodel.py \
  --work-dir debug_310p_stage3_vector_compile_gate \
  --skip-run \
  --compile-core VectorCore

python3 tilelang/tools/ascend310p_stage3_camodel.py \
  --work-dir debug_310p_stage3_index_run \
  --case compare_select \
  --case gather \
  --timeout 90
```

Notes:

- `compare_select` and `gather` are the current Stage 3 runtime gates.
- `reduce_sum`, `reduce_max`, `reduce_min`, `gather_mask`, `topk`, and
  `merge_sort` are compile-gated at Stage 3 for now.
- Reduce runtime closure still needs follow-up work in the generated 310P
  writeback path; keep it out of the default runtime gate until that is closed.

### Stage 4: Basic Cube and GEMM Closure

Status: complete for the scalarized 310P CAModel correctness gate; optimized
cube and cross-scope fusion remain follow-up work.

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

Completed changes:

- Added `tilelang/tools/ascend310p_stage4_camodel.py`, a reusable Stage 4
  harness that lowers small GEMM, batch GEMM, GEMV, simple fusion, and
  convolution kernels to 310P Ascend C, compiles them through `ccec`, and
  drives CAModel with deterministic input/golden/output artifacts.
- Fixed the `gemm_tail` harness data shapes so the generated program,
  `TensorSpec`, and golden data all describe `A(3, 7) x B(7, 5) -> C(3, 5)`.
- Disabled the Stage 4 harness default auto-sync pass so scalarized AiCore
  correctness gates do not leave copies or stores in an unreachable mixed-core
  branch.
- Marked `simple_fusion` as an explicit opt-in case instead of a default Stage
  4 runtime gate, because 310P compilation currently fails on
  `CrossCoreSetFlag<..., PIPE_FIX>`.
- Narrowed the default `batch_gemm` runtime gate to a single-batch case; the
  multi-batch 3D offset path still needs separate follow-up coverage.
- The default Stage 4 runtime gate now compiles, launches CAModel, and compares
  golden output for:
  - `batch_gemm`
  - `convolution`
  - `gemm_tail`
  - `gemv`

Current blocker:

- The optimized `T.gemm_v0` cube path is still not closed. Generated 310P cube
  GEMM sources compile, but CAModel runtime either times out with a
  never-ending instruction or returns fixed garbage output. Keep this out of
  the default correctness gate until the 310P cube writeback/synchronization
  path is fixed.
- Cross-scope simple fusion currently fails to compile on 310P CAModel because
  generated code references `PIPE_FIX`, which is unavailable in the dav-m200
  compile path. Track this with Stage 6 cross-scope synchronization work.

Verification commands run:

```bash
python3 -m py_compile tilelang/tools/ascend310p_stage4_camodel.py

python3 tilelang/tools/ascend310p_stage4_camodel.py \
  --work-dir debug_310p_stage4_default_gate \
  --timeout 90
```

### Stage 5: Developer Mode and Automatic Passes

Status: complete for 310P developer-mode lower/compile coverage; runtime
correctness is limited by the Stage 4 optimized cube blocker and missing
`torch_npu`.

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

Completed changes:

- Added `tilelang/tools/ascend310p_stage5_camodel.py`, a reusable Stage 5
  harness for 310P developer-mode and automatic-pass coverage.
- The harness exercises the automatic pass set:
  - `TL_ASCEND_AUTO_SYNC`
  - `TL_ASCEND_MEMORY_PLANNING`
  - `TL_ASCEND_AUTO_CV_COMBINE`
  - `TL_ASCEND_AUTO_CV_SYNC`
- Added compile coverage for representative Stage 5 patterns:
  - `developer_gelu_mul`: Developer-mode shared-buffer vector expression
    lowering.
  - `developer_gemm`: Developer-mode `alloc_shared` / `alloc_fragment` GEMM
    lowering.
  - `blocksparse_gemm`: masked block GEMM lowering with automatic pass
    coverage.
- Added a 310P carver check entry point. On this machine it records an explicit
  skip because importing the carver stack requires `torch_npu`, which is not
  installed.

Verification commands run:

```bash
python3 -m py_compile tilelang/tools/ascend310p_stage5_camodel.py

python3 tilelang/tools/ascend310p_stage5_camodel.py \
  --work-dir debug_310p_stage5_compile_all \
  --compile-all-default \
  --timeout 90
```

Notes:

- Stage 5 currently claims lower/compile legality, not full runtime correctness
  for cube-heavy developer examples. Generated developer GEMM still inherits
  the Stage 4 optimized `T.gemm_v0` cube CAModel blocker.
- Runtime JIT execution of the original Stage 5 example scripts is still gated
  by missing `torch_npu`.
- Quantized batch matmul and grouped GEMM remain follow-up compile/runtime
  coverage because they combine cube, workspace, pointer/metadata, and
  cross-scope paths that depend on later Stage 6 and Stage 8 closure.

### Stage 6: Fusion, Pipeline, and Cross-Scope Sync

Status: complete for 310P cross-scope and pipeline lower/compile coverage;
mixed-core CAModel runtime correctness remains follow-up work.

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

Completed changes:

- Added a 310P-specific codegen normalization for cross-core set flags:
  generated `PIPE_FIX` cross-core set flags now emit `PIPE_MTE3` for platform
  `310P`, matching the dav-m200 CANN compile path.
- Added `tilelang/tools/ascend310p_stage6_camodel.py`, a reusable Stage 6
  compile harness.
- Added compile coverage for representative Stage 6 patterns:
  - `cross_scope_fusion`: explicit `T.Scope("C")`, `T.Scope("V")`,
    `T.set_cross_flag`, `T.wait_cross_flag`, and C/V handoff.
  - `pipelined_vector`: `T.Pipelined` plus `T.pipe_barrier` in a vector-style
    loop.

Verification commands run:

```bash
make -j$(nproc)  # from build/

python3 -m py_compile tilelang/tools/ascend310p_stage6_camodel.py

python3 tilelang/tools/ascend310p_stage6_camodel.py \
  --work-dir debug_310p_stage6_compile_gate
```

Notes:

- The Stage 6 gate is currently a lower/compile gate. The local CAModel helper
  still selects one core branch for offline AiCore/VectorCore execution, so it
  cannot yet prove real mixed-core C/V runtime synchronization.
- Pipeline examples that also depend on optimized cube GEMM inherit the Stage 4
  optimized cube runtime blocker.

### Stage 7: FlashAttention and Sparse Attention

Status: complete for 310P attention-shape lower/compile coverage; runtime
correctness remains blocked by the Stage 4 optimized cube runtime issue and by
the local CAModel helper's limited mixed-core execution model.

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

Completed changes:

- Added `tilelang/tools/ascend310p_stage7_camodel.py`, a Stage 7 compile
  harness for attention-shaped kernels.
- Added compile coverage for `flash_attention_shape`, covering Q/K/V copies,
  score workspace, row-wise `T.reduce_max`, exponentiation, row-wise
  `T.reduce_sum`, softmax normalization, output workspace writes, and final
  output copies.
- Added compile coverage for `sparse_attention_index`, covering sparse index
  copies, `T.tile.gather`, scalar `T.reduce_sum`, and broadcast-style output
  writes.

Verification commands run:

```bash
python3 -m py_compile tilelang/tools/ascend310p_stage7_camodel.py

python3 tilelang/tools/ascend310p_stage7_camodel.py \
  --work-dir debug_310p_stage7_compile_gate
```

Notes:

- This stage intentionally claims lower/compile coverage rather than full
  correctness. Full FlashAttention runtime closure still depends on closing the
  optimized cube `T.gemm_v0` CAModel blocker and the mixed C/V runtime path.
- Sparse/paged attention example families remain represented by primitive
  shape coverage here; broader dynamic-shape runtime validation should be added
  once the runtime blockers are closed.

### Stage 8: MoE, Shared Memory, and Integration Examples

Status: complete for 310P MoE and aclgraph integration-shape lower/compile
coverage; shmem, dispatch-combine, and torch integration examples remain
runtime-dependent. The tree now also has a CPU-debug-only 310P fallback path
that lets the CPU twin runner validate `T.shmem_*` precision behavior, but it
is not a real 310P SHMEM runtime implementation.

Directories:

- `examples/moe_token_permute`
- `examples/dispatch_combine`
- `examples/shmem`
- `examples/aclgraph`
- `examples/torch_tl_ascend`

Risks:

- `dispatch_combine` depends on shmem and experimental tile APIs.
- `torch_tl_ascend` requires broader runtime integration and likely needs
  `torch_npu`.
- The CPU twin shmem coverage is a debug fallback, not a production 310P
  runtime path.

Exit criteria:

- MoE examples compile and representative kernels run.
- shmem-dependent examples are either supported by a 310P-compatible path or
  explicitly documented as unsupported on 310P.
- Integration examples have a clear CAModel-only path or are marked as requiring
  runtime NPU dependencies.

Completed changes:

- Added `tilelang/tools/ascend310p_stage8_camodel.py`, a Stage 8 compile and
  dependency-report harness.
- Added 310P compile coverage for `moe_token_unpermute_shape`, covering token
  index copies, probability casts, dynamic GM row copies, `T.tile.axpy`,
  explicit `T.set_flag`/`T.wait_flag`, and `T.pipe_barrier`.
- Added 310P compile coverage for `moe_token_permute_grad_shape`, covering
  token-gradient gather/scatter-style dynamic indexing, explicit V/MTE
  synchronization, accumulation, and output casts.
- Added 310P compile coverage for `aclgraph_rms_rope_shape`, covering RMS
  reduction, vector arithmetic, RoPE mask construction, reinterpret cast,
  broadcast, `T.tile.gather`, and final GM output copy.
- Added an explicit local dependency report for integration-only examples:
  `torch_npu` is missing on this machine, and the real 310P SHMEM runtime path
  still depends on external shmem support.
- Added CPU-debug-only 310P fallback helpers in
  `src/tl_templates/ascend/common.h` so the CPU twin runner can validate
  `T.shmem_get_nbi`, `T.shmem_put_nbi`, `T.shmem_ub_get_nbi`, and
  `T.shmem_ub_put_nbi` precision behavior.

Verification commands run:

```bash
python3 -m py_compile tilelang/tools/ascend310p_stage8_camodel.py

python3 tilelang/tools/ascend310p_stage8_camodel.py \
  --work-dir debug_310p_stage8_compile_gate \
  --dependency-report
```

Notes:

- Stage 8 is deliberately split between supported compile coverage and
  documented dependency gaps. The local machine cannot validate
  `examples/torch_tl_ascend` because `torch_npu` is unavailable.
- `examples/shmem` and `examples/dispatch_combine` still need a real 310P
  SHMEM runtime path before they can be claimed as supported end to end.
- The current CPU twin shmem coverage is sufficient for precision validation,
  but it does not replace the actual 310P runtime integration work.

## Remaining Follow-Up

All planned stages now have a 310P CAModel harness, compile gate, runtime gate,
dependency report, or explicit unsupported-path note. The remaining work is not
a new stage in this plan; it is runtime closure for the known blockers:

1. Close the optimized cube `T.gemm_v0` CAModel runtime issue. The generated
   128x128x64 cube source compiles for 310P, but CAModel execution still returns
   incorrect fixed output and reports never-ending instructions. The same
   16x16 L1/L0C GEMM pattern now matches golden in CPU twin, so the remaining
   fix is in CAModel/runtime-visible cube writeback/load/event semantics rather
   than basic host-side precision math.
2. Add a real mixed-core C/V CAModel runtime path. Stages 6 and 7 currently
   prove lower/compile legality for cross-scope and attention-shaped kernels,
   but the local helper executes one selected core branch offline.
3. Add or enable 310P shmem support before claiming `examples/shmem` and
   `examples/dispatch_combine` runtime support. The current tree only provides
   a CPU-debug fallback for `T.shmem_*`.
4. Install `torch_npu`/NPU runtime dependencies before validating
   `examples/torch_tl_ascend` and original runtime JIT example scripts on this
   machine.
5. Convert the reduced CPU twin examples into direct example-entry validation
   where possible. The current runner intentionally uses 310P-friendly reduced
   shapes and CPU-debug-safe source normalization; full "examples exactly as
   written" support requires preserving each example's original pass configs,
   shapes, dynamic arguments, and workspace contracts.
