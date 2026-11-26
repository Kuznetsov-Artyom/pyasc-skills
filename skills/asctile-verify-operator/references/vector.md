# Verify reference: Vector-route checklist

Apply this after the independent oracle and verification procedure are valid. API facts
carry provenance from the validated installed package (`pyasc-dev
1.1.1.dev20261123081345`), principally `language/memory_ops.py`,
`runtime/compiler.py`, and `asc/runtime/launcher.py`. This checklist adds verification
criteria, not new design choices.

The only published Vector acceptance template is `vector_add_fp16`; other Vector
material is candidate practice, not supported production coverage.

## Route and structural checklist

- [ ] All operator compute is on UB `LocalTensor`s; GM descriptors use true logical
      extents. There is no `matmul`/`matmul_acc`, L0A/L0B/L0C residency, `DistribMode`,
      or `cv_strategy`. `cv_ratio=1` is the expected pure-Vector choice; installed
      `cv_ratio=2` is expressible, so require brief/source justification rather than
      misreporting it as an API impossibility.
- [ ] `copy_in` local shapes and reshape/broadcast result shapes are compile-time;
      runtime GM extents/offsets remain runtime. One-dimensional copies need no
      alignment; every 2-D+ UB local last dimension is 32-byte aligned. `real_shape`
      rank/extents match the local tile and never exceed it.
- [ ] Actual `@asctile.jit` and call-site options match the reported specialization.
      `vf_fusion` and `reuse_alloc` values are in their installed ranges; L2 policy is
      reported per GM tensor. No `always_compile=True` or debug-only setting is silently
      used for the delivered measurement.
- [ ] Compiler-reported consumption and `Launcher.check_memory_overflow` pass for the
      specialization each variant actually used. Do not infer per-variant cache counts
      from runtime scalar shapes; audit the actual specialization and its configuration.

## Coverage and boundary audit

- [ ] Prove from tile count, grid, and index equations that every logical output is
      produced the intended number of times, with no gaps, unintended overlap, or empty
      out-of-range tile. For grid-stride 1-D work, expect a logical tile count such as
      `ceildiv(N, tile)` and offsets `t * tile`; the final valid extent is
      `min(tile, N - offset)`.
- [ ] Each boundary load/store uses the brief's exact tail family and valid extent.
      Do not accept implicit GM clamping alone as proof of no out-of-bounds access.
      Full-tile accesses at/past an edge or simulator DMA/MTE boundary errors are defect
      evidence even when `assert_close` and output guards pass.
- [ ] Output guards are the required 32 fp16 elements/64 bytes **per side**, filled with
      fixed sentinels and compared bit-for-bit to a pre-launch snapshot. Dry run compares
      the entire output allocation, not just the guards. Guards do not prove input-read
      safety; source/index proofs and logs cover that separately.
- [ ] Tail/pad lanes have defined values where consumed. Reduction pads use the correct
      identity (sum 0, max `-inf`, min `+inf`, product 1), and other arithmetic cannot
      turn the chosen pad into NaN/Inf that reaches a valid output.
- [ ] Input generation, output shapes, reduction axes and `keep_dims`, broadcast axes,
      flattening order, and output cast match Design. The reference is computed in fp32
      from the same fp16 inputs and cast as specified; tolerance comparison uses exactly
      `atol`/`rtol` from the brief on every real element.

## Configuration audit

Reconcile source, launch, `CONFIG`, `RESULT.md`, and logs per variant for:

- grid/launch bounds and kernel use of `asctile.block_num()`/`block_idx()`;
- actual tile family/shapes, loop nesting/order, `unroll_factor`, `gm_barrier` where
  relevant, tail strategy and valid extents;
- `vf_fusion`, `reuse_alloc`, `cv_ratio`, per-tensor `l2_cache`, and all other options;
- complete `ConstExpr` set vs runtime scalars, dtype/internal accumulation, seed and draw
  order, cache directory and truthful cache-warm state;
- identical tensors, kernel, grid, tile and options for warmup correctness and all five
  measured launches.

A missing field, stale shared-specialization description, or disagreement with executable
source is a reporting/submission defect, not permission for Verify to repair it. Report
per variant even when variants legitimately share one compiled specialization.

## Defect routing

Before routing any defect, apply Verify SKILL.md's cap gate: classify first, then check
`local_strikes`, `structural_reissues`, and `uncertain_reductions`; a transition beyond
its cap publishes `stopped` instead of an owner handoff.

- **Local:** incorrect offset/grid/tile/unroll/option, missing valid extent or identity
  pad, UB overflow, rank/broadcast mistake, host oracle/input/guard/report bug, or other
  error fixable while preserving every structural brief field. Return the smallest
  failing variant and evidence to Implement.
- **Structural:** the brief requires a non-UB Vector computation, an unsupported GM↔UB
  path, impossible fixed-extent alignment/tiling, or another dataflow/residency choice
  that no legal local configuration can satisfy. Return installed-package evidence to
  Design for a complete brief reissue; Implement must then synchronize kernel and full
  harness before re-verification.
- **Not a defect yet:** an invalid Verify oracle, wrong environment/command, incomplete
  log, or non-reproduced observation. Correct or repeat only the verification; do not
  alter or request alteration of implementation on that evidence.
