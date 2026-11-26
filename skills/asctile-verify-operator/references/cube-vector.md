# Verify reference: Cube+Vector-route acceptance checklist

Apply this checklist **on top of** `cube.md` and `simulator.md`, after the independent
oracle and verification procedure are valid. For `cube_vector`, load exactly
`cube.md + cube-vector.md + simulator.md`; **never load the full `vector.md` playbook**.
This is an acceptance audit, not an implementation recipe.

API expectations below carry provenance from the validated installed package
(`pyasc-dev 1.1.1.dev20261123081345`), including `memory_ops.py`, `prog_model_ops.py`,
`runtime/compiler.py`, the CV-strategy pass diagnostics, and the interpreter's explicitly
sequential sub-block model. Recheck the installed package if it changes.

The only published mix acceptance template is `gemm_bias_relu_vector_fp16` with
manual `SplitByM` at `cv_ratio=2`; other mechanisms are candidate practice.

## 1. Route and real-Vector-epilogue validity

- [ ] The entire K reduction is on the Cube path required by `cube.md`; no host or
      Pure-Vector GEMM substitutes for it. The claimed epilogue is in the same submitted
      kernel/program, not a second kernel or host post-processing.
- [ ] A completed fp32 L0C result crosses **L0C→UB**, and the brief-mandated Vector work
      actually consumes that UB result. For GEMM+bias+ReLU, both the per-column bias add
      and ReLU occur in UB after the Cube reduction, followed by the output store.
- [ ] Bias for that task is staged **GM→UB**, not solely through BT,
      `matmul(..., bias=...)`, or `zeros_acc(bias=...)`. ReLU is not performed only on L0C.
      A UB copy or cast without the specified UB add/ReLU does not establish the epilogue.
- [ ] A Cube-only bias fold, Cube-only ReLU, direct L0C→GM epilogue bypass, host/second-
      kernel epilogue, or Pure-Vector contraction is classified **`invalid_submission`**
      when the frozen task requires real Vector work—even if values pass. Missing Vector
      work is not repaired by calling the variant numerically `correct`.
- [ ] `cv_ratio` is actually 1 or 2 and matches the delivered specialization and report.
      `cv_ratio=2` is not accepted as proof of Vector work by itself. Conversely, an
      unsplit `cv_ratio=1` UB epilogue is route-valid when the mandated operations really
      execute there; two-AIV engagement is not a validity requirement.
- [ ] `CONFIG`/`RESULT.md` exposes the mechanism and Vector path: `cv_ratio`, manual
      `distrib`/CV strategy/unsplit, `DistribMode` and split axis, sub-block mapping, UB
      bias staging and broadcast, operation locations/dtypes, and output-store handling.

## 2. Strategy consistency and legal distribution

Audit the actual source/lowered specialization, not merely option names.

- [ ] **Manual distribution:** each `SplitByM` is on L0C→UB with even M and result
      `[M/2,N]`; each `SplitByN` is on L0C→UB with N divisible by 32 elements and result
      `[M,N/2]`. An explicit local `shape`, if present, is the split shape, not the source
      shape. `distrib=None` is not reported as a split.
- [ ] **Join/back-to-Cube:** every UB→L1 `JoinByM`/`JoinByN` has a 2-D per-AIV source and
      the corresponding doubled result shape. When a split result re-enters Cube, its Join
      uses the same axis and is followed by L1→L0A/L0B; no L1→L0C, L1↔L1, or distributed
      UB→UB path is invented. A Join is not required for an ordinary UB→GM epilogue store.
- [ ] **`FullVec*`:** the selected whole-tile sub-block and its guard agree with the actual
      Vector work/store. The other AIV does not duplicate output or an intended one-AIV
      Join. No UB-size saving is claimed for whole-tile delivery.
- [ ] **CV strategy:** the region argument is only `SplitByM` or `SplitByN`, with
      `cv_ratio=2`. The L0C→UB copy has the full source shape and **omits** `distrib=`.
      Any explicit `distrib` on a copy inside the region is a defect, not a second way to
      request the same split. `FullVec*`/`JoinBy*` is not accepted as the region argument.
- [ ] CV propagation stays within supported elementwise/reduction/reshape/broadcast and
      L0C→UB, UB→L1, UB→GM operations. Elementwise LocalTensor operands have the required
      full logical shape; the active split axis is not changed, broadcast, or reduced.
      A full-shape broadcast operand is not confused with manual per-half staging.
- [ ] A CV-managed store uses only the full tile origin; it does not add a manual
      `sub_block_idx()` half offset. The candidate has not stacked CV and manual
      distribution on one transfer or otherwise double-applied a split.
- [ ] Any CV-dependent UB→GM store with `real_shape` is rejected as a valid implementation:
      the installed pass reports `with real shape is not eligible for the CV strategy with
      splitting`. Merely moving the dependent store outside the lexical `with` block is
      not accepted as proof of avoiding propagation. Manual-distribution stores are audited
      separately and are not subject to this CV-only ban.
- [ ] If no Split/Join is used, the report calls the path unsplit and the source proves
      ownership/synchronization. `cv_ratio=2` plus an unguarded whole-tile copy is not
      assumed to give the AIVs disjoint work.

## 3. Sub-block ownership, bias mapping, and boundaries

- [ ] For a two-AIV cooperative tile, the logical task is derived consistently with
      `block_idx()/sub_block_num()` and the launch grid counts logical tasks, not
      `tasks*2`. `sub_block_idx()` is used for the half offset/FullVec selection, not to
      create a second logical tile. Cube-only ownership/K-loop checks from `cube.md` still
      pass.
- [ ] For manual SplitByM, sub-block `sb` owns rows beginning at
      `m_off + sb*(base_m/2)` and both AIVs address the same bias columns. For manual
      SplitByN, its columns begin at `n_off + sb*(base_n/2)` and its bias origin receives
      the same column-half offset. The candidate's equations, actual shapes, and stores
      agree; no overlap, gap, double-offset, or cross-half bias is hidden by a passing test.
- [ ] Every manual output store clamps to the **per-sub-block** valid extent, not merely
      the full-tile extent. `real_shape` has the local rank, does not exceed the local
      tile, and writes no padded C lanes. A sub-block whose origin is outside M/N is
      guarded from an empty store; output guards are not treated as proof of input safety.
- [ ] Rank-2+ UB GM load/store local last dimensions meet the **32-byte** alignment rule.
      This check is distinct from SplitByN's 32-element rule and is checked for the actual
      fp16/fp32 tile, including a column half.
- [ ] Bias GM descriptors and loads stay within logical N. A tail UB load uses a valid
      origin/extent; zero padding (explicit or the installed `real_shape` default) is the
      additive identity. No `real_shape`/`pad_value` appears on BT/L1/L0A/L0B loads. The
      Cube in-flight zero-padding and true-extent proofs from `cube.md` remain intact.
- [ ] The tail variant exercises the same real Vector epilogue as the primary variant;
      there is no primary-only Vector path with a Cube-only or host tail fallback. A CV
      design that cannot clamp a split tail is not accepted merely because the primary
      full tiles pass.

## 4. Numerics, residency, and capacity acceptance

- [ ] L0C→UB preserves the fp32 GEMM result. For the MVP, bias is converted to fp32 before
      the add, the order is **bias add → ReLU → one fp16 cast**, and that order matches the
      frozen fp32 reference/final cast and tolerance. An unexplained early fp16 cast is a
      numeric defect even if the sampled inputs happen to pass.
- [ ] Manual per-half bias broadcast shapes match the actual per-AIV result
      (`[base_m/2,base_n]` or `[base_m,base_n/2]`). A separate full-shape broadcast is not
      assumed to be sliced by a manual Split. CV full-shape broadcast handling is checked
      against the strategy's propagation instead.
- [ ] ReLU is invoked on the UB result after the add. The final cast/store is from UB. Any
      UB→L1 Join used by a later Cube phase preserves the full operand shape/dtype and is
      included in the report.
- [ ] Compiler-reported UB/L1/L0A/L0B/L0C/BT consumption passes the exact C310 capacities
      from `cube.md`, including **UB 253952 bytes**. UB accounting includes accumulator,
      bias/broadcast, live Vector
      intermediates, output cast, and carried state at their **actual** full or split
      shapes; `cv_ratio=2` alone is not treated as proof of halved UB usage.
- [ ] Reported carried state, if any, has an explicit UB location at initialization and
      every back edge, a stable split layout, and its live footprint included. A later Cube
      consumer has the matching same-axis Join. `gm_barrier=True` is present only for a
      genuine GM recurrence, not as a substitute for UB-state synchronization.
- [ ] No claim of split/join correctness rests solely on the installed interpreter: its
      sub-blocks run sequentially and its `cv_strategy` override is a no-op. Legacy blanket
      simulator core-count/deadlock claims are not accepted as installed-package facts;
      require valid frozen-environment compile/runtime evidence for the actual launch.

## 5. Evidence reconciliation and defect routing

- [ ] Per variant, reconcile executable source, launch, memory metadata, logs, `CONFIG`,
      and `RESULT.md` for `KernelType`, `cv_ratio`, strategy style, all Split/Join modes,
      grid/sub-block mapping, bias residency/broadcast, operation order/dtypes, Vector tile
      shapes, carried state, output extents, `reuse_alloc`/`vf_fusion`, and cache state.
      Warmup correctness and every measured launch use the identical epilogue and options.
- [ ] The frozen primary and tail both have independent compile/memory, warmup correctness,
      guard, tick, and reproducibility evidence under `simulator.md`. Numerical success
      never erases a route, ownership, out-of-bounds, or stale-configuration defect.

For a validly observed failure, first apply Verify SKILL.md's cap gate: classify the
defect, then check `local_strikes`, `structural_reissues`, and `uncertain_reductions`; a
transition beyond its cap publishes `stopped` instead of an owner handoff. After that,
preserve the reproducer and route ownership as follows:

- **Local → Implement:** a wrong half origin/extent, bias offset/broadcast, operation order,
  illegal local Split/Join shape, accidental `distrib` inside CV, double CV store offset,
  CV `real_shape` misuse when the brief permits manual distribution, UB overflow, or a
  source/report mismatch that can be fixed without changing the brief. Return the smallest
  failing primary/tail case, exact command/environment/cache, actual configuration, first
  error/log, and expected-versus-observed evidence. Verify does not patch the candidate.
- **Structural → Design:** the brief's fixed route/dataflow/distribution cannot legally
  deliver the Cube tile to the required Vector phase or respect the frozen boundaries. One
  concrete example is a brief that both mandates CV splitting for every variant and forbids
  the manual alternative while requiring clamped split stores. Design reissues the complete
  brief and re-derives dependent fields; Implement then synchronizes **both kernel and
  canonical harness** before a fresh Verify run.
- **Uncertain → Verify triage:** reduce the case and recheck oracle validity, installed
  package facts, actual specialization, and ownership proof. A single tile overflow, an
  unverified simulator restriction, or one illegal candidate option is not by itself proof
  that no legal configuration exists.
