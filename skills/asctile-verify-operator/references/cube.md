# Verify reference: Cube-route checklist

Apply this after the independent oracle and verification procedure are valid. API facts
carry provenance from the validated installed package (`pyasc-dev
1.1.1.dev20261123081345`), principally
`language/{binary_ops,creation_ops,memory_ops,shape_ops}.py`, `runtime/compiler.py`,
`asc/runtime/launcher.py`, and installed Cube-transpose diagnostics. This checklist
adds verification criteria, not implementation choices.

The only published Cube acceptance template is `gemm_fp16`; transpose and other Cube
material are candidate practice, not supported production coverage.

## Route, residency, and output checklist

- [ ] The entire K reduction is performed by compatible 2-D `matmul`/`matmul_acc`
      operations on L0A/L0B operands, producing or updating an fp32 L0C result. There
      is no host-side contraction and no substitute multiply-accumulate implementation
      outside the Cube path.
- [ ] A and B follow the declared GM→L1→L0A/L0B residency. Direct GM→L0A/L0B
      `copy_in` is legal only with the compiler's implicit L1 split; it is not evidence
      that L0 staging was skipped. No path relies on L1↔L1, L1→L0C, or a GM store from
      L1/L0A/L0B.
- [ ] Operand dtypes/ranks/shapes satisfy `input.shape[1] == other.shape[0]`, and the
      accumulator shape matches `[input.shape[0],other.shape[1]]`. MVP fp16 inputs use
      `hf32=False`; fp32 L0C accumulation and the final GM cast are reported accurately.
- [ ] A Cube-only direct-output candidate casts the completed L0C result and stores it
      directly to GM. An undeclared L0C→UB result hop, `DistribMode`, or `cv_strategy`
      is a structure/report mismatch. `cv_ratio=1` is the expected direct-path choice.
- [ ] Actual inferred or explicit `KernelType`, `matmul_cube_only`, `cv_ratio`,
      `reuse_alloc`, debug/recompile settings, and all other JIT/call-site options match
      the delivered specialization and `CONFIG`/`RESULT.md`.

## Accumulation and K-loop audit

- [ ] Determine accumulation from **executed semantics**, not source call-site counts.
      A single `matmul_acc` statement nested in a loop represents every dynamic
      iteration. `matmul` is appropriate only where exactly one product contributes to
      the output tile; repeated fresh `matmul` results must not overwrite or be added
      as a substitute for L0C accumulation.
- [ ] For K-split code, one fp32 `zeros_acc` is initialized outside every K loop, all
      outer/inner K slices update that same accumulator, and the store occurs after the
      complete reduction. Reinitialization inside a K loop, an early store, or an
      omitted K slice is a numerical defect even if a particular input happens to pass.
- [ ] Reconstruct outer/inner K trip counts and offsets from the actual runtime K,
      `k_L1`, and `base_k`. Check full logical-K coverage exactly once, with only the
      declared zero-padded tail contributing extra products. Record actual loop order
      and every `unroll_factor`, not merely the name of a buffering tuple.
- [ ] Reconstruct output tile ownership from tile counts, grid, block indices, and M/N
      sub-tile loops. Every logical C element has exactly one owner and one store; no
      empty out-of-range tile, overlap, or gap is hidden by a passing value comparison.

## Independent memory-capacity checklist

- [ ] The specialization used for each variant passes `Launcher.check_memory_overflow`
      and reports all six actual compiler-consumption values. Source sizing arithmetic
      is supporting evidence, never a replacement for the compiled specialization's
      metadata.
- [ ] Capacities on Ascend950PR_9599/c310 are exactly **UB 253952**, **L1 524288**,
      **L0A 65536**, **L0B 65536**, **L0C 262144**, and **BT 4096 bytes**. Reject any
      audit that treats UB as 262144 bytes, inflates L1, or substitutes a different
      platform's limits.
- [ ] L0A and L0B are checked as **separate 65536-byte banks**. Do not pool their
      capacities, compare their combined footprint with one 64-KiB budget, or excuse an
      individual overflow because the other bank has spare room.
- [ ] Independently recompute expected operand footprints from actual local shapes,
      dtypes, loop nesting, and simultaneously live buffers. L0C accounting uses fp32;
      L1 includes every live A/B staging tile and other L1 allocation. Investigate a
      material mismatch with compiler metadata rather than forcing a guessed buffering
      multiplier to fit it.
- [ ] A direct-output Cube-only plan normally has no result-tile UB working set, but UB
      consumption/capacity must still appear in `MEMORY_CHECK`. Nonzero UB consumption
      prompts a source/path reconciliation; it is not silently omitted because the
      report says "Cube-only".
- [ ] A single oversized tile/unroll choice is a **local** capacity defect. Structural
      backtrack requires evidence that no legal tile family can satisfy the brief's
      fixed dataflow/residency or fixed extents.

## Boundary and transpose checklist

- [ ] GM descriptors and actual allocations agree on true logical A `[M,K]`, B `[K,N]`,
      and C `[M,N]` extents and strides. A padded allocation paired with an incompatible
      logical descriptor is not a valid tail mechanism.
- [ ] No `copy_in` to **BT, L1, L0A, or L0B** uses `real_shape` or `pad_value`, including a
      direct GM→L0 load and a `pad_value`-only call. Do not accept source comments that
      claim masked Cube operands when the wrapper rejects them.
- [ ] For the declared static-padded tail family, plain GM→L1 loads use true descriptors
      and static padded tiles. On the validated Model path, use the tail GEMM's compiled/
      runtime correctness and DMA evidence rather than interpreter semantics. Prove that
      each load origin belongs to a real logical tile and that only valid M/N extents are
      stored. Do not infer input-read safety from output guards or numerical correctness
      alone, and do not generalize this Model evidence to NPU.
- [ ] Every output `copy_out(real_shape=...)` has rank 2, does not exceed its local tile,
      and equals the exact valid M/N extent at that tile origin. Stores cannot write
      padded C lanes. Both required 64-byte output guards remain bit-identical, while
      source/index proofs separately cover input reads.
- [ ] For the frozen no-transpose GEMM, any actual transpose is unjustified and must be
      reconciled with the brief and report. Design labels such as NONE/L1/L0 are not
      mistaken for installed API enums.
- [ ] If a candidate-practice brief requires **L1 transpose fusion**, the transpose
      input is a direct GM→L1 load result whose only user is the transpose. Transpose
      users must all be copies to the **same** L0 destination; multiple same-location
      copy users are legal. Arbitrary resident L1 transpose or mixed destinations are
      not accepted as the same supported pattern. This is not a published MVP template.
- [ ] If the brief requires **L0 transpose fusion**, the transpose input is the direct
      result of a copy to L0A/L0B. Cube fusion is 2-D only; the generic 2-D/3-D/4-D
      `copy_in`+transpose note does not broaden the Cube rule. Result shapes and all
      subsequent copy/matmul offsets reflect the permutation.

## Configuration and evidence reconciliation

Reconcile executable source, launch, memory metadata, `CONFIG`, `RESULT.md`, and logs per
variant for all of the following:

- [ ] grid/launch bounds, runtime core query, kernel `block_num()`/`block_idx()` use,
      M/N tile ownership, and the actual `m_L1/n_L1/k_L1/base_m/base_n/base_k` values;
- [ ] K-loop nesting and trip equations, each range's unroll value, any `gm_barrier`,
      dynamic accumulation choice, fp32 accumulator, output dtype/cast, and direct L0C
      output path;
- [ ] per-tensor L1/L0/UB residency, transpose placement/fusion chain, true GM extents,
      tail padding and output clamp, and per-tensor `l2_cache` policy;
- [ ] complete `ConstExpr` set versus runtime scalars, actual kernel/compile options,
      seed and operand draw order, cache directory and truthful cache-warm state;
- [ ] identical input tensors, kernel, grid, tile family, and options for warmup
      correctness and every measured launch. Report each variant even when two variants
      legitimately share a compiled specialization.

A missing, stale, or contradictory field is a reporting/submission defect, not permission
for Verify to repair the candidate. A passing `assert_close` never erases a route,
residency, out-of-bounds, or configuration violation.

## Defect routing

Before routing any defect, apply Verify SKILL.md's cap gate: classify first, then check
`local_strikes`, `structural_reissues`, and `uncertain_reductions`; a transition beyond
its cap publishes `stopped` instead of an owner handoff.

- **Local → Implement:** concrete tile/grid/offset/K-bound error, accumulator init/store
  placement, output clamp, per-bank overflow, option/report mismatch, or a locally
  incorrect transpose chain that can be fixed while preserving every structural brief
  field. Return the smallest failing variant, actual configuration, and evidence.
- **Structural → Design:** the brief itself requires masked L1/L0A/L0B loads, non-fp32
  L0C accumulation, an unsupported Cube data path, a non-2-D or standalone resident-Cube
  transpose, or a fixed dataflow/residency that no legal tiling can satisfy. Return the
  installed-package/platform evidence and require a complete brief reissue.
- **Uncertain → Verify triage:** reduce the case and revalidate the oracle, environment,
  memory metadata, and source/index proof before choosing an owner. Never call one tile
  overflow structural or repair a candidate from Verify.
