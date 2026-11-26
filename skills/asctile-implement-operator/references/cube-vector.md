# Implement reference: Cube+Vector mix route

Use this file **together with** `cube.md`: that reference still owns the GM→L1→L0A/L0B
operand path, fp32 L0C accumulation, K-loop, grid, memory checks, and Cube tail rules.
This file owns the composition after a Cube result must also undergo real Vector work.
For `cube_vector`, load exactly `cube.md + cube-vector.md`; **never load the full
`vector.md` playbook**. The Vector rules needed by the mix are included below.

Facts carry provenance from the validated installed package (`pyasc-dev
1.1.1.dev20261123081345`), principally
`language/{memory_ops,prog_model_ops,binary_ops,unary_ops,shape_ops,utils}.py`,
`runtime/compiler.py` and the installed CV-strategy diagnostics. Recheck them if the
installed package changes.

**Production support in this MVP is exactly `gemm_bias_relu_vector_fp16`** with
manual `SplitByM` at `cv_ratio=2`. Other mix patterns, automatic `cv_strategy`,
SplitByN, FullVec and carried-state material are candidate practice for later
validation, not supported production recipes.

## Route invariant and output boundary

A `cube_vector` program performs the complete K reduction on the Cube unit and then
executes the brief's required Vector work **in UB, in the same kernel/program**. For the
MVP GEMM+bias+ReLU, the required path is:

```
A/B: GM → L1 → L0A/L0B → fp32 L0C accumulator
result: L0C → UB → (fp32 + broadcast bias) → ReLU → one fp16 cast → GM
bias: GM → UB (not BT)
```

A direct L0C→GM store, a `matmul`/`zeros_acc` bias operand, or `asctile.relu(acc)` on
L0C does not satisfy a brief that requires the Vector unit to perform that work. A cast
or copy through UB is not by itself proof of the epilogue: the specified add and ReLU
must consume the Cube result as UB operations. Conversely, `cv_ratio=1` with an unsplit
UB epilogue is still a legal Cube+Vector implementation; using both AIVs is a performance
choice, not the route-validity test.

The local-copy boundaries are narrower than a free-form memory graph:

- `L0C→UB` carries the completed fp32 result to Vector. `UB→GM copy_out` writes the
  epilogue result. Direct `L0C→GM copy_out` bypasses Vector and is therefore not the MVP
  epilogue path.
- For a later Cube phase, use `UB→L1`, then `L1→L0A/L0B`; there is no `L1→L0C` or
  L1↔L1 shortcut. `distrib` is not supported on UB→UB copies.
- `SplitByM`/`SplitByN` apply only to L0C→UB. `JoinByM`/`JoinByN` apply only to UB→L1.
  `copy_out` has no `distrib` argument; GM output offsets/extents are handled separately.
- `FullVec0`/`FullVec1` are legal on L0C→UB and UB→L1 and deliver the whole tensor to
  the named Vector sub-block. Guard dependent Vector work/stores to that sub-block so the
  other AIV does not duplicate it.

## `cv_ratio` and same-task sub-block mapping

Installed `CompileOptions.cv_ratio` accepts only **1 or 2** on C310. Ascend950PR_9599 has
one Cube sub-core and two Vector sub-cores per AIC; `sub_block_num()` is unit-sensitive
(1 for Cube, and 2 for Vector at `cv_ratio=2`). `cv_ratio=2` does not replace a data
distribution mechanism, and an unsplit `distrib=None` copy does not manufacture disjoint
ownership for the two AIVs.

When both AIVs cooperate on one logical output tile, derive the logical task from the
unit-sensitive indices rather than treating raw AIV `block_idx()` as a second task:

```python
start = asctile.block_idx() / asctile.sub_block_num()
for tile in range(start, tiles_num, asctile.block_num(), unroll_factor=tile_uf):
    ...
```

Launch the grid for logical tasks, **not** `tasks * 2`, and retain `cube.md`'s positive
launch-bound and tile-coverage rules. Use `sub_block_idx()` only for the local half-tile
offset (and for a `FullVec*` selector), not to subtract from an already adjusted task
index. At `cv_ratio=1`, the same-task formula reduces to the ordinary Cube mapping.

For manual distribution, with `half_m = base_m // 2` and `half_n = base_n // 2`:

| Mode | Per-AIV UB result | GM tile origin for sub-block `sb` | Bias GM origin |
|---|---|---|---|
| `SplitByM` | `[half_m, base_n]` | `[m_off + sb*half_m, n_off]` | `[n_off]` (same columns on both AIVs) |
| `SplitByN` | `[base_m, half_n]` | `[m_off, n_off + sb*half_n]` | `[n_off + sb*half_n]` |

A CV strategy inserts its corresponding store offset automatically; do **not** add the
`sub_block_idx()` term a second time inside that strategy.

The installed interpreter executes blocks/sub-blocks sequentially and makes its
`cv_strategy` override a no-op. It can help arithmetic debugging, but sequential
interpreter output is not evidence of parallel split/join synchronization or performance.
Do not import old blanket simulator core-count/deadlock restrictions from legacy notes;
use the frozen environment and actual compile/runtime evidence for the selected launch.

## Manual `DistribMode` versus a CV strategy

### Manual `distrib=` on local `copy`

This is the controllable form and the recommended MVP form when a split tail store needs
`real_shape`:

- `SplitByM`: 2-D source, M divisible by **2**, result `[M/2,N]`.
- `SplitByN`: 2-D source, N divisible by **32 elements**, result `[M,N/2]`.
- For either Split, omit `shape` or pass exactly that split result shape, never the full
  source shape.
- `JoinByM`/`JoinByN`: 2-D per-AIV UB source on UB→L1, with result shape doubled on the
  corresponding axis; omit `shape` or pass exactly the joined shape. Use distribution on
  UB→L1 at `cv_ratio=2`. Pair `SplitByM` with `JoinByM`, or `SplitByN` with `JoinByN`, when
  the halves must reconstruct one Cube operand. A Join is not needed merely to store to GM.
- `distrib=None` preserves the whole shape and applies no distribution. It is the natural
  unsplit choice at `cv_ratio=1`; at `cv_ratio=2`, do not mistake it for single-AIV
  selection or for disjoint ownership.

### Automatic `with asctile.cv_strategy(split)` — candidate practice only

The following facts explain the platform contract and preserve real developer
experience, but automatic CV is not a selectable production recipe in this MVP.
The region accepts **only** `SplitByM` or `SplitByN` and is intended for `cv_ratio=2`.
`FullVec*` and `JoinBy*` are not accepted as the region argument. Mark the L0C→UB copy and
its dependent UB operations; the compiler splits the copy, propagates the split through
supported elementwise/reduction/reshape/broadcast operations, adds the matching Join to a
UB→L1 copy, and adjusts a dependent UB→GM store offset.

The installed implementation resolves several contradictory legacy claims as follows:

1. **Omit `distrib=` inside the region.** An explicit `distrib` on a copy in a CV region is
   rejected (`'distrib' argument must be omitted inside the CV strategy`). Do not combine
   `with cv_strategy(...)` and `distrib=...` on the same transfer. Conversely, a plain
   L0C→UB copy *inside* the region is exactly how automatic splitting is requested; it is
   not an unsplit manual copy.
2. **Give a CV-managed store the full tile origin only.** The strategy adds the sub-block
   offset. Manual `+ sub_block_idx()*half` in that store double-offsets sub-block 1.
3. **A CV split store cannot use `real_shape`.** The installed apply pass rejects a
   propagated store having a real extent: `with real shape is not eligible for the CV
   strategy with splitting`. This is a CV-strategy restriction, not a blanket
   `copy_out(real_shape=...)` restriction. Moving a dependent store just outside the
   `with` block is not a workaround because CV results continue to propagate. Use manual
   distribution for a clamped split store, or use CV only where the entire static output
   tile is within the logical C extents.
4. **Split-compatible shapes are required throughout propagation.** The L0C→UB copy uses
   the full source shape. Other LocalTensor operands of an elementwise operation must
   have that same full logical shape. Reshape/broadcast/reduction must preserve the active
   split axis, and a reduction may not reduce that axis. For a broadcast epilogue operand,
   materialize the full-shape UB tile before combining it with the CV result; do not assume
   a bare 1-D bias will be reinterpreted independently per half.

For this MVP, choose one coherent style for the epilogue dependency chain. Manual and CV
are alternatives, not two annotations to stack on one copy. More involved independent
regions are outside the MVP and require their own compile evidence.

## Canonical manual MVP epilogue and tail stores

After `cube.md`'s complete K reduction has produced `acc` as fp32 `[base_m,base_n]` in
L0C, the following core pattern uses both AIVs and keeps bias/ReLU on Vector. It assumes
`base_m` is even and the rank-2 UB store's last dimension satisfies the separate 32-byte
alignment rule. It is an epilogue fragment, not a replacement for the Cube loop:

```python
sb = asctile.sub_block_idx()
half_m = base_m // 2
acc_ub = asctile.copy(acc, location=asctile.TensorLocation.UB,
                      distrib=asctile.DistribMode.SplitByM)

# Both AIVs need the same per-column bias, but only their own row-half result.
bias_valid = (n - n_off).min(base_n)
bias_ub = asctile.copy_in(bias_gm, [n_off], [base_n],
                          asctile.TensorLocation.UB, real_shape=[bias_valid])
bias_2d = bias_ub.to(asctile.float32).broadcast_to([half_m, base_n])
y_ub = asctile.relu(acc_ub + bias_2d).to(asctile.float16)

m_sb = m_off + half_m * sb
if m_sb < m:  # do not emit an empty store for a fully padded sub-block
    store_m = (m - m_sb).min(half_m)
    store_n = (n - n_off).min(base_n)
    asctile.copy_out(y_ub, c_gm, [m_sb, n_off],
                     real_shape=[store_m, store_n])
```

`real_shape` without `pad_value` on the UB bias load supplies zero padding, which is the
additive identity. The Cube operand zero-padding rules remain those in `cube.md`; do not
move `real_shape`/`pad_value` onto BT/L1/L0A/L0B loads. Every output tile origin must be a
real logical tile, and any sub-block whose origin is beyond M/N must not store. For
`SplitByN`, apply the analogous independent clamp to each sub-block's column extent and
guard an empty column half.

A CV-strategy epilogue is simpler only when no split store needs clamping: pre-stage the
bias as fp32 broadcast to the **full** `[base_m,base_n]` UB shape, use a plain L0C→UB copy
inside the region (no `distrib`), apply add→ReLU→cast, and store at `[m_off,n_off]` with no
`real_shape`. The frozen tail shape makes the manual pattern above the safer uniform MVP
choice; do not silently drop its output clamp to make CV strategy accepted.

## Vector numerics, UB shapes, and budgets

- Preserve the Cube result in **fp32** on L0C→UB. Convert fp16 bias to fp32, perform
  `acc + bias` and then ReLU in fp32, and cast once to the required fp16 GM dtype after
  ReLU. This matches the MVP reference order; an early fp16 cast is a different numeric
  choice and needs explicit justification against the tolerance.
- `asctile.relu` supports fp16/fp32 in UB. Use it on the UB result after the bias add.
  Although it also supports L0C, that L0C form is not the required Vector epilogue.
- A per-column 1-D bias may be loaded into UB directly. For **manual** SplitByM, broadcast
  it to `[half_m,base_n]` on each AIV; for manual SplitByN, load each half at its own bias
  offset and broadcast to `[base_m,half_n]`. Do not broadcast a full result tile and then
  assume manual Split magically slices that separate operand.
- For **CV strategy**, the corresponding broadcast input instead has the full logical
  result shape, because the compiler slices dependent operands. Keep this distinction
  visible in source review and `CONFIG`.
- Rank-2+ UB GM loads/stores require the last local dimension aligned to **32 bytes**
  (16 fp16 or 8 fp32 elements); this is distinct from SplitByN's 32-element constraint.
  `real_shape` rank must equal the local rank and no extent may exceed the local shape.
- UB is **253952 bytes**. Count all simultaneously live per-AIV tiles: the split/full
  accumulator, bias and broadcast tile, add/ReLU/cast intermediates, and any carried state.
  A Split reduces UB footprint only for tensors actually represented at the split shape.
  Compiler consumption plus `Launcher.check_memory_overflow`, for every variant, is
  authoritative; reduce tile/live-buffer choices rather than declaring structural failure
  from one overflow.

## Split/Join and carried-state concepts

The frozen GEMM+bias+ReLU needs no carried state. When another mix brief does need it:

- Keep a loop-carried Vector accumulator/state explicitly in UB, normally fp32 where its
  numeric role requires it, and include it in the live UB budget for the whole loop.
  Give both its initialization and every back-edge value an explicit UB location rather
  than relying on `Auto` to reconcile the carried type.
- Keep the state partitioned in the same axis/half shape as the incoming L0C→UB Split.
  If the next Cube operation consumes the full state, reconstruct it with the matching
  same-axis UB→L1 Join and then copy L1→L0A/L0B. Split is not a substitute for a Join.
- Under CV strategy, a carried value must preserve the strategy's split axis/shape and be
  used only by compatible same-split regions. Use manual distribution when the algorithm
  needs different ownership or axis handling; do not flatten/index a full logical tile and
  assume the strategy will repair it.
- Avoid unrolling a genuinely serial carried-state loop without proving live ranges and
  dependencies. Use `gm_barrier=True` only for an actual GM write-then-read recurrence,
  not for UB-resident state or ordinary read-only GEMM inputs.

These are composition rules only; they do not change `cube.md`'s fp32 L0C
`matmul`/`matmul_acc` accumulation selection or K-loop lifetime rules.

## Configuration report and handoff

For each variant, add to the Cube `CONFIG`/`RESULT.md` fields:

- actual `cv_ratio`, inferred/explicit `KernelType`, and the mechanism used:
  `manual_distrib`, `cv_strategy`, or `unsplit`;
- every actual `DistribMode`, Split/Join axis and source/result shapes, plus any `FullVec*`
  selector and its guard;
- logical task mapping (`block_idx()/sub_block_num()`), grid, and the per-sub-block GM
  output/bias origin equations;
- the L0C→UB and (if present) UB→L1 path, bias GM→UB staging, broadcast shape, where add,
  ReLU, and final cast execute, and their dtypes;
- UB load/store alignment, padding, per-sub-block `real_shape`/empty-store guard, and the
  reason a CV-managed store has no `real_shape`;
- Vector tile/live-buffer/working-set facts, carried-state layout if any, and all actual
  compile options including `reuse_alloc`/`vf_fusion` without promising a performance win.

A wrong offset, broadcast, dtype order, illegal local strategy combination, or tail extent
is a local defect: preserve the smallest failing variant, exact command/cache/config, and
log for Implement. Evidence that the brief's fixed distribution/dataflow cannot express a
legal epilogue is structural: Design reissues the complete brief and Implement synchronizes
both kernel and canonical harness before fresh Verify. A brief that mandates CV strategy
for every variant **and** forbids the manual alternative while requiring clamped split
stores is structurally contradictory; a candidate merely using CV `real_shape` when the
brief permits manual distribution is a local implementation error.
