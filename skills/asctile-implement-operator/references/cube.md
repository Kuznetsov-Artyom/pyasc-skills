# Implement reference: Cube route

Facts carry provenance from the validated installed package (`pyasc-dev
1.1.1.dev20261123081345`). Primary sources are `language/binary_ops.py`,
`creation_ops.py`, `memory_ops.py`, `shape_ops.py`, `global_tensor.py`, `range.py`,
`runtime/compiler.py`, and `asc/runtime/launcher.py`. Cube-transpose fusion
restrictions were also checked against installed pass diagnostics. Recheck the
installed package when it changes.

**Production support in this MVP is exactly `gemm_fp16`**: canonical row-major fp16
K-split GEMM with direct L0C→GM output. Other Cube facts are candidate practice for
later validation, not additional supported production recipes.

## Cube memory model and legal data path

- The canonical operand path is `GM→L1→L0A/L0B`, followed by
  `L0A×L0B→L0C`. `copy_in(..., L1)` plus `copy(..., L0A/L0B)` makes the staging
  explicit. A direct `copy_in(..., L0A/L0B)` is also legal: the compiler's
  `add_split_cube_load` pass splits it into GM→L1→L0 internally.
- `copy` supports L1→L0A/L0B/BT, L0C→L1/UB, UB→L1, and UB→UB. It does not support
  L1↔L1 or L1→L0C. `copy_out` can store an L0C tile directly to GM; no L1 result
  staging is needed.
- `global_tensor` describes contiguous GM storage. Declare A, B, and C with their
  **true logical extents** and matching allocation strides, while local tile shapes
  remain compile-time constants. GM extents and offsets may be runtime scalars.
  Do not host-pad an allocation and then describe it with a different logical shape.
- For Cube-only GEMM, A is `[M,K]` in L0A-shaped slices, B is `[K,N]` in
  L0B-shaped slices, and the result accumulator is `[base_m,base_n]` in L0C.
  There is no host-side contraction and no non-Cube reimplementation of the K loop.

### Exact C310 capacities and independent budgets

Query `get_platform_info(Platform.Ascend950PR_9599)` and check actual compiler
consumption with `Launcher.check_memory_overflow`. The C310 values are:

| Bank | Capacity (bytes) | Budget independently from |
|---|---:|---|
| UB | **253952** (248 KiB) | any planned UB live tiles; normally none for direct-output GEMM |
| L1 | **524288** (512 KiB) | all simultaneously live A/B staging tiles and other L1 tensors |
| L0A | **65536** (64 KiB) | A operand tiles only |
| L0B | **65536** (64 KiB) | B operand tiles only |
| L0C | **262144** (256 KiB) | live fp32 accumulators/results |
| BT | **4096** (4 KiB) | any Cube-side bias tiles |

L0A and L0B each have a separate 64-KiB budget; do **not** pool their footprints into
one 128-KiB limit. L1 is 512 KiB and UB is 248 KiB, not interchangeable
"about 256/512 KiB" allowances.

For initial sizing, a live `[r,c]` tile costs `r*c*itemsize` bytes in its own bank.
Thus one A slice costs `base_m*base_k*itemsize` in L0A, one B slice costs
`base_k*base_n*itemsize` in L0B, and one accumulator costs `base_m*base_n*4` in L0C.
Use the **actual GM→L1 load shapes**, plus every simultaneously live staging tile, for
L1. Multiply only by live copies justified by the loop/buffering structure; an unroll
factor is not an automatic, exact physical-buffer multiplier. These are sizing
heuristics, not acceptance: compiler-reported per-variant consumption and the six-bank
check are authoritative. If a tile overflows, first reduce tile sizes or unrolling;
one overflowing choice is not evidence that the brief is structurally infeasible.

## Accumulation and `matmul`/`matmul_acc`

- `matmul` and `matmul_acc` require compatible 2-D operands. The MVP's fp16/fp16
  inputs produce an fp32 L0C result; Cube accumulation is fp32. `zeros_acc` likewise
  accepts only fp32 and creates the L0C accumulator. `hf32=True` is legal only for
  fp32 inputs, so keep it false for fp16 GEMM.
- Use `matmul` only when exactly one matrix product contributes to that output tile
  by construction. It returns a new L0C result.
- Use one `zeros_acc` followed by `matmul_acc` for a K-tiled reduction. Initialize the
  accumulator once **outside all K loops**, accumulate every L0 K slice into that same
  object, and store only after all K iterations.
- Select from the **dynamic accumulation behavior**, not the number of syntactic call
  sites. One `matmul_acc` statement inside a loop may execute many times and is still
  a multi-product reduction. Conversely, a loop whose trip count is not proven one
  must not overwrite a fresh `matmul` result on each iteration or attempt to sum
  separate L0C results with elementwise operations.

## Canonical M/N-tiled, K-split GEMM

Use the brief's mapping. For one-output-tile ownership with grid-stride output tiles,
the following core structure keeps coverage and accumulation explicit. Tile sizes and
unroll values are `ConstExpr`; `m`, `n`, `k`, tile counts, and offsets are runtime:

```python
a_gm = asctile.global_tensor(a_ptr, [m, k])
b_gm = asctile.global_tensor(b_ptr, [k, n])
c_gm = asctile.global_tensor(c_ptr, [m, n])
m_blocks = asctile.ceildiv(m, m_L1)
n_blocks = asctile.ceildiv(n, n_L1)
tiles_num = m_blocks * n_blocks
tile_uf, m_uf, n_uf, k_l1_uf, k_l0_uf = double_buffering
for tile in range(asctile.block_idx(), tiles_num, asctile.block_num(),
                  unroll_factor=tile_uf):
    m_tile = (tile // n_blocks) * m_L1
    n_tile = (tile % n_blocks) * n_L1
    for i in range(asctile.ceildiv(m_L1, base_m), unroll_factor=m_uf):
        m_off = m_tile + i * base_m
        for j in range(asctile.ceildiv(n_L1, base_n), unroll_factor=n_uf):
            n_off = n_tile + j * base_n
            acc = asctile.zeros_acc([base_m, base_n], dtype=asctile.float32)
            for outer_k in range(asctile.ceildiv(k, k_L1), unroll_factor=k_l1_uf):
                k_off = outer_k * k_L1
                a_l1 = asctile.copy_in(a_gm, [m_off, k_off], [base_m, k_L1],
                                       asctile.TensorLocation.L1)
                b_l1 = asctile.copy_in(b_gm, [k_off, n_off], [k_L1, base_n],
                                       asctile.TensorLocation.L1)
                for inner_k in range(asctile.ceildiv(k_L1, base_k),
                                     unroll_factor=k_l0_uf):
                    ka = inner_k * base_k
                    a_l0 = asctile.copy(a_l1, [0, ka], [base_m, base_k],
                                        asctile.TensorLocation.L0A)
                    b_l0 = asctile.copy(b_l1, [ka, 0], [base_k, base_n],
                                        asctile.TensorLocation.L0B)
                    asctile.matmul_acc(acc, a_l0, b_l0)
            store_m = (m - m_off).min(base_m)
            store_n = (n - n_off).min(base_n)
            asctile.copy_out(acc.to(asctile.float16), c_gm, [m_off, n_off],
                             real_shape=[store_m, store_n])
```

Keep `m_L1`/`n_L1` as whole numbers of base tiles and `k_L1` as a whole number of
base-K slices for this regular structure. The simple initial choice is
`m_L1=base_m`, `n_L1=base_n`. Derive launch bounds from the same tile equations:
`tiles_num=ceildiv(M,m_L1)*ceildiv(N,n_L1)` and use a positive
`block_num <= min(Launcher.get_core_num(), tiles_num)`. Kernel iteration must use
`asctile.block_num()`/`block_idx()`, never a baked host core count. Every output tile
must be owned once, with no gaps, duplicate stores, or empty out-of-range tiles.

A conservative fp16 starting heuristic for this GEMM family is
`base_m=min(128,align_up(M,16))`, `base_n=min(128,align_up(N,16))`, and
`base_k=min(64,align_up(K,16))`; choose a positive `k_L1` that is a multiple of
`base_k`, initially at most 256 and no larger than `align_up(K,base_k)`. The 16-element
rounding here is a tile-selection heuristic, not a general API alignment rule. Check
all banks and shrink values when actual consumption requires it.

### Buffering, loops, and options

- The project-convention `double_buffering=(tile_uf,m_uf,n_uf,k_l1_uf,k_l0_uf)`
  tuple is only five values passed to the corresponding `range(...,
  unroll_factor=...)`; it is not an installed API object and does not itself prove
  double buffering. Installed `unroll_factor` must be an integer at least 1.
  `(1,1,1,2,2)` is a conventional starting point for this nested GEMM; all ones are
  the conservative fallback under memory pressure. Budget every live tile and report
  actual values without claiming a universal performance win.
- Preserve outer-K GM→L1 staging followed by inner-K L1→L0 consumption. Use
  `gm_barrier=True` only for a real cross-iteration GM dependency, not as a general
  buffering switch. Read-only A/B and one-owner C tiles do not create such a dependency.
- For direct-output Cube-only GEMM keep `cv_ratio=1`; no L0C distribution is needed.
  Let `KernelType` be inferred unless the brief requires an explicit value, and report
  the resulting type plus all actual `CompileOptions`. `matmul_cube_only` is available
  for explicit Cube-only compilation; do not confuse it with accumulation selection.
  `reuse_alloc` accepts 0/1/2 and may trade performance for lower memory use. Keep
  debug/always-recompile settings out of delivered measurements unless required and
  fully reported. Record per-tensor `l2_cache` policy as well.

## Transpose placement and fusion boundary

The frozen no-transpose GEMM needs **no transpose operation**; A `[M,K]` and B `[K,N]`
already have the expected logical orientation. `NONE`/`L1`/`L0` are useful design
labels for the three placements, not installed AscTile enum values.

When a different complete Cube brief does require transpose, use only a supported
single-purpose fusion chain:

- **L1 placement:** transpose the direct result of a GM→L1 load. The loaded L1 tensor
  must have exactly one user — the transpose — and no other load consumer. The
  transpose may feed one or more copy operations, provided every user is a copy to
  the **same** L0 destination location. This is a candidate-practice note for later
  transpose validation, not a selectable MVP recipe.
- **L0 placement:** transpose the direct result of a local `copy` to L0A or L0B.
  Transpose on L0A/L0B is supported for copy results, not arbitrary resident tensors.
- Cube transpose fusion is **2-D only**. The generic `copy_in`+single-use `transpose`
  documentation admits 2-D/3-D/4-D fusion, but the Cube pass additionally rejects
  non-2-D Cube operands. The generic standalone-transpose rule therefore does not
  legalize standalone transpose on L1/L0A/L0B.
- Do not insert a standalone transpose after unrelated computation, reuse a transposed
  L1 tile for unrelated consumers, or assume that spelling `.transpose()` guarantees
  fusion. Recompute result shapes and copy offsets after the permutation and preserve
  the brief's declared GM layout.

## Cube tail handling and direct output

- `real_shape` and `pad_value` are **not supported on copy_in to BT, L1, L0A, or L0B**;
  they are legal only on UB/Auto.
  Supplying `pad_value` alone does not bypass the restriction: the installed wrapper
  internally supplies `real_shape=shape`. This prohibition also applies to a direct
  GM→L0A/L0B load even though the compiler later splits it.
- For the canonical GEMM tail family, use plain GM→L1 `copy_in` with true GM
  descriptors and a static padded tile shape. On the validated Model path, the tail
  GEMM with M=72, N=80, K=56 passed correctness with exact output extents and no
  recorded illegal-DMA diagnostics. This is compiled/runtime Model evidence, not a
  proof based on interpreter semantics and not an NPU guarantee. Do not replace this
  with a masked Cube load, and do not claim arbitrary pad values are available on that
  path. Only valid M/N output extents are stored.
- Clamp only the **output store** with rank-2
  `real_shape=[min(base_m,M-m_off),min(base_n,N-n_off)]`. The runtime-scalar `.min`
  form avoids Python-only type reassignment in JIT tracing. `real_shape` must not
  exceed the local tile shape. Iterate only output tiles whose origins are within the
  logical M/N extents; padding does not justify launching an empty tile.
- For Cube-only GEMM, cast the completed fp32 accumulator to the required GM dtype and
  call `copy_out` directly from L0C, as in the skeleton. Do not add an L0C→UB hop,
  `DistribMode`, or `cv_strategy` to this direct path. UB may therefore be unused, but
  its 253952-byte bank is still included in the compiler memory report.
- Prove both input-read and output-write boundaries in source. Correct values and GM
  output guards do not by themselves prove safe input reads. Record the actual tail
  mechanism, padded local shapes, valid store extents, and K-loop trip equations per
  variant.
