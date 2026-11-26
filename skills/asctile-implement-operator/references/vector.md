# Implement reference: Vector (AIV) route

Facts carry provenance from the validated installed package (`pyasc-dev
1.1.1.dev20261123081345`). Primary sources are `language/memory_ops.py`,
`global_tensor.py`, `range.py`, `reduction_ops.py`, `shape_ops.py`, `unary_ops.py`,
and `runtime/compiler.py`. Recheck the installed package when it changes.

**Production support in this MVP is exactly `vector_add_fp16`**: `out = a + b` with
fp16 contiguous 1-D tensors. Other Vector facts below are retained as candidate
practice for later validation, not as supported production recipes.

## Route and GM ↔ UB data path

- Pure Vector compute operates on UB `LocalTensor`s. `copy_in(..., location=Auto)`
  resolves to UB; global tensors describe contiguous GM storage. Do not introduce
  `matmul`/`matmul_acc`, L0A/L0B/L0C residency, `DistribMode`, or `cv_strategy`.
- `global_tensor(ptr, shape, l2_cache=True)` accepts runtime GM extents. Its declared
  shape must be the **true logical extent**, not a padded allocation. `l2_cache` is a
  per-tensor bool; choose and report it rather than assuming one policy fits all data.
- `copy_in` local `shape` is compile-time (`int`/`ConstExpr`); GM shapes and offsets may
  be runtime. Omitting local `shape` loads a scalar. One-dimensional copies have no
  alignment restriction; 2-D+ UB copies require the local last dimension to be a
  32-byte multiple (16 fp16/bf16, 8 fp32/int32, 32 int8 elements).
- `copy_in(..., real_shape=r, pad_value=v)` loads only `r` and fills the rest with `v`;
  `copy_out(..., real_shape=r)` stores only `r`. `real_shape` has local-tensor rank and
  cannot exceed static local shape. If `real_shape` is present without `pad_value`, the
  installed API defaults the pad to zero; without either, pad lanes are uninitialized.
- Never rely on implicit GM clamping for the boundary proof. Iterate only logical tiles,
  give each load/store its exact valid extent, and do not launch empty tiles. Full-tile
  accesses at or beyond the GM edge can produce illegal DMA even when values and output
  guards happen to pass.

## Bounded flat elementwise skeleton

Use the brief's selected mapping. For flat elementwise work, grid-stride tiles make the
coverage and tail proof direct:

```python
@asctile.jit(vf_fusion=1)
def elementwise(a_ptr: asctile.GlobalAddress, b_ptr: asctile.GlobalAddress,
                out_ptr: asctile.GlobalAddress, n: int,
                tile: asctile.ConstExpr[int], uf: asctile.ConstExpr[int]):
    a_gm = asctile.global_tensor(a_ptr, [n])
    b_gm = asctile.global_tensor(b_ptr, [n])
    out_gm = asctile.global_tensor(out_ptr, [n])
    tile_count = asctile.ceildiv(n, tile)
    for t in asctile.range(asctile.block_idx(), tile_count, asctile.block_num(),
                           unroll_factor=uf):
        offset = t * tile
        valid = min(tile, n - offset)  # installed JIT maps builtin min to PlainValue.min for scalars
        a = asctile.copy_in(a_gm, [offset], [tile], real_shape=[valid], pad_value=0)
        b = asctile.copy_in(b_gm, [offset], [tile], real_shape=[valid], pad_value=0)
        asctile.copy_out(a + b, out_gm, [offset], real_shape=[valid])
```

On the host choose `block_num = min(Launcher.get_core_num(), ceildiv(n, tile))` (or a
smaller positive, reported choice) and launch `elementwise[block_num](...)`. The kernel
must use the launched `asctile.block_num()`, not that host value as a baked constant.
For `out = a + b`, zero padding is neutral and only valid lanes are stored. A contiguous
slab or per-core-row mapping is also valid only when its per-block iteration bounds and
`real_shape` exclude every empty/out-of-extent access; derive slab length from the exact
same loop count.

## Exact tiling, grid, loops, and options

- Query `get_platform_info(Platform.Ascend950PR_9599).ub_size` (253952 bytes on this
  platform) and `Launcher.get_core_num()` at run time. A conservative initial elementwise
  budget is `tile * itemsize * live * uf <= ub_size - reserve`, where `live` includes all
  simultaneously live inputs, outputs, and temporaries. This is a sizing heuristic, not
  the acceptance check: compile and run `Launcher.check_memory_overflow` on actual
  compiler consumption.
- One-dimensional tiles need no legal alignment, but a 32-byte-aligned size is a useful
  initial choice. For 2-D tiles, align the **local** last dimension and use `real_shape`
  for the true row width. For a narrow contiguous reduction, a legal alternative is one
  contiguous 1-D load followed by `reshape`; `reshape` preserves element count and uses
  static shapes.
- `asctile.range(..., unroll_factor=uf)` requires integer `uf >= 1` (installed default 1).
  Start with the brief's loop structure, budget higher unrolling against all live tiles,
  and report the actual value. Use `gm_barrier=True` when iterations depend on prior GM
  writes; it prevents cross-iteration load/store reordering and is not a general tuning
  switch.
- `vf_fusion` accepts 0/1/2 and is C310-only when enabled; installed default is 0. Use 1
  as the initial Vector choice unless the brief/evidence says otherwise, and report what
  was measured. Do not claim a universal performance win.
- `reuse_alloc` accepts 0/1/2 (default 0) and may reduce memory at a performance cost.
  Prefer shrinking tile/unroll first; if reuse is needed, pass the chosen option at that
  shape's call site and report it. JIT options may be supplied at the decorator or call
  site and are part of the compiled specialization.
- Keep `cv_ratio=1` for pure Vector unless the complete brief explicitly justifies
  another legal choice. The installed compiler accepts 1 or 2; 2 is not intrinsically
  invalid, but there is no L0C transfer for `DistribMode`/`cv_strategy` to split on this
  route.
- Record actual grid, tile shapes, loop order, `uf`, `vf_fusion`, `reuse_alloc`, L2 flags,
  `cv_ratio`, tail mechanism, and the complete `ConstExpr` set per variant. Runtime scalar
  values do not by themselves create distinct JIT specializations; do not build harness
  validity around a guessed cache-entry count.

## Reduction, normalization, and shape correctness

These are candidate-practice facts for future recipes, not a supported reduction or
normalization implementation in this MVP.

- Axis `reduce_sum` accepts int32/int64/fp32; scalar whole-tile sum accepts
  int64/fp16/fp32. Axis `reduce_max`/`reduce_min` accept int8/int16/int32/int64/fp16/
  bf16/fp32; their scalar paths exclude int8 and bf16. `reduce_prod` is fp32-only on the
  axis path and has no scalar path. Promote fp16/bf16 accumulation to fp32 where the
  selected API requires it; cast back as the output contract requires.
- Pad with the consuming operation's identity: 0 for sum, `-inf` for max, `+inf` for min,
  1 for product. If padded lanes enter other arithmetic, choose a value that also keeps
  that arithmetic defined; never treat omitted padding as zero.
- `keep_dims=True` retains reduced axes as size 1 for broadcasting; false squeezes them.
  A no-axis reduction returns a scalar. Check the stored shape and broadcast axes against
  the brief rather than relying on tolerance comparison to catch rank mistakes.
- `asctile.softmax` supports fp16/fp32, 1-D or 2-D, reducing the whole 1-D tile or each
  2-D row. `layer_norm` supports fp16/bf16/fp32; `rms_norm` supports fp16/fp32. Use these
  only when their shape/dtype semantics match the brief, and account for padded lanes.
- Hoist invariant loads when the brief's dataflow permits. Keep immutable tile values and
  live temporaries inside the UB budget. For each variant, prove coverage, boundary
  extents, reduction shape, and numeric dtype before using ticks for any comparison.
