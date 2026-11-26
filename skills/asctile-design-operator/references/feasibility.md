# Design reference: feasibility check

Feasibility is decided **before** a brief is emitted. Facts below carry provenance
from the validated installed package (`pyasc-dev 1.1.1.dev20261123081345`,
Python 3.13.5, `Platform.Ascend950PR_9599` → `CompilationArch.C310`). They are not a
runtime version-equality gate. Sources are named per fact. When in doubt, re-derive
the fact from the installed package — never trust old skills or memory.

## Platform and on-chip capacities

Verified in `asc/runtime/launcher.py::get_platform_info` (PlatformInfo dataclass) and
confirmed live against the simulator via `Launcher.get_core_num()` and
`get_platform_info(Platform.Ascend950PR_9599)`:

| Memory | Capacity (bytes) | Role |
|---|---:|---|
| UB | 253 952 (248 KiB) | Vector working set; only location Vector ops run on |
| L1 | 524 288 (512 KiB) | Cube operand staging (GM→L1), L0C→L1 result staging, JoinBy* rejoin |
| L0A | 65 536 (64 KiB) | Left matmul operand |
| L0B | 65 536 (64 KiB) | Right matmul operand |
| L0C | 262 144 (256 KiB) | fp32 accumulator bank — produced by `matmul`/`matmul_acc`/`zeros_acc`; the accumulator can also be consumed in place by the L0C-capable op `asctile.relu` and by the output cast before `copy`/`copy_out` (see Cube path facts) |
| BT | 4 096 (4 KiB) | 1-D bias operand for matmul/zeros_acc |

- `Launcher.check_memory_overflow(memory_consumed)` compares compiler-reported
  consumption against exactly these six banks — it is the protocol's memory-capacity
  check (`asc/runtime/launcher.py:159`).
- Physical structure: the installed `prog_model_ops` contract documents two Vector
  sub-cores and one Cube sub-core per AIC on Ascend950PR_9599. `cv_ratio` is declared
  as `CompileOptions.cv_ratio: Literal[1, 2]` (`compiler.py:22`) and the runtime
  membership check is at `compiler.py:87-88`; use the symbol + contract rather than a
  simulator-only config field as the source.
- `Launcher.get_core_num()` queries the runtime at run time (32 AICs observed on the
  Ascend950PR_9599 simulator). The brief must not bake a core count into any
  structural decision — grids are derived at launch time.

## Route-required facts

### Cube path (`cube`, `cube_vector`)

Verified in `asc/experimental/asctile/language/binary_ops.py::matmul` /
`matmul_acc`, `creation_ops.py::zeros_acc`, `memory_ops.py::copy`/`copy_in`/`copy_out`:

- `matmul`/`matmul_acc` operands must be 2-D `LocalTensor`s with dtype `float16`,
  `bfloat16`, or `float32`; operands are internally relocated to `L0A`/`L0B`. The
  checker also accepts fp8 pairs, but fp8 is not end-to-end validated and is not a
  supported MVP dtype. A request for a Cube contraction on int8/int32/fp64 is
  **infeasible**; other platform-feasible dtypes outside the published fp16 templates
  are candidate/unsupported, not automatically infeasible.
- The accumulator is **always `float32` in `L0C`** (`zeros_acc` dtype restriction;
  `matmul` docstring: "Result tensor type is always `float32`"). A spec demanding
  fp16 accumulation on the Cube path is infeasible — the fp32→output cast happens on
  the way out.
- Bias, when used on the Cube path, is a **1-D tensor in `BT`** whose size equals the
  N dimension of the output (`check_bias` in `language/utils.py`); dtypes
  fp16/bf16/fp32 are auto-promoted to fp32 inside the mmad. BT storage is charged at
  the **loaded** dtype, not at fp32: the 4 096-byte bank allows 2 048 fp16 elements.
  This is a structural constraint on `base_n` only when the Cube-side bias path is
  chosen.
- `copy` supports relocations: `L1→L0A/L0B/BT`, `L0C→L1`, `L0C→UB`, `UB→L1`,
  `UB→UB` (`memory_ops.py::copy` docstring, "Supported location transfers").
  **`L1↔L1` is unsupported**, and there is **no `L1→L0C`** — every Cube operand
  goes through `L0A`/`L0B`.
- `copy_in` accepts locations `UB, L1, L0A, L0B, BT` (`copy_in` docstring). A
  direct `copy_in(..., L0A/L0B)` is legal — the compiler splits it into
  `GM→L1→L0A/L0B` internally (`add_split_cube_load` pass, `compiler.py:129`), so a
  brief may choose implicit L1 staging for a single-use operand or explicit
  `copy_in(..., L1)` + `copy(..., L0A/L0B)` when the L1 tile shape must be
  controlled (transpose fusion, full-load reuse, double-buffered K loops).
- `real_shape`/`pad_value` masking is legal only on **UB/Auto** loads. It is illegal
  on BT as well as L1/L0A/L0B: `memory_ops.py`'s docstring omits BT, but the installed
  `LoadOp::verify` enforces UB/Auto and rejects BT. Cube operand tails use true GM
  descriptors and the supported static-padded/static-tile Model path; never plan a
  masked load into BT/L1/L0A/L0B.
- `copy_out` stores from **`UB` and `L0C`** locations (`copy_out` docstring). A spec
  that would require storing directly from `L1`/`L0A`/`L0B` has no legal path.
- `DistribMode` constraints (`memory_ops.py::copy` + `prog_model_ops.py::cv_strategy`):
  - `SplitByM` (L0C→UB): source 2-D, **M axis multiple of 2**; each sub-block gets
    `[M/2, N]`.
  - `SplitByN` (L0C→UB): source 2-D, **N axis multiple of 32**; each sub-block gets
    `[M, N/2]`.
  - `JoinByM`/`JoinByN` (UB→L1): doubles the axis back; pairs with the same-axis
    Split when data re-enters the Cube.
  - `FullVec0`/`FullVec1` (L0C→UB, UB→L1): whole tensor to one sub-block — no UB
    halving.
  - `cv_strategy(split)` accepts **only** `SplitByM`/`SplitByN`
    (`CVStrategyContext.__init__` raises otherwise) and marks a *region*: an explicit
    `distrib=` is rejected inside that region, not automatically on every unrelated
    transfer in the kernel. For the published MVP, use manual `SplitByM` at
    `cv_ratio=2`; other distribution styles are candidate practice only.
- In-kernel `transpose` as a standalone op is supported only for **2-D tensors in
  UB**, and the input shape must additionally be a **multiple of 16 for 2- or
  4-byte elements, or 32 for 1-byte elements** (`shape_ops.py::transpose`
  docstring). The fused form — a `copy_in(...)` directly consumed by
  `.transpose()`, fused into one data copy at compile time — supports any 2-D/3-D/
  4-D tensor without the standalone constraints. A spec that needs a transpose on
  an already-resident Cube tile cannot express it — plan it at `copy_in` or on the
  host; this is a structural-design constraint, not per se infeasible.
- `relu` supports locations `(UB, L0C)` (`unary_ops.py::relu`,
  `support_loc=(TensorLocation.UB, TensorLocation.L0C)`). It is the **only**
  elementwise op that runs on `L0C` — every other unary/binary/reduce op defaults
  `support_loc` to `(UB,)` (verified by grep across `language/*.py`). This is the
  L0C-fusion boundary the route decision uses: a lone `relu` on the GEMM result can
  stay on the Cube side (`cube`); any UB-only epilogue op makes fused on-device
  work a Vector-unit requirement (`cube_vector`).
- `hf32=True` on `matmul`/`matmul_acc` requires `float32` inputs
  (`check_matmul_arguments`); on fp16/bf16 inputs it raises.

### Vector path (`vector`)

Verified in `memory_ops.py`, `unary_ops.py`, `reduction_ops.py`, `shape_ops.py`:

- The only compute location is `UB` — `support_loc=(TensorLocation.UB,)` is the
  default for elementwise ops. `copy_in(..., location=Auto)` resolves to UB.
- `copy_in` local `shape` must be compile-time (ConstExpr/int); GM shapes and offsets
  may be runtime. 1-D copies have no alignment constraint; **2-D+ UB copies require
  the last dimension aligned to 32 bytes** (16 fp16, 8 fp32, 32 int8 elements).
- `reduce_*` dtypes differ between the axis-reduction path and the scalar
  (whole-tensor, no-dims) path — verified against `reduction_ops.py`
  `support_dtypes` / `support_dtypes_as_1d`:
  - `reduce_sum`: axes → `int32`/`int64`/`float32`; scalar → `int64`/`float16`/
    `float32` (**int32 excluded** on the scalar path).
  - `reduce_max`/`reduce_min`: axes → `int8`/`int16`/`int32`/`int64`/`float16`/
    `bfloat16`/`float32`; scalar → `int16`/`int32`/`int64`/`float16`/`float32`
    (**int8 and bf16 excluded** on the scalar path).
  - `reduce_prod`: axes → `float32` only; **scalar prod is unsupported**
    (`support_dtypes_as_1d=()`).
  fp16/bf16 reductions that must accumulate must promote to fp32 first — only the
  scalar path reads fp16 inputs directly, and bf16 has no scalar path at all.
- `softmax` accepts `float16`/`float32`, 1-D or 2-D (`unary_ops.py::softmax`).
- `cv_ratio=2` is *expressible* on a `vector` route — `CompileOptions` accepts
  `Literal[1, 2]` regardless of kernel content, and `sub_block_num()`/`sub_block_idx()`
  exist for sub-block-aware indexing — but it is not recommended: a pure-Vector
  kernel has no L0C, so `DistribMode`/`cv_strategy` (which operate on `L0C→UB` /
  `UB→L1` transfers) have nothing to split, and `cv_ratio=2` only changes the AIV
  indexing convention without adding a compute pipe. Keep `cv_ratio=1` for `vector`;
  its presence would be a structural oddity to justify, not an illegality.
  Conversely, `DistribMode`/`cv_strategy` are defined only on `L0C→UB` and `UB→L1`
  transfers (`memory_ops.py::copy`, `prog_model_ops.py::cv_strategy`) — they have no
  operand to act on without a Cube-produced tensor, so do not put them in a
  `vector` brief. (`sub_block_idx()`/`sub_block_num()` themselves are just index
  queries and remain callable regardless.)

## Structural impossibilities → infeasibility

These are the requests that cannot be made to work with **any** tile sizes under the
**installed AscTile API/contract**, so the answer is a report, not a brief. This is an
AscTile feasibility verdict, not a claim that CANN or the hardware lacks the operator
entirely:

- A Cube contraction (`matmul`/`matmul_acc`) on int8 or another dtype rejected by the
  installed argument contract. fp8 is not end-to-end validated in this MVP and must not
  be treated as a supported production dtype merely because the checker accepts it.
- A spec requiring an accumulation dtype other than fp32 in L0C.
- A spec requiring masked (`real_shape`) loads into `BT`, `L1`, `L0A` or `L0B`.
- A spec requiring Vector work that cannot be expressed through UB-resident ops
  (e.g. an op on L0C that `support_loc` does not include, other than the documented
  matmul/zeros_acc/relu set).
- A `cube_vector` structure where the *spec-mandated* split is unsatisfiable —
  e.g. it requires `SplitByM` but the spec's fixed shape/extents make an M-even
  tile family impossible, requires `SplitByN` but `base_n % 32 == 0` cannot be
  met, or simultaneously requires `cv_strategy` semantics and manual `distrib=`
  control on the same transfers (the two mechanisms are mutually exclusive only within
  the same transfer/CV region, not as a blanket whole-kernel rule). A spec-mandated split
  that merely constrains tile
  selection is a brief constraint, not infeasibility.
- A single-tensor working set that cannot fit the relevant bank at the minimum legal
  tile (e.g. a row length whose *aligned* UB footprint exceeds 253 952 bytes with no
  legal way to tile the axis — verify the axis really cannot be tiled before
  declaring infeasibility; most such requests are merely tile-constrained).
- Any data path outside the supported `copy` relocation set — e.g. a requirement to
  write GM directly from L1, or a required L1↔L1 copy.
- A spec that forbids both the Cube path (for its contraction) and the Vector path
  (for its elementwise work) simultaneously — nothing is left to run on.

Capacity pressure that **is not** infeasibility: tiles that don't fit at first guess
are a tile-family sizing matter for Implement. Infeasibility requires showing no
legal tile family exists — typically because a *spec-fixed* extent (not a tile)
exceeds a bank or violates an alignment the spec also fixes.

## Infeasibility report

When a check above fails, emit the report **instead of** a brief:

```markdown
# Infeasibility report — <task id>

- blocking constraint: <the exact platform/API rule that fails, quoting the
  verified fact and its source file>
- spec clause: <the part of the user's request that trips it>
- smallest relaxation: <the minimal spec change that would unblock — e.g.
  "allow fp32 accumulation output then cast", "bias via BT instead of UB",
  "drop the masked-load requirement for Cube operands">
- verification: <where the fact was checked in the installed package>
```

No brief, no route, no tile family accompanies an infeasibility report. If the user
relaxes the spec, Design re-runs from step 1.

## Quick verification checklist for the brief's §8

Before writing `verdict: feasible`, confirm and quote:

- [ ] Every dtype in the spec is supported on the chosen route's ops (list the ops).
- [ ] Every residency/transfer in §4 uses a supported `copy`/`copy_in`/`copy_out`
      path (list the locations).
- [ ] Every `real_shape`/`pad_value` in §7 is on a **UB/Auto** copy, never BT,
      L1, L0A or L0B.
- [ ] The `cube_vector` L0C→UB transfer mechanism is named and legal: a
      `DistribMode` (`SplitByM` needs `base_m % 2 == 0`, `SplitByN` needs
      `base_n % 32 == 0`, `FullVec*` needs none) or `distrib=None` unsplit —
      and if a split is chosen its divisibility constraints can be met by the
      tile family or mandated by the spec.
- [ ] Accumulation is fp32 wherever L0C is involved.
- [ ] `cv_ratio` ∈ {1, 2} (`CompileOptions.cv_ratio: Literal[1, 2]`; Ascend950PR_9599
      has two vector sub-cores per cube sub-core). It sets `sub_block_num()` — the
      active AIV sub-block count per AIC — and is accepted on any route
      (expressibility). Recommendation: `2` is the default for `cube_vector` to
      engage both AIVs and halve UB tiles via `SplitByM`/`SplitByN`/`cv_strategy`;
      `1` is the default for `vector` and `cube`. `cv_ratio=2` on `cube` wastes an
      AIV; `cv_ratio=1` on `cube_vector` forfeits the second AIV but remains
      expressible (`distrib=None`); `cv_ratio=2` on pure `vector` is legal but
      structurally unjustified (no L0C transfers to split — see the Vector path
      section).
- [ ] No `L1↔L1`, no GM store from L1/L0A/L0B, no masked Cube-operand load, no
      standalone transpose on Cube memory in the structural plan.
- [ ] Nothing in the brief hardcodes a core count or an exact tile size.
