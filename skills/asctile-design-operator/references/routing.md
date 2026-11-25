# Design reference: route selection, brief schema, targeted questions

This reference holds the three things SKILL.md delegates here:

1. the **route decision procedure** (step 2),
2. the **design-brief schema** the brief must fill (step 4),
3. the **targeted-question menu** for missing semantics (step 1).

All API facts below are verified against the installed `asc.experimental.asctile`
package in the evaluation venv (`pyasc==1.1.1+g63ed594e.d20261122`, Python 3.13.5).
Where a fact has a single authoritative source, that source is named inline.

## Route enum

The brief emits exactly one route:

| Route | Meaning | Implement loads | Verify loads |
|---|---|---|---|
| `vector` | All compute on the Vector (AIV) pipe through UB — elementwise/activation/cast chains, reductions, normalization, index/data movement. No `matmul`/`matmul_acc` anywhere. | `vector.md` | `vector.md` + `simulator.md` |
| `cube` | Dominant compute is a Cube matmul (`matmul`/`matmul_acc`, L0A×L0B→L0C). No Vector elementwise work on the result — at most a `.to(dtype)` cast or an L0C-fusible op (`asctile.relu`, the only elementwise op with `support_loc=(UB, L0C)` in the installed package) executed on the Cube side, then a direct `L0C→GM copy_out`. | `cube.md` | `cube.md` + `simulator.md` |
| `cube_vector` | Cube matmul **plus** spec-mandated Vector-unit work on the result (epilogue, multi-phase, carried state, multi-tile combine) inside the same program — an L0C→UB hop with a named transfer mechanism (`DistribMode` split/join, a `cv_strategy` region, or `distrib=None` unsplit); `cv_ratio=2` + a split is the recommended (not mandatory) structure to engage both AIVs. | `cube.md` + `cube-vector.md` | `cube.md` + `cube-vector.md` + `simulator.md` |

`cube_vector` never loads `vector.md`: the mix reference carries only the Vector delta
the route needs.

## Route decision procedure

Apply these tests **in order** to the spec's *mandated* semantics. The route is what
the spec requires to run on each unit, not what the math could be coerced into.

### Test 1 — Is a K-dimension reduction on the Cube unit required?

The spec requires the Cube path when either holds:

- It explicitly mandates a Cube/MatMul unit reduction (e.g. "the K-dimension
  reduction must be performed by the Cube unit (`asctile.matmul` /
  `asctile.matmul_acc` or equivalent)"), **or**
- The operator semantics are a dense contraction `C[...] = Σ_k A[...,k]·B[k,...]`
  (GEMM, BMM, matmul, attention QK/PV-style fragments, `x @ W`) and nothing in the
  spec forbids or reassigns the reduction.

If no K-contraction exists at all → go to Test 4.

If a K-contraction exists → **Test 2**.

Notes:
- A spec that only says "implement `A @ B`" without naming a unit still resolves to
  the Cube path: `matmul`/`matmul_acc` is the only AscTile mechanism for a dense
  K-reduction, and a Vector multiply-accumulate re-implementation is an implementation
  detail the spec is not asking for.
- If the spec *explicitly* forbids the Cube unit for a contraction (unheard-of, but
  possible), treat it as a missing-semantics case and ask a targeted question — do not
  silently route `vector`.

### Test 2 — Is real Vector work required on the Cube result?

`cube_vector` requires **both**: a Cube K-reduction (Test 1) **and** work the spec
mandates run on the **Vector unit**, inside the same program. "Mandated on the
Vector unit" is the decisive phrase — fused elementwise work that *can* stay on the
Cube side does not by itself force `cube_vector`:

- `asctile.relu` is the **only** elementwise op whose `support_loc` includes `L0C`
  (`unary_ops.py:255`, `support_loc=(UB, L0C)` — verified by grep: no other op
  declares L0C). `relu(A @ B)` can therefore execute entirely on the Cube side —
  apply `relu` to the L0C accumulator immediately after `matmul`, `.to(dtype)`,
  direct `L0C→GM copy_out` — and that execution structure is the `cube` route.
  Whether a spec gets that structure still depends on what it mandates: explicit
  Cube-side allowance → `cube`; explicit Vector requirement → `cube_vector`;
  ambiguous → unit-placement question (see boundary cases).
- Everything else elementwise (`add`, `sub`, `mul`, `maximum`, `exp`, `where`,
  `reduce_*`, `broadcast_to`, `.to` chains on a UB tile, …) is UB-only — it
  genuinely needs the AIV pipe and makes the route `cube_vector` when required
  on-device on the GEMM result.

The spec forces `cube_vector` when:

- It mandates an on-device **Vector** epilogue on the GEMM result (e.g. "a real
  Vector epilogue applies `+bias` and `relu` inside the same kernel" — the unit is
  named, and a Cube-side fold is disallowed), **or**
- The operator math is `epilogue(A @ B)` where `epilogue` contains UB-only work
  (bias add, softmax, scale chains, where/select, reductions — anything except a
  lone relu) **and** the spec requires it fused on-device (not host-side, not a
  second kernel), **or**
- The structure is inherently Cube↔Vector multi-phase (Cube result feeds Vector work
  that feeds another Cube op — attention patterns, chained matmuls with on-chip
  elementwise between them) or keeps loop-carried state the Vector unit updates.

If Test 1 passed and mandated Vector-unit work exists → **`cube_vector`**.

If Test 1 passed and no spec-mandated Vector-unit work exists → **`cube`**
— **except** when the spec contains an epilogue but leaves its unit unstated:
that is ambiguity about the route itself, not absence of Vector work, and it
goes to the unit-placement targeted question (see boundary cases / question
menu), never to a `cube` default.

Boundary cases:

- **Cast-only is not an epilogue.** `C = (A @ B).to(fp16)` — a `.to(dtype)` on the L0C
  accumulator then `copy_out` direct to GM — is `cube`. The cast can run on the Cube
  core; no AIV work is needed.
- **Lone `relu`: explicit Cube-side → `cube`; explicit Vector → `cube_vector`;
  ambiguous → ask.** `relu` is L0C-fusible (see above), so `C = relu(A @ B)`
  routes `cube` when the spec explicitly permits a Cube/L0C-side epilogue (e.g.
  "apply relu on the L0C accumulator", "no Vector epilogue required", or "relu may
  fuse wherever it can"). A spec that *explicitly requires the relu to execute on
  the Vector unit* (as `gemm-bias-relu` does — its spec bans any implementation
  "such that the Vector unit never executes the elementwise work") routes
  `cube_vector`. If the spec contains the epilogue but is **ambiguous about which
  unit** runs it, that is the unit-placement targeted question below — do not
  resolve it by defaulting to `cube`.
- **Cube-side bias is not a Vector epilogue.** `asctile.matmul` accepts a `bias`
  operand (1-D, `BT` location) and `zeros_acc` accepts `bias=` initialization; both
  keep the add on the Cube path. If the spec *permits but does not require* the Vector
  unit to do the add, `cube` is correct. If the spec explicitly requires the bias add
  to run on the Vector unit, the route is `cube_vector`.
- **No epilogue at all → `cube`; epilogue with unstated unit → ask.** A GEMM whose
  spec has no fused elementwise math at all routes `cube` — do not invent a Vector
  stage. A GEMM+epilogue spec that is silent on *which unit* runs the epilogue is
  a **targeted-question** case (unit placement decides the route) — see the
  question menu; never default such a spec to `cube`.
- **Parallel-but-independent Vector work.** A spec that requires Vector work with no
  data dependency on the Cube result inside the same kernel still routes
  `cube_vector` — the deciding factor is shared-program Cube+AIV structure, not
  strictly an "epilogue".

### Test 3 — Cube-side structural sanity (quick check)

Before committing to `cube`/`cube_vector`, confirm the Cube path can express the
operands: `matmul`/`matmul_acc` inputs must be 2-D `LocalTensor`s of `float16`,
`bfloat16`, or `float32` (fp8 `float8_e4m3fn`/`float8_e5m2` pairs also accepted by
`check_matmul_arguments`), accumulator is always `float32` in L0C. A spec demanding a
Cube contraction on an unsupported dtype (e.g. int8 matmul) fails feasibility — go to
`references/feasibility.md`.

### Test 4 — Pure-Vector confirmation

No Cube contraction → the route is `vector` **if** the op is expressible through the
Vector vocabulary: elementwise binary/unary ops, `reduce_*`, `softmax`/`layer_norm`/
`rms_norm`, `broadcast_to`/`reshape`/`where`/`gather`/`transpose` (2-D UB), `copy_in`/
`copy_out` through UB. That covers the vast majority of non-matmul ops.

If the op cannot be expressed on either pipe (e.g. requires a data movement AscTile
doesn't expose), it is a feasibility failure — `references/feasibility.md`, not a
fourth route.

### Worked calibration examples

| Spec | Tests | Route |
|---|---|---|
| `out[i] = a[i] + b[i]`, fp16 | no K-contraction → T4, pure elementwise | `vector` |
| `C[i,j] = Σ_k A[i,k]·B[k,j]`, fp16, "reduction must run on Cube unit" | T1 yes → T2 no epilogue | `cube` |
| `C = relu(A @ B + bias[j])`, spec mandates on-device Vector epilogue, forbids Cube-side bias fold | T1 yes → T2 real Vector work required | `cube_vector` |

## Brief schema

The brief is a single markdown document. Sections 1–10 must be **concrete** — the
brief fixes **structure**; Implement fixes **numbers** inside the structure's
constraints. `deferred-to-implement` is allowed **only** in §11 (Implement-owned
numerical and open items: tile values, unroll values, launch options, exact
offsets, cache policy). An absent or deferred structural field is a defect —
if a structural fact is unavailable, that is a targeted-question case, not a
deferral.

```markdown
# Design brief — <task id>

## 1. Semantic contract
- math: <exact formula, e.g. C[i,j] = relu( Σ_k A[i,k]·B[k,j] + bias[j] )>
- tensors: <name → shape / dtype / layout / broadcast semantics for every input
  and output; note which dims are compile-time vs runtime>
- variants/shapes: <per-variant shape table if the spec fixes several>
- boundary rule: <which logical extents must be respected; e.g. "no GM access
  outside [0,N) of any tensor">

## 2. Route
- route: vector | cube | cube_vector
- evidence: <the spec clauses / decision-procedure tests that forced this route>

## 3. Algorithm family
<The structural skeleton, not the code. Pick exactly one.>
- vector: flat elementwise slab | row-block reduce/normalize | per-element
  index-driven | per-row accumulate | <other named pattern>
- cube: K-split accumulate (zeros_acc + matmul_acc K loop) | single-shot matmul |
  M/N-split | StreamK family
- cube_vector: GEMM + Vector epilogue | Cube→Vector→Cube multi-phase | stateful/
  loop-carried | multi-tile combine — plus the L0C→UB transfer mechanism:
  `cv_strategy` region | manual `distrib=` on L0C↔UB copies (recommended —
  engages both AIVs at `cv_ratio=2`) | `distrib=None` unsplit (legal at
  `cv_ratio=1`; whole tile to the single vector context — valid but leaves the
  second AIV unused, a performance choice not a validity one)

## 4. Structural dataflow and residency
<Per tensor/result: where it lives and how it moves — the pipe, not the sizes.>
- e.g. A: GM → L1 staging → L0A operand; B: GM → L1 → L0B;
  acc: L0C fp32; result: L0C →UB (fixpipe, distrib=…) → Vector epilogue → GM;
  or: a/b: GM → UB; out: UB → GM.
- Explicitly state the L0C output path for cube routes:
  direct L0C→GM (cast on Cube core) | L0C→UB→GM. When the brief routes
  `cube_vector` and the L0C result is consumed by Vector work, the L0C→UB hop
  must name its transfer mechanism: a DistribMode (`SplitByM`/`SplitByN`/
  `FullVec*`) or `distrib=None` (unsplit) — never leave it implicit. (A
  `cube_vector` brief whose Vector work is parallel and does not consume the L0C
  result states the direct `L0C→GM` path plus the independent Vector dataflow.)

## 5. Parallelism
- block mapping: contiguous slab | grid-stride tiles | per-core rows |
  one-output-tile-per-block | <named>
- cv_ratio: 1 | 2 — expressible anywhere but recommended only where Vector work
  exists: `cube_vector` defaults to 2 (engages both AIVs, halves UB tiles via the
  split), `vector`/`cube` stay 1
- L0C→UB transfer mechanism and split axis: `cv_strategy(SplitByM|SplitByN)` |
  manual `distrib=SplitByM|SplitByN` | `FullVec*` | `distrib=None` unsplit —
  with the divisibility constraint the chosen split imposes (see §6 note;
  `distrib=None`/`FullVec*` impose none)
- grid source: derived from tile counts at launch time (never a baked constant)

## 6. Tile family
<Tile parameter NAMES and their constraints — never final values.>
- vector: e.g. `tile` (ConstExpr, multiple of 32/itemsize), `uf`, live-tile count;
- cube: `m_L1, n_L1, k_L1, base_m, base_n, base_k` (all ConstExpr), accumulator
  dtype fp32; constraints e.g. `base_m % 2 == 0` iff SplitByM chosen,
  `base_n % 32 == 0` iff SplitByN, `base_m·base_n·4 ≤ 262144` (L0C capacity);
  per-loop `unroll_factor` parameters as ConstExpr — optionally bundled as the
  project-convention `double_buffering = (tile_uf, m_uf, n_uf, k_l1_uf, k_l0_uf)`
  5-tuple (one unroll factor per nested loop: output-tile grid, L1-M, L1-N,
  GM→L1 K-chunk, L1→L0 K-slice). This is a **brief convention**, not an installed
  API field — `unroll_factor` itself is a parameter of `asctile.range`.
- which values are ConstExpr vs runtime scalars.

## 7. Boundary / tail strategy family
<The mechanism, not the offsets.>
- e.g. clamped copies vs declared GM extents | copy_in real_shape + pad_value
  (identity where it enters math) | copy_out real_shape clamp |
  in-flight zero-pad on GM→L1 copy_in | per-sub-block real_shape on stores.
- Note the platform constraint: real_shape is illegal on L1/L0A/L0B copy_in
  (legal on UB/BT loads; verified in memory_ops.py).

## 8. Feasibility verdict
- verdict: feasible (a brief only ever says feasible; infeasible produces an
  infeasibility report instead of a brief)
- checks relied on: <the specific feasibility.md rules consulted, with the
  capacities/dtype facts quoted at check time>

## 9. Correctness contract
- reference: <exact formula string, e.g. "(a.float() + b.float()).half()">
- tolerance: atol/rtol values and the comparison method
- internal-precision note: <e.g. "accumulation dtype is an implementation choice;
  output must meet the fp16 reference check">

## 10. Baseline availability
- measured_baseline: available | unavailable | partial
- location: <path to baseline artifacts, e.g. mvp/baseline/<task>/>
- note: baseline numbers are comparison references for A/B, never targets the
  brief optimizes toward; a failed baseline still counts as "available" evidence
  and is marked partial/failed accordingly.

## 11. Open items for Implement
<Anything deliberately deferred: tile VALUES, unroll values, launch options,
exact offsets, cache policy — flagged so Implement knows it owns them.>
```

## Targeted questions

When a semantic fact is missing, ask **one** targeted question naming exactly the
missing fact and why the brief cannot proceed without it. The menu — pick the closest,
never ask about things the spec already answers:

| Missing / ambiguous fact | Ask |
|---|---|
| dtype of any tensor | "What dtype is `<tensor>`? The brief cannot fix tolerance, residency sizing, or dtype support checks without it." |
| layout / transpose | "Is `A`/`B` stored row-major `[M,K]`/`[K,N]`, or is a transpose flag expected? This changes the staging path, not just the formula." |
| epilogue placement | "Must `+bias`/`relu`/`<op>` run on-device inside the same kernel, or may it be host-side / a second kernel? This alone decides `cube` vs `cube_vector`." |
| epilogue unit placement | "The epilogue is fused on-device — must it execute on the **Vector unit** specifically, or may it stay on the Cube side? Both are expressible (`relu` is L0C-fusible via `support_loc=(UB, L0C)`; a bias add can fold into `matmul`/`zeros_acc(bias=…)` on BT), and the answer decides `cube` vs `cube_vector` — the brief cannot be emitted without it." |
| Cube-side bias allowed? | "May the bias add use the Cube-side path (`matmul`/`zeros_acc` bias operand), or must the Vector unit execute it? If the Vector unit must do it, the route is `cube_vector`." |
| broadcast semantics | "How does `<tensor>` broadcast — per-row, per-column, scalar? The residency plan and reference both depend on it." |
| reduction axis | "Which axis is reduced and is `keep_dims` expected? This fixes the skeleton family and output shape." |
| shape(s) | "What are the concrete shapes/variants? Structural choices (tile family, split axis) depend on the shape class even though final sizes are deferred." |
| tolerance / reference | "What reference formula and tolerances define correctness? Design must fix them; 'close enough' is not a contract." |
| accumulation precision | "Is fp32 accumulation mandatory, or is it an implementation choice as long as output meets the reference?" |
| output overwrite/accumulate | "Does the output get overwritten, or accumulated into (`C += …`)? This changes whether `zeros_acc` init is legal." |
| forbidden mechanisms | "Are any mechanisms explicitly disallowed (e.g. Cube-side bias fold, host precompute)? The route check enforces these, so they must be in the contract." |

Rules for asking:

- **One question, one fact.** Ask the smallest question that unblocks the brief;
  batch only genuinely independent facts in one message.
- **Never guess and record it later.** A guessed dtype or epilogue placement produces
  a brief that is wrong in a way Implement cannot detect.
- **Quote the spec.** Cite the exact sentence that is ambiguous so the user answers
  the real gap, not a re-derivation.
- **Questions suspend output.** Do not emit a partial brief while a question is
  open; the brief is emitted once, complete.

## Route-recheck on structural backtrack

A structural defect report from Verify can invalidate the route itself (e.g. a fusion
the platform cannot express, or a `cube` brief that actually required Vector work).
When that happens, re-run this decision procedure against the *defect evidence + the
original spec*, revise whichever test was misjudged, and reissue the full brief — per
SKILL.md's structural-backtrack rule.
