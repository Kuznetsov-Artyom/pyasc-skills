# Design reference: route selection, brief schema, targeted questions

This reference holds the three things SKILL.md delegates here:

1. the **route decision procedure** (step 2),
2. the **design-brief schema** the brief must fill (step 4),
3. the **targeted-question menu** for missing semantics (step 1).

API facts below are provenance from the validated installed package (`pyasc-dev`
1.1.1.dev20261123081345). They are not a runtime version-equality gate. Re-derive the
facts from the installed package when it changes; preflight records the actual
versions and capabilities. Where a fact has a single authoritative source, that source
is named inline.

## Route enum

The brief emits exactly one route:

| Route | Meaning | Implement loads | Verify loads |
|---|---|---|---|
| `vector` | All compute on the Vector (AIV) pipe through UB — elementwise/activation/cast chains, reductions, normalization, index/data movement. No `matmul`/`matmul_acc` anywhere. | `vector.md` | `vector.md` + `simulator.md` |
| `cube` | Dominant compute is a Cube matmul (`matmul`/`matmul_acc`, L0A×L0B→L0C). No Vector elementwise work on the result — at most a `.to(dtype)` cast or an L0C-fusible op (`asctile.relu`, the only elementwise op with `support_loc=(UB, L0C)` in the installed package) executed on the Cube side, then a direct `L0C→GM copy_out`. | `cube.md` | `cube.md` + `simulator.md` |
| `cube_vector` | Cube matmul **plus** spec-mandated Vector-unit work on the result (epilogue, multi-phase, carried state, multi-tile combine) inside the same program — an L0C→UB hop with a named transfer mechanism (`DistribMode` split/join, a `cv_strategy` region, or `distrib=None` unsplit); `cv_ratio=2` + a split is the recommended (not mandatory) structure to engage both AIVs. | `cube.md` + `cube-vector.md` | `cube.md` + `cube-vector.md` + `simulator.md` |

`cube_vector` never loads `vector.md`: the mix reference carries only the Vector delta
the route needs.

### Validated production templates

Only these exact templates are selectable for a production brief in the current MVP:

- `vector_add_fp16` — `out[i] = a[i] + b[i]`, fp16 contiguous 1-D;
- `gemm_fp16` — row-major fp16 `C = A @ B`, canonical K-split direct output;
- `gemm_bias_relu_vector_fp16` — fp16 `C = relu(A @ B + bias[j])`, Vector epilogue
  in UB with manual `SplitByM` at `cv_ratio=2`.

Other route vocabulary, structures and mechanisms are **candidate practice**, not
production support. Real developer experience is preserved as such: it is not called
false, but it must be applied and verified before promotion. A request outside the
three templates produces `candidate_recipe` when such real experience exists, or
`unsupported_in_current_iteration` when no applicable candidate exists. Neither is
`infeasible` unless the installed platform/API actually contradicts the request.

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
- If the spec *explicitly* forbids the Cube unit for a contraction, preserve that
  constraint rather than turning it into a hardware question. Apply the normal support
  boundary: use a validated recipe if one applies; otherwise use `candidate_recipe` when
  real experience exists, or `unsupported_in_current_iteration`. Do not use `infeasible`
  without a proven installed-API/platform contradiction.

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
  Whether a spec gets that structure depends on what it mandates: explicit
  Cube-side allowance → `cube` if a validated Cube recipe exists; explicit Vector
  requirement → `cube_vector`; otherwise apply the deterministic template default
  or candidate/unsupported status. No unit-placement question is sent to the user.
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

If Test 1 passed and no Vector work is required → **`cube`**. If the spec contains a
fused epilogue but does not state its unit, Design uses the deterministic default:
`cube_vector` for the published `gemm_bias_relu_vector_fp16` template. Other epilogues
are candidate/unsupported according to the recipe registry; they are not a question
about hardware for the user.

Boundary cases:

- **Cast-only is not an epilogue.** `C = (A @ B).to(fp16)` — a `.to(dtype)` on the L0C
  accumulator then `copy_out` direct to GM — is `cube`. The cast can run on the Cube
  core; no AIV work is needed.
- **Lone `relu`:** `relu` is L0C-fusible, but a lone-relu GEMM is not one of the three
  published templates. Do not silently turn it into a supported `cube` recipe. Treat
  real experience as candidate practice; otherwise report
  `unsupported_in_current_iteration`. An explicit Vector requirement is accepted by the
  published `gemm_bias_relu_vector_fp16` template only when the full math matches
  `relu(A @ B + bias)`.
- **Cube-side bias is not a Vector epilogue.** `asctile.matmul` accepts a `bias`
  operand (1-D, `BT` location) and `zeros_acc` accepts `bias=` initialization; both
  keep the add on the Cube path. That is platform knowledge, not a published template.
  If the user explicitly requires Vector bias work and the math matches the published
  template, use `cube_vector`; otherwise keep the candidate/unsupported distinction.
- **No epilogue at all → `cube`; unstated epilogue unit → deterministic default.** A
  GEMM with no fused elementwise math routes `cube`. For a spec matching
  `relu(A @ B + bias)` without a unit restriction, Design chooses the published
  `cube_vector` template and records the decision as `agent_decision`. No unit-placement
  question is sent to the user.
- **Parallel-but-independent Vector work.** This is platform/structure knowledge, not
  one of the published templates. Preserve it as candidate practice rather than
  promising an implementation path that has not been validated.

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
`copy_out` through UB. That vocabulary describes platform capability, not current MVP
support: only `vector_add_fp16` is a validated production template. Everything else
follows the candidate/unsupported status rules above.

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

- task_id: <id>
- brief_revision: <monotonic revision>
- protocol_revision: asctile-model-v1
- status: complete | candidate_recipe
- route: vector | cube | cube_vector
- template: vector_add_fp16 | gemm_fp16 | gemm_bias_relu_vector_fp16 | <candidate-specific>
- artifact_file: vector_add.py | gemm.py | gemm_bias_relu.py | <candidate-specific>.py
- kernel_symbol: vector_add | gemm | gemm_bias_relu | <candidate-specific>
- launch_symbol: launch
- main_symbol: main
- next_stage: implement
- supersedes: <previous brief revision or none>
- local_strikes: 0
- structural_reissues: 0
- uncertain_reductions: 0
- provenance: user | default | agent_decision | verified

## 1. Semantic contract
- math: <exact formula, e.g. C[i,j] = relu( Σ_k A[i,k]·B[k,j] + bias[j] )>
- tensors: <name → shape / dtype / layout / broadcast semantics for every input
  and output; note which dims are compile-time vs runtime>
- variants/shapes: <per-variant shape table if the spec fixes several>
- boundary rule: <which logical extents must be respected; e.g. "no GM access
  outside [0,N) of any tensor">

## 2. Route
Production only (`status=complete`):
- route: vector | cube | cube_vector
- template: vector_add_fp16 | gemm_fp16 | gemm_bias_relu_vector_fp16
- artifact_file: vector_add.py | gemm.py | gemm_bias_relu.py
- kernel_symbol: vector_add | gemm | gemm_bias_relu
- launch_symbol: launch
- main_symbol: main
- evidence: <the spec clauses / decision-procedure tests that forced this route>

Evaluation-only (`status=candidate_recipe`):
- route: a route selected by the same decision procedure if applicable; otherwise
  `unsupported_in_current_iteration` — never an unknown route
- template: <candidate-specific name, not one of the three public templates>
- artifact_file: <candidate-specific>.py
- kernel_symbol: <candidate-specific>
- launch_symbol: launch
- main_symbol: main

## 3. Algorithm family
<The structural skeleton, not the code. For `status=complete`, pick exactly one from
the validated allowlist. For `status=candidate_recipe`, use the candidate-specific
structure named by the candidate brief and keep it evaluation-only.>
- `vector_add_fp16`: flat grid-stride elementwise addition with exact boundary extents.
- `gemm_fp16`: canonical K-split accumulate (`zeros_acc` + `matmul_acc` K loop),
  direct L0C→GM output and final fp16 cast.
- `gemm_bias_relu_vector_fp16`: GEMM + Vector UB epilogue, manual `SplitByM`
  on L0C→UB copies at `cv_ratio=2`, per-sub-block clamped stores.

Other patterns (row reduce/normalize, per-element index-driven, per-row accumulate,
M/N-split, StreamK, multi-phase, carried state, `SplitByN`, `FullVec*`,
`cv_strategy`, unsplit) are retained only as **candidate practice**. They are not
production-selectable until a later validation task proves them through the same
Implement→Verify pipeline.

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
- L0C→UB transfer mechanism and split axis: **only for `cube_vector`**. For the published
  MVP use manual `distrib=SplitByM` at `cv_ratio=2`; for `vector`/`cube` write
  `not_applicable`, do not invent a distribution mechanism. Other mechanisms are
  candidate practice only.
- grid source: derived from tile counts at launch time (never a baked constant)

## 6. Tile family
<Tile parameter NAMES and their constraints — never final values.>
- vector: e.g. `tile` (ConstExpr; 1-D copies have no alignment requirement, an aligned
  tile is only a heuristic), `uf`, live-tile count;
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
- Note the platform constraint: `real_shape` is legal only on **UB/Auto** loads.
  It is illegal on BT as well as L1/L0A/L0B: the Python docstring's omission of BT is
  misleading, while the installed IR verifier enforces UB/Auto (`LoadOp::verify`).

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

Questions are for missing **user semantics**, not hardware implementation. Ask only when
the answer cannot be inferred from the request or an explicit MVP default. Use plain
language, one fact per question, and do not emit a partial brief while a question is open.

| Missing / ambiguous fact | Ask |
|---|---|
| dtype of a required tensor | "Which dtype should `<tensor>` use? This MVP currently publishes fp16 templates; dtype is needed for feasibility and correctness." |
| concrete shape(s) | "What are the concrete shapes/variants? The tile family and boundary checks depend on them." |
| mathematical semantics | "Which exact operation should be computed? The current request leaves `<operator>` ambiguous." |
| layout / transpose | "Should tensors be contiguous row-major? If a transpose is intended, describe it explicitly; this MVP does not silently assume one." |
| broadcast semantics | "How should `<tensor>` broadcast — per-row, per-column, or scalar? This affects the reference and output shape." |
| reduction axis | "Which axis is reduced, and should the reduced dimension be kept? This changes the output shape." |
| user-specific non-default correctness requirement, if and only if it is ambiguous | "Do you have a specific non-default correctness contract that you require? If not, the labelled MVP default is used and recorded, with no tolerance question." |
| output overwrite/accumulate | "Should the output be overwritten, or accumulated into (`C += …`)? This changes accumulator initialization." |

Rules:

- **One question, one fact.** Batch only genuinely independent missing facts.
- **Never guess and record it later.** A guessed dtype or semantic contract produces a
  brief that Implement cannot detect as wrong.
- **Quote the ambiguous sentence** when one exists; if a fact is simply absent, name the
  missing field instead of inventing a quotation.
- **Questions suspend output.** Emit `DESIGN_REPORT.md` with `needs_user_input`; no
  partial brief is produced.
- **Do not ask about Cube vs Vector, unit placement, tile sizes, `cv_ratio`,
  `DistribMode`, `cv_strategy`, or other hardware details.** These belong to Design.

## Candidate validation path

An explicit user request to validate a candidate recipe creates a complete,
revisioned **evaluation-only candidate brief** with `status=candidate_recipe`. It has
math, shapes/dtypes, reference, variants, artifact identity and candidate provenance.
It follows the same Implement → Verify ownership and defect rules as a production brief.

Before authoritative Verify/acceptance passes, the candidate is not supported. Passing
produces `promotion_evidence` for a future extension and does not silently add a fourth
public template to this MVP. Failure follows the same defect/terminal lifecycle.

## Route-recheck on structural backtrack

A structural defect report from Verify can invalidate the route itself (e.g. a fusion
the platform cannot express, or a `cube` brief that actually required Vector work).
When that happens, re-run this decision procedure against the *defect evidence + the
original spec*, revise whichever test was misjudged, and reissue the full brief — per
SKILL.md's structural-backtrack rule.
