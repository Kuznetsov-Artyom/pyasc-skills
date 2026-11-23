---
name: asctile-implement-operator
description: Use when a complete AscTile design brief exists and a kernel plus full plain-Python host harness must be produced — choosing exact tiling/grid/options, implementing the canonical standalone artifact, and iterating on early compile/smoke defects before Verify judges it. Middle stage of the Design → Implement → Verify pipeline. Not for incomplete briefs, final correctness/performance verdicts, or structural redesign.
compatibility: Requires the installed pyasc package (`from asc.experimental import asctile`), CANN toolkit environment, and the Ascend950PR_9599 simulator for compile/smoke runs. Plain Python only — no pytest, no conftest.py, no pyasc checkout.
---

# AscTile: Implement Operator

## Overview

Implement consumes one **complete, feasible design brief** and owns the whole candidate:
kernel source **and** the full host-side standalone harness, including exact tiling,
grid, compile/launch options, boundary handling, checks, and configuration reporting.
It hands Verify a compile/smoke-tested candidate, not a final correctness or performance
verdict. Verify owns acceptance and Deliver; Design owns structural choices.

## Entry Preconditions

Proceed only when the original task/protocol and a brief with **concrete sections 1–10**
of the Design schema are available:

- exact semantics, every tensor's shape/dtype/layout/broadcast rule, variants, and GM
  boundary rule;
- route, algorithm family, per-tensor dataflow/residency, parallelism/grid source, and
  tile-family constraints, including which values are `ConstExpr` or runtime scalars;
- boundary/tail strategy family, explicit feasibility verdict, exact correctness
  reference/comparison/tolerances, and baseline availability.

Only section 11 items (tile values, unroll values, launch options, exact offsets, cache
policy) may be `deferred-to-implement`. A missing, hollow, contradictory, or infeasible
structural field goes back to `asctile-design-operator`; do not guess or emit a partial
candidate. The standalone installed-package environment must also be available.

## Route → References

After validating the brief, load **exactly** the row selected by its `route`. Read the
listed files directly, one level below this skill; do not browse the other route files.

| Route | Load | Do not load |
|---|---|---|
| `vector` | [references/vector.md](references/vector.md) | `cube.md`, `cube-vector.md` |
| `cube` | [references/cube.md](references/cube.md) | `vector.md`, `cube-vector.md` |
| `cube_vector` | [references/cube.md](references/cube.md) + [references/cube-vector.md](references/cube-vector.md) | `vector.md` |

An unknown route is an incomplete-brief defect, not permission to load every reference.

## Procedure

Work in this order.

1. **Choose and record the exact configuration.** Stay inside the brief's structure and
   tile constraints. On the host, derive per-variant tile values, loop/unroll counts,
   positive launch bounds/grid, and options from the actual shapes, dtypes, queried
   platform resources, and the selected boundary strategy. Pass local tile shapes and
   unroll values as required by the installed API; keep runtime extents runtime. Record
   every choice for `CONFIG`/`RESULT.md`. Never bake a queried core count into kernel
   indexing: use `asctile.block_num()` / `asctile.block_idx()` and prove logical coverage
   without overlap, gaps, empty out-of-range tiles, or GM accesses outside true extents.
2. **Write kernel and canonical host harness together.** Implement the brief's dataflow
   and exact reference contract. The submission contains only `entry.py` plus optional
   `kernel.py` and `host.py`, and `RESULT.md`; `entry.py` is the single entry point. It
   may embed all code. It reads no hidden inputs, checkout paths, network, or test
   fixtures and imports the installed package with `from asc.experimental import asctile`.
3. **Compile early, then smoke.** Use the proven block below with a fresh cache. Run
   compile-only and a runtime smoke for every selected variant before handoff. Fix local
   errors by changing one hypothesis at a time and rerunning the smallest failing case.
   Smoke diagnostics are preserved, but are not Verify's correctness/tick verdict.
4. **Hand off or backtrack.** Provide the artifact, exact commands, cache state, complete
   per-variant configuration, memory/smoke output, and any unresolved defect. Do not tune
   against ticks or declare success from the candidate's own `PASS` lines.

### Canonical harness contract

Implement, not Verify, owns all of these executable details:

- Behavior is selected only by `PYASC_DRY_RUN`, `PYASC_CACHE_DIR`, and `MVP_SHAPES`.
  `primary`/`tail` select one variant; `all` or unset runs primary then tail; any other
  value fails. Apply the complete compile → correctness → ticks sequence per variant.
- Select `asctile.Backend.Model` / `asctile.Platform.Ascend950PR_9599` once per process.
  Print the protocol's complete `ENVIRONMENT` and stop on an environment deviation.
- Generate each variant's CPU inputs exactly as the task states (seed once per variant,
  operands in declared order). Use the brief's exact fp32-computed/reference-cast formula
  and tolerances, never a baseline's or a self-invented reference.
- Allocate output inside 32-fp16-element (64-byte) guards on both sides, fill fixed
  sentinels, and snapshot the whole buffer before launch. Compare guards bit-for-bit.
- Under `PYASC_DRY_RUN=1`, exercise the real compile/launch path, check compiler memory
  consumption with `Launcher.check_memory_overflow` plus the six platform capacities,
  and prove the entire output allocation unchanged. Print `MEMORY_CHECK` and
  `COMPILE_ONLY` only after those checks pass.
- At runtime, synchronize before and after one warmup launch; use that output for
  `torch.testing.assert_close` and guard checks. Print `WARMUP` and `CORRECTNESS` with
  the exact reference string, tolerances, max absolute error, variant/shape, and per-side
  guard bytes. Then collect the protocol's five synchronized samples and print raw
  `TICK_SAMPLE` records and `TICK_SUMMARY`; correctness and measured launches must use
  identical tensors, kernel, grid, and options.
- Print flushed, single-line JSON via `json.dumps(..., sort_keys=True)` for
  `ENVIRONMENT`, `MEMORY_CHECK`, `CORRECTNESS`, `TICK_SUMMARY`, and `CONFIG`, using the
  protocol's literal forms for the
  other lines. `CONFIG` includes task/variant/shape, grid/launch bounds, actual tile
  family and loop/unroll order, tail mechanism, kernel/compile options, `ConstExpr` and
  runtime-scalar sets, seed/draw order, cache directory and cache-warm state, and every
  other implementation choice.
- Any failure exits nonzero with a reason. Never swallow exceptions, silently retry,
  emit `PASS` after a failure, or report ticks for an incorrect variant. `RESULT.md`
  starts as Implement's configuration/smoke handoff; Verify owns its final verdict/report.

## Proven standalone compile/smoke

This is the Task 1 environment, not a pytest substitute. Replace only `candidate` with
the artifact directory; run from the frozen execution root.

```bash
cd /home/artyomka/repos/work/test_skills/mvp
set -e
source /home/artyomka/repos/work/test_skills/.venv/bin/activate
source /opt/ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
candidate=/path/to/candidate
cache=$(mktemp -d /tmp/opencode/asctile-implement.XXXXXX)
env -u PYTHONPATH PYASC_DRY_RUN=1 PYASC_CACHE_DIR="$cache" MVP_SHAPES=all python -I "$candidate/entry.py"
env -u PYTHONPATH -u PYASC_DRY_RUN PYASC_CACHE_DIR="$cache" MVP_SHAPES=all python -I "$candidate/entry.py"
```

Expected environment: Python 3.13.5, `pyasc==1.1.1+g63ed594e.d20261122`, torch
2.15.0+cpu, CANN 9.2.2, runtime/SoC `Ascend950PR_9599`, compilation arch `c310`, and
`rt.is_model() == True`. Dry-run CANN model setup/teardown messages are expected; they
do not prove a launch. The unchanged-output check does.

## Backtracking, defect report, and no-progress stop

- **Local defect:** wrong tile/grid/offset/option, compile error, UB overflow, harness
  check, tail predication, or guard violation that can be fixed without changing the
  brief. Implement may fix it and rerun the affected compile/smoke case.
- **Structural evidence:** no legal configuration can satisfy the brief's route,
  residency, fusion, or boundary structure. Stop editing and return evidence to Verify;
  Design must reissue the **complete** brief. Implement then synchronizes both kernel and
  canonical harness to that reissued brief. Verify never makes those edits.
- **Invalid verification:** wrong oracle/environment/procedure produces no implementation
  change. Wait for a valid Verify result.

Every defect report contains: task/variant and brief revision; classification
(`local`, `structural`, or `uncertain`); exact command, environment and cache state;
minimal reproducer; first failing error and preserved log; actual configuration; changes
already tried with their outcomes; and the proposed next owner. `uncertain` goes to
Verify for triage, not directly to speculative redesign.

**No-progress stop:** after two consecutive hypothesis-driven local changes leave the
same minimal failure and error signature unchanged, stop and send the defect report to
Verify. Also stop before any change that would alter the frozen contract or brief.
Do not convert thrashing into silent retuning.

## Output

A self-contained candidate directory (`entry.py` [+ `kernel.py`/`host.py`], draft
`RESULT.md`) and a handoff note with configuration and evidence above. Final correctness,
outcome classification, reproducibility, ticks, and Deliver belong to Verify.
