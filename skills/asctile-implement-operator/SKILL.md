---
name: asctile-implement-operator
description: Use when a complete AscTile design brief exists and one algorithm-specific Python file must be produced — choosing exact tiling/grid/options, implementing kernel plus operator-facing launch wrapper and protocol-facing main, and iterating on early compile/smoke defects before Verify judges it. Middle stage of the Design → Implement → Verify pipeline. Not for incomplete briefs, final correctness/performance verdicts, or structural redesign.
compatibility: Requires the installed pyasc package (`from asc.experimental import asctile`), CANN toolkit environment, Torch host backend and the Ascend950PR_9599 simulator for compile/smoke runs. Plain Python only — no pytest, no conftest.py, no pyasc checkout.
---

# AscTile: Implement Operator

## Overview

Implement consumes one **complete, feasible design brief** and owns the whole candidate:
one algorithm-specific Python file containing kernel, `launch` wrapper, `main` protocol
orchestrator, exact tiling, grid, compile/launch options, boundary handling, checks and
configuration reporting.

It hands Verify a compile/smoke-tested candidate, not a final correctness or performance
verdict. Verify owns acceptance and Deliver; Design owns structural choices.

## Entry Preconditions

Proceed only when the original task and a brief with concrete sections 1–10 of the Design
schema are available:

- exact semantics, every tensor's shape/dtype/layout/broadcast rule, variants and GM
  boundary rule;
- route, exact template, algorithm family, per-tensor dataflow/residency, parallelism/grid
  source and tile-family constraints, including which values are `ConstExpr` or runtime
  scalars;
- boundary/tail strategy family, explicit feasibility verdict, exact correctness
  reference/comparison/tolerances and optional baseline status;
- artifact identity: `artifact_file`, `kernel_symbol`, `launch_symbol=launch`,
  `main_symbol=main`.

Only section 11 items (tile values, unroll values, launch options, exact offsets, cache
policy) may be `deferred-to-implement`. A missing, hollow, contradictory or infeasible
structural field goes back to `asctile-design-operator`; do not guess or emit a partial
candidate. Before returning it, write `DESIGN_REPORT.md` with `status=needs_design`,
the failed field/reason, current revisions and any candidate hash so a fresh agent selects
Design for reissue instead of bouncing to Implement. The user's currently configured
installed-package environment must be available.

An evaluation-only candidate brief with `status=candidate_recipe` is accepted only when
the user explicitly requested validation. It follows the same Implement → Verify rules,
but does not become supported before acceptance.

Do not activate Implement over a current matching `stopped` RESULT. A terminal task can be
reopened only by the explicit user-driven path with a new brief revision. An
`environment_failed` RESULT is the exception: Verify owns retry/preflight; after a
successful retry, Verify publishes a revision-and-hash-matching successor RESULT before
archiving the environment failure. Implement may resume only from that successor if its
artifact and revision match. A terminal-looking RESULT with missing identity/counters or a
candidate hash mismatch is never reopened by Implement; Verify must first repair or
quarantine it using a persistent `QUARANTINE.md` marker with `status=quarantined`.

## Route → References

After validating the brief, load the route reference **plus** `references/harness.md`.
Read the listed files directly, one level below this skill; do not browse other route
files.

| Route | Load | Do not load |
|---|---|---|
| `vector` | [references/vector.md](references/vector.md) + [references/harness.md](references/harness.md) | `cube.md`, `cube-vector.md` |
| `cube` | [references/cube.md](references/cube.md) + [references/harness.md](references/harness.md) | `vector.md`, `cube-vector.md` |
| `cube_vector` | [references/cube.md](references/cube.md) + [references/cube-vector.md](references/cube-vector.md) + [references/harness.md](references/harness.md) | `vector.md` |

An unknown route is an incomplete-brief defect, not permission to load every reference.

## Procedure

Work in this order.

1. **Choose and record the exact configuration.** Stay inside the brief's structure and
   tile constraints. Derive per-variant tile values, loop/unroll counts, positive launch
   bounds/grid and options from actual shapes, dtypes, queried platform resources and the
   selected boundary strategy. Keep runtime extents runtime. Record every choice for
   `CONFIG`/`RESULT.md`. Never bake a queried core count into kernel indexing: use
   `asctile.block_num()` / `asctile.block_idx()` and prove logical coverage without
   overlap, gaps, empty out-of-range tiles or GM accesses outside true extents.
2. **Write the one algorithm file.** Use the exact `artifact_file` and `kernel_symbol`
   from the brief. The file has this order:
   `@asctile.jit` kernel → `def launch(...)` → `def main()` → standard main guard.
   `launch` accepts runtime tensors/scalars/options, prepares output, selects tiling/grid/
   core count and compile/runtime parameters, performs every JIT launch through
   `kernel_symbol[grid](...)` and returns the result. `main` orchestrates environment,
   variants, input/reference generation, guards, dry-run/runtime/ticks,
   `CONFIG`/`RESULT.md` and calls `launch`. No direct JIT launch outside `launch`.
   The file is self-contained: no `entry.py`, `kernel.py`, `host.py` or sibling modules.
3. **Compile early, then smoke.** Use the generic command pattern from `harness.md` with a
   fresh cache per variant. Run compile-only and runtime smoke for every selected variant
   before handoff. Fix local errors by changing one hypothesis at a time and rerunning the
   smallest failing case. Smoke diagnostics are preserved, but are not Verify's
   correctness/tick verdict.
4. **Hand off or backtrack.** Provide the artifact, exact commands, cache state, complete
   per-variant configuration, memory/smoke output, candidate SHA256 and any unresolved
   defect. Do not tune against ticks or declare success from the candidate's own `PASS`
   lines.

## Canonical harness contract

The complete executable contract is in [references/harness.md](references/harness.md).
Implement, not Verify, owns all executable details: environment preflight, deterministic
Torch inputs/reference, 64-byte guards, dry-run no-launch proof, five synchronized tick
samples, two independent runtime processes, unique run manifests, captured outputs,
nonzero failure exits and truthful `CONFIG`.

The canonical production file is exactly one of `vector_add.py`, `gemm.py`, or
`gemm_bias_relu.py`, with kernel → launch → main structure. An evaluation-only candidate
brief instead uses its candidate-specific `artifact_file`/`kernel_symbol` with the same
one-file structure. Do not create sibling modules or a second entry point.

## Environment handling

Use the user's actual configured environment. Do not require a fresh or isolated
environment. Do not install packages or modify CANN yourself.

Preflight and revalidation are defined in `harness.md`. On a failed capability, record
and report the exact missing dependency or wrong configuration, publish
`environment_failed` without a result, and wait for user remediation. That RESULT blocks
ordinary defect/RESULT routing. Only Verify reruns preflight after an explicit retry; on
success Verify **first publishes a revision-and-hash-matching successor RESULT**, then
archives the failed RESULT. Implement may resume only from that successor when its
artifact/revision match. If the successor cannot be published or preflight fails again, the
failed RESULT remains current, no result/ticks are emitted, and old defects do not route.
Implement does not perform an unowned retry. Re-evaluate environment identity and baseline
comparability after any drift.

## Backtracking, defect report and no-progress stop

- **Local defect:** wrong tile/grid/offset/option, compile error, UB overflow, harness
  check, tail predication or guard violation that can be fixed without changing the brief.
  Implement may fix it and rerun the affected compile/smoke case.
- **Structural evidence:** no legal configuration can satisfy the brief's route,
  residency, fusion or boundary structure. Stop editing and return evidence to Verify for
  classification; only after Verify classifies structural does Design reissue the
  complete brief. Implement then synchronizes both kernel and canonical harness to that
  reissued brief. Verify never makes those edits.
- **Invalid verification:** wrong oracle/environment/procedure produces no implementation
  change. Wait for a valid Verify result.

Every defect report contains: `defect_id`, `status=active|resolved|superseded`,
task/variant, brief/protocol revisions, candidate SHA256, the counters `local_strikes`,
`structural_reissues`, `uncertain_reductions`, exact command/environment/cache,
minimal reproducer, first failing error and preserved log, actual configuration, changes
already tried with outcomes, and the proposed next owner. `uncertain` goes to Verify for
triage, not directly to speculative redesign.

A local defect becomes `resolved` only atomically with a new `ready_for_verify` RESULT for
the latest revisions and current candidate hash. Archive the old RESULT with the defect.
If a resolved defect has no such matching successor, the next stage is Implement to
produce one.

**No-progress stop:** persist `local_strikes` in the defect report/RESULT. After two
consecutive hypothesis-driven local changes leave the same minimal failure and error
signature unchanged, stop and send the defect report to Verify. Do not write a new
`ready_for_verify` result after that stop. Also stop before any change that would alter
the frozen contract or brief. Do not convert thrashing into silent retuning. Exhausting
the stop produces `stopped` through Verify, superseding and archiving all active defects.

## Output

A self-contained algorithm file named by the brief, plus draft `RESULT.md` only when no
terminal cap has been exhausted and no current `stopped` RESULT exists:

- `status=ready_for_verify`, `next_stage=verify`;
- task id, brief/protocol revisions and `candidate_sha256`;
- the current counters `local_strikes`, `structural_reissues`, `uncertain_reductions`;
- commands, actual configuration and evidence paths.

Final correctness, outcome classification, reproducibility, ticks and Deliver belong to
Verify. Implement does not edit a final Verify report; a new local iteration starts a new
draft with the current identity.
