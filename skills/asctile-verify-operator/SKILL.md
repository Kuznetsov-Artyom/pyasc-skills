---
name: asctile-verify-operator
description: Use when a complete AscTile design brief and candidate kernel plus standalone harness exist and must be judged — validating the verification oracle, running the protocol's compile/runtime/correctness/tick sequence, applying the route checklist, classifying outcomes and defects, and owning the final Deliver report. Final stage of the Design → Implement → Verify pipeline. Not for writing or patching kernels or harnesses; invalid verification must never trigger implementation changes.
compatibility: Requires the installed pyasc package (`from asc.experimental import asctile`), CANN toolkit environment, and the Ascend950PR_9599 simulator. Plain Python only — no pytest, no conftest.py, no pyasc checkout.
---

# AscTile: Verify Operator

## Overview

Verify consumes a complete design brief and an unmodified candidate, independently checks
correctness against the **brief's reference**, executes the frozen protocol, applies the
route checklist, classifies each variant, and owns Deliver. Verify never edits kernel or
harness code. Local defects return to Implement; structural defects return to Design for
a complete brief reissue, after which Implement synchronizes kernel and harness.

## Entry Preconditions

Require all of the following before judging the candidate:

- the original task/frozen protocol and a complete, feasible brief with concrete Design
  sections 1–10, especially exact variants/tensors/boundaries and correctness formula,
  comparison method, and tolerances;
- a self-contained candidate claiming that brief: `entry.py` [+ `kernel.py`/`host.py`]
  and `RESULT.md`, with no runtime dependencies beyond the installed environment;
- the proven installed-package/CANN/simulator environment and isolated execution mode.

An absent/incomplete brief returns to Design; an absent candidate returns to Implement.
Only Implement-owned numerical/open items may remain deferred. Do not fill gaps by
inferring intent from candidate code.

## Route → References

Load **exactly** the brief's route row plus the shared simulator reference. These are
one-level direct reads; do not inspect another route's checklist.

| Route | Load | Do not load |
|---|---|---|
| `vector` | [references/vector.md](references/vector.md) + [references/simulator.md](references/simulator.md) | `cube.md`, `cube-vector.md` |
| `cube` | [references/cube.md](references/cube.md) + [references/simulator.md](references/simulator.md) | `vector.md`, `cube-vector.md` |
| `cube_vector` | [references/cube.md](references/cube.md) + [references/cube-vector.md](references/cube-vector.md) + [references/simulator.md](references/simulator.md) | `vector.md` |

## Procedure

Perform these gates **in order**. A failed validity gate cannot generate an
implementation backtrack.

### 1. Verification validity precedes backtracking

Establish an independent oracle from the original task and Design brief, not from the
candidate's `RESULT.md`, printed `PASS`, or baseline behavior. Check:

- exact math/reference string, input dtypes/shapes/layouts, CPU seed and operand draw
  order, fp32 computation/final cast, comparison method, and tolerances;
- true GM boundary rules and every frozen variant, including its tail case;
- verifier environment, commands, cache state, and protocol sequence against
  `simulator.md`; complete logs with nonzero failures preserved;
- candidate artifact layout and whether its executable harness actually implements that
  same oracle and protocol rather than merely printing the required labels.

If **Verify's** reference, environment, procedure, or evidence is wrong/incomplete, mark
only the verification attempt invalid, discard its verdict and ticks, and correct/re-run
the verification. Do not send a defect or change implementation. If the brief contradicts
the original frozen task, return that contract defect to Design without touching the
candidate. Once the independent verification is valid, a candidate harness that violates
the artifact/oracle/protocol contract is `invalid_submission`; return its host defect to
Implement without editing it.

### 2. Audit structure and configuration

Apply only the loaded route checklist. Read kernel and host source, compare actual launch
and loop structure with `CONFIG`/`RESULT.md`, and prove coverage, legal residency, memory
capacity, tail extents, and numeric semantics. A numerically passing output does not erase
an out-of-bounds access or route violation. Record mismatches before running performance
comparisons.

### 3. Run the protocol per variant

Follow `simulator.md` exactly:

1. In a separate process with a fresh cache, run `PYASC_DRY_RUN=1`; require real
   compilation, the compiled specialization's six-bank memory check, and unchanged full
   output storage. Compilation or memory failure is `compile_failed` when the submission
   itself is otherwise valid.
2. In a separate runtime process, require explicit Model/9599 platform and environment
   facts, exact input generation, one synchronized warmup, correctness against the Design
   reference at its tolerances, and bit-identical 64-byte output guards on both sides.
   A launched numerical/guard failure is `incorrect`; a crash, hang, or unavailable tick
   after successful compilation is `run_failed`.
3. Only after correctness passes, collect five synchronized tick samples with identical
   tensors/kernel/grid/options, positive deltas, and raw start/end/delta plus summary.
   Repeat the complete runtime pass in a second independent process. Both must be correct,
   with five measurements and medians within ±1 tick. If a valid environment/procedure
   does not yield the required reproducible tick result, classify that measurement
   failure `run_failed`, retain the reproducibility concern, and never average deviations
   away.
4. Preserve each invocation and failure. Do not silently retry, edit tolerances, repair
   source, or report ticks for an incorrect variant. A protocol/artifact violation is
   `invalid_submission`; missing proven environment is an environment deviation, not a
   candidate performance result.

### 4. Classify outcome and backtrack

Assign exactly one protocol outcome to each variant: `correct`, `incorrect`,
`compile_failed`, `run_failed`, or `invalid_submission`. `correct` requires every check
and reproducibility requirement, not just a printed `CORRECTNESS PASS`.

For a validly observed failure, classify the defect:

- **local → Implement:** a concrete kernel/host/configuration error fixable without
  changing the brief. Return a reproducible report containing task/variant/brief revision,
  exact command/environment/cache, minimal case, first error and log, actual config,
  expected-vs-observed evidence, and attempted fixes if supplied.
- **structural → Design:** evidence that no legal implementation can satisfy the brief's
  route/dataflow/residency/fusion/boundary structure. Name the failed structural field
  and installed-package/platform evidence. Design reissues the complete brief and
  re-derives dependents; Implement then synchronizes both kernel and canonical harness
  before a fresh Verify run.
- **uncertain → Verify triage:** reduce the reproducer and check verification validity
  again before choosing an owner. Never guess structural from a tile overflow or local
  from a wrong route.

No defect classification authorizes Verify to patch the candidate. An invalid verification
never enters this backtrack process.

## Deliver

Verify owns the final `RESULT.md`/Deliver report and preserved run evidence; this report
is the only candidate artifact Verify may update. Include:

- task id, brief/protocol revision, route, candidate and exact commands/log locations;
- per-variant outcome and validity/route-check findings;
- environment versions/platform/arch and cache directory/cache-warm state;
- actual per-variant configuration and memory consumption/capacities;
- reference/tolerances, correctness error and guard results;
- for correct variants, both runs' raw start/end/delta samples, five raw deltas,
  median/min/max, warmup/measurement/synchronization facts, and median difference;
- concerns, whole-launch measurement limitation, environment deviations, and any local or
  structural defect report/backtrack. Preserve failed outcomes rather than hiding them.

Simulator ticks are not hardware latency and the interval includes the complete
synchronized launcher path, not proven compute-only cycles. Never convert ticks to time
or compute speedup from a failed/incomparable baseline. Verify delivers evidence and
classification; it does not modify implementation to obtain a better report.

## Hard Boundaries

- No kernel or harness edits, no pytest/conftest/checkout dependency.
- Correctness is judged against Design's exact reference, not candidate self-report.
- Verification validity comes before any backtrack; invalid verification changes no code.
- Structural backtrack means full Design brief reissue plus Implement-owned kernel and
  harness synchronization, followed by fresh verification.
