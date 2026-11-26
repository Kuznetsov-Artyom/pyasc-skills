---
name: asctile-verify-operator
description: Use when a complete AscTile design brief and candidate algorithm file exist and must be judged — validating the verification oracle, running the protocol's compile/runtime/correctness/tick sequence, applying the route checklist, classifying outcomes and defects, and owning the final Deliver report. Final stage of the Design → Implement → Verify pipeline. Not for writing or patching kernels or harnesses; invalid verification must never trigger implementation changes.
compatibility: Requires the installed pyasc package (`from asc.experimental import asctile`), CANN toolkit environment, Torch host backend and the Ascend950PR_9599 simulator. Plain Python only — no pytest, no conftest.py, no pyasc checkout.
---

# AscTile: Verify Operator

## Overview

Verify consumes a complete design brief and an unmodified candidate, independently checks
correctness against the **brief's reference**, executes the Model protocol, applies the
route checklist, classifies each variant, and owns Deliver. Verify never edits kernel or
harness code. Local defects return to Implement; structural defects return to Design for
a complete brief reissue, after which Implement synchronizes kernel and harness.

## Entry Preconditions

Require all of the following before judging the candidate:

- the original task and a complete, feasible brief with concrete Design sections 1–10,
  especially exact variants/tensors/boundaries and correctness formula, comparison method
  and tolerances;
- the exact brief-selected algorithm file (production: `vector_add.py`, `gemm.py`, or
  `gemm_bias_relu.py`; evaluation-only: the candidate-specific `artifact_file`) containing
  kernel, `launch` and `main`, with no sibling candidate modules and no runtime
  dependencies beyond the installed environment;
- the user's currently configured installed-package/CANN/simulator environment and
  isolated execution mode.

An absent/incomplete brief returns to Design; an absent candidate returns to Implement.
Only Implement-owned numerical/open items may remain deferred. Do not fill gaps by
inferring intent from candidate code.

An evaluation-only candidate brief is accepted only when the user explicitly requested
validation. Its result is promotion evidence, not a supported public template.

## Route → References

Load exactly the brief's route row plus the shared simulator reference. These are
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

- exact math/reference string, input dtypes/shapes/layouts, Torch seed and operand draw
  order, fp32 computation/final cast, comparison method and tolerances;
- true GM boundary rules and every frozen variant, including its tail case;
- verifier environment, commands, cache state and protocol sequence against
  `simulator.md`; complete logs with nonzero failures preserved;
- candidate artifact layout and whether its executable harness actually implements that
  same oracle and protocol rather than merely printing the required labels.

If **Verify's** reference, environment, procedure or evidence is wrong/incomplete, mark
only the verification attempt invalid, discard its verdict and ticks, and correct/re-run
the verification. Do not send a defect or change implementation. If the brief contradicts
the original task, return that contract defect to Design without touching the candidate.
Once the independent verification is valid, a candidate harness that violates the
artifact/oracle/protocol contract is `invalid_submission`; return its host defect to
Implement without editing it.

### 2. Audit structure and configuration

Apply only the loaded route checklist. Read kernel and host source, compare actual launch
and loop structure with `CONFIG`/`RESULT.md`, and prove coverage, legal residency, memory
capacity, tail extents and numeric semantics. A numerically passing output does not erase
an out-of-bounds access or route violation. Record mismatches before running performance
comparisons.

The candidate must have the complete kernel→launch→main call-return contract: `launch`
accepts runtime tensors/scalars/options, prepares output, selects tiling/grid/core count
and compile/runtime parameters, performs every JIT launch and returns the result. `main`
orchestrates environment/variants/input/reference/guards/measurements/reporting and calls
`launch`. A direct JIT launch outside `launch` is a structure defect.

### 3. Run the protocol per variant

Follow `simulator.md` exactly:

1. In a separate process with a fresh cache, run `PYASC_DRY_RUN=1`; require real
   compilation, the compiled specialization's six-bank memory check and unchanged full
   output storage. Compilation or memory failure is `compile_failed` when the submission
   itself is otherwise valid.
2. In a separate runtime process, require explicit Model/9599 platform and environment
   facts, exact input generation, one synchronized warmup, correctness against the Design
   reference at its tolerances and bit-identical 64-byte output guards on both sides.
   A launched numerical/guard failure is `incorrect`; a crash, hang or unavailable tick
   after successful compilation is `run_failed`.
3. Only after correctness passes, collect five synchronized tick samples with identical
   tensors/kernel/grid/options, positive deltas and raw start/end/delta plus summary.
   Repeat the complete runtime pass in a second independent process. Both must be correct,
   with five measurements and medians within ±1 tick. If a valid environment/procedure
   does not yield the required reproducible tick result, classify that measurement
   failure `run_failed`, retain the reproducibility concern and never average deviations
   away.
4. Preserve each invocation and failure. Do not silently retry, edit tolerances, repair
   source or report ticks for an incorrect variant. A protocol/artifact violation is
   `invalid_submission`; missing proven environment is an environment deviation, not a
   candidate performance result.

### 4. Classify outcome and backtrack

Assign exactly one protocol outcome to each variant: `correct`, `incorrect`,
`compile_failed`, `run_failed` or `invalid_submission`. `correct` requires every check
and reproducibility requirement, not just a printed `CORRECTNESS PASS`.

For a validly observed failure, classify the defect first. Before routing to the selected
owner, read the persisted counters `local_strikes`, `structural_reissues`, and
`uncertain_reductions` from `RESULT.md`/`DEFECT_REPORT.md` and check the cap for that
specific transition. Another local handoff is forbidden once `local_strikes >= 2`; a
structural reissue is permitted only while `structural_reissues < 2`; another uncertain
reduction is permitted only while `uncertain_reductions < 2`. If the transition would
exceed its cap, do not route back to work: publish `status=stopped` first, then supersede
and archive every active defect. An uncertain reduction is one concrete attempt to reduce
the reproducer and re-check verification validity.

The counters are initialized to 0 and required on every RESULT/defect report and on every
brief/reissue, so they survive the structural handoff. Implement increments
`local_strikes` after each same-signature failed local hypothesis and resets it when the
signature changes or a new `ready_for_verify` RESULT is produced. Design increments
`structural_reissues` on each permitted reissue and records it in the new brief. Verify
increments `uncertain_reductions` after each reduction attempt and resets it after
successful classification; the uncertain cap applies per uncertain episode, not as a
hidden per-task reset. Counters reset only on explicit terminal reopen with a new brief
revision, except for the signature-change/local-success resets and the uncertain-episode
classification reset just stated.

After classification and the cap gate:

- **local → Implement:** a concrete kernel/host/configuration error fixable without
  changing the brief. Return a reproducible report containing `defect_id`,
  `status=active`, allowed `next_owner=implement`, task/variant, brief/protocol revisions,
  `candidate_sha256`, counters `local_strikes`, `structural_reissues`,
  `uncertain_reductions`, exact command/environment/cache, minimal case, first error and log,
  actual config, expected-vs-observed evidence and attempted fixes if supplied.
- **structural → Design:** evidence that no legal implementation can satisfy the brief's
  route/dataflow/residency/fusion/boundary structure. Name the failed structural field
  and installed-package/platform evidence. When the evidence came from Implement, Verify
  writes the current `needs_design` or `needs_implement` RESULT **first**, then
  supersedes/archives the incoming `next_owner=verify` report. This prevents both a Verify
  loop and a crash window: before the RESULT exists, the still-active report selects Verify
  again. Design reissues the complete brief
  and re-derives dependents; Implement then synchronizes both kernel and canonical harness
  before a fresh Verify run.
- **uncertain → Verify triage:** reduce the reproducer and check verification validity
  again before choosing an owner. Never guess structural from a tile overflow or local
  from a wrong route.

No defect classification authorizes Verify to patch the candidate. An invalid verification
never enters this backtrack process. Implement-originated structural evidence always goes
Implement → Verify for classification → Design; never shortcut that hop.

A valid candidate-caused launch failure before a correct result returns to Implement as a
local defect. A crash in a second runtime process is classified the same way when it is
candidate-caused; if the harness/protocol itself is invalid, it is `invalid_submission`.
A measurement failure after correct output (including unavailable or non-reproducible
ticks) is terminal `stopped`: no candidate edit and no speedup. Invalid environment/oracle
never produces an implementation request.

## Deliver

Verify owns the final `RESULT.md`/Deliver report and preserved run evidence. Verify may
create verification-owned reports, oracle and run evidence, but never modifies
`request.md`, `brief.md` or the candidate algorithm file.

Every Verify final or intermediate `RESULT.md` carries `task_id`, `brief_revision`,
`protocol_revision`, `candidate_sha256`, and the counters `local_strikes`,
`structural_reissues`, `uncertain_reductions`. Without these fields it is not eligible for
precedence decisions.

Include in Deliver:

- task id, brief/protocol revisions, route/template, candidate file and exact commands/log
  locations;
- per-variant outcome and validity/route-check findings;
- environment versions/platform/arch and cache directory/cache-warm state;
- actual per-variant configuration and memory consumption/capacities;
- reference/tolerances, correctness error and guard results;
- for correct variants, both runs' raw start/end/delta samples, five raw deltas,
  median/min/max, warmup/measurement/synchronization facts and median difference;
- immutable run manifest paths binding candidate hash, revisions, variant, process/run id
  and captured outputs;
- concerns, whole-launch measurement limitation, environment deviations and any local or
  structural defect report/backtrack. Preserve failed outcomes rather than hiding them.

Simulator ticks are not hardware latency and the interval includes the complete
synchronized launcher path, not proven compute-only cycles. Never convert ticks to time
or compute speedup from a failed/incomparable baseline. Verify delivers evidence and
classification; it does not modify implementation to obtain a better report.

Correct + reproducible + slower than a valid comparable baseline remains `correct` with a
visible regression concern. There is **no automatic performance backtrack** in this MVP;
Tune is a Target extension.

## Persistent handoff state

- `ready_for_verify` from Implement selects Verify.
- `delivered` means every variant passed; no next stage.
- `needs_implement` selects Implement for a valid local/invalid-submission defect.
- `needs_design` selects Design for a Verify-classified structural/contract defect.
- `verify_retry` selects Verify after an invalid verification; candidate is unchanged.
- `needs_user_input` selects Design after the user answers; the old report is superseded
  and archived.
- `environment_failed` records a failed preflight without a correctness result and blocks
  ordinary defect/RESULT routing. After user remediation and an explicit retry, only Verify
  re-runs preflight. On success Verify **first publishes a revision-and-hash-matching
  successor RESULT**, then archives the failed RESULT and routes that successor's stage. If
  the successor cannot be published or preflight fails again, the failed RESULT remains
  current, no result/ticks are emitted, and old defects do not route. Baseline comparability
  is re-evaluated after drift.
- `stopped` is terminal for no-progress/measurement reports; no next stage.
- A terminal-looking RESULT without required identity/counters or with a candidate hash
  mismatch is not eligible for precedence decisions: Verify must write persistent
  `QUARANTINE.md` with `status=quarantined`, reason and identity, and either repair it or
  archive it as inconsistent. The marker outranks old defects and prevents their revival;
  it also supersedes/archives coexisting active defects. It never falls through to an
  active defect or Implement work by default.

Current matching `delivered` or `stopped` has absolute terminal precedence. On entering
`stopped`, publish the terminal RESULT first, then supersede and archive all active
defects, not falsely resolved. `delivered` also supersedes any remaining active defects.
Terminal states are left only by a new brief revision/user-driven task change, never by an
old defect. If a crash leaves an active defect beside a terminal RESULT, resume repairs the
lifecycle by superseding and archiving the defect rather than resuming work.

The local two-strike stop, structural-reissue cap and uncertain-triage cap all produce
`stopped` through Verify. Concretely, at most **two full structural reissues** per task
and at most **two uncertain-triage reductions per uncertain episode** are allowed;
exhausting any of the three caps produces `stopped` and supersedes/archives every active
defect.

A user-remediated environment retry re-runs preflight and resumes at the
artifact-appropriate current stage without restoring old defect routing.

## Hard Boundaries

- No kernel or harness edits, no pytest/conftest/checkout dependency.
- Correctness is judged against Design's exact reference and the Verify-owned independent
  oracle, not candidate self-report.
- Verification validity comes before any backtrack; invalid verification changes no code.
- Structural backtrack means full Design brief reissue plus Implement-owned kernel and
  harness synchronization, followed by fresh verification.
