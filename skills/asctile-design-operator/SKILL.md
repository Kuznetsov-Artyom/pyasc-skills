---
name: asctile-design-operator
description: Use when starting a new greenfield AscTile/C310 operator request before any kernel code is written — turning a user task into a complete design brief with route, exact supported template, structural dataflow/residency/parallelism, tile family constraints, feasibility verdict, and correctness reference/tolerances. Entry stage of the Design → Implement → Verify pipeline. Not for writing or debugging kernel code, choosing final tile sizes, or modifying an existing operator. The user does not need to know Cube/Vector, UB/L0/L1, cv_ratio, DistribMode, MLIR, IR or AscendC.
compatibility: Requires the installed pyasc package (`from asc.experimental import asctile`) for fact-checking routing and feasibility. No pyasc source checkout, no pytest. AscTile and C310/Ascend950PR_9599 only.
---

# AscTile: Design Operator

## Overview

Design is the entry stage of the Design → Implement → Verify pipeline. It converts a
greenfield operator request into a **complete, unambiguous design brief** that Implement
can consume without guessing, or into an explicit terminal report.

Design does **not** write kernel code, pick final tile sizes or grid numbers, or launch
anything. It may read installed-package contracts and perform read-only capability
introspection, but never compiles or launches a kernel.

## Entry Preconditions

Activate this skill only when **all** of the following hold:

- The user asks for a **new** AscTile operator (greenfield creation, not debugging or
  refactoring an existing kernel).
- Target is AscTile on C310 / `Ascend950PR_9599` — no other backend or platform.
- No design brief exists yet for this request, or a structural backtrack from Verify
  has invalidated the previous brief (in that case the brief is **reissued in full**,
  not patched — see "Structural backtrack" below), or Implement has rejected an existing
  incomplete/hollow/contradictory brief for reissue, or the user has explicitly reopened a
  terminal task with a changed contract, which also requires a new brief revision.

If any precondition fails, emit `out_of_scope` and stop. Do not forward a brief-less
request to Implement or Verify: their entry conditions reject it and would create a
dead-end.

## Procedure

Do these steps **in order**. Steps 2–4 are detailed in the references — load them when
you reach the step, not before.

1. **Extract the semantic contract.** Record the exact math, every input/output tensor
   (name, shape, dtype, layout/broadcast semantics), the correctness reference and
   tolerances. If a mathematical or required user fact is missing or ambiguous, ask a
   **plain-language targeted question** — never guess. Do not ask the user about Cube
   vs Vector, unit placement, tile sizes, `cv_ratio`, `DistribMode`, or other hardware
   implementation details: those are Design decisions.
2. **Choose the route and exact template**: `vector`, `cube`, or `cube_vector`, plus
   the exact published template from `routing.md`. Route is chosen by what the spec
   requires and by the validated implementation recipe, not by asking the user to
   choose hardware.
3. **Check feasibility** against the installed pyasc package per
   `references/feasibility.md`. If the request is structurally infeasible under the
   installed AscTile contract, emit an infeasibility report immediately — do not
   produce a brief. Platform-feasible but unvalidated experience is candidate practice,
   not infeasibility.
4. **Fill in the brief**: algorithm/template, structural dataflow (per-tensor GM ↔
   on-chip residency), parallelism choice, tile family (names + constraints, **not**
   values), boundary/tail strategy family, feasibility verdict, correctness
   reference/tolerances, optional baseline status, and all defaults/assumptions with
   provenance. The schema is in `references/routing.md`.

## Defaults and questions

- If dtype or shape is missing, ask one short question naming only the missing fact.
- If the user did not specify a correctness reference, derive it from the exact math and
  record it as `default`/`agent_decision`.
- For the published fp16 templates, missing tolerances use the labelled default
  `atol=2e-3`, `rtol=2e-3`; the user may override them.
- Default layout is contiguous row-major unless the request says otherwise.
- Default input generation is deterministic; record the generator, seed and draw order.
- Do not ask about reference/tolerance preferences when the defaults are sufficient.
- Never guess an ambiguous mathematical contract and record it later as if it were a fact.

## Output and artifact status

Exactly one of the following is emitted:

| Status | Condition | Artifact |
|---|---|---|
| `complete` | A published template applies and semantics are known | Full `brief.md` with `next_stage=implement` |
| `needs_user_input` | Missing math/shape/dtype or another user-only fact | `DESIGN_REPORT.md` with the question; no partial brief |
| `candidate_recipe` | Real experience exists but is not validated in this pipeline | `DESIGN_REPORT.md` by default, or an evaluation-only candidate brief when explicitly asked to validate |
| `unsupported_in_current_iteration` | No published template and no applicable candidate recipe | `DESIGN_REPORT.md` with reason and closest supported alternative |
| `infeasible` | Installed AscTile API/platform contract contradicts the request | Infeasibility report with constraint, evidence and minimal relaxation |
| `out_of_scope` | Not a greenfield AscTile/C310 operator request | Terminal refusal without redirect |

The precedence is: `out_of_scope` → proven `infeasible` → `candidate_recipe` for real
experience awaiting validation → `unsupported_in_current_iteration` →
`needs_user_input` when missing semantics prevent classification. No terminal status
creates a partial brief.

## Artifact contract

Design creates `<task-workdir>/request.md` containing the original request, user answers
and confirmed semantics. A complete result is `<task-workdir>/brief.md`; other outcomes
use `<task-workdir>/DESIGN_REPORT.md`. Candidate evidence and superseded artifacts are
preserved under `<task-workdir>/runs/`.

Exception: when the user explicitly asks to validate a candidate recipe, the
**evaluation-only candidate brief** is also `<task-workdir>/brief.md`, but with
`status=candidate_recipe`, its own `brief_revision`, and `next_stage=implement`. A
`DESIGN_REPORT.md` describing a candidate is terminal until the user asks for validation.
The exact production template→file/symbol mapping applies only to `status=complete`
briefs. Evaluation-only candidates instead choose a candidate-specific `artifact_file`
and `kernel_symbol` (for example `softmax.py`/`softmax`) while retaining the same
one-file kernel→launch→main contract. They are not supported public templates.

Only current top-level artifacts participate in routing. `runs/` is immutable archived
evidence and is never scanned for routing. There is one current brief: the highest
revision. Superseded briefs are archived as `runs/brief-r<revision>.md`.

Every complete `brief.md` contains the **expected route/template**. An acceptance gate
compares the actual route against that expectation and fails on mismatch; it does not
merely observe whichever route the agent happened to choose.

Every published template requires **positive N/M/K**. Any zero required extent maps to
`unsupported_in_current_iteration` unless an evaluation-only no-op candidate policy is
explicitly requested and validated. This is distinct from installed-API `infeasible`.

Every complete `brief.md` contains:

- `task_id`, `brief_revision`, `protocol_revision=asctile-model-v1`;
- `local_strikes`, `structural_reissues`, `uncertain_reductions` (initialized to 0 on the
  first brief and carried across every handoff/reissue);
- production briefs use `status=complete`; evaluation-only candidate briefs use
  `status=candidate_recipe` and may name a candidate-specific `template` outside the three
  published templates. Their `route` must still be `vector`, `cube`, or `cube_vector`,
  selected by `routing.md`; if none applies, return `unsupported_in_current_iteration`;
- `route`, `template`;
- `artifact_file`, `kernel_symbol`, `launch_symbol=launch`, `main_symbol=main`;
- `next_stage=implement`, and `supersedes` when it is a reissue;
- provenance for structural facts: `user`, `default`, `agent_decision`, or `verified`.

`next_stage` in a brief describes the handoff created by that brief. It does not override
a newer candidate/result state or an active defect with matching current revisions.

## Persistent handoff state

Implement's draft `RESULT.md` must contain:

- `status=ready_for_verify`, `next_stage=verify`;
- task id, brief/protocol revisions;
- `candidate_sha256` of the exact brief-selected algorithm file;
- commands, actual configuration and evidence paths.

The algorithm file contains kernel, launch wrapper and protocol harness in one file, so
its hash covers both kernel and harness changes. No separate harness identity is needed.

A local defect becomes `resolved` only atomically with a new `ready_for_verify` RESULT
for the latest revisions and current candidate hash. The previous RESULT is archived with
the resolved defect. If a resolved defect has no such matching successor, the next stage
is Implement to produce one, not Verify.

Every Verify final or intermediate `RESULT.md` carries `task_id`, `brief_revision`,
`protocol_revision`, `candidate_sha256`, and counters `local_strikes`,
`structural_reissues`, `uncertain_reductions`, including `delivered`, `stopped`, `needs_*`
and `verify_retry`. Without these fields it is not eligible for precedence decisions.

Counters are initialized to 0 on the first brief/RESULT. Implement increments
`local_strikes` after each failed hypothesis-driven local change with the same failure
signature and resets it when the signature changes or when a new `ready_for_verify` RESULT
is produced. Design increments `structural_reissues` when publishing each permitted full
reissue and carries it in the new brief. Verify increments `uncertain_reductions` after
each concrete reproducer-reduction attempt and resets it when a classification is reached;
the uncertain cap applies per uncertain episode, not as a hidden per-task reset. Counters
persist across normal local/structural cycles and are reset on an explicit terminal reopen
with a new brief revision, except for the signature-change/local-success resets and the
uncertain-episode classification reset just stated.

Verify's final `RESULT.md` uses:

- `status=delivered` — every variant is correct and the protocol passed; no next stage;
- `status=needs_implement`, `next_stage=implement` — valid local or invalid-submission
  candidate defect;
- `status=needs_design`, `next_stage=design` — Verify-classified structural or contract
  defect;
- `status=verify_retry`, `next_stage=verify` — invalid verification; candidate is
  unchanged;
- `status=needs_user_input`, `next_stage=design` — waiting for a user answer;
- `status=stopped` — terminal no-progress/measurement report; no next stage.

Each `DEFECT_REPORT.md` contains `defect_id`, `status=active|resolved|superseded`,
brief/protocol revisions, candidate SHA256, the counters `local_strikes`,
`structural_reissues`, `uncertain_reductions`, and `next_owner`. Allowed `next_owner` values
are exactly `implement`, `verify`, `design`. Only an active report that matches the latest
revisions and candidate identity participates in routing. Resolved or superseded reports
are archived under `runs/` and have no routing power.

For Implement-originated structural evidence, Implement writes `next_owner=verify`. Verify
performs classification and **publishes the current RESULT first** (`needs_design` or
`needs_implement`), then supersedes/archives the incoming evidence report. This ordering
makes the transition crash-safe: if Verify crashes before the RESULT exists, the still
active `next_owner=verify` report simply selects Verify again. It also forbids a direct
Implement → Design shortcut. If verification is invalid, no evidence report is resolved and
the result is `verify_retry`.

Producer provenance is fixed for structural evidence: Implement may produce
`next_owner=verify`; Verify may produce `next_owner=implement` or `next_owner=design`
after classification. An incomplete/hollow/contradictory brief rejected by Implement is a
separate contract-defect path: Implement returns it directly to Design for reissue and
does not count as structural evidence requiring Verify classification.

Fresh-agent resume uses this **terminal precedence** order:

0. A current matching `delivered` or `stopped` `RESULT.md` is terminal and absolute. If a
   terminal-looking RESULT lacks required identity/counters, Verify owns state repair or
   quarantine; it never silently falls back to old-defect or Implement work.
1. An open, unanswered question invalidates the current brief and suspends all other
   stages. No active defect or ready RESULT may route while the question is open. An
   answered tombstone with `answered=true,next_stage=design` also takes precedence over
   old defects, ready RESULTs and the invalidated brief until the successor brief/report
   is published; only then is it superseded and archived.
2. An existing incomplete/hollow/contradictory brief with a current `DESIGN_REPORT.md`
   carrying `status=needs_design` selects Design for reissue and blocks Implement/Verify
   until the new brief is published.
3. A current `environment_failed` RESULT suspends ordinary defect/RESULT routing. Verify
   owns any preflight retry; only after a successful retry does it publish a revision-and-
   hash-matching successor RESULT and archive the failed one. If retry fails again, the
   failed RESULT remains current and no result/ticks are produced.
4. An active defect matching the latest revisions and candidate identity selects its
   `next_owner`.
5. A `RESULT.md` matching the latest revisions and current candidate hash selects its
   status/next stage.
6. A newer brief without a matching ready candidate selects Implement. The old candidate
   is ineligible for the new revision. If the candidate hash changed but no matching RESULT
   exists after a crash/local edit, Implement also owns reconciliation and must produce the
   matching current RESULT before any Verify — **unless a current terminal RESULT exists**,
   in which case rule 0/quarantine wins and Implement must not reopen work.
7. `DESIGN_REPORT.md` with an answered `needs_user_input` selects Design after the user's
   answer; an unanswered report stays a barrier.

An environment failure uses `status=environment_failed`, **not** `stopped`. On user
remediation plus explicit retry, only Verify re-runs preflight. On success, Verify first
publishes a revision-and-hash-matching successor RESULT, then archives the old
environment-failure RESULT and routes the successor's stage. If publication or preflight
fails, the failed RESULT remains current and no result/ticks are emitted. Old defects remain
without routing power and baseline comparability is re-evaluated.

A terminal RESULT whose candidate hash or identity mismatches the current artifact is
stale/inconsistent and is quarantined for Verify-owned repair, never treated as permission
for Implement to reopen work. Verify writes a persistent top-level `QUARANTINE.md` with
`status=quarantined`, task/revision/hash and reason. That marker has terminal precedence
until a fully identified terminal RESULT or an authorized new brief revision exists.
Archiving an inconsistent RESULT atomically preserves that quarantine marker and
supersedes/archives every coexisting active defect, so a later resume cannot rediscover
old work.

Entering `stopped` publishes the terminal RESULT **first**, then supersedes and archives
every active defect for the task (not falsely `resolved`). This write order is crash-safe:
terminal precedence keeps the task stopped even if cleanup is interrupted. No-progress and
measurement stops can be left only by a new user request or a changed task contract, never
by an old defect. An `environment_failed` state is the explicit exception: after user
remediation plus retry, Verify may rerun preflight, but must publish the matching successor
RESULT before archiving that failed state and resuming its stage. The local two-strike stop, structural
reissue cap and uncertain-triage cap all produce `stopped` through Verify. `delivered`
also supersedes any remaining active defects as a consistency guard.

A terminal restart requires a new brief revision; a repeated user request alone does not
make an old terminal state current. If a crash leaves an active defect beside a terminal
RESULT, resume first repairs the lifecycle by superseding and archiving that defect; it
does not resume work.

When a user answers a question, keep a top-level answered marker with
`status=needs_user_input`, `answered=true`, and `next_stage=design` until the successor
brief/report is published. Only then mark the old `DESIGN_REPORT.md` or Verify
`needs_user_input` RESULT superseded and archive it. Any changed brief after an answer
increments `brief_revision` and archives the previous brief only after the new one is
published.

An open question invalidates the current brief and suspends **all** other stages, not only
the "newer brief → Implement" rule. No active defect or ready RESULT may route work while
the question is open.

## Candidate validation path

A user may explicitly ask to validate a candidate recipe. In that case Design produces a
complete, revisioned **evaluation-only candidate brief** with `status=candidate_recipe`,
math, shapes/dtypes, reference, variants, artifact identity and candidate provenance.
That brief follows the same Implement → Verify ownership and defect rules.

Before authoritative Verify/acceptance passes, neither Design nor README/RESULT may call
the candidate supported. Passing supplies `promotion_evidence` for a future extension; it
does not silently add a fourth public template to this MVP. Failure follows the existing
defect/terminal lifecycle.

## Structural backtrack

When Verify returns a defect classified as **structural** (the brief's dataflow cannot
meet the contract — wrong route, wrong residency choice, a fusion the platform cannot
express), Design reissues only if Verify's cap gate permits it: `structural_reissues < 2`.
If the cap is exhausted, Verify writes `stopped` and supersedes/archives every active
defect; Design does not reissue and does not overwrite that terminal state.

Otherwise Design **reissues the complete brief in full** with the failed decision revised
and every dependent field re-derived. Never patch just the failed field: the brief is one
consistent structure, and a partial edit leaves stale fields that contradict the new
structure. The same `structural_reissues < 2` cap and counter transport also apply to a
full reissue of an incomplete/hollow/contradictory brief.

A reissue increments `brief_revision`, sets `next_stage=implement`, marks every active
defect from the old revision `superseded`, and makes the old candidate ineligible for the
new revision. Publish the new brief first, then archive the old brief, candidate and
RESULT under `runs/`. This ordering prevents a crash from reviving a stale structure.
Implement then synchronizes both kernel and harness before fresh Verify.

## Hard Boundaries

- No kernel source, no `@asctile.jit` code, no final tile-size or grid numbers.
  Tile families and their constraints belong to the brief; the numbers belong to
  Implement.
- No hardware questions to the user. Hardware choices are agent-owned and recorded.
- All routing and feasibility facts are verified against the **installed** pyasc
  package (`from asc.experimental import asctile`); never copied unchecked from old
  skills, docs, or a pyasc checkout. Every hardware/API fact the brief relies on must
  be traceable to the installed package — see `references/feasibility.md`.
- No pytest, no repo `conftest.py`, no `pytest --compile-only` dependency anywhere in
  the produced brief.
- Missing semantics → targeted question. Structural infeasibility → immediate report.
  Real but unvalidated experience → candidate practice, never a false supported claim.
  Never resolve any of these by guessing.
