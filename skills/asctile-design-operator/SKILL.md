---
name: asctile-design-operator
description: Use when starting a new greenfield AscTile/C310 operator request before any kernel code is written — turning a user task into a complete design brief with route (vector, cube, cube_vector), structural dataflow/residency/parallelism choices, tile family, feasibility verdict, and correctness reference/tolerances. Entry stage of the Design → Implement → Verify pipeline. Not for writing or debugging kernel code, choosing final tile sizes, or modifying an existing operator.
compatibility: Requires the installed pyasc package (`from asc.experimental import asctile`) for fact-checking routing and feasibility. No pyasc source checkout, no pytest. AscTile and C310/Ascend950PR_9599 only.
---

# AscTile: Design Operator

## Overview

Design is the entry stage of the Design → Implement → Verify pipeline. It converts a
greenfield operator request into a **complete, unambiguous design brief** that Implement
can consume without guessing, or into an immediate infeasibility report.

Design does **not** write kernel code, pick final tile sizes or grid numbers, or launch
anything.

## Entry Preconditions

Activate this skill only when **all** of the following hold:

- The user asks for a **new** AscTile operator (greenfield creation, not debugging or
  refactoring an existing kernel).
- Target is AscTile on C310 / `Ascend950PR_9599` — no other backend or platform.
- No design brief exists yet for this request, or a structural backtrack from Verify
  has invalidated the previous brief (in that case the brief is **reissued in full**,
  not patched — see "Structural backtrack" below).

If any precondition fails, do not activate: route implementation work to
`asctile-implement-operator` and verification work to `asctile-verify-operator`.

## Procedure

Do these steps **in order**. Steps 2–4 are detailed in the references — load them when
you reach the step, not before.

1. **Extract the semantic contract.** Record the exact math, every input/output tensor
   (name, shape, dtype, layout/broadcast semantics), the correctness reference and
   tolerances. If any of these is missing or ambiguous, ask the user a **targeted
   question** — never guess. The question menu is in
   [references/routing.md](references/routing.md#targeted-questions).
2. **Choose the route**: `vector`, `cube`, or `cube_vector`, by the verified decision
   procedure in [references/routing.md](references/routing.md). Route is decided by
   what the *spec mandates about where compute runs*, not by what the math could
   theoretically be lowered to.
3. **Check feasibility** against the installed pyasc package per
   [references/feasibility.md](references/feasibility.md). If the request is
   structurally infeasible, emit an infeasibility report immediately — do not produce
   a brief.
4. **Fill in the brief**: algorithm family, structural dataflow (per-tensor GM ↔
   on-chip residency), parallelism choice, tile family (names + constraints, **not**
   values), boundary/tail strategy family, feasibility verdict, correctness
   reference/tolerances, available-baseline status. The schema is in
   [references/routing.md](references/routing.md#brief-schema).

## Output

Exactly one of:

- **A complete design brief** following the schema in
  [references/routing.md](references/routing.md#brief-schema) — complete means the
  structural fields (semantics, route, algorithm family, dataflow/residency,
  parallelism, tile family constraints, boundary strategy family, feasibility,
  correctness contract, baseline status) are **all concrete**; only Implement-owned
  numerical/open items (tile values, unroll values, launch options, exact offsets,
  cache policy) may be marked `deferred-to-implement`. A brief with guessed or
  hollow structure is worse than no brief.
- **An infeasibility report** naming the blocking constraint and the smallest spec
  relaxation that would unblock it, per
  [references/feasibility.md](references/feasibility.md#infeasibility-report).

A targeted question (step 1) suspends Design output entirely until the user answers —
a partial brief is not emitted in the meantime.

## Structural backtrack

When Verify returns a defect classified as **structural** (the brief's dataflow cannot
meet the contract — wrong route, wrong residency choice, a fusion the platform cannot
express), Design **reissues the complete brief in full** with the failed decision
revised and every dependent field re-derived. Never patch just the failed field: the
brief is one consistent structure, and a partial edit leaves stale fields that
contradict the new structure.

## Hard Boundaries

- No kernel source, no `@asctile.jit` code, no final tile-size or grid numbers.
  Tile families and their constraints (e.g. "`base_m % 2 == 0` iff `SplitByM` is the
  chosen split") belong to the brief; the numbers belong to Implement.
- All routing and feasibility facts are verified against the **installed** pyasc
  package (`from asc.experimental import asctile`); never copied unchecked from old
  skills, docs, or a pyasc checkout. Every hardware/API fact the brief relies on must
  be traceable to the installed package — see
  [references/feasibility.md](references/feasibility.md).
- No pytest, no repo `conftest.py`, no `pytest --compile-only` dependency anywhere in
  the produced brief.
- Missing semantics → targeted question. Structural infeasibility → immediate report.
  Never resolve either by guessing.
