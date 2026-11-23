# pyasc-skills

A standalone three-skill project that helps a fresh agent develop correct AscTile/C310
operators and obtain reproducible simulator performance estimates.

**Scope:** greenfield creation of new operators only; AscTile and C310 /
`Ascend950PR_9599` (simulator) only. Everything runs against the installed pyasc
package via `from asc.experimental import asctile` — no pyasc checkout, no pytest,
no repo `conftest.py`.

## Skills

| Skill | Stage | Loads |
|---|---|---|
| `skills/asctile-design-operator` | Design — turn an operator request into a complete brief (route, structure, feasibility, reference) | `references/routing.md`, `references/feasibility.md` |
| `skills/asctile-implement-operator` | Implement — kernel + standalone harness for the brief's route | `references/{vector,cube,cube-vector}.md` by route |
| `skills/asctile-verify-operator` | Verify — run the protocol, classify the outcome, own the report | `references/{vector,cube,cube-vector}.md` by route + `references/simulator.md` |

Pipeline: `Design → Implement → Verify → Deliver` — no orchestrator skill.

## Evaluation workspace

Calibration tasks, baselines, and candidate results live outside this project in
`/home/artyomka/repos/work/test_skills/mvp` (frozen task specs,
`evaluation-protocol.md`, `baseline/`, `candidate/`). This repository contains only
the skills themselves.

Simulator ticks are simulator-relative evidence, never hardware latency.
