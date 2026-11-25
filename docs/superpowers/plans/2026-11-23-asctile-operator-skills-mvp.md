# AscTile Operator Skills MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone three-skill project that helps a fresh agent develop correct AscTile/C310 operators and obtain reproducible simulator performance estimates, improving on the current `is/matmul_skill` skills.

**Architecture:** `Design → Implement → Verify → Deliver`; three skills with route-specific references, one canonical harness, local/structural backtracking, no orchestrator.

**Tech Stack:** Agent Skills Specification, pyasc-dev, CANN, plain Python harnesses, Ascend950PR_9599 simulator, simulator ticks.

**Spec:** The agreed design in the conversation, represented by this plan's global constraints, file map, gates and acceptance tasks.

## Global Constraints

- Project root: `/home/artyomka/repos/work/pyasc-skills`; execution uses the isolated worktree.
- Evaluation workspace: `/home/artyomka/repos/work/test_skills`.
- Greenfield creation of new operators only; no arbitrary debugging/refactoring.
- AscTile and C310/Ascend950PR_9599 only.
- No pyasc checkout/build dependency; use `from asc.experimental import asctile`.
- No repo `conftest.py`, pytest fixtures, or `pytest --compile-only` dependency.
- No assets/, scripts/, orchestrator, or tuning skill in the MVP.
- Verify all technical facts against the installed package; do not blindly copy old skills.
- Simulator ticks are not hardware latency.
- First A/B is calibration, without an invented universal numeric performance threshold.
- No git mutations without explicit user permission.

## Review Focus

- Cube-only must not load Vector knowledge.
- Cube+Vector must not load the full Pure Vector playbook.
- Invalid verification must not trigger implementation changes.
- Structural backtrack must reissue the brief and synchronize the canonical harness.
- Simulator measurements must be reproducible and record environment versions.

---

### Task 1: Prove the standalone environment and simulator tick protocol

**Files:**
- Create: `/home/artyomka/repos/work/test_skills/mvp/environment_probe.py`
- Create: `/home/artyomka/repos/work/test_skills/mvp/environment-report.md`

- [ ] Activate the supplied venv/CANN environment and record Python, pyasc, CANN, backend and platform versions.
- [ ] Verify `from asc.experimental import asctile` and explicit `set_platform(Backend.Model, Platform.Ascend950PR_9599)`.
- [ ] Implement a plain-Python fp16 vector-add probe without pytest.
- [ ] Verify compile-only via `PYASC_DRY_RUN=1`, simulator correctness against Torch, and memory checks.
- [ ] Establish tick collection with warmup, five measurements, synchronization, raw values and median.
- [ ] Record exact commands and limitations in `environment-report.md`.

**Gate:** standalone compile/run/correctness works and ticks are reproducible. If ticks are unavailable, stop the performance portion and report before authoring skills.

### Task 2: Freeze three calibration tasks and capture the as-is baseline

**Files:**
- Create: `/home/artyomka/repos/work/test_skills/mvp/tasks/vector-add.md`
- Create: `/home/artyomka/repos/work/test_skills/mvp/tasks/gemm.md`
- Create: `/home/artyomka/repos/work/test_skills/mvp/tasks/gemm-bias-relu.md`
- Create: `/home/artyomka/repos/work/test_skills/mvp/evaluation-protocol.md`
- Create: `/home/artyomka/repos/work/test_skills/mvp/baseline/`

Calibration tasks:
- Vector `y = a + b`, fp16, primary `N=16384`, tail `N=16003`.
- Cube GEMM, fp16, primary `M=128,N=128,K=64`, tail `M=72,N=80,K=56`.
- Cube+Vector `relu(A @ B + bias)`, fp16, same shapes, with a real Vector epilogue.
- Torch references compute in fp32 then cast to fp16; `atol=2e-3`, `rtol=2e-3`.

- [ ] Freeze task specifications and the common evaluation protocol.
- [ ] Run each task with fresh agents exposing only the current `is/matmul_skill` skills.
- [ ] Preserve baseline kernels, host-side code, harnesses, commands, correctness, ticks and process logs.
- [ ] Do not fix the old skills or coach them about known defects.
- [ ] Mark incorrect baseline outcomes as baseline process failures; never compute speedup from them.
- [ ] Passively record user hints, compile attempts and backtracks.

**Gate:** each task has an unambiguous baseline outcome; a failed baseline is a valid experimental result.

### Task 3: Scaffold the project and enable standards validation

**Files:**
- Modify: `README.md`
- Create: `skills/asctile-design-operator/SKILL.md`
- Create: `skills/asctile-design-operator/references/routing.md`
- Create: `skills/asctile-design-operator/references/feasibility.md`
- Create: `skills/asctile-implement-operator/SKILL.md`
- Create: `skills/asctile-implement-operator/references/vector.md`
- Create: `skills/asctile-implement-operator/references/cube.md`
- Create: `skills/asctile-implement-operator/references/cube-vector.md`
- Create: `skills/asctile-verify-operator/SKILL.md`
- Create: `skills/asctile-verify-operator/references/vector.md`
- Create: `skills/asctile-verify-operator/references/cube.md`
- Create: `skills/asctile-verify-operator/references/cube-vector.md`
- Create: `skills/asctile-verify-operator/references/simulator.md`

- [ ] Create exactly the agreed tree.
- [ ] Add valid name/description/compatibility frontmatter and stage-specific entry conditions.
- [ ] Document project purpose, AscTile/C310 scope and external evaluation workspace in README.
- [ ] Check that no content requires pyasc checkout/conftest/repo-only pytest options.
- [ ] Run `skills-ref validate` for every skill.

**Gate:** valid structure, distinguishable activation descriptions, no extra directories.

### Task 4: Implement `asctile-design-operator`

**Files:**
- `skills/asctile-design-operator/SKILL.md`
- `skills/asctile-design-operator/references/routing.md`
- `skills/asctile-design-operator/references/feasibility.md`

- [ ] Define the complete brief schema, algorithm choice, route enum, structural dataflow/residency/parallelism, tile family, feasibility, correctness reference/tolerances and available baseline.
- [ ] Specify targeted user questions for missing semantic contract and full-brief reissue after structural backtrack.
- [ ] Verify all routing/feasibility facts against installed pyasc.
- [ ] Give the three frozen specifications to fresh agents; expect routes `vector`, `cube`, `cube_vector`.
- [ ] Verify Design writes no kernel and chooses no final tiling.
- [ ] Verify missing dtype/semantics elicits a targeted question and infeasibility produces an immediate Report.

**Gate:** complete correctly routed briefs, no guessing or premature implementation.

### Task 5: Implement the common Implement/Verify loop on the Vector route

**Files:**
- `skills/asctile-implement-operator/SKILL.md`
- `skills/asctile-implement-operator/references/vector.md`
- `skills/asctile-verify-operator/SKILL.md`
- `skills/asctile-verify-operator/references/vector.md`
- `skills/asctile-verify-operator/references/simulator.md`
- Candidate artifacts: `/home/artyomka/repos/work/test_skills/mvp/candidate/vector-add/`

- [ ] Define brief preconditions, route→references table, kernel/host ownership, exact tiling/grid/options, canonical plain-Python harness, early compile/smoke, defect reports and no-progress stop.
- [ ] Distil verified Vector knowledge without confirmed baseline errors.
- [ ] Put the standalone environment/run block in Implement SKILL.md and the proven tick protocol in simulator.md.
- [ ] Define correctness, route checklist, verification-validity checks and Verify-owned Deliver/report.
- [ ] Run the Vector task with a fresh agent exposing only new skills; test `N=16384` and `N=16003`.
- [ ] Confirm no Cube references were loaded; capture correctness and ticks.
- [ ] Probe an intentionally invalid reference and confirm Verify does not modify the kernel.

**Gate:** Vector pipeline works end-to-end and invalid verification cannot alter implementation.

### Task 6: Add the Cube route

**Files:**
- `skills/asctile-implement-operator/references/cube.md`
- `skills/asctile-verify-operator/references/cube.md`
- Candidate artifacts: `/home/artyomka/repos/work/test_skills/mvp/candidate/gemm/`

- [ ] Distil Cube memory, accumulation, transpose, tiling, buffering and output paths.
- [ ] Correct L1/UB budgets, separate L0A/L0B budgets, dynamic `matmul_acc` selection, Cube real_shape prohibition and supported fusible transpose patterns.
- [ ] Run GEMM with a fresh agent through the new pipeline on primary and tail shapes.
- [ ] Confirm no Vector references were loaded.
- [ ] Capture simulator correctness, ticks and Verify report.

**Gate:** correct measured GEMM without Vector knowledge.

### Task 7: Add the Cube+Vector route

**Files:**
- `skills/asctile-implement-operator/references/cube-vector.md`
- `skills/asctile-verify-operator/references/cube-vector.md`
- Candidate artifacts: `/home/artyomka/repos/work/test_skills/mvp/candidate/gemm-bias-relu/`

- [ ] Implement `cube_vector → cube.md + cube-vector.md`; never load full `vector.md`.
- [ ] Keep cv_ratio, Split/Join, DistribMode, cv_strategy/manual distrib, sub-block offsets, Vector numerics and carried-state rules together in cube-vector.md.
- [ ] Correct cv_strategy contradictions and document the copy_out(real_shape) restriction.
- [ ] Run GEMM+bias+ReLU with a real Cube+Vector epilogue on primary and tail shapes.
- [ ] Verify local defects return a reproducible case to Implement and structural returns reissue brief/synchronize harness.

**Gate:** Cube+Vector works without Pure Vector playbook; backtracks preserve brief/harness consistency.

### Task 8: Run final A/B and fix only observed regressions

**Files:**
- Update: `/home/artyomka/repos/work/test_skills/mvp/evaluation-protocol.md`
- Create: `/home/artyomka/repos/work/test_skills/mvp/ab-results.md`

- [ ] Rerun final new skills on all three tasks in clean candidate workspaces.
- [ ] Use identical external tasks, references/tolerances, environment versions and tick protocol.
- [ ] Allow different internal signatures, grid, tiles, cv_ratio and options; measure each final configuration.
- [ ] Record correctness and median ticks per shape.
- [ ] Compute `speedup = baseline_ticks / candidate_ticks` only for correct comparable results.
- [ ] Record baseline failures without speedup; attach passive process metadata.
- [ ] Use existing backtrack for regressions; fix only proven problems and rerun affected routes.

**Gate:** all candidate results are correct and reproducible; improvement over baseline is reported per task without hiding regressions in an average or inventing a threshold.

### Task 9: Final independent package review

- [ ] Run `skills-ref validate` for all skills.
- [ ] Check directory/name agreement, frontmatter, entry descriptions, direct one-level references and compact route-neutral SKILL.md files.
- [ ] Check absence of repo-only paths, pytest dependency and incorrect imports.
- [ ] Give the project and ab-results.md to a fresh independent reviewer.
- [ ] Fix only confirmed blocker/high findings and rerun affected checks.

**Final result:** three valid skills, three reproducible operator results, standalone simulator protocol, A/B report and documented limitations/future extension points.
