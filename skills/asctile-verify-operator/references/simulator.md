# Verify reference: simulator run and tick protocol

This is the accepted Task 1 protocol for `pyasc==1.1.1+g63ed594e.d20261122` on
`Backend.Model` / `Platform.Ascend950PR_9599` (arch `c310`). It is shared by all routes.
Use the original frozen protocol as authority; this file is its executable checklist.

## Exact environment and commands

Proven facts: Python 3.13.5, torch 2.15.0+cpu, CANN runtime/simulator 9.2.2,
`rt.is_model() == True`, runtime SoC `Ascend950PR_9599`. Imports must resolve to the
supplied venv's installed `asc` / `asc.experimental.asctile`, with `PYTHONPATH` absent.
Any mismatch is an environment deviation: stop and record it instead of a result.

```bash
cd /home/artyomka/repos/work/test_skills/mvp
set -e
source /home/artyomka/repos/work/test_skills/.venv/bin/activate
source /opt/ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
candidate=/path/to/candidate
variant=primary
cache=$(mktemp -d /tmp/opencode/asctile-verify.XXXXXX)
env -u PYTHONPATH PYASC_DRY_RUN=1 PYASC_CACHE_DIR="$cache" MVP_SHAPES="$variant" python -I "$candidate/entry.py"
env -u PYTHONPATH -u PYASC_DRY_RUN PYASC_CACHE_DIR="$cache" MVP_SHAPES="$variant" python -I "$candidate/entry.py"
env -u PYTHONPATH -u PYASC_DRY_RUN PYASC_CACHE_DIR="$cache" MVP_SHAPES="$variant" python -I "$candidate/entry.py"
```

Replace `candidate` with the candidate directory. Repeat the complete three-invocation
sequence with a fresh cache for `tail` and for every other brief variant. The compile and
runtime passes are separate processes; the last two commands are independent runtime
processes for reproducibility. Using the compile-populated cache makes these runtime
passes cache-warm; report that fact and the concrete cache path. Capture each stdout/
stderr and exit status separately, preserving failures and CANN messages without parsing
them as harness facts. `MVP_SHAPES=all`/unset means primary then tail with the full
sequence per variant; an unknown value must fail.

Inside each process, the harness selects
`asctile.set_platform(asctile.Backend.Model, asctile.Platform.Ascend950PR_9599)` once.
Dry-run model setup/teardown messages are expected because platform selection initializes
the runtime; only unchanged output proves that no launch occurred.

## Compile-only and runtime checks

Per variant, in order:

1. **Compile-only:** use a fresh empty cache and `PYASC_DRY_RUN=1`. Exercise the actual
   JIT launch path. Require the protocol's compiled-kernel check, compiler memory
   metadata, `Launcher.check_memory_overflow`, and explicit consumption/capacity reporting
   for UB/L1/L0A/L0B/L0C/BT. C310 capacities are UB 253952, L1 524288, L0A 65536,
   L0B 65536, L0C 262144, BT 4096 bytes. Snapshot output before the dry-run call and
   require bit-identical full output storage afterward. Print `MEMORY_CHECK` and the
   literal `COMPILE_ONLY PASS ...` only after all checks pass.
2. **Runtime setup:** print complete `ENVIRONMENT`, including Python/executable, pyasc,
   asc/asctile module paths, torch, CANN home/runtime/simulator versions, backend/platform/
   arch, `runtime_is_model`, runtime SoC, dry-run/cache/PYTHONPATH values, task/variant/
   shape/dtype. Generate CPU inputs exactly once per variant in the task's seed/draw order.
3. **Guards and warmup:** allocate output within a guard-padded buffer with 32 fp16
   elements (64 bytes) on each side, fixed sentinels, and a pre-launch whole-buffer
   snapshot. Execute `rt.synchronize()` → one launch → `rt.synchronize()` and print
   `WARMUP 1 synchronized launch complete` once per variant.
4. **Correctness:** on that warmup output, run `torch.testing.assert_close` against the
   exact Design formula/comparison/tolerances and compare both guard regions bit-for-bit.
   Print `CORRECTNESS` with status, exact reference string, atol/rtol, max absolute error,
   per-side `output_guard_bytes=64`, guard status, variant and shape. Any mismatch fails
   the variant; do not proceed to report ticks for it.
5. **Configuration:** print `CONFIG` at least once in the runtime pass for each variant,
   with actual grid/launch bounds, tile family, loop/unroll order, tail handling, options,
   `ConstExpr`/runtime split, seed/draw order, cache path/warm state, and all choices a
   different correct implementation could make.

## Tick protocol

Only after warmup correctness and guards pass, keep the same tensors, kernel, grid and
options and perform exactly five measurements:

1. `rt.synchronize()`;
2. `start = rt.current_tick()`; fail if it is `None`;
3. launch the same kernel/configuration;
4. `rt.synchronize()`;
5. `end = rt.current_tick()`; fail if it is `None`;
6. `delta = end - start`; fail unless `delta > 0`, then print
   `TICK_SAMPLE <i> start=<s> end=<e> ticks=<d>` with variant/shape identifiers.

Print `TICK_SUMMARY` sorted-key JSON containing status, variant/shape,
`warmup_launches=1`, `measurements=5`, all five `raw_ticks`, `median_ticks`, `min_ticks`,
`max_ticks`, and `synchronize_before_and_after_each_launch=true`. All harness facts are
single-line `KEY <json>` (or the protocol's literal forms), `sort_keys=True`, flushed.
CANN's own `Total tick` and model messages are recorded, not parsed.

Run the complete runtime pass in two independent processes. Reproducible means both pass
correctness, both summaries contain five measurements, and medians are within **±1 tick**
(identical is expected; Task 1 reproduced raw deltas exactly). Preserve both raw sets and
report the median difference. A larger deviation is a reproducibility concern and the
variant is not `correct`; after verification validity/environment checks pass, record the
measurement failure as `run_failed`, never hide it through averaging or selective retries.

## Interpretation and failure handling

- Ticks are simulator ticks only, **never hardware latency**; never convert them to ns/ms.
- The interval surrounds the complete synchronized launcher path. The installed launcher
  performs argument copies, launch, synchronization and result copies, so this is a
  consistent whole-launch comparison, not proven isolated compute-only cycles.
- Task 1 proved one deterministic fp16 Vector configuration. It did not prove other
  shapes/dtypes, NPU behavior, or universal performance targets.
- Any check failure exits nonzero with a reason; do not swallow exceptions, silently retry,
  or print later `PASS` records that conceal it. Preserve the protocol outcome and raw
  evidence. Only a `correct`, reproducible variant supplies ticks for comparison.
