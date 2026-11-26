# Verify reference: simulator run and tick protocol

This is the self-contained Model protocol `asctile-model-v1` for `Backend.Model` /
`Platform.Ascend950PR_9599` (arch c310). It is shared by all routes. No external
protocol document, host-specific path or validator command is required.

## Environment and preflight

Use the user's actual configured environment. Do not require a clean or isolated
environment. Do not install packages or modify CANN yourself.

The candidate and Verify oracle use the installed package and Torch host backend.
Record actual versions and capabilities in `ENVIRONMENT`, `runs/revalidation.json` and
`RESULT.md`:

- supported target from the selected wheel manifest (Linux AArch64/x86_64, matching
  CPython/glibc), actual release host/interpreter;
- package dependencies `pybind11==2.10.3`, `numpy<2`, `typing_extensions`;
- C++17 compiler (verified example: `g++`) and matching Python development headers;
- Python/executable, `pyasc-dev` or `pyasc` distribution/version, module paths, Torch,
  CANN home/runtime/simulator versions;
- backend/platform/arch, `rt.is_model()`, pyasc-reported SoC;
- dry-run/cache/PYTHONPATH values, task/variant/shape/dtype and cache-warm state.

Required capability facts:

1. `from asc.experimental import asctile` and protocol imports resolve from the installed
   package.
2. `set_platform(Backend.Model, Platform.Ascend950PR_9599)` works; `rt.is_model()` is
   true; pyasc-reported SoC is 9599; compilation arch is c310.
3. Runtime capacities match C310: UB 253952, L1 524288, L0A/L0B 65536, L0C 262144,
   BT 4096 bytes.
4. `Launcher.get_core_num()` is positive; the value is never baked into the kernel.
5. `rt.synchronize`/`rt.current_tick` exist and Model tick is non-None.
6. Torch tensor launch works in the current environment.
7. A minimal supported kernel performs real dry-run compilation, exposes memory metadata
   and passes `Launcher.check_memory_overflow`.

On a failed capability, publish `status=environment_failed` without a correctness or
performance result. Report the exact missing dependency/configuration and minimal
user-actionable fix. This RESULT blocks ordinary defect/RESULT routing. After user
remediation and an explicit retry, only Verify re-runs preflight/revalidation. On success
Verify **first publishes a revision-and-hash-matching successor RESULT**, then archives the
failed RESULT and routes the successor's stage. If the successor cannot be published or
preflight fails again, the failed RESULT remains current, no result/ticks are emitted, and
old defects do not route. Re-evaluate environment identity and baseline comparability after
drift.

## Per-variant compile/runtime sequence

Each brief variant is selected by `ASCTILE_VARIANT=<brief-variant-name>`. Unset means all
variants in brief order. Unknown names fail nonzero without a result or ticks.

For every variant:

1. Create a fresh compile cache and run compile-only with `PYASC_DRY_RUN=1`.
2. Run the complete runtime pass in one process.
3. Run the same complete runtime pass in a second independent process.

The two runtime processes use the compile-populated cache and are cache-warm; record the
cache path and that fact. Preserve each invocation's stdout/stderr and exit status
separately. CANN messages are preserved, not parsed as harness facts.

The same mechanical sequence may be used by Implement for smoke, but only Verify issues
an authoritative verdict.

## Compile-only and runtime checks

Per variant, in order:

1. **Compile-only:** exercise the actual JIT launch path. Require real compilation,
   compiler memory metadata, `Launcher.check_memory_overflow` and explicit
   consumption/capacity reporting for UB/L1/L0A/L0B/L0C/BT. Snapshot output before the
   dry-run call and require bit-identical full output storage afterward. Print
   `MEMORY_CHECK` and `COMPILE_ONLY` only after those checks pass.
2. **Runtime setup:** print complete `ENVIRONMENT`, select Model/9599 once per process,
   generate CPU inputs exactly once per variant in the brief's seed/draw order.
3. **Guards and warmup:** allocate output with 64-byte guards per side, fixed sentinels
   and a pre-launch whole-buffer snapshot. Execute `rt.synchronize()` → one launch through
   `launch` → `rt.synchronize()` and print `WARMUP` once per variant.
4. **Correctness:** on that warmup output, use `torch.testing.assert_close` against the
   exact Design reference/tolerances and compare both guard regions bit-for-bit. Print
   `CORRECTNESS` with status, reference, atol/rtol, max absolute error, per-side
   `output_guard_bytes=64`, guard status, variant and shape. Any mismatch fails the
   variant; do not proceed to ticks.
5. **Configuration:** print `CONFIG` at least once per variant with actual grid/launch
   bounds, tile family, loop/unroll order, tail handling, options, ConstExpr/runtime split,
   seed/draw order, cache path/warm state and all choices another correct implementation
   could make.

## Tick protocol

Only after warmup correctness and guards pass, keep the same tensors, kernel, grid and
options and perform exactly five measurements:

1. `rt.synchronize()`;
2. `start = rt.current_tick()`; fail if None;
3. launch the same kernel/configuration through `launch`;
4. `rt.synchronize()`;
5. `end = rt.current_tick()`; fail if None;
6. `delta = end - start`; fail unless `delta > 0`, then print raw `TICK_SAMPLE` with
   variant/shape identifiers.

Print `TICK_SUMMARY` as sorted-key JSON containing status, variant/shape,
`warmup_launches=1`, `measurements=5`, all five raw ticks, median/min/max and
`synchronize_before_and_after_each_launch=true`. All machine-readable harness facts are
single-line `KEY <json>`, flushed. CANN's own messages are recorded, not parsed.

Run the complete runtime pass in two independent processes. Reproducible means both pass
correctness, both summaries contain five measurements, and medians are within ±1 tick.
Preserve both raw sets and report the median difference. A larger deviation is a
reproducibility concern and the variant is not `correct`; after verification validity and
environment checks pass, record the measurement failure as `run_failed`, never hide it
through averaging or selective retries.

## Independent oracle and immutable evidence

Verify creates `runs/verify_oracle.py` outside the candidate code. It regenerates inputs
from `request.md` and `brief.md`, computes the reference and compares the captured outputs.
It also audits the embedded harness source. This is an independent oracle, not a second
kernel harness.

Every run uses a unique `runs/<run_id>/` and `manifest.json` binding full candidate hash,
brief/protocol revisions, variant, process/run id and every captured output path.
Overwriting an existing run is forbidden. The oracle and final report validate this
binding. Candidate hash is recorded before and after Verify and must not change.

## Interpretation and failure handling

- Ticks are simulator ticks only, never hardware latency; never convert them to ns/ms.
- The interval surrounds the complete synchronized launcher path, not proven compute-only
  cycles.
- Any check failure exits nonzero with a reason; do not swallow exceptions, silently retry
  or print later `PASS` records that conceal it.
- Preserve protocol outcome and raw evidence. Only a correct, reproducible variant supplies
  ticks for comparison.
- A protocol/artifact violation is `invalid_submission`; missing proven environment is an
  environment deviation, not a candidate performance result.
