# Implement reference: canonical standalone harness

This reference is loaded on every route together with the selected route reference. It is
the authoring recipe for the one algorithm-specific Python file. It contains no IR,
MLIR or AscendC debugging workflow.

## File and symbol contract

A production brief chooses exactly one of:

| Template | `artifact_file` | `kernel_symbol` |
|---|---|---|
| `vector_add_fp16` | `vector_add.py` | `vector_add` |
| `gemm_fp16` | `gemm.py` | `gemm` |
| `gemm_bias_relu_vector_fp16` | `gemm_bias_relu.py` | `gemm_bias_relu` |

An evaluation-only candidate brief instead chooses a candidate-specific `artifact_file`
and `kernel_symbol` appropriate to the candidate algorithm (for example `softmax.py` /
`softmax`). It remains one self-contained Python file with kernel, `launch` and `main`.

The file contains exactly these roles in this order:

1. `@asctile.jit` kernel;
2. `def launch(...)`;
3. `def main()`;
4. `if __name__ == "__main__": main()`.

No `entry.py`, `kernel.py`, `host.py` or sibling candidate module is allowed. Additional
local helpers, if needed, remain inside the same file and do not introduce another
execution entry point.

## Exact imports

```python
from asc.experimental import asctile
from asc.runtime.launcher import Launcher, get_platform_info
from asc.lib import runtime as rt
import torch
```

Torch is the canonical host backend for this iteration. Do not claim that NumPy is
unsupported; it is simply not the canonical accepted host backend. `.npy` files, when
used for captured outputs, are an evidence serialization format, not a second host
backend.

## `launch(...)` contract

`launch` is the operator-facing wrapper. It accepts runtime tensors/scalars/options and
owns the complete launch configuration:

- prepare the output tensor with the required dtype and true logical extents;
- choose or validate tile values, grid/core count and compile/runtime parameters;
- pack parameters in the kernel's declared order;
- perform every JIT launch through `kernel_symbol[grid](...)`;
- return the result to `main`.

No direct JIT launch is allowed outside `launch`. `main` must not bypass the wrapper.

## `main()` contract

`main` is the protocol-facing orchestrator. It owns:

- environment preflight and capability revalidation;
- variant selection;
- deterministic input and reference generation;
- guard allocation and snapshots;
- dry-run, runtime and tick protocol;
- `CONFIG`, `RESULT.md` and user-facing reporting;
- calls to `launch` for execution.

`main` runs only through the standard main guard.

## Environment preflight

Use the user's actual configured environment. Do not require a fresh or isolated
environment and do not install packages or modify CANN yourself.

Record in `ENVIRONMENT` and `runs/revalidation.json`:

- supported target from the selected wheel manifest (Linux AArch64/x86_64, matching
  CPython/glibc), plus the actual release host and interpreter;
- package dependencies `pybind11==2.10.3`, `numpy<2`, `typing_extensions`;
- C++17 compiler (verified example: `g++`) and matching Python development headers;
- Python executable and version;
- `pyasc-dev` or `pyasc` distribution/version; the two must not be co-installed;
- actual `asc`/`asctile` module paths;
- Torch version;
- CANN home/runtime/simulator versions;
- backend, platform, compilation arch, `rt.is_model()` and pyasc-reported SoC;
- current `PYASC_DRY_RUN`, `PYASC_CACHE_DIR`, `PYTHONPATH` and cache-warm state.

Minimal capability checks:

1. `from asc.experimental import asctile` and all protocol imports resolve from the
   installed package.
2. `set_platform(Backend.Model, Platform.Ascend950PR_9599)` works; `rt.is_model()` is
   true; pyasc-reported SoC is 9599; compilation arch is c310.
3. Runtime capacities match the C310 contract: UB 253952, L1 524288, L0A/L0B 65536,
   L0C 262144, BT 4096 bytes.
4. `Launcher.get_core_num()` returns a positive value; never bake that value into the
   kernel.
5. `rt.synchronize` and `rt.current_tick` exist; Model tick is non-None.
6. Torch tensor launch works in the current environment.
7. A minimal supported kernel passes real dry-run compilation, exposes memory metadata
   and passes `Launcher.check_memory_overflow`.

For CANN setup, use the user's configured environment. If setup instructions are needed,
use generic placeholders: `source <CANN_INSTALL>/set_env.sh` and add
`$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib` to `LD_LIBRARY_PATH`.
SoC/arch come from pyasc, not from CANN driver log strings.

## Detect, report, remediate and resume

On a failed capability:

1. Record the failed check and actual evidence.
2. Report the exact missing dependency or wrong configuration and the minimal
   user-actionable fix.
3. Do not install packages, modify CANN or silently switch backend.
4. Publish `status=environment_failed` without a correctness or performance result.

After the user remediates and explicitly requests a retry, only Verify re-runs preflight
and revalidation. The failed RESULT blocks ordinary defect/RESULT routing. On success,
Verify **first publishes a revision-and-hash-matching successor RESULT**, then archives the
old environment-failure RESULT and routes the successor's stage. If the successor cannot be
published or preflight fails again, the failed RESULT remains current, no result/ticks are
emitted, and old defects do not route. Re-evaluate environment identity and baseline
comparability: old ticks are incomparable by default after package/runtime/CANN drift.

## Variants, guards and dry run

Canonical invocation pattern (replace only the exact brief-selected algorithm filename
and the brief's actual variant names; no host-specific paths):

```bash
candidate=/absolute/path/to/task-workdir   # no host-specific path is embedded in the skills
algorithm_file=<artifact_file-from-brief> # e.g. vector_add.py
variants=(primary tail)                   # exact names from the brief
for variant in "${variants[@]}"; do
  cache=$(mktemp -d /tmp/opencode/asctile-mvp.XXXXXX)
  env -u PYTHONPATH PYASC_DRY_RUN=1 PYASC_CACHE_DIR="$cache" \
    ASCTILE_VARIANT="$variant" python -I "$candidate/$algorithm_file"
  env -u PYTHONPATH -u PYASC_DRY_RUN PYASC_CACHE_DIR="$cache" \
    ASCTILE_VARIANT="$variant" python -I "$candidate/$algorithm_file"
  env -u PYTHONPATH -u PYASC_DRY_RUN PYASC_CACHE_DIR="$cache" \
    ASCTILE_VARIANT="$variant" python -I "$candidate/$algorithm_file"
done
```

Each variant has its own fresh compile cache and two independent compile-cache-warm
runtime processes. `python -I` proves there are no sibling imports. The same sequence is
used for smoke and authoritative Verify; only Verify issues the verdict.

- `ASCTILE_VARIANT=<brief-variant-name>` runs exactly that variant; unset runs all
  variants in brief order. An unknown name exits nonzero with no result/ticks.
- Every variant gets a fresh compile cache for compile-only and two independent runtime
  processes. Record cache paths and warm state truthfully.
- Allocate output with **64-byte guards on both sides**, deriving the element count from
  output dtype. Fill fixed sentinels and snapshot the whole buffer before launch.
- Under `PYASC_DRY_RUN=1`, exercise the real compile/launch path, run the six-bank memory
  check and prove the entire output allocation unchanged. CANN setup/teardown messages
  are not launch evidence.
- A failure in one variant prevents an all-variant result from being presented as
  delivered. Never emit `PASS` after a failure.

## Correctness and measurements

- Generate each variant's inputs exactly as the brief states: Torch generator, seed,
  distribution/range and operand draw order are recorded.
- Compute the reference in fp32 and apply the brief's final cast. Record the exact
  reference string, tolerances, NaN/Inf policy and the dtype used for max error.
- Use `torch.testing.assert_close` with the brief's `atol`/`rtol`; report max absolute
  error and guard status. An incorrect output never receives ticks or a `correct` verdict.
- After one synchronized warmup used for correctness, collect exactly five synchronized
  samples with identical tensors, kernel, grid and options.
- For each sample: synchronize, read `rt.current_tick()` (fail on None), launch through
  `launch`, synchronize, read tick again, require positive delta and print raw
  `TICK_SAMPLE` plus `TICK_SUMMARY`.
- Repeat the complete runtime pass in a second independent process. Medians must differ
  by at most ±1 tick on the validated environment; a larger deviation is a measurement
  `run_failed`, not something to average away.
- Ticks are simulator-relative, whole-launch evidence, never hardware latency.

## Immutable evidence and independent oracle

Each run uses a unique `runs/<run_id>/` directory. `run_id` includes candidate SHA256
prefix, brief/protocol revisions, variant and process/run id. Its `manifest.json` binds
full candidate hash, revisions, variant, process/run id and every captured output path.
Overwriting an existing run is forbidden.

Verify owns `runs/verify_oracle.py`. It regenerates inputs from `request.md` and
`brief.md`, computes the reference and compares the manifest-bound captured outputs. It
also audits the embedded harness source. Verify never edits the candidate file; its hash
is recorded before and after verification.

## Failure and reporting discipline

- Any failure exits nonzero with a reason; never swallow exceptions or silently retry.
- Preserve raw logs, failure signatures and environment evidence.
- `MemoryOverflowError` → reduce tile/unroll;
- 32-byte alignment error → fix local last dimension or valid extent;
- scalar-clamp tracing issue → use the installed scalar-min idiom proven by the active
  probe;
- illegal MTE/DMA diagnostics → check logical tiles and exact extents;
- manual split error → check SplitByM shape, origin and per-sub-block offsets.
- Do not respond to these failures with IR/MLIR/AscendC inspection.
