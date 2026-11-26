# pyasc-skills

A standalone three-skill project that helps an ordinary user create new AscTile/C310
operators through a production-quality `Design → Implement → Verify` pipeline.

The user supplies the operator semantics, shapes and dtypes. The agent owns routing,
residency, tiling and verification. No knowledge of Cube/Vector, MLIR, IR or generated
AscendC is required.

## Skills

| Skill | Stage | Responsibility |
|---|---|---|
| `asctile-design-operator` | Design | Turn a request into a complete brief with an exact supported template, structure, feasibility and correctness contract |
| `asctile-implement-operator` | Implement | Produce one algorithm-specific Python file with kernel, `launch` wrapper and `main` protocol orchestrator |
| `asctile-verify-operator` | Verify | Independently validate correctness, reproducibility and deliver the final report |

Pipeline: `Design → Implement → Verify`. Deliver is Verify's report, not a fourth skill.
There is no orchestrator.

## Supported MVP templates

| Template | Route | Contract |
|---|---|---|
| `vector_add_fp16` | `vector` | `out[i] = a[i] + b[i]`, fp16 contiguous 1-D |
| `gemm_fp16` | `cube` | Row-major fp16 `C = A @ B`, canonical K-split GEMM |
| `gemm_bias_relu_vector_fp16` | `cube_vector` | `C = relu(A @ B + bias[j])`, fp16, UB Vector epilogue |

Other operations and mechanisms are not silently claimed as supported. Real developer
experience is retained as **candidate practice** for later implementation and
verification. It is not treated as false merely because it has not yet been validated.

## Quick start

1. Use a supported Linux AArch64 or x86_64 target from the selected wheel manifest and a
   matching CPython/glibc environment. A dedicated venv is recommended, not required.
2. Install the package without co-installing `pyasc` and `pyasc-dev`:

   ```bash
   python -m pip install --pre pyasc-dev
   ```

   The package dependencies are `pybind11==2.10.3`, `numpy<2`, and `typing_extensions`.
   Kernel execution also needs a C++17 compiler (for example `g++`) and matching Python
   development headers.
3. Configure CANN and the simulator:

   ```bash
   source <CANN_INSTALL>/set_env.sh
   export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH:-}"
   ```

4. Expose the three directories under `skills/` to your agent harness, using the harness's
   normal skills-loading mechanism. If no such mechanism is available, point the agent at
   the repository root and let it read those skills directly.
5. Ask for a new operator in plain language (for example, `out = a + b` with fp16
   tensors). The agent will create `request.md`, `brief.md`, one algorithm file and
   `RESULT.md` through the pipeline.

If a prerequisite is missing, the agent reports it and stops without a result. It does not
install packages or modify CANN by itself. After you fix the issue and explicitly retry,
Verify re-runs preflight and resumes from the current artifact state.

## Runtime contract

- Current host backend: **Torch**.
- Platform: AscTile/C310, `Backend.Model`, `Ascend950PR_9599` simulator.
- No pyasc checkout, pytest, `conftest.py` or repo fixtures are required.
- The pipeline uses the user's currently configured environment. A fresh or isolated
  environment is not required.
- Preflight validates installed package, Torch, CANN, simulator, backend/SoC/arch and
  compiler prerequisites. Missing dependencies are reported to the user; the agent does
  not install packages or modify CANN by itself.
- After remediation and an explicit retry, the pipeline re-runs preflight and resumes
  from the current artifact state.

## Candidate artifacts

```text
<task-workdir>/
  request.md
  brief.md
  DESIGN_REPORT.md
  <algorithm>.py
  RESULT.md
  DEFECT_REPORT.md
  runs/
```

For production templates the algorithm file is exactly one of `vector_add.py`, `gemm.py`,
or `gemm_bias_relu.py`, with this structure. Evaluation-only candidates use their
candidate-specific algorithm filename and the same structure.

```text
@asctile.jit kernel
        ↓
launch(...) → kernel[grid](...)
        ↓
main() → launch(...)
        ↓
if __name__ == "__main__": main()
```

Verify owns the final `RESULT.md` and verification evidence; it never edits the
candidate code. Simulator ticks are simulator-relative evidence, never hardware latency.

## Acceptance evidence

All three published templates are **delivered** through the full `asctile-model-v1`
protocol on the configured `Backend.Model` / `Ascend950PR_9599` simulator: an independent
Verify oracle regenerates inputs/reference, each variant uses a fresh compile cache and two
independent cache-warm runtime processes with five synchronized samples, 64-byte output
guards, a zero illegal-DMA scan, immutable hash-bound manifests, and a two-process tick
median difference within ±1. Acceptance also covers arbitrary working directories, the
known/unknown variant selector, an intentional-incorrect-output rejection, the
environment-failure lifecycle, fresh-agent state recovery (caps, precedence, quarantine),
and tile/epilogue boundary cases. The same three templates were additionally delivered by a
**second, different model** in fresh clean-room runs, confirming the guidance is not
model-specific.

The consolidated record lives in the external evaluation workspace at
`mvp-publication-v9/acceptance-results.md` (with per-task `RESULT.md`, `runs/` manifests and
oracle reports). One degenerate boundary case (a 3-element `tile=1` micro-kernel) is recorded
as a measurement-reproducibility limitation with correctness validated; it is not a
production correctness failure.

**Limitations.** Simulator ticks are whole-launch, simulator-relative evidence — never
hardware latency, never converted to time, and no runtime speedup is claimed. This refactor
improves reliability, portability and ordinary-user accessibility, not operator runtime.

## Scope and roadmap

The first MVP publishes only the three templates above. Additional operator families,
NPU, debugging existing operators and Tune remain later extensions. Every extension must
pass the same Implement → Verify acceptance path before it is called supported.
