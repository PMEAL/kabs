# Warp backend readiness and development plan

## Objective

Finish the KABS integration with XLB's NVIDIA Warp backend and determine
whether it provides a useful single-GPU CUDA alternative to the Taichi and
JAX implementations.

The guiding constraint is to use XLB and Warp through their public APIs. KABS
should not modify or duplicate XLB's collision, streaming, equilibrium, or
boundary-condition internals.

## Why Warp is worth evaluating

XLB presents Warp as its high-performance single-GPU backend. Its Warp
Navier-Stokes stepper is already structured as a fused device kernel that
performs streaming, boundary-condition handling, macroscopic reconstruction,
equilibrium calculation, and collision. This execution model may suit CUDA
better than the JAX path observed in the initial KABS/PoreStore tests.

Official references:

- [XLB repository and installation instructions](https://github.com/Autodesk/XLB)
- [XLB macroscopic operator](https://github.com/Autodesk/XLB/blob/main/xlb/operator/macroscopic/macroscopic.py)
- [NVIDIA Warp installation guide](https://github.com/NVIDIA/warp/blob/main/docs/user_guide/installation.rst)
- [NVIDIA Warp device documentation](https://nvidia.github.io/warp/latest/user_guide/runtime.html#devices)

## Current KABS status

The public API already accepts the Warp selection:

```python
result = kabs.solve_flow(
    image,
    backend="xlb",
    compute_backend="warp",
    collision_model="srt",
)
```

The following pieces are already present:

- `solve_flow(..., backend="xlb", compute_backend="warp")` dispatch.
- XLB initialization with `ComputeBackend.WARP`.
- A D3Q19, FP32/FP32 velocity set.
- The XLB `IncompressibleNavierStokesStepper` using BGK/SRT collision.
- Pressure-boundary construction for all three flow directions.
- Full-way bounce-back boundaries for solid voxels.
- XLB population-field allocation and buffer swapping.
- Common `FlowResult` and permeability post-processing interfaces.

The core XLB timestep and KABS integration are now present. On the Windows
GTX development machine, KABS has native Warp result extraction, device-side
velocity convergence measurements, lazy Taichi imports, and an isolated CUDA
numerical validation suite. The remaining work is synchronized benchmarking,
setup-scalability measurement, RunPod validation, and PoreStore integration.

### Windows validation checkpoint (2026-08-27)

Verified environment:

- NVIDIA GeForce GTX 1660 Ti Max-Q, 6 GiB VRAM, compute capability `sm_75`.
- NVIDIA driver 538.92, reporting CUDA 12.2 support.
- Python 3.12.5, XLB 0.3.1, Warp 1.10.0, Taichi 1.7.4.
- Warp's bundled CUDA toolkit is 12.8 and `cuda:0` is the preferred device.

Completed implementation and validation:

- Native XLB Warp `Macroscopic` reconstruction into Warp grid fields.
- Explicit synchronization followed by direct Warp-to-NumPy final extraction.
- No `warp_array_to_jax` or JAX macroscopic operator in the Warp path.
- A KABS-owned criterion-aware Warp reduction that retains the previous
  velocity field on-device only when needed and transfers at most five FP32
  convergence scalars per check.
- Backend-neutral `FlowResult` and lazy package/Taichi solver imports.
- Fresh-process CUDA validation for all flow directions, pressure-face edges
  and corners, an internal obstacle, analytical permeability, Taichi SRT
  agreement, XLB/JAX agreement, and fixed-step versus early convergence.

Representative four-cylinder results:

- analytical permeability: 6.28319 lu^2;
- XLB Warp: 6.40577 lu^2 (1.95% analytical error);
- Taichi SRT/CUDA: 6.44774 lu^2 (0.65% difference from Warp);
- XLB/JAX: 6.40505 lu^2 (0.011% difference from Warp);
- all three implementations stopped at step 1000 for the validation case.

Merge-gate results:

- ordinary suite: 180 passed, 16 CUDA-only tests skipped;
- CUDA-specific suite: 26 passed;
- all files changed for the Warp work pass Ruff.

## Confirmed blockers and risks

### 1. Native result extraction (resolved in KABS)

KABS previously imported `warp_array_to_jax` from `xlb.utils`. That helper is
not included in the installed `xlb==0.3.1` wheel, although a version exists on
XLB's current development branch. Consequently, the old implementation could
not return a `FlowResult` from a Warp solve with the pinned dependency set.

KABS no longer depends on this conversion. It allocates Warp density and
velocity fields through XLB's public grid API, invokes the native Warp
`Macroscopic` operator, synchronizes, and copies final fields directly to
NumPy.

### 2. Warp-native macroscopic reconstruction (resolved)

KABS constructs `Macroscopic` with `ComputeBackend.JAX`, even when the
timestep backend is Warp. At each convergence interval it attempts this path:

```text
Warp populations -> JAX array -> JAX macroscopic operator -> host scalars
```

The Warp path now constructs `Macroscopic` with `ComputeBackend.WARP` and
supplies preallocated `rho` and `u` fields. No JAX array or JAX macroscopic
operator is used after selecting Warp.

### 3. Device-side convergence monitoring (resolved for velocity metrics)

The current convergence reducer is implemented with `jax.jit` and
`jax.device_get`. A finished Warp backend needs either:

1. a Warp-native reduction and velocity snapshot, transferring only two
   scalars to the host at each check; or
2. as an initial correctness implementation, a synchronized velocity copy at
   each relatively infrequent convergence interval.

KABS implements the first option. The reduction and snapshot mechanism is
separated from the stopping policy so future permeability-change metrics can
be added without coupling their calculation to the current velocity rule.

### 4. Runtime isolation (Taichi import resolved; XLB state risk remains)

A minimal Warp smoke run on macOS aborted in LLVM initialization after KABS
had imported Taichi. KABS previously imported the Taichi solver eagerly, even
when only the XLB backend was requested.

This may be platform-specific, but it must be checked on Windows and Linux
CUDA. The robust KABS-side solution is lazy backend importing:

- Move `FlowResult` and shared constants into a backend-neutral module.
- Import `SinglePhaseSolver` only when the Taichi backend is selected.
- Avoid importing `_single_phase_solver` eagerly from `kabs.__init__`.

KABS now uses backend-neutral result structures and lazy-loads
`SinglePhaseSolver`; importing KABS or running XLB/Warp does not load Taichi.

XLB 0.3.1 nevertheless retains process-global operator and boundary state.
Switching XLB between JAX and Warp, or running differently configured Warp
boundary cases sequentially in one interpreter, produced invalid results on
Windows. Numerical validation therefore launches each physical case in a
fresh subprocess. Production integration must either preserve one stable XLB
configuration per process or formalize subprocess isolation.

### 5. CUDA test coverage (Windows complete; RunPod pending)

CUDA tests remain opt-in with `KABS_TEST_WARP_CUDA=1`. They now validate native
extraction, device convergence, permeability, boundary behavior, all three
directions, obstacles, analytical behavior, and agreement with Taichi SRT and
XLB/JAX. Ordinary GitHub CI still does not exercise Warp CUDA, and the suite
has not yet run on RunPod/Linux.

A degenerate one-voxel pore channel placed entirely on a pressure-face corner
became non-finite under XLB/Warp 0.3.1. Normal pressure faces containing
interior, edge, and corner pore nodes pass in all three directions.

### 6. No synchronized Warp benchmark exists

Warp launches CUDA work asynchronously. Valid timing requires a warm-up and
`wp.synchronize()` immediately before starting and after stopping each timed
sample. Setup/JIT time, fixed-step throughput, and time-to-convergence must be
reported separately.

### 7. Solid-boundary setup may consume substantial host memory

KABS currently converts every solid coordinate into three Python lists before
constructing XLB's full-way bounce-back boundary. This is acceptable for
small tests but may become expensive for production volumes. It should be
measured separately from warmed solver throughput before any redesign is
considered.

## Windows GTX development machine

A Windows x86-64 laptop with an NVIDIA GTX discrete GPU is suitable for Warp
development and functional CUDA validation, subject to the checks below.

NVIDIA's current Warp documentation lists these relevant requirements:

- Python 3.10 or newer.
- Windows x86-64 is supported.
- A CUDA-capable NVIDIA GPU; the documented minimum is GeForce GTX 9xx.
- CUDA 12 Warp packages require an NVIDIA driver version of at least 525.
- Prebuilt Warp wheels include the required CUDA runtime components, so a
  separately installed CUDA Toolkit is normally unnecessary.

Four GB of VRAM is sufficient for small development cases and the current
scale-2 benchmark. Six to eight GB is preferable for medium-sized images. The
laptop is appropriate for correctness and comparative development testing;
RunPod remains the authoritative production-performance environment.

### Initial machine checks

Run the following in PowerShell:

```powershell
nvidia-smi
```

Record:

- GPU model
- driver version
- total and free VRAM
- any active GPU processes

Clone or update KABS, select the `finalizing-warp-option` branch, and create a
fresh environment:

```powershell
git fetch origin
git switch finalizing-warp-option
uv sync --group dev
```

Verify that Warp detects CUDA:

```powershell
uv run python -c "import warp as wp; wp.init(); wp.print_diagnostics()"
```

Also run:

```powershell
uv run python -c "import warp as wp; wp.init(); print(wp.get_devices()); print('CUDA devices:', wp.get_cuda_device_count())"
```

The output must include a device such as `cuda:0`, and the CUDA-device count
must be at least one. If Warp reports only `cpu`, update the NVIDIA driver
before changing KABS code.

For consistent laptop measurements:

- Connect AC power.
- Select the high-performance NVIDIA GPU in Windows Graphics Settings or the
  NVIDIA control panel.
- Close other GPU workloads.
- Record the Windows power mode and GPU driver with each benchmark.

Native Windows is the simplest initial target. WSL2 is not required for Warp
and adds another CUDA configuration layer.

## Development plan

### Issue 1: Establish a CUDA smoke-test environment

Status: complete on the Windows GTX machine; repeat on RunPod.

Goal: prove that the supported packages can compile and launch XLB Warp code
on the GTX GPU before restructuring KABS.

Tasks:

- Capture `nvidia-smi` and `wp.print_diagnostics()` output.
- Confirm the versions of Python, XLB, Warp, and KABS.
- Run a minimal standalone XLB D3Q19 Warp stepper outside KABS.
- Synchronize after the step and verify that the population arrays remain
  finite.
- Run the same minimal call after importing KABS to expose any Taichi/Warp
  runtime conflict.

Acceptance criteria:

- Warp lists `cuda:0`.
- A native XLB Warp step compiles and executes.
- Any KABS import conflict is reproduced with a small deterministic command.

### Issue 2: Make the KABS Warp path native end to end

Status: complete.

Goal: return a valid `FlowResult` without converting Warp populations to JAX.

Tasks:

- Construct XLB `Macroscopic` with `ComputeBackend.WARP`.
- Allocate `rho` and `u` using the XLB Warp grid/public field API.
- Run fixed-step solves initially with `velocity_tol=None`.
- Invoke the native macroscopic operator after the final step.
- Call `wp.synchronize()` before host extraction.
- Convert `rho` and `u` directly to NumPy using Warp's public array API.
- Preserve KABS array shapes, dtypes, directions, viscosity, iteration count,
  and `collision_model="srt"` metadata.
- Remove `warp_array_to_jax` from the Warp execution path.

Acceptance criteria:

- A small Warp solve returns finite `rho` and velocity fields.
- No JAX array or JAX macroscopic operator is used after Warp selection.
- `compute_permeability()` accepts the result.
- The JAX backend remains unchanged and its tests continue to pass.

### Issue 3: Isolate backend imports

Status: complete for Taichi imports. Fresh-process isolation is still used for
XLB backend/configuration changes because of XLB 0.3.1 global state.

Goal: ensure selecting XLB/Warp does not initialize Taichi unnecessarily.

Tasks:

- Move backend-neutral result structures and constants out of the Taichi
  implementation module where necessary.
- Lazy-import Taichi solver classes only in the Taichi dispatch branch.
- Update package exports without breaking the public API.
- Add subprocess tests, since native-runtime aborts cannot be caught reliably
  by an in-process `pytest.raises` assertion.

Acceptance criteria:

- Importing and running the Warp backend does not import or initialize Taichi.
- Taichi-only workflows remain unchanged.
- A subprocess Warp smoke test exits successfully on Windows and RunPod.

### Issue 4: Implement Warp convergence monitoring

Status: complete for the current velocity-change semantics. The implementation
is structured so permeability-change measurements can be added separately.

Goal: reproduce KABS convergence semantics without JAX and without copying a
full velocity field at every check.

Recommended production design:

- Keep current and previous velocity arrays on the Warp device.
- Use a small KABS-owned Warp reduction kernel to calculate:
  - `sum(abs(u_current))`
  - `sum(abs(u_current - u_previous))`
- Copy only those two scalars to the host.
- Copy or swap the device velocity snapshot after each check.

This is KABS orchestration around XLB output arrays; it does not alter XLB's
solver internals.

Acceptance criteria:

- Warp and JAX convergence criteria agree within floating-point tolerance.
- Warp stops at the same logging interval as the reference implementation.
- Only scalar-sized host transfers occur during periodic checks.
- `velocity_tol=None` still performs no convergence reduction.

### Issue 5: Add CUDA numerical validation

Status: complete on Windows except for the documented degenerate one-voxel
corner channel; repeat the suite on RunPod/Linux CUDA.

Goal: establish that Warp produces the expected porous-media physics.

Required cases:

- Tiny smoke geometry with finite fields and positive axial flow.
- Pressure-driven flow in `x`, `y`, and `z`.
- Pore voxels on pressure-face edges and corners.
- Internal full-way bounce-back obstacles.
- Four-cylinder analytical permeability case.
- Direct agreement with Taichi SRT/BGK.
- Direct agreement with XLB JAX where practical.
- Early convergence and fixed-step execution.

Record for each comparison:

- permeability in lattice units
- Darcy velocity
- mean density
- iteration count
- convergence criterion
- maximum or normed field differences where useful

Initial tolerances should match the existing XLB-versus-Taichi tests. Tighten
them only after observing repeatable CUDA results.

### Issue 6: Build a reproducible Warp benchmark

Status: not started.

Goal: compare warmed solver performance fairly on the GTX laptop and RunPod.

Timing rules:

- Compile every required kernel before collecting a sample.
- Call `wp.synchronize()` before and after each timed region.
- Exclude result extraction from fixed-step MLUPS.
- Report result extraction and setup/JIT separately.
- Use multiple repeats and report the median plus raw samples.
- Do not compare a cold Warp run with a warmed Taichi or JAX run.

Required measurements:

- fixed-step seconds and total MLUPS
- time to convergence
- convergence iterations
- cold setup/JIT time
- final extraction time
- peak allocated GPU memory if available
- permeability and its difference from Taichi SRT

Test at least:

1. the standard scale-2 benchmark (600,000 lattice sites);
2. a larger case that exercises the GTX memory bandwidth; and
3. the representative PoreStore production image on RunPod.

Compare:

- optimized Taichi SRT
- XLB Warp
- XLB JAX, if a correct CUDA JAX environment is available

### Issue 7: Measure setup scalability

Status: not started.

Goal: determine whether Python solid-coordinate construction limits real
PoreStore images.

Measure separately:

- conversion of the input image to the solid mask
- `np.where` and Python-list construction
- XLB boundary-mask preparation
- device-field allocation
- kernel compilation

Only redesign boundary ingestion if measurements show it is material. Avoid
changing XLB internals speculatively.

## CUDA test command convention

The existing optional Warp test uses `KABS_TEST_WARP_CUDA=1`. In PowerShell:

```powershell
$env:KABS_TEST_WARP_CUDA = "1"
uv run pytest -q tests/test_convergence_xlb.py
```

Keep CPU/JAX tests in ordinary CI and mark only tests that truly require CUDA.
The current CUDA-specific merge gate is:

```powershell
$env:KABS_TEST_WARP_CUDA = "1"
uv run pytest -q tests/test_backend_import_isolation.py tests/test_convergence_xlb.py tests/test_warp_cuda_validation.py
```

## RunPod benchmark record

For every authoritative run, store:

- KABS commit
- XLB version or exact Git commit
- Warp version
- Python version
- operating system
- GPU model and VRAM
- NVIDIA driver
- CUDA runtime reported by Warp
- image shape and porosity
- viscosity, relaxation time, tolerance, and logging interval
- raw timing samples
- permeability and iteration count

Do not change XLB versions between backend comparisons without recording that
fact. The latest PyPI release is currently `0.3.1`, while XLB's development
branch contains API additions not present in that wheel.

## Definition of ready

The Warp backend is ready for normal KABS/PoreStore use when all of the
following are true:

- It runs end to end on Windows CUDA and RunPod CUDA without JAX conversion.
- It does not initialize Taichi in the Warp process.
- All three flow directions pass numerical tests.
- Permeability agrees with Taichi SRT within the established tolerance.
- Device-side convergence agrees with the reference implementation.
- Fixed-step timing uses explicit Warp synchronization.
- Performance is repeatably competitive with optimized Taichi SRT on the
  intended production workload.
- Peak GPU memory and maximum practical image size are documented.
- PoreStore explicitly records `backend="xlb"`, `compute_backend="warp"`,
  and `collision_model="srt"` with each result.

## Recommended next action

Merge the validated Warp integration checkpoint into `dev`, then build the
synchronized benchmark harness on the Windows laptop. Use explicit Warp
warm-up and synchronization to measure setup/JIT, fixed-step MLUPS, final
extraction, time-to-convergence, and memory separately. Once the harness is
stable, run the authoritative production-scale comparison and memory study on
RunPod/Linux CUDA, followed by setup-scalability measurement and PoreStore
metadata integration.
