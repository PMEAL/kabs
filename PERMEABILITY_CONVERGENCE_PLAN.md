# Backend-Neutral Permeability Convergence Plan

## Status

Deferred until the Taichi, XLB-JAX, and XLB-Warp backends are all functioning
and have stable state/macroscopic-field interfaces.

An initial Taichi-only prototype was deliberately abandoned. The convergence
idea should be implemented once as a backend-neutral policy, with small
backend-native device reducers, rather than growing separate stopping logic in
each solver.

## Objective

Reduce time-to-permeability by stopping when the calculated permeability has
stabilized, while retaining safeguards against a misleading scalar plateau.

The existing convergence check measures the relative change of the entire
velocity field. That is conservative when permeability is the only required
quantity. However, permeability alone is a single volume-averaged scalar and
can appear stable while local velocities are still redistributing or mass is
still accumulating. The proposed stopping policy therefore combines three
relative criteria:

1. Permeability change
2. Velocity-field change
3. Inlet/outlet mass-flux imbalance

All three criteria must pass on two consecutive checks.

## Proposed Public API

```python
result = solve_flow(
    im,
    direction="x",
    backend="taichi",       # or "xlb"
    compute_backend="jax",  # or "warp" for XLB
    convergence="permeability",
    k_tol=5e-3,
    velocity_tol=1e-2,
    flux_tol=5e-3,
    convergence_every=100,
)
```

The default tolerances correspond to:

- `k_tol=5e-3`: permeability changes by less than 0.5%.
- `velocity_tol=1e-2`: the velocity field changes by less than 1%.
- `flux_tol=5e-3`: inlet/outlet mass-flow mismatch is below 0.5%.

`convergence_every` must be independent of `log_every`. Logging frequency
should not alter the numerical meaning of a convergence tolerance.

The existing velocity-only behavior should remain available for backward
compatibility:

```python
solve_flow(..., convergence="velocity", tol=1e-3)
```

## Metric Definitions

For checks separated by `convergence_every` timesteps, define:

```text
epsilon_u = sum_pore(abs(u_current - u_previous))
            / sum_pore(abs(u_current))

epsilon_K = abs(K_current - K_previous) / abs(K_current)

epsilon_Q = abs(mass_flux_in - mass_flux_out)
            / (0.5 * (abs(mass_flux_in) + abs(mass_flux_out)))
```

The directional Darcy velocity and permeability are:

```text
u_D = sum_pore(u_direction) / number_of_all_domain_voxels

K = u_D * nu / abs(pressure_gradient)
```

The face fluxes should use mass flux rather than volume flux:

```text
mass_flux_face = sum_face_pores(rho * u_direction)
```

Undefined ratios, including zero total velocity, zero permeability, or two
zero face fluxes, must not satisfy convergence.

### Stopping Policy

At every convergence check:

1. Obtain the five reduced observables from the active backend.
2. Calculate `epsilon_u`, `epsilon_K`, and `epsilon_Q` on the host.
3. Increment the pass streak only if all three criteria are defined and below
   their tolerances.
4. Reset the pass streak to zero if any criterion fails.
5. Stop after two consecutive passing checks.

There should also be:

- A minimum of three permeability samples, which follows naturally from two
  consecutive passing intervals.
- The existing `n_steps` maximum as a hard fallback.
- Explicit reporting of whether the solve converged or exhausted `n_steps`.

## Shared Architecture

Backend code should calculate observables. Shared code should decide whether
the simulation has converged.

```text
Taichi fields -> Taichi reducer ----+
JAX arrays ----> JAX reducer --------+--> five host scalars
Warp arrays ---> Warp reducer -------+          |
                                                v
                                  shared ConvergenceController
                                                |
                                      continue / stop / report
```

Create a shared module, for example `kabs/_convergence.py`, containing data
structures similar to:

```python
@dataclass(frozen=True)
class ConvergenceConfig:
    mode: str
    check_every: int
    velocity_tol: float
    k_tol: float
    flux_tol: float
    required_passes: int = 2


@dataclass(frozen=True)
class ConvergenceObservables:
    velocity_total: float
    velocity_change: float
    directional_flow: float
    inlet_mass_flux: float
    outlet_mass_flux: float


@dataclass(frozen=True)
class ConvergenceReport:
    velocity_criterion: float | None
    k_criterion: float | None
    flux_criterion: float | None
    permeability: float | None
    consecutive_passes: int
    converged: bool
```

`ConvergenceController` should own the previous permeability, pass streak,
metric formulas, stopping policy, validation, and reporting. It must not know
about Taichi, JAX, Warp, or XLB array types.

Each backend adapter should return the same logical five-scalar
`ConvergenceObservables` value. Small floating-point differences caused by
parallel reduction order are expected and should not change the definitions.

## Backend Adapters

### Taichi

Taichi already keeps `rho`, velocity, and the previous velocity snapshot on
the device for convergence monitoring.

Implement one device reduction that accumulates a five-element result:

```text
[sum(abs(u)),
 sum(abs(u - u_previous)),
 sum(u_direction),
 sum_inlet(rho * u_direction),
 sum_outlet(rho * u_direction)]
```

Requirements:

- Iterate over logical pore voxels only.
- Support x, y, and z flow directions.
- Work for dense, tiled, and sparse storage.
- Transfer only the five reduced values to the host.
- Snapshot velocity on-device after every non-terminal check.
- Allocate convergence-only fields only when convergence monitoring is
  enabled.

The earlier prototype used separate velocity and flow reductions. A combined
reducer is preferable because it gives one definition, one launch path, and
one small host transfer per check.

### XLB-JAX

Extend the existing JIT-compiled JAX convergence reducer to accept:

- `rho_current`
- `u_current`
- `u_previous`
- a pore mask resident on the device
- flow-axis metadata

It should return one stacked five-element JAX array. Use exactly one
`jax.device_get()` per check. Do not convert full density or velocity arrays to
NumPy during convergence checks.

For multi-GPU JAX arrays, verify that reductions cover all shards and return
global totals.

### XLB-Warp

Implement a native Warp convergence reduction after the XLB-Warp state and
macroscopic-field interface has been finalized.

The preferred design is a Warp kernel that atomically accumulates into a
five-element device array, followed by a transfer of only that array. Keep the
previous velocity on the GPU, copying it with Warp primitives when necessary.

Do not make Warp convergence depend on converting full fields to JAX. The
current development version of the XLB wrapper routes Warp populations through
a JAX macroscopic operator, and XLB utility names for Warp/JAX interoperation
have varied between versions. Native Warp reduction avoids that fragile
dependency and preserves the purpose of the Warp backend.

Questions to resolve after XLB-Warp is complete:

- Does the native XLB Warp macroscopic operator allocate new `rho` and `u`
  arrays on each call or reuse output buffers?
- What is the exact Warp velocity layout: component-first or component-last?
- Can the previous velocity retain a reference safely, or is an explicit
  device copy required?
- What public XLB/Backend APIs should be used so KABS does not depend on XLB
  internals?

## Solver-Loop Integration

Both solver loops should follow the same lifecycle even though their state
types differ:

```python
controller = ConvergenceController(config, domain, direction, nu, pressure_drop)
sampler = backend_specific_sampler(...)

for step in range(...):
    advance_one_timestep()

    if controller.check_due(step):
        observables = sampler.sample()
        report = controller.update(observables)

        if report.converged:
            break

        sampler.snapshot_velocity()
```

Progress logging remains independently controlled by `log_every`.

## Result Metadata

`FlowResult` should record:

```text
convergence_mode
converged
n_iterations
k_criterion
velocity_criterion
flux_criterion
convergence_every
consecutive_passes
```

Retain `convergence_criterion` as a backward-compatible alias for the velocity
criterion until a deprecation decision is made.

The final progress message should report each measured value alongside its
tolerance. PoreStore should be able to distinguish a converged solve from one
that merely reached `n_steps`.

## Pressure and Driving-Mode Considerations

The initial implementation targets the existing inlet/outlet pressure-driven
solver.

If a periodic body-force driving mode is added later, inlet/outlet flux
imbalance is not directly defined. Options include:

1. Compare mass flux through two separated cross-sections.
2. Use the maximum variation of mass flux over several cross-sections.
3. Define a driving-mode-specific continuity residual.

Do not silently disable `flux_tol`. The selected driving mode should provide a
documented continuity observable with the same role.

Pressure boundary values should eventually be parameters stored with the
result rather than duplicated constants. The convergence controller and final
permeability calculation must use the same pressure gradient.

## Validation Plan

### Shared Controller Tests

- A scripted sequence where all criteria pass twice stops on the expected
  check.
- Failure of each individual criterion resets the pass streak.
- Zero denominators never pass.
- NaN and infinite observables never pass.
- Invalid tolerance and interval values fail clearly.
- `log_every` does not affect convergence results.

### Reducer Tests

For synthetic `rho`, velocity, previous velocity, and solid-mask arrays:

- Compare all five outputs with NumPy formulas.
- Test x, y, and z directions.
- Include solid voxels with deliberately nonzero test values to prove they are
  excluded.
- Test non-cubic domains.
- Test Taichi dense, tiled, and sparse storage.
- Confirm one small synchronized transfer per check for JAX and Warp.
- Confirm no full-field host extraction during periodic checks.

### Cross-Backend Integration Tests

Use a small straight channel and a small porous sample:

- Run a deliberately long reference solve for each backend.
- Run permeability convergence with the proposed defaults.
- Compare stopped permeability with the corresponding long-run permeability.
- Compare final criteria and stopping steps within one check interval.
- Verify inlet/outlet mass conservation.
- Verify all flow directions.
- Verify SRT Taichi, XLB-JAX, and XLB-Warp.
- Run Warp tests on a CUDA machine.
- Run JAX tests on both one GPU and multiple GPUs when available.

Backend implementations need not stop at exactly the same iteration because
parallel reduction order differs. Acceptance should focus on final permeability
accuracy and correctly satisfied criteria.

### Production Calibration

Run representative PoreStore images spanning:

- porosity
- anisotropy
- connected and dead-end pores
- image size and aspect ratio
- permeability range
- viscosity/relaxation time

For each image, compare the early-stopped result with a long reference run.
Record time-to-K, stopped-versus-reference permeability error, velocity-field
error, flux imbalance, and backend.

Only consider making permeability convergence the default after these results
show that the selected tolerances meet the scientific accuracy requirement.

## Prototype Evidence

The abandoned Taichi prototype was useful as a feasibility test:

- It kept array-sized work on the device.
- It successfully reduced directional flow and boundary mass flux for all
  three axes.
- On the local four-tube Metal test, the proposed default tolerances stopped at
  800 steps with approximately:

```text
K                  = 6.444431
K change           = 0.093%
velocity change    = 0.095%
flux imbalance     = 0.461%
```

The result was close to the longer-run tube permeability, but this single
geometry is not enough to establish safe production defaults.

## Suggested Issue Breakdown

1. **Create shared convergence controller and policy tests**
2. **Implement combined Taichi convergence reducer**
3. **Implement XLB-JAX convergence adapter**
4. **Implement XLB-Warp convergence adapter on CUDA**
5. **Add cross-backend numerical and transfer-regression tests**
6. **Calibrate tolerances on representative PoreStore jobs**
7. **Decide whether permeability convergence should become the default**

## Preconditions for Resuming

Resume this work when:

- XLB-Warp can complete a KABS flow solve on CUDA.
- Its macroscopic density and velocity fields can be obtained through a stable
  public XLB interface.
- The field layouts and ownership/lifetime rules are known.
- Baseline long-run permeability cases exist for Taichi, XLB-JAX, and
  XLB-Warp.

