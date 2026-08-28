# Backend-Neutral Composable Convergence

## Status

Implemented for Taichi, XLB-JAX, and XLB-Warp. Permeability and flux criteria
remain opt-in pending calibration on representative PoreStore images.

## Public API

Convergence is controlled by three independently optional tolerances:

```python
result = solve_flow(
    im,
    direction="x",
    backend="taichi",       # or "xlb"
    compute_backend="jax",  # or "warp" for XLB
    velocity_tol=1e-2,
    k_tol=5e-3,
    flux_tol=5e-3,
    convergence_every=500,
)
```

Every non-`None` tolerance enables that criterion. All enabled criteria must
pass on two consecutive checks. Setting all three to `None` disables convergence
monitoring and its device allocations. The effective default is velocity-only
convergence with `velocity_tol=1e-3`.

The old `tol` argument is a deprecated alias for `velocity_tol`. Supplying both
explicitly is an error. `convergence_every` is independent of `log_every`.

## Definitions

Checks use a componentwise L1 velocity ratio over pore voxels:

```text
epsilon_u = sum_pore_components(abs(u_current - u_previous))
            / sum_pore_components(abs(u_current))
```

Permeability uses pore-only directional flow divided by the number of all
domain voxels:

```text
u_D = sum_pore(u_direction) / number_of_all_domain_voxels
K = u_D * nu / abs((rho_in - rho_out) * (1/3) / domain_length)
epsilon_K = abs(K_current - K_previous) / abs(K_current)
```

Continuity uses axis-oriented mass flux on the actual pressure faces:

```text
Q_in  = sum_inlet_face_pores(rho * u_direction)
Q_out = sum_outlet_face_pores(rho * u_direction)
epsilon_Q = abs(Q_in - Q_out)
            / (0.5 * (abs(Q_in) + abs(Q_out)))
```

Missing history, zero denominators, NaN, and infinite values cannot pass.
Velocity and permeability criteria need a baseline, so they can first converge
on the third scheduled sample. Flux-only convergence can occur on the second.

## Architecture

Backend code only measures enabled observables. The shared controller calculates
criteria, owns permeability history and the pass streak, and decides whether to
stop.

```text
Taichi fields -> criterion-sized Taichi reduction --+
JAX arrays ----> criterion-sized JAX reduction ------+--> host observables
Warp arrays ---> criterion-sized Warp reduction -----+          |
                                                               v
                                                shared controller
```

The possible observables are velocity total/change, directional flow, and the
two pressure-face fluxes. A check performs at most one small device-to-host
transfer. Unused observables are not reduced. Previous velocity remains on the
device and is allocated only when the velocity criterion is enabled.

Taichi supports dense, tiled, and sparse storage. JAX reductions preserve global
array/sharding semantics. Warp uses native Warp fields and kernels without
routing full fields through JAX.

## Loop and Result Semantics

`n_steps` is the exact maximum number of timestep updates. Checks occur at
completed steps `convergence_every`, `2 * convergence_every`, and so on. Logging
has its own schedule, and `log_every=0` disables it. A terminal passing check does
not take another velocity snapshot.

`FlowResult` records the pressure densities, normalized tolerances, final
criteria, convergence interval, pass streak, exact iteration count, and a
three-state status:

- `True`: every enabled criterion passed twice;
- `False`: an enabled policy exhausted `n_steps`;
- `None`: monitoring was disabled or provenance is unavailable.

`convergence_criterion` remains a compatibility alias for
`velocity_criterion`. VTR metadata round-trips all convergence and pressure
provenance while remaining compatible with older files.

## Validation and Calibration

Automated coverage includes shared-policy sequences, invalid/undefined ratios,
argument compatibility, all flow axes, non-cubic and masked synthetic fields,
Taichi storage layouts, JAX transfer regression, optional Warp CUDA tests,
exact loop counts, independent logging, and VTR round trips.

Before enabling permeability or flux tolerances by default, calibrate against
long reference runs across representative porosity, anisotropy, connected and
dead-end pores, image shapes and sizes, permeability ranges, and viscosities.
Record stopped-versus-reference permeability and velocity error, final flux
imbalance, stopping step, and time-to-permeability for every backend.
