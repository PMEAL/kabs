"""High-level entry point: run a single-phase LBM flow simulation using XLB."""

from functools import lru_cache
import time

import numpy as np

from ._convergence import (
    ConvergenceConfig,
    ConvergenceController,
    ConvergenceObservables,
    TOL_UNSET,
    VELOCITY_TOL_UNSET,
    normalize_convergence_tolerances,
    validate_solver_intervals,
)
from ._flow_common import FlowResult, _RHO_IN, _RHO_OUT
from ._solve_flow import _print_convergence_report


__all__ = ["solve_flow_xlb"]

# XLB 3D face names for each flow direction.
# In XLB's grid convention: left=x0, right=x1, front=y0, back=y1, bottom=z0, top=z1
_FACE_NAMES = {
    "x": ("left", "right"),
    "y": ("front", "back"),
    "z": ("bottom", "top"),
}


@lru_cache(maxsize=None)
def _get_jax_convergence_reducer(
    needs_velocity=True, needs_permeability=False, needs_flux=False, flow_axis=0
):
    """Return a criterion-aware compiled device reduction."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def reduce_observables(rho, current, previous, pore):
        values = []
        if needs_velocity:
            component_mask = pore
            values.extend(
                (
                    jnp.sum(jnp.where(component_mask, jnp.abs(current), 0.0)),
                    jnp.sum(
                        jnp.where(
                            component_mask,
                            jnp.abs(current - previous),
                            0.0,
                        )
                    ),
                )
            )
        directional = current[flow_axis]
        spatial_pore = pore[0]
        if needs_permeability:
            values.append(jnp.sum(jnp.where(spatial_pore, directional, 0.0)))
        if needs_flux:
            mass_flux = jnp.where(spatial_pore, rho[0] * directional, 0.0)
            values.extend(
                (
                    jnp.sum(jnp.take(mass_flux, 0, axis=flow_axis)),
                    jnp.sum(jnp.take(mass_flux, -1, axis=flow_axis)),
                )
            )
        return jnp.stack(values)

    return reduce_observables


def _convergence_sums_to_host(current, previous):
    """Backward-compatible test helper for the velocity-only reduction."""
    import jax
    import jax.numpy as jnp

    spatial_shape = current.shape[-3:]
    rho = jnp.ones((1, *spatial_shape), dtype=current.dtype)
    pore = jnp.ones((1, *spatial_shape), dtype=bool)
    sums = jax.device_get(
        _get_jax_convergence_reducer()(rho, current, previous, pore)
    )
    return float(sums[0]), float(sums[1])


class _JaxConvergenceMonitor:
    def __init__(self, pore_mask, config, flow_axis):
        self._pore_host = np.asarray(pore_mask, dtype=bool)[None, ...]
        self._pore = None
        self._config = config
        self._flow_axis = flow_axis
        self._previous = None

    def sample(self, rho, velocity):
        if (
            self._config.needs_velocity
            and self._previous is None
            and not self._config.needs_permeability
            and not self._config.needs_flux
        ):
            return ConvergenceObservables()
        import jax
        import jax.numpy as jnp

        if self._pore is None:
            try:
                self._pore = jax.device_put(self._pore_host, velocity.sharding)
            except (AttributeError, TypeError, ValueError):
                self._pore = jnp.asarray(self._pore_host)
        previous = velocity if self._previous is None else self._previous
        reducer = _get_jax_convergence_reducer(
            self._config.needs_velocity,
            self._config.needs_permeability,
            self._config.needs_flux,
            self._flow_axis,
        )
        values = jax.device_get(reducer(rho, velocity, previous, self._pore))
        offset = 0
        velocity_total = velocity_change = directional_flow = None
        inlet_mass_flux = outlet_mass_flux = None
        if self._config.needs_velocity:
            velocity_total = float(values[offset])
            velocity_change = (
                float(values[offset + 1]) if self._previous is not None else None
            )
            offset += 2
        if self._config.needs_permeability:
            directional_flow = float(values[offset])
            offset += 1
        if self._config.needs_flux:
            inlet_mass_flux = float(values[offset])
            outlet_mass_flux = float(values[offset + 1])
        return ConvergenceObservables(
            velocity_total=velocity_total,
            velocity_change=velocity_change,
            directional_flow=directional_flow,
            inlet_mass_flux=inlet_mass_flux,
            outlet_mass_flux=outlet_mass_flux,
        )

    def snapshot_velocity(self, velocity):
        self._previous = velocity


def solve_flow_xlb(
    im,
    direction="x",
    n_steps=15000,
    nu=1.0 / 6.0,
    log_every=500,
    verbose=True,
    tol=TOL_UNSET,
    compute_backend="jax",
    *,
    velocity_tol=VELOCITY_TOL_UNSET,
    k_tol=None,
    flux_tol=None,
    convergence_every=500,
):
    """
    Run a pressure-driven single-phase LBM simulation using the XLB library.

    This is the XLB-specific implementation used by
    ``solve_flow(backend='xlb')``. It returns the same ``FlowResult`` object
    and is fully compatible with the downstream ``compute_permeability()`` and
    ``solve_hydraulic_conductance()`` functions.

    Parameters
    ----------
    im : np.ndarray, shape (nx, ny, nz)
        Binary image of the pore space.  1 (or True) = pore, 0 (or False) = solid.
        This matches the PoreSpy convention.
    direction : {'x', 'y', 'z'}
        Axis along which the pressure gradient is applied.  Default ``'x'``.
    n_steps : int
        Maximum number of LBM time steps to run.  Default 15000.
    nu : float
        Kinematic viscosity in lattice units.  Default 1/6.
    log_every : int
        Print a progress line every this many completed steps. Zero disables
        progress logging. Default 500.
    verbose : bool
        Print progress to stdout.  Default True.
    tol : float or None
        Deprecated alias for ``velocity_tol``.
    velocity_tol, k_tol, flux_tol : float or None
        Independently optional convergence tolerances. All enabled criteria
        must pass on two consecutive checks.
    convergence_every : int
        Number of completed steps between convergence samples. Default 500.
    compute_backend : {'jax', 'warp'}
        XLB compute backend.  ``'jax'`` enables multi-GPU via JAX (CPU also
        works).  ``'warp'`` uses NVIDIA Warp for single-GPU runs.
        Default ``'jax'``.

    Returns
    -------
    result : FlowResult
        Result object containing ``solid``, ``rho``, ``velocity``,
        ``direction``, and ``nu`` as numpy arrays/values.  Pass directly to
        ``compute_permeability()`` or ``solve_hydraulic_conductance()``,
        or call ``result.export_to_vtk(prefix)`` to save a VTR file.

    Notes
    -----
    **Collision model**: this solver uses the BGK (single relaxation time)
    operator.  BGK is appropriate for Stokes-regime (low Reynolds number)
    porous-media flow and produces physically equivalent results to the MRT
    operator used by ``solve_flow_taichi()`` in that regime. If you suspect
    inertial effects (higher flow rates, large pores), prefer
    ``solve_flow(backend='taichi')``, which uses MRT.

    **Boundary conditions**: inlet and outlet faces receive an equilibrium
    pressure (density) BC with ρ_in = 1.00 and ρ_out = 0.99, matching the
    Taichi implementation. Unassigned transverse faces are automatically
    periodic (XLB's streaming operator wraps around by default).

    **JIT compilation**: JAX will compile the stepper on the first call.
    Expect an additional overhead of 10–60 s on the first invocation.

    **Initialization**: ``xlb.init()`` is called internally.  If you need to
    call it with custom settings before invoking this function, set
    ``compute_backend`` to match your prior ``xlb.init()`` call.

    **Installation**::

        pip install "xlb>=0.3.1" "warp-lang==1.10.0"  # CPU/JAX
        pip install "xlb[cuda]>=0.3.1" "warp-lang==1.10.0"  # NVIDIA GPU via JAX
    """
    if not isinstance(direction, str) or direction.lower() not in _FACE_NAMES:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")
    direction = direction.lower()

    if not isinstance(compute_backend, str) or compute_backend.lower() not in {
        "jax",
        "warp",
    }:
        raise ValueError(
            f"compute_backend must be 'jax' or 'warp', got {compute_backend!r}"
        )
    backend_key = compute_backend.lower()
    n_steps, log_every, convergence_every = validate_solver_intervals(
        n_steps, log_every, convergence_every
    )
    velocity_tol, k_tol, flux_tol = normalize_convergence_tolerances(
        tol=tol,
        velocity_tol=velocity_tol,
        k_tol=k_tol,
        flux_tol=flux_tol,
    )
    config = ConvergenceConfig(
        check_every=convergence_every,
        velocity_tol=velocity_tol,
        k_tol=k_tol,
        flux_tol=flux_tol,
    )

    try:
        import xlb
        from xlb.compute_backend import ComputeBackend
        from xlb.precision_policy import PrecisionPolicy
        from xlb.grid import grid_factory
        from xlb.operator.stepper import IncompressibleNavierStokesStepper
        from xlb.operator.boundary_condition import (
            EquilibriumBC,
            FullwayBounceBackBC,
            ZouHeBC,
        )
        from xlb.operator.macroscopic import Macroscopic
    except ImportError as exc:
        raise ImportError(
            "XLB is required for solve_flow_xlb. Install compatible dependencies with:\n"
            "    pip install 'xlb>=0.3.1' 'warp-lang==1.10.0'\n"
        ) from exc

    _backend_map = {"jax": ComputeBackend.JAX, "warp": ComputeBackend.WARP}
    _backend = _backend_map[backend_key]

    # --- Initialize XLB (safe to call multiple times) ---
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q19(
        precision_policy=precision_policy,
        compute_backend=_backend,
    )
    xlb.init(
        velocity_set=velocity_set,
        default_backend=_backend,
        default_precision_policy=precision_policy,
    )

    # --- Prepare image ---
    # Public (PoreSpy) convention: 1 = pore, 0 = solid.
    # Internal (FlowResult) convention: 1 = solid, 0 = pore.
    solid_im = (im == 0).astype(np.int8)
    solid_mask = solid_im.astype(bool)  # True where solid
    nx, ny, nz = im.shape

    # BGK relaxation parameter from viscosity: tau = 3*nu + 0.5, omega = 1/tau
    omega = 1.0 / (3.0 * nu + 0.5)

    # --- Build XLB grid and resolve boundary indices ---
    grid = grid_factory((nx, ny, nz), compute_backend=_backend)

    # Pressure boundaries must include every pore voxel on their face. Zou-He
    # is well-defined on face interiors, while edge and corner nodes need a
    # full equilibrium condition because they have multiple missing directions.
    box_face = grid.bounding_box_indices(remove_edges=False)

    inlet_name, outlet_name = _FACE_NAMES[direction]

    flow_axis = {"x": 0, "y": 1, "z": 2}[direction]

    def _pore_face_indices(face_key):
        """Return interior and edge/corner pore indices for a domain face."""
        idx = np.array(box_face[face_key])  # shape (3, n)
        xs, ys, zs = idx[0], idx[1], idx[2]
        is_pore = ~solid_mask[xs, ys, zs]
        is_edge = np.zeros(idx.shape[1], dtype=bool)
        for axis, size in enumerate(im.shape):
            if axis != flow_axis:
                is_edge |= (idx[axis] == 0) | (idx[axis] == size - 1)
        interior = is_pore & ~is_edge
        edge = is_pore & is_edge
        return (
            [idx[d][interior].tolist() for d in range(3)],
            [idx[d][edge].tolist() for d in range(3)],
        )

    inlet_indices, inlet_edge_indices = _pore_face_indices(inlet_name)
    outlet_indices, outlet_edge_indices = _pore_face_indices(outlet_name)

    # All solid voxels throughout the domain
    solid_coords = np.where(solid_mask)
    solid_indices = [solid_coords[d].tolist() for d in range(3)]

    # --- Boundary conditions ---
    # Pressure BCs: Zou-He pressure BC at inlet/outlet pore voxels.
    # This sets the density to the prescribed value and determines velocity
    # from mass conservation — analogous to the Taichi pressure BC.
    # Solid BCs: full-way bounce-back throughout the domain.
    # Transverse faces have no explicit BC → streaming wraps around (periodic).
    boundary_conditions = []
    # XLB 0.3.1's Warp ZouHeBC path calls np.nonzero() on prescribed_value.
    # NumPy rejects that operation for scalar/0-D values, although scalar
    # pressure values are part of ZouHeBC's public API.  A one-element array is
    # an equivalent supported input and avoids the upstream Warp-only bug.
    rho_in = (
        np.asarray([_RHO_IN], dtype=np.float64)
        if _backend is ComputeBackend.WARP
        else float(_RHO_IN)
    )
    rho_out = (
        np.asarray([_RHO_OUT], dtype=np.float64)
        if _backend is ComputeBackend.WARP
        else float(_RHO_OUT)
    )
    if inlet_indices[0]:
        boundary_conditions.append(
            ZouHeBC(
                bc_type="pressure",
                prescribed_value=rho_in,
                indices=inlet_indices,
            )
        )
    if outlet_indices[0]:
        boundary_conditions.append(
            ZouHeBC(
                bc_type="pressure",
                prescribed_value=rho_out,
                indices=outlet_indices,
            )
        )
    if inlet_edge_indices[0]:
        boundary_conditions.append(
            EquilibriumBC(
                rho=float(_RHO_IN), u=(0.0, 0.0, 0.0), indices=inlet_edge_indices
            )
        )
    if outlet_edge_indices[0]:
        boundary_conditions.append(
            EquilibriumBC(
                rho=float(_RHO_OUT), u=(0.0, 0.0, 0.0), indices=outlet_edge_indices
            )
        )

    if solid_coords[0].size > 0:
        bc_solid = FullwayBounceBackBC(indices=solid_indices)
        boundary_conditions.append(bc_solid)

    # --- Macroscopic operator ---
    macro = Macroscopic(
        compute_backend=_backend,
        precision_policy=precision_policy,
        velocity_set=velocity_set,
    )

    if _backend is ComputeBackend.WARP:
        import warp as wp

        rho_device = grid.create_field(1, dtype=precision_policy.store_precision)
        u_device = grid.create_field(3, dtype=precision_policy.store_precision)

        def _update_macroscopic(f):
            macro(f, rho_device, u_device)
            return rho_device, u_device

    else:

        def _update_macroscopic(f):
            return macro(f)

    # --- Stepper ---
    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="BGK",
    )
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    controller = None
    monitor = None
    if config.monitoring_enabled:
        controller = ConvergenceController(
            config,
            domain_shape=solid_im.shape,
            direction=direction,
            nu=nu,
            rho_in=_RHO_IN,
            rho_out=_RHO_OUT,
        )
        if _backend is ComputeBackend.WARP:
            from ._warp_convergence import WarpConvergenceMonitor

            monitor = WarpConvergenceMonitor(
                u_device, ~solid_mask, config, flow_axis
            )
        else:
            monitor = _JaxConvergenceMonitor(~solid_mask, config, flow_axis)

    time_init = time.time()
    time_pre = time_init
    final_step = 0
    final_report = None

    for completed_steps in range(1, n_steps + 1):
        f_0, f_1 = stepper(
            f_0, f_1, bc_mask, missing_mask, omega, completed_steps - 1
        )
        f_0, f_1 = f_1, f_0  # swap buffers; f_0 now holds the updated state
        final_step = completed_steps

        if controller is not None and controller.check_due(completed_steps):
            rho_current, u_current = _update_macroscopic(f_0)
            observables = monitor.sample(rho_current, u_current)
            final_report = controller.update(observables)
            if verbose:
                _print_convergence_report(final_report, config)
            if final_report.converged:
                if verbose:
                    print(f"Converged at step {completed_steps}")
                break
            if config.needs_velocity:
                monitor.snapshot_velocity(u_current)

        if log_every and completed_steps % log_every == 0:
            time_now = time.time()
            diff = int(time_now - time_pre)
            elap = int(time_now - time_init)
            m_d, s_d = divmod(diff, 60)
            h_d, m_d = divmod(m_d, 60)
            m_e, s_e = divmod(elap, 60)
            h_e, m_e = divmod(m_e, 60)

            if verbose:
                print(
                    f"Step {completed_steps:6d}/{n_steps}  "
                    f"interval {h_d:02d}h{m_d:02d}m{s_d:02d}s  "
                    f"elapsed {h_e:02d}h{m_e:02d}m{s_e:02d}s"
                )
            time_pre = time_now

    converged = None if controller is None else bool(final_report and final_report.converged)
    if verbose and final_step == n_steps and not converged:
        if controller is None:
            print(f"Completed {n_steps} steps with convergence monitoring disabled")
        else:
            print(f"Reached n_steps={n_steps} without convergence")

    # --- Extract final fields ---
    rho_field, u_field = _update_macroscopic(f_0)
    if _backend is ComputeBackend.WARP:
        wp.synchronize()
        rho_field = rho_field.numpy()
        u_field = u_field.numpy()
    # rho: (1, nx, ny, nz) → (nx, ny, nz); u: (3, nx, ny, nz) → (nx, ny, nz, 3)
    rho_np = np.array(rho_field[0]).astype(np.float32)
    vel_np = np.moveaxis(np.array(u_field), 0, -1).astype(np.float32)

    result = FlowResult.from_arrays(
        solid=solid_im,
        rho=rho_np,
        velocity=vel_np,
        direction=direction,
        nu=nu,
        collision_model="srt",
        rho_in=_RHO_IN,
        rho_out=_RHO_OUT,
        converged=converged,
        n_iterations=final_step,
        velocity_tol=velocity_tol,
        k_tol=k_tol,
        flux_tol=flux_tol,
        velocity_criterion=(
            None if final_report is None else final_report.velocity_criterion
        ),
        k_criterion=None if final_report is None else final_report.k_criterion,
        flux_criterion=(
            None if final_report is None else final_report.flux_criterion
        ),
        convergence_every=(convergence_every if controller is not None else None),
        consecutive_passes=(
            0 if final_report is None else final_report.consecutive_passes
        ),
    )

    return result
