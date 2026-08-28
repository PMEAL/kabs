"""High-level entry point: run a single-phase LBM flow simulation using XLB."""

from functools import lru_cache
import time

import numpy as np

from ._flow_common import FlowResult, _RHO_IN, _RHO_OUT


__all__ = ["solve_flow_xlb"]

# XLB 3D face names for each flow direction.
# In XLB's grid convention: left=x0, right=x1, front=y0, back=y1, bottom=z0, top=z1
_FACE_NAMES = {
    "x": ("left", "right"),
    "y": ("front", "back"),
    "z": ("bottom", "top"),
}


@lru_cache(maxsize=1)
def _get_jax_convergence_reducer():
    """Return a compiled reduction that keeps array-sized work on the device."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def reduce_velocity_change(current, previous):
        return jnp.stack(
            (
                jnp.sum(jnp.abs(current)),
                jnp.sum(jnp.abs(current - previous)),
            )
        )

    return reduce_velocity_change


def _convergence_sums_to_host(current, previous):
    """Synchronize and transfer the two convergence scalars together."""
    import jax

    sums = jax.device_get(_get_jax_convergence_reducer()(current, previous))
    return float(sums[0]), float(sums[1])


def solve_flow_xlb(
    im,
    direction="x",
    n_steps=15000,
    nu=1.0 / 6.0,
    log_every=500,
    verbose=True,
    tol=1e-3,
    compute_backend="jax",
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
        Print a progress line and check convergence every this many steps.
        Default 500.
    verbose : bool
        Print progress to stdout.  Default True.
    tol : float or None
        Convergence tolerance.  The simulation stops early when the relative
        change in the total velocity magnitude between log intervals falls
        below this value: ``delta|v| / |v| < tol``.  Set to ``None`` to
        always run the full ``n_steps``.  Default 1e-3.
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

        from ._warp_convergence import WarpConvergenceMonitor

        rho_device = grid.create_field(1, dtype=precision_policy.store_precision)
        u_device = grid.create_field(3, dtype=precision_policy.store_precision)
        warp_convergence = WarpConvergenceMonitor(u_device)

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

    # --- Time loop ---
    time_init = time.time()
    time_pre = time_init
    u_previous = None
    enable_convergence_monitor = (
        log_every != 0
        and n_steps >= abs(log_every)
        and (_backend is not ComputeBackend.WARP or tol is not None)
    )
    final_step = n_steps
    final_criterion = None

    for i in range(n_steps + 1):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, i)
        f_0, f_1 = f_1, f_0  # swap buffers; f_0 now holds the updated state

        if i % log_every == 0:
            time_now = time.time()
            diff = int(time_now - time_pre)
            elap = int(time_now - time_init)
            m_d, s_d = divmod(diff, 60)
            h_d, m_d = divmod(m_d, 60)
            m_e, s_e = divmod(elap, 60)
            h_e, m_e = divmod(m_e, 60)

            if verbose:
                print(
                    f"Step {i:6d}/{n_steps}  "
                    f"interval {h_d:02d}h{m_d:02d}m{s_d:02d}s  "
                    f"elapsed {h_e:02d}h{m_e:02d}m{s_e:02d}s"
                )

            if enable_convergence_monitor:
                # rho shape: (1, nx, ny, nz); u shape: (3, nx, ny, nz)
                _, u_current = _update_macroscopic(f_0)
                if _backend is ComputeBackend.WARP:
                    metrics = warp_convergence.sample(u_current)
                    if metrics is not None:
                        v_total = metrics.velocity_total
                        v_change = metrics.velocity_change
                elif u_previous is not None:
                    metrics = None
                    v_total, v_change = _convergence_sums_to_host(
                        u_current, u_previous
                    )
                else:
                    metrics = None

                has_comparison = (
                    metrics is not None
                    if _backend is ComputeBackend.WARP
                    else u_previous is not None
                )
                if has_comparison:
                    if v_total > 0:
                        final_criterion = v_change / v_total
                    if verbose:
                        print(f"         |v|={v_total:.4e}  delta|v|={v_change:.4e}")
                    if tol is not None and v_total > 0 and final_criterion < tol:
                        if verbose:
                            print(
                                f"Converged at step {i} "
                                f"(delta|v|/|v| = {final_criterion:.2e} < tol={tol:.2e})"
                            )
                        final_step = i
                        break
                if _backend is not ComputeBackend.WARP:
                    u_previous = u_current
            time_pre = time_now

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
    )
    result.n_iterations = final_step
    result.convergence_criterion = final_criterion

    return result
