"""High-level entry point for pressure-driven single-phase LBM flow."""

import time
from typing import Literal

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
from ._flow_common import (
    FlowResult,
    _DEFAULT_SPARSE,
    _DEFAULT_STORAGE,
    _RHO_IN,
    _RHO_OUT,
    _normalize_collision_model,
)

__all__ = ["solve_flow", "solve_flow_taichi", "FlowResult"]

_BC_SETTERS = {
    "x": ("set_bc_rho_x0", "set_bc_rho_x1"),
    "y": ("set_bc_rho_y0", "set_bc_rho_y1"),
    "z": ("set_bc_rho_z0", "set_bc_rho_z1"),
}


def _print_convergence_report(report, config):
    parts = []
    for label, criterion, tolerance in (
        ("velocity", report.velocity_criterion, config.velocity_tol),
        ("K", report.k_criterion, config.k_tol),
        ("flux", report.flux_criterion, config.flux_tol),
    ):
        if tolerance is None:
            continue
        value = "undefined" if criterion is None else f"{criterion:.3e}"
        parts.append(f"{label}={value} (tol={tolerance:.3e})")
    print("         " + "  ".join(parts) + f"  streak={report.consecutive_passes}")


def _taichi_observables(solver, config, has_velocity_snapshot):
    solver.reset_convergence_sums()
    solver.accumulate_convergence_sums(has_velocity_snapshot)
    sums = solver.get_convergence_sums()
    return ConvergenceObservables(
        velocity_total=(
            sums[solver._convergence_velocity_offset]
            if config.needs_velocity
            else None
        ),
        velocity_change=(
            sums[solver._convergence_velocity_offset + 1]
            if config.needs_velocity and has_velocity_snapshot
            else None
        ),
        directional_flow=(
            sums[solver._convergence_flow_offset]
            if config.needs_permeability
            else None
        ),
        inlet_mass_flux=(
            sums[solver._convergence_flux_offset] if config.needs_flux else None
        ),
        outlet_mass_flux=(
            sums[solver._convergence_flux_offset + 1]
            if config.needs_flux
            else None
        ),
    )


def solve_flow_taichi(
    im,
    direction="x",
    n_steps=15000,
    nu=1.0 / 6.0,
    log_every=500,
    verbose=True,
    sparse=_DEFAULT_SPARSE,
    tol=TOL_UNSET,
    *,
    velocity_tol=VELOCITY_TOL_UNSET,
    k_tol=None,
    flux_tol=None,
    convergence_every=500,
    storage: Literal["dense", "tiled", "sparse"] = _DEFAULT_STORAGE,
    tile_size: int | tuple[int, int, int] = 16,
    collision_model: Literal["mrt", "srt"] = "mrt",
):
    """Run a pressure-driven flow solve using Taichi.

    Each non-``None`` tolerance enables a convergence criterion. All enabled
    criteria must pass on two consecutive checks. ``tol`` is a deprecated
    alias for ``velocity_tol``.

    Parameters
    ----------
    im : np.ndarray, shape (nx, ny, nz)
        Binary pore image using the PoreSpy convention: 1 is pore and 0 is
        solid.
    direction : {'x', 'y', 'z'}
        Positive coordinate direction of the pressure-driven flow.
    n_steps : int
        Exact maximum number of LBM updates. Default 15000.
    nu : float
        Kinematic viscosity in lattice units. Default 1/6.
    log_every : int
        Completed steps between progress messages. Zero disables logging.
    verbose : bool
        Print progress and convergence reports.
    sparse : bool
        Backward-compatible alias for the Taichi storage option.
    tol : float or None
        Deprecated alias for ``velocity_tol``.
    velocity_tol, k_tol, flux_tol : float or None, keyword-only
        Relative convergence tolerances. ``None`` disables that criterion;
        setting all three to ``None`` disables monitoring entirely.
    convergence_every : int, keyword-only
        Completed steps between convergence samples, independent of logging.
    storage : {'dense', 'tiled', 'sparse'}, keyword-only
        Taichi field layout. Default ``'dense'``.
    tile_size : int or tuple of 3 ints, keyword-only
        Tile dimensions for tiled and sparse layouts. Default 16.
    collision_model : {'mrt', 'srt'}, keyword-only
        Taichi collision operator. Default ``'mrt'``.

    Returns
    -------
    FlowResult
        Final fields plus exact iteration and convergence provenance.
    """
    from ._single_phase_solver import SinglePhaseSolver

    if not isinstance(direction, str) or direction.lower() not in _BC_SETTERS:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")
    direction = direction.lower()
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

    solid_im = (im == 0).astype(np.int8)
    solver = SinglePhaseSolver(
        solid_im,
        sparse_storage=sparse,
        storage=storage,
        tile_size=tile_size,
        collision_model=collision_model,
        _enable_convergence_monitor=config.monitoring_enabled,
        _convergence_needs_velocity=config.needs_velocity,
        _convergence_needs_permeability=config.needs_permeability,
        _convergence_needs_flux=config.needs_flux,
        _convergence_direction=direction,
    )
    set_inlet, set_outlet = _BC_SETTERS[direction]
    getattr(solver, set_inlet)(_RHO_IN)
    getattr(solver, set_outlet)(_RHO_OUT)
    solver.set_viscosity(nu)
    solver.init_simulation()

    controller = None
    if config.monitoring_enabled:
        controller = ConvergenceController(
            config,
            domain_shape=solid_im.shape,
            direction=direction,
            nu=nu,
            rho_in=_RHO_IN,
            rho_out=_RHO_OUT,
        )

    time_init = time.time()
    time_pre = time_init
    has_velocity_snapshot = False
    final_step = 0
    final_report = None

    for completed_steps in range(1, n_steps + 1):
        solver.step()
        final_step = completed_steps

        if controller is not None and controller.check_due(completed_steps):
            velocity_only_first_sample = (
                config.needs_velocity
                and not has_velocity_snapshot
                and not config.needs_permeability
                and not config.needs_flux
            )
            observables = (
                ConvergenceObservables()
                if velocity_only_first_sample
                else _taichi_observables(solver, config, has_velocity_snapshot)
            )
            final_report = controller.update(observables)
            if verbose:
                _print_convergence_report(final_report, config)
            if final_report.converged:
                if verbose:
                    print(f"Converged at step {completed_steps}")
                break
            if config.needs_velocity:
                solver.snapshot_velocity()
                has_velocity_snapshot = True

        if log_every and completed_steps % log_every == 0:
            time_now = time.time()
            diff = int(time_now - time_pre)
            elapsed = int(time_now - time_init)
            h_d, remainder = divmod(diff, 3600)
            m_d, s_d = divmod(remainder, 60)
            h_e, remainder = divmod(elapsed, 3600)
            m_e, s_e = divmod(remainder, 60)
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

    return FlowResult(
        solver,
        direction,
        nu,
        n_iterations=final_step,
        rho_in=_RHO_IN,
        rho_out=_RHO_OUT,
        converged=converged,
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


def solve_flow(
    im,
    direction="x",
    n_steps=15000,
    nu=1.0 / 6.0,
    log_every=500,
    verbose=True,
    sparse=_DEFAULT_SPARSE,
    tol=TOL_UNSET,
    *,
    velocity_tol=VELOCITY_TOL_UNSET,
    k_tol=None,
    flux_tol=None,
    convergence_every=500,
    storage: Literal["dense", "tiled", "sparse"] = _DEFAULT_STORAGE,
    tile_size: int | tuple[int, int, int] = 16,
    backend: Literal["taichi", "xlb"] = "taichi",
    compute_backend: Literal["jax", "warp"] = "jax",
    collision_model: Literal["mrt", "srt"] | None = None,
):
    """Run a pressure-driven single-phase LBM simulation to steady state.

    Convergence arguments have the same composable semantics as
    :func:`solve_flow_taichi`. ``backend='taichi'`` uses the native Taichi
    solver; ``backend='xlb'`` selects XLB with either its JAX or Warp compute
    backend. Taichi storage options are rejected for XLB.
    """
    try:
        backend_key = backend.lower()
    except AttributeError as exc:
        raise ValueError(
            f"backend must be 'taichi' or 'xlb', got {backend!r}"
        ) from exc

    convergence_args = dict(
        tol=tol,
        velocity_tol=velocity_tol,
        k_tol=k_tol,
        flux_tol=flux_tol,
        convergence_every=convergence_every,
    )
    if backend_key == "taichi":
        effective_collision_model = "mrt" if collision_model is None else collision_model
        return solve_flow_taichi(
            im,
            direction=direction,
            n_steps=n_steps,
            nu=nu,
            log_every=log_every,
            verbose=verbose,
            sparse=sparse,
            storage=storage,
            tile_size=tile_size,
            collision_model=effective_collision_model,
            **convergence_args,
        )

    if backend_key == "xlb":
        if collision_model is not None:
            effective_collision_model = _normalize_collision_model(collision_model)
            if effective_collision_model != "srt":
                raise ValueError("backend='xlb' only supports collision_model='srt'")
        if sparse != _DEFAULT_SPARSE or storage != _DEFAULT_STORAGE or tile_size != 16:
            raise ValueError(
                "sparse, storage, and tile_size are only supported with backend='taichi'"
            )
        from ._solve_flow_xlb import solve_flow_xlb

        return solve_flow_xlb(
            im,
            direction=direction,
            n_steps=n_steps,
            nu=nu,
            log_every=log_every,
            verbose=verbose,
            compute_backend=compute_backend,
            **convergence_args,
        )

    raise ValueError(f"backend must be 'taichi' or 'xlb', got {backend!r}")
