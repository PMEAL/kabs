"""High-level entry point: run a single-phase LBM flow simulation."""

import time

import numpy as np

from ._single_phase_solver import SinglePhaseSolver


__all__ = ["solve_flow"]

# Pressure BCs are hardcoded: only the difference matters for Darcy's law,
# and both u_D and gradP scale with Δρ so they cancel in k = u_D*mu/gradP.
_RHO_IN  = 1.00
_RHO_OUT = 0.99

_BC_SETTERS = {
    "x": ("set_bc_rho_x0", "set_bc_rho_x1"),
    "y": ("set_bc_rho_y0", "set_bc_rho_y1"),
    "z": ("set_bc_rho_z0", "set_bc_rho_z1"),
}


def solve_flow(
    im,
    direction="x",
    n_steps=15000,
    nu=1.0 / 6.0,
    log_every=500,
    export_vtk=True,
    output_prefix="LB_SingelPhase",
    verbose=True,
):
    """Run a pressure-driven single-phase LBM simulation to steady state.

    Parameters
    ----------
    im : np.ndarray, shape (nx, ny, nz)
        Binary image of the pore space.  0 = pore, non-zero = solid.
    direction : {'x', 'y', 'z'}
        Axis along which the pressure gradient is applied.  Default ``'x'``.
    n_steps : int
        Number of LBM time steps to run.  Default 15000.
    nu : float
        Kinematic viscosity in lattice units.  Default 1/6.
    log_every : int
        Print a progress line every this many steps.  Default 500.
    export_vtk : bool
        If True (default), write ``{output_prefix}_{n_steps}.vtr`` at the end.
    output_prefix : str
        Filename prefix for the VTR output.  Default ``'LB_SingelPhase'``.
    verbose : bool
        Print progress to stdout.  Default True.

    Returns
    -------
    solver : SinglePhaseSolver
        The solver object after the run.  Call ``solver.export_VTK(n)``
        manually if you set ``export_vtk=False`` and want to save later.

    Notes
    -----
    Taichi must be initialised by the caller before invoking this function::

        import taichi as ti
        ti.init(arch=ti.cpu)
    """
    direction = direction.lower()
    if direction not in _BC_SETTERS:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

    solver = SinglePhaseSolver(im)
    set_inlet, set_outlet = _BC_SETTERS[direction]
    getattr(solver, set_inlet)(_RHO_IN)
    getattr(solver, set_outlet)(_RHO_OUT)
    solver.set_viscosity(nu)
    solver.init_simulation()

    time_init = time.time()
    time_pre = time_init
    v_prev = None

    for i in range(n_steps + 1):
        solver.step()

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

            v_now = solver.v.to_numpy()
            if v_prev is not None and verbose:
                v_total  = np.sum(np.abs(v_now))
                v_change = np.sum(np.abs(v_now - v_prev))
                print(f"         |v|={v_total:.4e}  delta|v|={v_change:.4e}")
            v_prev = v_now
            time_pre = time_now

    if export_vtk:
        solver.export_VTK(n_steps, prefix=output_prefix)
        if verbose:
            print(f"Exported {output_prefix}_{n_steps}.vtr")

    return solver
