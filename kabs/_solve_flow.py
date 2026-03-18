"""High-level entry point: run a single-phase LBM flow simulation."""

import time

import numpy as np

from ._single_phase_solver import SinglePhaseSolver
from .utils import write_flow_vtr


__all__ = ["solve_flow", "FlowResult"]

# Pressure BCs are hardcoded: only the difference matters for Darcy's law,
# and both u_D and gradP scale with Δρ so they cancel in k = u_D*mu/gradP.
_RHO_IN = 1.00
_RHO_OUT = 0.99

_BC_SETTERS = {
    "x": ("set_bc_rho_x0", "set_bc_rho_x1"),
    "y": ("set_bc_rho_y0", "set_bc_rho_y1"),
    "z": ("set_bc_rho_z0", "set_bc_rho_z1"),
}


class FlowResult:
    """Container for a converged single-phase LBM flow simulation.

    Attributes
    ----------
    solid : np.ndarray, shape (nx, ny, nz), dtype int8
        Solid mask in internal convention: 1 = solid, 0 = pore.
    rho : np.ndarray, shape (nx, ny, nz), dtype float32
        Density field.
    velocity : np.ndarray, shape (nx, ny, nz, 3), dtype float32
        Velocity field (vx, vy, vz) at each voxel.
    direction : str
        Flow direction used in the simulation ('x', 'y', or 'z').
    nu : float
        Kinematic viscosity in lattice units.
    """

    def __init__(self, solver, direction, nu):
        self.direction = direction
        self.nu = nu
        self._solver = solver
        self.solid = solver.solid.to_numpy()
        self.rho = solver.rho.to_numpy()
        self.velocity = solver.v.to_numpy()  # shape (nx, ny, nz, 3)

    @classmethod
    def from_arrays(cls, solid, rho, velocity, direction=None, nu=None):
        """Construct a FlowResult directly from numpy arrays (e.g. read from a VTR file).

        Parameters
        ----------
        solid : np.ndarray, shape (nx, ny, nz), dtype int8
            Solid mask: 1 = solid, 0 = pore.
        rho : np.ndarray, shape (nx, ny, nz), dtype float32
            Density field.
        velocity : np.ndarray, shape (nx, ny, nz, 3), dtype float32
            Velocity field.
        direction : {'x', 'y', 'z'} or None
            Flow direction.  Can be supplied later to ``compute_permeability``.
        nu : float or None
            Kinematic viscosity in lattice units.  Can be supplied later.
        """
        obj = object.__new__(cls)
        obj.direction = direction
        obj.nu = nu
        obj._solver = None
        obj.solid = solid
        obj.rho = rho
        obj.velocity = velocity
        return obj

    def export_to_vtk(self, prefix):
        """Write a VTK Rectilinear Grid (.vtr) file.

        Parameters
        ----------
        prefix : str
            Output path without extension (pyevtk appends ``.vtr``).
        """
        write_flow_vtr(prefix, self)


def solve_flow(
    im,
    direction="x",
    n_steps=15000,
    nu=1.0 / 6.0,
    log_every=500,
    export_vtk=True,
    output_prefix="LB_SinglePhase",
    verbose=True,
    sparse=False,
    tol=1e-3,
):
    """
    Run a pressure-driven single-phase LBM simulation to steady state.

    Parameters
    ----------
    im : np.ndarray, shape (nx, ny, nz)
        Binary image of the pore space.  1 (or True) = pore, 0 (or False) = solid.
        This matches the PoreSpy convention.
    direction : {'x', 'y', 'z'}
        Axis along which the pressure gradient is applied.  Default ``'x'``.
    n_steps : int
        Number of LBM time steps to run.  Default 15000.
    nu : float
        Kinematic viscosity in lattice units.  Default 1/6.
    log_every : int
        Print a progress line every this many steps.  Default 500.
    export_vtk : bool
        If True (default), write ``{output_prefix}-{final_step}-{direction}.vtr``
        at the end.
    output_prefix : str
        Filename prefix for the VTR output.  Default ``'LB_SinglePhase'``.
    verbose : bool
        Print progress to stdout.  Default True.
    sparse : bool
        If True, use Taichi sparse (pointer-backed) storage for the
        distribution fields.  Only pore cells are allocated, which can
        significantly reduce memory on images with high solid fractions.
        Default False (dense storage).
    tol : float or None
        Convergence tolerance.  The simulation stops early when the relative
        change in the total velocity magnitude between log intervals falls
        below this value: ``delta|v| / |v| < tol``.  Set to ``None`` to
        always run the full ``n_steps``.  Default 1e-3.

    Returns
    -------
    result : FlowResult
        Result object containing ``solid``, ``rho``, ``velocity``, ``direction``,
        and ``nu`` as numpy arrays/values.  Pass directly to
        ``compute_permeability()`` or ``compute_hydraulic_conductance()``,
        or call ``result.export_to_vtk(prefix)`` to save a VTR file.

    Notes
    -----
    Taichi must be initialized by the caller before invoking this function:

        import taichi as ti
        ti.init(arch=ti.cpu)
    """
    direction = direction.lower()
    if direction not in _BC_SETTERS:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

    # Public convention: 1=pore, 0=solid (PoreSpy-compatible).
    # SinglePhaseSolver uses the opposite, so flip here.
    solid_im = (im == 0).astype(np.int8)
    solver = SinglePhaseSolver(solid_im, sparse_storage=sparse)
    set_inlet, set_outlet = _BC_SETTERS[direction]
    getattr(solver, set_inlet)(_RHO_IN)
    getattr(solver, set_outlet)(_RHO_OUT)
    solver.set_viscosity(nu)
    solver.init_simulation()

    time_init = time.time()
    time_pre = time_init
    v_prev = None
    final_step = n_steps

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
            if v_prev is not None:
                v_total = np.sum(np.abs(v_now))
                v_change = np.sum(np.abs(v_now - v_prev))
                if verbose:
                    print(f"         |v|={v_total:.4e}  delta|v|={v_change:.4e}")
                if tol is not None and v_total > 0 and v_change / v_total < tol:
                    if verbose:
                        print(
                            f"Converged at step {i} "
                            f"(delta|v|/|v| = {v_change / v_total:.2e} < tol={tol:.2e})"
                        )
                    final_step = i
                    break
            v_prev = v_now
            time_pre = time_now

    result = FlowResult(solver, direction, nu)

    if export_vtk:
        vtk_path = f"{output_prefix}-{final_step}-{direction}"
        result.export_to_vtk(vtk_path)
        if verbose:
            print(f"Exported {vtk_path}.vtr")

    return result
