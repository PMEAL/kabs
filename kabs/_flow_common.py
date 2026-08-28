"""Backend-neutral flow result types, constants, and argument helpers."""

from .utils import write_flow_vtr


__all__ = ["FlowResult"]


_RHO_IN = 1.00
_RHO_OUT = 0.99


def _darcy_permeability_from_flow(
    *, directional_flow, domain_shape, direction, nu, rho_in, rho_out
):
    """Return lattice permeability from a pore-only directional-flow sum."""
    axis = {"x": 0, "y": 1, "z": 2}[direction]
    domain_volume = int(domain_shape[0]) * int(domain_shape[1]) * int(domain_shape[2])
    u_darcy = float(directional_flow) / domain_volume
    pressure_gradient = abs((float(rho_in) - float(rho_out)) / 3.0 / domain_shape[axis])
    if pressure_gradient == 0.0:
        raise ValueError("rho_in and rho_out must define a nonzero pressure drop")
    return u_darcy * float(nu) / pressure_gradient


class _DefaultValue:
    """Sentinel whose repr keeps backwards-compatible signature defaults readable."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)


_DEFAULT_STORAGE = _DefaultValue("dense")
_DEFAULT_SPARSE = _DefaultValue(False)


def _normalize_collision_model(collision_model):
    if not isinstance(collision_model, str):
        raise ValueError(
            "collision_model must be 'mrt' or 'srt', "
            f"got {collision_model!r}"
        )

    collision_model = collision_model.lower()
    if collision_model not in ("mrt", "srt"):
        raise ValueError(
            "collision_model must be 'mrt' or 'srt', "
            f"got {collision_model!r}"
        )
    return collision_model


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
    direction : {'x', 'y', 'z'} or None
        Flow direction used by the simulation.
    nu : float or None
        Kinematic viscosity in lattice units.
    n_iterations : int or None
        Number of LBM time steps that were executed.
    converged : bool or None
        True when all enabled criteria passed twice, False when an enabled
        policy exhausted ``n_steps``, or None when monitoring was disabled.
    velocity_criterion, k_criterion, flux_criterion : float or None
        Final measured convergence criteria. ``convergence_criterion`` is a
        backward-compatible alias for ``velocity_criterion``.
    velocity_tol, k_tol, flux_tol : float or None
        Effective tolerances used by the solve.
    rho_in, rho_out : float or None
        Pressure-boundary densities used by the solve.
    collision_model : {'mrt', 'srt'} or None
        Collision operator used by the simulation.
    """

    def __init__(
        self,
        solver,
        direction,
        nu,
        n_iterations=None,
        convergence_criterion=None,
        *,
        rho_in=_RHO_IN,
        rho_out=_RHO_OUT,
        converged=None,
        velocity_tol=None,
        k_tol=None,
        flux_tol=None,
        velocity_criterion=None,
        k_criterion=None,
        flux_criterion=None,
        convergence_every=None,
        consecutive_passes=0,
    ):
        self.direction = direction
        self.nu = nu
        self.n_iterations = n_iterations
        self.rho_in = rho_in
        self.rho_out = rho_out
        self.converged = converged
        self.velocity_tol = velocity_tol
        self.k_tol = k_tol
        self.flux_tol = flux_tol
        self.velocity_criterion = (
            velocity_criterion
            if velocity_criterion is not None
            else convergence_criterion
        )
        self.k_criterion = k_criterion
        self.flux_criterion = flux_criterion
        self.convergence_every = convergence_every
        self.consecutive_passes = consecutive_passes
        self.collision_model = getattr(solver, "collision_model", None)
        self._solver = solver
        self.solid = solver.solid.to_numpy()
        self.rho = solver.get_rho()
        self.velocity = solver.get_velocity()

    @classmethod
    def from_arrays(
        cls,
        solid,
        rho,
        velocity,
        direction=None,
        nu=None,
        collision_model=None,
        *,
        rho_in=None,
        rho_out=None,
        converged=None,
        n_iterations=None,
        velocity_tol=None,
        k_tol=None,
        flux_tol=None,
        velocity_criterion=None,
        k_criterion=None,
        flux_criterion=None,
        convergence_every=None,
        consecutive_passes=None,
    ):
        """Construct a result directly from NumPy-compatible arrays.

        This path is used by non-Taichi backends and VTR imports. Direction,
        viscosity, and collision metadata may be omitted when unavailable.
        """
        obj = object.__new__(cls)
        obj.direction = direction
        obj.nu = nu
        obj.n_iterations = n_iterations
        obj.rho_in = rho_in
        obj.rho_out = rho_out
        obj.converged = converged
        obj.velocity_tol = velocity_tol
        obj.k_tol = k_tol
        obj.flux_tol = flux_tol
        obj.velocity_criterion = velocity_criterion
        obj.k_criterion = k_criterion
        obj.flux_criterion = flux_criterion
        obj.convergence_every = convergence_every
        obj.consecutive_passes = consecutive_passes
        obj.collision_model = (
            None
            if collision_model is None
            else _normalize_collision_model(collision_model)
        )
        obj._solver = None
        obj.solid = solid
        obj.rho = rho
        obj.velocity = velocity
        return obj

    @property
    def convergence_criterion(self):
        """Backward-compatible alias for ``velocity_criterion``."""
        return self.velocity_criterion

    @convergence_criterion.setter
    def convergence_criterion(self, value):
        self.velocity_criterion = value

    def export_to_vtk(self, prefix):
        """Write a VTK Rectilinear Grid (``.vtr``) file.

        Parameters
        ----------
        prefix : str or path-like
            Output path without the extension added by pyevtk.
        """
        write_flow_vtr(prefix, self)
