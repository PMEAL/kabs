"""Backend-neutral flow result types, constants, and argument helpers."""

from .utils import write_flow_vtr


__all__ = ["FlowResult"]


_RHO_IN = 1.00
_RHO_OUT = 0.99


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
    convergence_criterion : float or None
        Final value of ``delta|v| / |v|``, when measured.
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
    ):
        self.direction = direction
        self.nu = nu
        self.n_iterations = n_iterations
        self.convergence_criterion = convergence_criterion
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
    ):
        """Construct a result directly from NumPy-compatible arrays.

        This path is used by non-Taichi backends and VTR imports. Direction,
        viscosity, and collision metadata may be omitted when unavailable.
        """
        obj = object.__new__(cls)
        obj.direction = direction
        obj.nu = nu
        obj.n_iterations = None
        obj.convergence_criterion = None
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

    def export_to_vtk(self, prefix):
        """Write a VTK Rectilinear Grid (``.vtr``) file.

        Parameters
        ----------
        prefix : str or path-like
            Output path without the extension added by pyevtk.
        """
        write_flow_vtr(prefix, self)
