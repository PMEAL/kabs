"""
Compute hydraulic and diffusive conductance from LBM simulation output.

Each function accepts either a result object (``FlowResult`` / ``DiffusionResult``)
returned by ``solve_flow`` / ``solve_diffusion``, or a path to a .vtr file.

Hydraulic conductance (single-phase flow):

    $$Q = g_h * dP$$

where Q is the volumetric flow rate and dP = P_in - P_out is the total pressure
drop across the conduit.  For a circular cylinder this reduces to the
Hagen-Poiseuille result:  $g_h = pi * R^4 / (8 * mu * L)$.

Unit conversion (lattice → physical):
    g_h_phys [m^3/(Pa·s)]  =  g_h_lu  *  nu_lu  *  dx_m^3  /  mu_phys

where:
    dx_m    = physical voxel size in metres
"""


import numpy as np


__all__ = ["compute_hydraulic_conductance"]


_RHO_IN  = 1.00
_RHO_OUT = 0.99


def compute_hydraulic_conductance(
    source,
    direction=None,
    nu=None,
    dx_m=None,
    mu_phys=None,
    verbose=True,
):
    """Compute hydraulic conductance g from a single-phase LBM simulation.

    The conductance is defined by  $Q = g * (P_in - P_out)$,  where $Q$ is the
    volumetric flow rate through the conduit.  For a circular cylinder of
    radius $R$ and length $L$ this equals the Hagen-Poiseuille value
    $g = pi * R^4 / (8 * mu * L)$.

    Parameters
    ----------
    source : FlowResult or str/path-like
        Either a ``FlowResult`` returned by ``solve_flow()``, or a path to a
        ``.vtr`` file written by ``SinglePhaseSolver.export_VTK()``.
        When a ``FlowResult`` is given, ``direction`` and ``nu`` default to the
        values stored in the result.
    direction : {'x', 'y', 'z'} or None
        Flow direction.  If *None* and ``source`` is a ``FlowResult``, taken
        from ``source.direction``; otherwise defaults to ``'x'``.
    nu : float or None
        Kinematic viscosity used in the LBM simulation (lattice units).
        If *None* and ``source`` is a ``FlowResult``, taken from ``source.nu``;
        otherwise defaults to 1/6.
    dx_m : float or None
        Physical voxel size in metres.  Required for physical-unit output.
    mu_phys : float or None
        Dynamic viscosity of the fluid in Pa·s (e.g. water at 20 °C ≈ 1e-3).
        Required for physical-unit output.  Ignored if ``dx_m`` is None.
    verbose : bool
        Print a summary of results to stdout.  Default True.

    Returns
    -------
    dict with keys:
        Q_lu       – volumetric flow rate in lattice units (lu^3/ts)
        dP_lu      – total pressure drop in lattice units
        g_lu       – conductance in lattice units (lu^3/ts / lu_pressure)
        Q_m3s      – volumetric flow rate in m^3/s  (None if dx_m/mu_phys not given)
        dP_Pa      – total pressure drop in Pa       (None if dx_m/mu_phys not given)
        g_SI       – conductance in m^3/(Pa·s)       (None if dx_m/mu_phys not given)
    """
    _dir = direction if direction is not None else source.direction
    _nu  = nu        if nu        is not None else source.nu
    solid    = source.solid
    velocity = source.velocity

    _dir = _dir.lower()
    if _dir not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {_dir!r}")

    cs2 = 1.0 / 3.0  # D3Q19 speed-of-sound squared

    pore_mask = solid == 0
    vx = velocity[..., 0]
    vy = velocity[..., 1]
    vz = velocity[..., 2]
    nx, ny, nz = solid.shape

    v_flow  = {"x": vx,      "y": vy,      "z": vz     }[_dir]
    L_flow  = {"x": nx,      "y": ny,      "z": nz     }[_dir]
    A_cross = {"x": ny * nz, "y": nx * nz, "z": nx * ny}[_dir]

    u_darcy = float(np.mean(v_flow))
    Q_lu    = u_darcy * A_cross
    dP_lu   = (_RHO_IN - _RHO_OUT) * cs2
    g_lu    = Q_lu / dP_lu

    Q_m3s = dP_Pa = g_SI = None
    can_convert = (dx_m is not None) and (mu_phys is not None)
    if can_convert:
        g_SI = g_lu * _nu * dx_m**3 / mu_phys

    if verbose:
        n_pore   = int(pore_mask.sum())
        porosity = n_pore / pore_mask.size
        print(f"\nFlow direction           = {_dir}")
        print(f"Conduit length           = {L_flow}  [lu]")
        print(f"Cross-section area       = {A_cross}  [lu^2]")
        print(f"Pore voxels              = {n_pore}  (porosity = {porosity:.4f})")
        print(f"\nDarcy velocity  u_D      = {u_darcy:.6e}  [lu/ts]")
        print(f"Volumetric flow Q        = {Q_lu:.6e}  [lu^3/ts]")
        print(f"Pressure drop   dP       = {dP_lu:.6f}  [lu pressure]")
        print(f"\nConductance     g        = {g_lu:.6e}  [lu^3/ts / lu_pressure]")
        if can_convert:
            print(f"\nWith dx = {dx_m:.4e} m  and  mu = {mu_phys:.4e} Pa·s:")
            print(f"  g = {g_SI:.4e}  m^3/(Pa·s)")
        elif dx_m is None:
            print("\nTo get physical units: pass dx_m (voxel size in metres)"
                  " and mu_phys (dynamic viscosity in Pa·s).")
        else:
            print("\nTo get physical units: also pass mu_phys (dynamic viscosity in Pa·s).")

        ax_idx = {"x": 0, "y": 1, "z": 2}[_dir]
        slices = np.array([np.sum(np.take(v_flow, i, axis=ax_idx)) for i in range(L_flow)])
        print(f"\n--- Sanity check: per-slice Q (should be constant) ---")
        print(f"  Q_slice min={slices.min():.4e}  max={slices.max():.4e}"
              f"  mean={slices.mean():.4e}  std={slices.std():.4e}")

    return {
        "Q_lu":  Q_lu,
        "dP_lu": dP_lu,
        "g_lu":  g_lu,
        "Q_m3s": Q_m3s,
        "dP_Pa": dP_Pa,
        "g_SI":  g_SI,
    }


