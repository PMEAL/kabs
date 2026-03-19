"""
Compute Darcy permeability from LBM single-phase simulation output.

Accepts a ``FlowResult`` object returned by ``solve_flow``.

Darcy's Law:  k = u_D * mu / |dP/dL|

  u_D  = Darcy (superficial) velocity = mean(vx) over the whole domain
           Solid voxels have vx=0, so this naturally accounts for porosity.
  mu   = dynamic viscosity (LBM lattice units); mu = rho * nu ~ nu since rho~1
  dP/dL = pressure gradient = (rho_in - rho_out) * cs^2 / L
           cs^2 = 1/3 in D3Q19 LBM,  L = nx (domain length in lattice units)

Result is in lattice units (voxels^2).  Multiply by dx_m**2 to get m^2,
then divide by 9.869e-16 to get milliDarcy.
"""

import numpy as np


__all__ = ["compute_permeability"]


_RHO_IN  = 1.00
_RHO_OUT = 0.99


def _compute_permeability_core(solid, velocity, direction, nu, dx_m, verbose):
    """Compute permeability given numpy arrays (shared by file and object paths)."""
    pore_mask = solid == 0
    porosity  = pore_mask.sum() / pore_mask.size

    vx = velocity[..., 0]
    vy = velocity[..., 1]
    vz = velocity[..., 2]
    nx, ny, nz = solid.shape

    v_flow = {"x": vx, "y": vy, "z": vz}[direction]
    L_dir  = {"x": nx, "y": ny, "z": nz}[direction]

    u_darcy = float(np.mean(v_flow))
    u_pore  = float(np.mean(v_flow[pore_mask]))

    cs2   = 1.0 / 3.0
    L     = L_dir
    dP    = (_RHO_IN - _RHO_OUT) * cs2
    gradP = dP / L

    mu   = nu
    k_lu = u_darcy * mu / gradP

    k_m2 = k_mD = None
    if dx_m is not None:
        k_m2 = k_lu * dx_m ** 2
        k_mD = k_m2 / 9.869233e-16

    if verbose:
        print(f"\nFlow direction        = {direction}")
        print(f"Porosity (phi)        = {porosity:.4f}")
        print(f"Darcy velocity  u_D   = {u_darcy:.6e}  [lu/ts]")
        print(f"Mean pore vel   u_p   = {u_pore:.6e}  [lu/ts]")
        print(f"Check: u_D / phi      = {u_darcy / porosity:.6e}  (should ≈ u_p)")
        print(f"\nPressure drop   dP    = {dP:.6f}")
        print(f"Domain length   L     = {L}  [lu]")
        print(f"Pressure grad |dP/dL| = {gradP:.6e}")
        print(f"\nPermeability  k       = {k_lu:.6e}  [lu^2  i.e. voxels^2]")
        if dx_m is not None:
            print(f"\nWith dx = {dx_m * 1e6:.3f} microns:")
            print(f"  k = {k_m2:.4e}  m^2")
            print(f"  k = {k_mD:.4f}   mD (milliDarcy)")
        else:
            print("\nTo get physical units: pass dx_m (voxel size in metres).")
        all_v = {"x": vx, "y": vy, "z": vz}
        transverse = [c for c in ("x", "y", "z") if c != direction]
        print("\n--- Sanity checks ---")
        print(f"v{direction} pore: min={v_flow[pore_mask].min():.3e}  max={v_flow[pore_mask].max():.3e}")
        for c in transverse:
            vc = all_v[c]
            print(f"v{c} pore: min={vc[pore_mask].min():.3e}  max={vc[pore_mask].max():.3e}  (should be ~0)")

    return {
        "porosity": float(porosity),
        "u_darcy": u_darcy,
        "u_pore": u_pore,
        "k_lu": k_lu,
        "k_m2": k_m2,
        "k_mD": k_mD,
    }


def compute_permeability(
    source,
    direction=None,
    nu=None,
    dx_m=None,
    verbose=True,
):
    """Compute Darcy permeability from a single-phase LBM simulation.

    Parameters
    ----------
    source : FlowResult
        A ``FlowResult`` returned by ``solve_flow()``.  ``direction`` and
        ``nu`` default to the values stored in the result.
    direction : {'x', 'y', 'z'} or None
        Flow direction.  If *None*, taken from ``source.direction``.
    nu : float or None
        Kinematic viscosity in lattice units.  If *None*, taken from
        ``source.nu``.
    dx_m : float or None
        Physical voxel size in metres.  If given, results are also reported
        in m² and milliDarcy.  E.g. ``dx_m=2.85e-6`` for a 2.85-µm scan.
    verbose : bool
        Print a summary of results to stdout.  Default True.

    Returns
    -------
    dict with keys:
        porosity   – pore fraction (dimensionless)
        u_darcy    – superficial velocity in the flow direction (lu/ts)
        u_pore     – mean pore-space velocity (lu/ts)
        k_lu       – permeability in lattice units (voxels²)
        k_m2       – permeability in m²  (None if dx_m is None)
        k_mD       – permeability in milliDarcy  (None if dx_m is None)
    """
    _dir = direction if direction is not None else source.direction
    _nu  = nu        if nu        is not None else source.nu
    solid    = source.solid
    velocity = source.velocity

    _dir = _dir.lower()
    if _dir not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {_dir!r}")

    return _compute_permeability_core(solid, velocity, _dir, _nu, dx_m, verbose)
