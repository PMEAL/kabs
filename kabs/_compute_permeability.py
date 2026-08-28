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

from ._flow_common import (
    _RHO_IN,
    _RHO_OUT,
    _darcy_permeability_from_flow,
)


__all__ = ["compute_permeability"]


def _compute_permeability_core(
    solid, velocity, direction, nu, dx_m, verbose, rho_in=_RHO_IN, rho_out=_RHO_OUT
):
    """Compute permeability given numpy arrays (shared by file and object paths)."""
    pore_mask = solid == 0
    porosity = pore_mask.sum() / pore_mask.size

    vx = velocity[..., 0]
    vy = velocity[..., 1]
    vz = velocity[..., 2]
    nx, ny, nz = solid.shape

    v_flow = {"x": vx, "y": vy, "z": vz}[direction]
    L_dir = {"x": nx, "y": ny, "z": nz}[direction]

    directional_flow = float(np.sum(v_flow[pore_mask]))
    u_darcy = directional_flow / solid.size
    u_pore = float(np.mean(v_flow[pore_mask]))

    cs2 = 1.0 / 3.0
    L = L_dir
    dP = (rho_in - rho_out) * cs2
    gradP = dP / L

    k_lu = _darcy_permeability_from_flow(
        directional_flow=directional_flow,
        domain_shape=solid.shape,
        direction=direction,
        nu=nu,
        rho_in=rho_in,
        rho_out=rho_out,
    )

    k_m2 = k_mD = None
    if dx_m is not None:
        k_m2 = k_lu * dx_m**2
        k_mD = k_m2 / 9.869233e-16

    all_v = {"x": vx, "y": vy, "z": vz}
    transverse = [c for c in ("x", "y", "z") if c != direction]
    lines = [
        "",
        f"Flow direction        = {direction}",
        f"Porosity (phi)        = {porosity:.4f}",
        f"Darcy velocity  u_D   = {u_darcy:.6e}  [lu/ts]",
        f"Mean pore vel   u_p   = {u_pore:.6e}  [lu/ts]",
        f"Check: u_D / phi      = {u_darcy / porosity:.6e}  (should ≈ u_p)",
        "",
        f"Pressure drop   dP    = {dP:.6f}",
        f"Domain length   L     = {L}  [lu]",
        f"Pressure grad |dP/dL| = {gradP:.6e}",
        "",
        f"Permeability  k       = {k_lu:.6e}  [lu^2  i.e. voxels^2]",
    ]
    if dx_m is not None:
        lines += [
            "",
            f"With dx = {dx_m * 1e6:.3f} microns:",
            f"  k = {k_m2:.4e}  m^2",
            f"  k = {k_mD:.4f}   mD (milliDarcy)",
        ]
    else:
        lines.append("To get physical units: pass dx_m (voxel size in metres).")
    lines += [
        "",
        "--- Sanity checks ---",
        f"v{direction} pore: min={v_flow[pore_mask].min():.3e}  max={v_flow[pore_mask].max():.3e}",
    ]
    for c in transverse:
        vc = all_v[c]
        lines.append(
            f"v{c} pore: min={vc[pore_mask].min():.3e}  max={vc[pore_mask].max():.3e}  (should be ~0)"
        )
    summary = "\n".join(lines)

    if verbose:
        print(summary)

    return {
        "porosity": float(porosity),
        "u_darcy": u_darcy,
        "u_pore": u_pore,
        "k_lu": k_lu,
        "k_m2": k_m2,
        "k_mD": k_mD,
        "summary": summary,
    }


def compute_permeability(
    soln,
    direction=None,
    nu=None,
    dx_m=None,
    verbose=True,
):
    """Compute Darcy permeability from a single-phase LBM simulation.

    Parameters
    ----------
    soln : FlowResult
        A ``FlowResult`` returned by ``solve_flow()``.  ``direction`` and
        ``nu`` default to the values stored in the result.
    direction : {'x', 'y', 'z'} or None
        Flow direction.  If *None*, taken from ``soln.direction``.
    nu : float or None
        Kinematic viscosity in lattice units.  If *None*, taken from
        ``soln.nu``.
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
        summary    – human-readable result summary string (always populated)
    """
    _dir = direction if direction is not None else soln.direction
    _nu = nu if nu is not None else soln.nu
    solid = soln.solid
    velocity = soln.velocity

    _dir = _dir.lower()
    if _dir not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {_dir!r}")

    rho_in = getattr(soln, "rho_in", None)
    rho_out = getattr(soln, "rho_out", None)
    if rho_in is None:
        rho_in = _RHO_IN
    if rho_out is None:
        rho_out = _RHO_OUT
    return _compute_permeability_core(
        solid, velocity, _dir, _nu, dx_m, verbose, rho_in, rho_out
    )
