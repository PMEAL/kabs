"""
Compute hydraulic conductance from LBM single-phase simulation output.

For a conduit of arbitrary cross-section, the hydraulic conductance g is defined by:

    Q = g * dP

where Q is the volumetric flow rate and dP = P_in - P_out is the total pressure
drop across the conduit.  For a circular cylinder this reduces to the
Hagen-Poiseuille result:  g = pi * R^4 / (8 * mu * L).

The LBM velocity field gives Q directly; dP follows from the fixed boundary
densities used by solve_flow (rho_in=1.00, rho_out=0.99, dP = Drho * cs^2).

Unit conversion (lattice → physical):
    g_phys [m^3/(Pa·s)]  =  g_lu  *  nu_lu  *  dx_m^3  /  mu_phys

where:
    nu_lu   = LBM kinematic viscosity (lattice units, default 1/6)
    dx_m    = physical voxel size in metres
    mu_phys = dynamic viscosity of the fluid in Pa·s
              (e.g. water at 20 °C ≈ 1.002e-3 Pa·s)
"""

import re
import struct

import numpy as np

# Import the shared VTR parsing helpers from the permeability module.
from ._compute_permeability import _parse_xml_arrays, _read_array


__all__ = ["compute_conductance"]


_RHO_IN  = 1.00
_RHO_OUT = 0.99


def compute_conductance(
    vtr_file,
    direction="x",
    nu=1.0 / 6.0,
    dx_m=None,
    mu_phys=None,
    verbose=True,
):
    """Compute hydraulic conductance g from a single-phase LBM .vtr output file.

    The conductance is defined by  Q = g * (P_in - P_out),  where Q is the
    volumetric flow rate through the conduit.  For a circular cylinder of
    radius R and length L this equals the Hagen-Poiseuille value
    g = pi * R^4 / (8 * mu * L).

    Parameters
    ----------
    vtr_file : str or path-like
        Path to the VTR file written by ``SinglePhaseSolver.export_VTK()``.
    direction : {'x', 'y', 'z'}
        Flow direction used in the simulation.  Default ``'x'``.
    nu : float
        Kinematic viscosity used in the LBM simulation (lattice units).
        Default 1/6 (the solver default).
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
    direction = direction.lower()
    if direction not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

    cs2 = 1.0 / 3.0  # D3Q19 speed-of-sound squared

    # ── Read VTR file ─────────────────────────────────────────────────────────
    if verbose:
        print(f"Reading {vtr_file} ...")
    with open(vtr_file, "rb") as fh:
        raw = fh.read()

    marker = raw.index(b'<AppendedData encoding="raw">')
    binary_start = raw.index(b"_", marker) + 1

    xml_header = raw[:marker].decode("utf-8", errors="replace")
    arrays = _parse_xml_arrays(xml_header)

    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    if verbose:
        print(f"  Grid: {nx} x {ny} x {nz} points")

    solid    = _read_array(raw, binary_start, arrays, "Solid",    nx, ny, nz)
    velocity = _read_array(raw, binary_start, arrays, "velocity", nx, ny, nz)
    vx = velocity[..., 0]
    vy = velocity[..., 1]
    vz = velocity[..., 2]
    if verbose:
        print("  Arrays loaded.")

    # ── Geometry ──────────────────────────────────────────────────────────────
    pore_mask = solid == 0
    v_flow    = {"x": vx, "y": vy, "z": vz}[direction]
    L_flow    = {"x": nx, "y": ny, "z": nz}[direction]  # conduit length
    A_cross   = {"x": ny * nz, "y": nx * nz, "z": nx * ny}[direction]  # cross-section area

    # ── Volumetric flow rate ───────────────────────────────────────────────────
    # In steady incompressible flow Q is the same through any cross-section.
    # Summing v over the whole domain and dividing by length gives the
    # per-slice average, i.e. Q = mean_slice(sum(v)) = u_darcy * A_cross.
    # Solid voxels contribute v=0 naturally.
    u_darcy = float(np.mean(v_flow))       # superficial (Darcy) velocity [lu/ts]
    Q_lu    = u_darcy * A_cross            # volumetric flow rate [lu^3/ts]

    # ── Pressure drop ─────────────────────────────────────────────────────────
    dP_lu = (_RHO_IN - _RHO_OUT) * cs2    # total dP across the conduit [lu pressure]

    # ── Conductance (lattice units) ───────────────────────────────────────────
    g_lu = Q_lu / dP_lu

    # ── Physical units ────────────────────────────────────────────────────────
    # Unit conversion:  g_phys [m^3/(Pa·s)] = g_lu * nu_lu * dx_m^3 / mu_phys
    # Derived from: dt_s = nu_lu/nu_phys * dx_m^2,  v_scale = dx_m/dt_s,
    #               P_scale = rho_phys * (dx_m/dt_s)^2,  mu_phys = nu_phys*rho_phys
    Q_m3s = dP_Pa = g_SI = None
    can_convert = (dx_m is not None) and (mu_phys is not None)
    if can_convert:
        # Time scale:  dt_s = nu_lu / nu_phys * dx_m^2
        # nu_phys = mu_phys / rho_phys — but we can fold everything into:
        #   g_phys = g_lu * nu_lu * dx_m^3 / mu_phys
        g_SI = g_lu * nu * dx_m**3 / mu_phys

        # Derive dP and Q in SI from the LBM scales for reporting
        # P_scale = mu_phys^2 / (rho_phys * dx_m^2) -- requires rho_phys separately.
        # Instead, back-calculate from g_SI and g_lu:
        #   g_SI = Q_SI / dP_SI  and  Q_SI / Q_lu = dx_m^3 / dt_s
        # We don't report Q_m3s and dP_Pa individually unless rho_phys is given,
        # so we leave them as None and only report g_SI.
        pass

    # ── Verbose output ────────────────────────────────────────────────────────
    if verbose:
        n_pore  = int(pore_mask.sum())
        porosity = n_pore / pore_mask.size
        print(f"\nFlow direction           = {direction}")
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

        # Per-slice flow rate check (should be constant in steady state)
        ax_idx = {"x": 0, "y": 1, "z": 2}[direction]
        slices = [np.sum(np.take(v_flow, i, axis=ax_idx)) for i in range(L_flow)]
        slices = np.array(slices)
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
