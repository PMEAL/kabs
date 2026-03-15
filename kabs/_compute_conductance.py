"""
Compute hydraulic and diffusive conductance from LBM simulation output.

Each function accepts either a result object (``FlowResult`` / ``DiffusionResult``)
returned by ``solve_flow`` / ``solve_diffusion``, or a path to a .vtr file.

Hydraulic conductance (single-phase flow):

    Q = g_h * dP

where Q is the volumetric flow rate and dP = P_in - P_out is the total pressure
drop across the conduit.  For a circular cylinder this reduces to the
Hagen-Poiseuille result:  g_h = pi * R^4 / (8 * mu * L).

Unit conversion (lattice → physical):
    g_h_phys [m^3/(Pa·s)]  =  g_h_lu  *  nu_lu  *  dx_m^3  /  mu_phys

Diffusive conductance (passive scalar diffusion):

    J_total = g_d * Δc

where J_total is the total diffusive flux through the conduit cross-section and
Δc = c_in - c_out is the concentration difference.  For a circular cylinder of
radius R this equals  g_d = pi * R^2 * D_eff / L  (Fick's law).

Unit conversion (lattice → physical):
    g_d_phys [m^3/s]  =  g_d_lu  *  D0_m2s  *  dx_m  /  D_lu

where:
    D_lu    = LBM bulk diffusivity (lattice units, default 1/4)
    D0_m2s  = physical bulk diffusivity in m^2/s
    dx_m    = physical voxel size in metres
"""

import re
import struct

import numpy as np

# Import the shared VTR parsing helpers from the permeability module.
from ._compute_permeability import _parse_xml_arrays, _read_array


__all__ = ["compute_hydraulic_conductance", "compute_diffusive_conductance"]


_RHO_IN  = 1.00
_RHO_OUT = 0.99


def _read_flow_vtr(vtr_file, verbose):
    """Read solid and velocity arrays from a flow .vtr file."""
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
    if verbose:
        print("  Arrays loaded.")
    return solid, velocity


def _read_diffusion_vtr(vtr_file, verbose):
    """Read solid and flux arrays from a diffusion .vtr file."""
    if verbose:
        print(f"Reading {vtr_file} ...")
    with open(vtr_file, "rb") as fh:
        raw = fh.read()

    marker       = raw.index(b'<AppendedData encoding="raw">')
    binary_start = raw.index(b"_", marker) + 1
    xml_header   = raw[:marker].decode("utf-8", errors="replace")
    arrays       = _parse_xml_arrays(xml_header)

    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    if verbose:
        print(f"  Grid: {nx} x {ny} x {nz} points")

    solid    = _read_array(raw, binary_start, arrays, "Solid", nx, ny, nz)
    flux_vec = _read_array(raw, binary_start, arrays, "flux",  nx, ny, nz)
    if verbose:
        print("  Arrays loaded.")
    return solid, flux_vec


def compute_hydraulic_conductance(
    source,
    direction=None,
    nu=None,
    dx_m=None,
    mu_phys=None,
    verbose=True,
):
    """Compute hydraulic conductance g from a single-phase LBM simulation.

    The conductance is defined by  Q = g * (P_in - P_out),  where Q is the
    volumetric flow rate through the conduit.  For a circular cylinder of
    radius R and length L this equals the Hagen-Poiseuille value
    g = pi * R^4 / (8 * mu * L).

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
    from ._solve_flow import FlowResult

    if isinstance(source, FlowResult):
        _dir = direction if direction is not None else source.direction
        _nu  = nu        if nu        is not None else source.nu
        solid    = source.solid
        velocity = source.velocity
    else:
        _dir = direction if direction is not None else "x"
        _nu  = nu        if nu        is not None else 1.0 / 6.0
        solid, velocity = _read_flow_vtr(source, verbose)

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


_C_IN  = 1.00
_C_OUT = 0.00


def compute_diffusive_conductance(
    source,
    direction=None,
    D_lu=None,
    D0_m2s=None,
    dx_m=None,
    verbose=True,
):
    """Compute diffusive conductance g_d from a D3Q7 BGK LBM simulation.

    The conductance is defined by  J_total = g_d * (c_in - c_out),  where
    J_total is the total diffusive flux through the conduit cross-section.
    For a circular cylinder of radius R and length L this equals the
    analytical value  g_d = pi * R^2 * D_eff / L.

    Parameters
    ----------
    source : DiffusionResult or str/path-like
        Either a ``DiffusionResult`` returned by ``solve_diffusion()``, or a
        path to a ``.vtr`` file written by ``DiffusionSolver.export_VTK()``.
        When a ``DiffusionResult`` is given, ``direction`` and ``D_lu`` default
        to the values stored in the result.
    direction : {'x', 'y', 'z'} or None
        Flow direction.  If *None* and ``source`` is a ``DiffusionResult``,
        taken from ``source.direction``; otherwise defaults to ``'x'``.
    D_lu : float or None
        Bulk diffusivity in lattice units used during the simulation.
        If *None* and ``source`` is a ``DiffusionResult``, taken from
        ``source.D``; otherwise defaults to 1/4.
    D0_m2s : float or None
        Physical bulk diffusivity in m^2/s.  Required together with
        ``dx_m`` for physical-unit output.
    dx_m : float or None
        Physical voxel size in metres.  Required together with ``D0_m2s``
        for physical-unit output.
    verbose : bool
        Print a summary of results to stdout.  Default True.

    Returns
    -------
    dict with keys:
        J_lu    – total diffusive flux through the cross-section [lattice units]
        dc_lu   – concentration difference c_in - c_out [lattice units]
        g_lu    – diffusive conductance in lattice units (lu^3/ts)
        g_SI    – diffusive conductance in m^3/s  (None if D0_m2s/dx_m not given)
    """
    from ._solve_diffusion import DiffusionResult

    if isinstance(source, DiffusionResult):
        _dir  = direction if direction is not None else source.direction
        _D_lu = D_lu      if D_lu      is not None else source.D
        solid    = source.solid
        flux_vec = source.flux
    else:
        _dir  = direction if direction is not None else "x"
        _D_lu = D_lu      if D_lu      is not None else 1.0 / 4.0
        solid, flux_vec = _read_diffusion_vtr(source, verbose)

    _dir = _dir.lower()
    if _dir not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {_dir!r}")

    pore_mask = solid == 0
    nx, ny, nz = solid.shape
    A_cross   = {"x": ny * nz, "y": nx * nz, "z": nx * ny}[_dir]
    L_flow    = {"x": nx,      "y": ny,      "z": nz     }[_dir]
    dir_idx   = {"x": 0, "y": 1, "z": 2}[_dir]
    flux      = flux_vec[..., dir_idx]

    J_lu  = float(np.mean(flux)) * A_cross
    dc_lu = _C_IN - _C_OUT  # = 1.0
    g_lu  = J_lu / dc_lu

    g_SI = None
    can_convert = (D0_m2s is not None) and (dx_m is not None)
    if can_convert:
        g_SI = g_lu * D0_m2s * dx_m / _D_lu

    if verbose:
        n_pore   = int(pore_mask.sum())
        porosity = n_pore / pore_mask.size
        print(f"\nFlow direction           = {_dir}")
        print(f"Conduit length           = {L_flow}  [lu]")
        print(f"Cross-section area       = {A_cross}  [lu^2]")
        print(f"Pore voxels              = {n_pore}  (porosity = {porosity:.4f})")
        print(f"\nMean flux (flow dir)     = {float(np.mean(flux)):.6e}  [lu]")
        print(f"Total flux  J            = {J_lu:.6e}  [lu^3/ts]")
        print(f"Conc. difference dc      = {dc_lu:.4f}  [lu]")
        print(f"\nDiffusive conductance g  = {g_lu:.6e}  [lu^3/ts / lu_conc]")
        if can_convert:
            print(f"\nWith D0 = {D0_m2s:.4e} m^2/s  and  dx = {dx_m:.4e} m:")
            print(f"  g_d = {g_SI:.4e}  m^3/s")
        elif D0_m2s is None:
            print("\nTo get physical units: pass D0_m2s (bulk diffusivity in m^2/s)"
                  " and dx_m (voxel size in metres).")
        else:
            print("\nTo get physical units: also pass D0_m2s (bulk diffusivity in m^2/s).")

        ax_idx = {"x": 0, "y": 1, "z": 2}[_dir]
        slices = np.array([np.sum(np.take(flux, i, axis=ax_idx)) for i in range(L_flow)])
        print("\n--- Sanity check: per-slice J (should be constant) ---")
        print(f"  J_slice min={slices.min():.4e}  max={slices.max():.4e}"
              f"  mean={slices.mean():.4e}  std={slices.std():.4e}")

    return {
        "J_lu":  J_lu,
        "dc_lu": dc_lu,
        "g_lu":  g_lu,
        "g_SI":  g_SI,
    }
