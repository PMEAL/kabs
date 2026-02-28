"""
Compute Darcy permeability from LBM single-phase simulation output.

Reads the .vtr file written by pyevtk using only numpy (no vtk/pyvista needed).

Darcy's Law:  k = u_D * mu / |dP/dL|

  u_D  = Darcy (superficial) velocity = mean(vx) over the whole domain
           Solid voxels have vx=0, so this naturally accounts for porosity.
  mu   = dynamic viscosity (LBM lattice units); mu = rho * nu ~ nu since rho~1
  dP/dL = pressure gradient = (rho_in - rho_out) * cs^2 / L
           cs^2 = 1/3 in D3Q19 LBM,  L = nx (domain length in lattice units)

Result is in lattice units (voxels^2).  Multiply by dx_m**2 to get m^2,
then divide by 9.869e-16 to get milliDarcy.
"""

import re
import struct

import numpy as np


__all__ = ["compute_permeability"]


def _parse_xml_arrays(xml):
    """Return a dict of {name: (offset, dtype, n_components)} from a VTK XML header."""
    arrays = {}
    for m in re.finditer(
        r'<DataArray Name="(\w+)"[^>]*NumberOfComponents="(\d+)"'
        r'[^>]*type="(\w+)"[^>]*offset="(\d+)"',
        xml,
    ):
        name, n_comp, dtype_str, offset = m.groups()
        dtype_map = {
            "Float32": np.float32,
            "Float64": np.float64,
            "Int8": np.int8,
            "Int32": np.int32,
            "UInt64": np.uint64,
        }
        arrays[name] = (int(offset), dtype_map[dtype_str], int(n_comp))
    return arrays


def _read_array(raw, binary_start, arrays, name, nx, ny, nz):
    """Read one named array from the raw VTR bytes.

    pyevtk writes vector data in AoS format (interleaved per point) with
    Fortran-order spatial indexing:
        vx[0,0,0], vy[0,0,0], vz[0,0,0], vx[1,0,0], ...
    Scalar arrays are written in Fortran order directly.
    """
    offset, dtype, n_comp = arrays[name]
    pos = binary_start + offset
    # 8-byte UInt64 header = number of *bytes* in this array
    n_bytes, = struct.unpack_from("<Q", raw, pos)
    pos += 8
    data = np.frombuffer(raw, dtype=dtype, count=n_bytes // dtype().itemsize, offset=pos)
    if n_comp > 1:
        data_pts = data.reshape(nx * ny * nz, n_comp)
        data = np.stack(
            [data_pts[:, c].reshape(nx, ny, nz, order="F") for c in range(n_comp)],
            axis=-1,
        )
    else:
        data = data.reshape((nx, ny, nz), order="F")
    return data


_RHO_IN  = 1.00
_RHO_OUT = 0.99


def compute_permeability(
    vtr_file,
    direction="x",
    nu=1.0 / 6.0,
    dx_m=None,
    verbose=True,
):
    """Compute Darcy permeability from a single-phase LBM .vtr output file.

    Parameters
    ----------
    vtr_file : str or path-like
        Path to the VTR file written by ``SinglePhaseSolver.export_VTK()``.
    direction : {'x', 'y', 'z'}
        Flow direction used in the simulation.  Determines which velocity
        component and domain length are used.  Default ``'x'``.
    nu : float
        Kinematic viscosity in lattice units.  Default 1/6.
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
    direction = direction.lower()
    if direction not in ("x", "y", "z"):
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

    cs2 = 1.0 / 3.0  # speed-of-sound squared for D3Q19

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
    rho      = _read_array(raw, binary_start, arrays, "rho",      nx, ny, nz)
    velocity = _read_array(raw, binary_start, arrays, "velocity", nx, ny, nz)
    vx = velocity[..., 0]
    vy = velocity[..., 1]
    vz = velocity[..., 2]
    if verbose:
        print("  Arrays loaded.")

    # ── Porosity ──────────────────────────────────────────────────────────────
    pore_mask = solid == 0
    porosity  = pore_mask.sum() / pore_mask.size

    # ── Darcy (superficial) velocity ──────────────────────────────────────────
    # Average the flow-direction component over the ENTIRE domain;
    # solid voxels have v=0, so this naturally accounts for porosity.
    v_flow = {"x": vx, "y": vy, "z": vz}[direction]
    L_dir  = {"x": nx, "y": ny, "z": nz}[direction]

    u_darcy = float(np.mean(v_flow))
    u_pore  = float(np.mean(v_flow[pore_mask]))

    # ── Pressure gradient ─────────────────────────────────────────────────────
    L     = L_dir
    dP    = (_RHO_IN - _RHO_OUT) * cs2
    gradP = dP / L

    # ── Darcy permeability ────────────────────────────────────────────────────
    mu   = nu  # dynamic viscosity; rho_ref = 1 in LBM so mu = nu
    k_lu = u_darcy * mu / gradP

    # ── Physical units ────────────────────────────────────────────────────────
    k_m2 = k_mD = None
    if dx_m is not None:
        k_m2 = k_lu * dx_m ** 2
        k_mD = k_m2 / 9.869233e-16  # 1 mD = 9.869e-16 m^2

    # ── Verbose output ────────────────────────────────────────────────────────
    if verbose:
        print(f"\nPorosity (phi)        = {porosity:.4f}")
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
        all_v   = {"x": vx, "y": vy, "z": vz}
        transverse = [c for c in ("x", "y", "z") if c != direction]
        print("\n--- Sanity checks ---")
        print(f"Mean rho (pore space) = {np.mean(rho[pore_mask]):.6f}  (expect ~{(_RHO_IN + _RHO_OUT) / 2:.3f})")
        print(f"v{direction} pore: min={v_flow[pore_mask].min():.3e}  max={v_flow[pore_mask].max():.3e}")
        for c in transverse:
            vc = all_v[c]
            print(f"v{c} pore: min={vc[pore_mask].min():.3e}  max={vc[pore_mask].max():.3e}  (should be ~0)")

    return {
        "porosity": float(porosity),
        "u_darcy":  u_darcy,
        "u_pore":   u_pore,
        "k_lu":     k_lu,
        "k_m2":     k_m2,
        "k_mD":     k_mD,
    }
