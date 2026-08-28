"""
Helpers for reading and writing VTR (VTK Rectilinear Grid) files written by pyevtk.
"""

import re
import struct

import numpy as np


__all__ = [
    "parse_xml_arrays",
    "read_array",
    "read_flow_vtr",
    "write_flow_vtr",
]


def parse_xml_arrays(xml):
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


def read_array(raw, binary_start, arrays, name, nx, ny, nz):
    """Read one named array from the raw VTR bytes.

    pyevtk writes vector data in AoS format (interleaved per point) with
    Fortran-order spatial indexing:
        vx[0,0,0], vy[0,0,0], vz[0,0,0], vx[1,0,0], ...
    Scalar arrays are written in Fortran order directly.
    """
    offset, dtype, n_comp = arrays[name]
    pos = binary_start + offset
    # 8-byte UInt64 header = number of *bytes* in this array
    (n_bytes,) = struct.unpack_from("<Q", raw, pos)
    pos += 8
    data = np.frombuffer(
        raw, dtype=dtype, count=n_bytes // dtype().itemsize, offset=pos
    )
    if n_comp > 1:
        data_pts = data.reshape(nx * ny * nz, n_comp)
        data = np.stack(
            [data_pts[:, c].reshape(nx, ny, nz, order="F") for c in range(n_comp)],
            axis=-1,
        )
    else:
        data = data.reshape((nx, ny, nz), order="F")
    return data


_META_RE = re.compile(r"<!--\s*kabs-meta\s+(.*?)\s*-->", re.DOTALL)


def _parse_meta_comment(xml_header):
    """Extract simulation metadata from a kabs-meta comment, if present."""
    m = _META_RE.search(xml_header)
    if not m:
        return {}
    raw = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    meta = {}
    if raw.get("direction") in ("x", "y", "z"):
        meta["direction"] = raw["direction"]
    if raw.get("collision_model") in ("mrt", "srt"):
        meta["collision_model"] = raw["collision_model"]
    for name in (
        "nu",
        "rho_in",
        "rho_out",
        "velocity_tol",
        "k_tol",
        "flux_tol",
        "velocity_criterion",
        "k_criterion",
        "flux_criterion",
    ):
        try:
            value = float(raw[name])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            meta[name] = value
    for name in ("n_iterations", "convergence_every", "consecutive_passes"):
        try:
            value = int(raw[name])
        except (KeyError, TypeError, ValueError):
            continue
        if value >= 0:
            meta[name] = value
    if raw.get("converged") in ("true", "false"):
        meta["converged"] = raw["converged"] == "true"
    return meta


def read_flow_vtr(vtr_file, verbose=False):
    """Read a flow .vtr file and return a :class:`~kabs.FlowResult`.

    Simulation, pressure, and convergence provenance are restored when the
    file was written by :func:`write_flow_vtr` with a ``FlowResult`` that
    carried them. Older files without those attributes remain supported.
    """
    from ._flow_common import FlowResult

    if verbose:
        print(f"Reading {vtr_file} ...")
    with open(vtr_file, "rb") as fh:
        raw = fh.read()

    marker = raw.index(b'<AppendedData encoding="raw">')
    binary_start = raw.index(b"_", marker) + 1

    xml_header = raw[:marker].decode("utf-8", errors="replace")
    arrays = parse_xml_arrays(xml_header)

    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    if verbose:
        print(f"  Grid: {nx} x {ny} x {nz} points")

    solid = read_array(raw, binary_start, arrays, "Solid", nx, ny, nz)
    rho = read_array(raw, binary_start, arrays, "rho", nx, ny, nz)
    velocity = read_array(raw, binary_start, arrays, "velocity", nx, ny, nz)
    metadata = _parse_meta_comment(xml_header)
    if verbose:
        print("  Arrays loaded.")
    return FlowResult.from_arrays(
        solid,
        rho,
        velocity,
        **metadata,
    )


def write_flow_vtr(path, result):
    """Write a FlowResult to a VTK Rectilinear Grid (.vtr) file.

    Parameters
    ----------
    path : str
        Output path without extension (pyevtk appends ``.vtr``).
    result : FlowResult
        A converged flow result containing ``solid``, ``rho``, and ``velocity``.
        Simulation provenance is embedded as a metadata comment when present.
    """
    from pyevtk.hl import gridToVTK

    nx, ny, nz = result.solid.shape
    x = np.linspace(0, nx, nx)
    y = np.linspace(0, ny, ny)
    z = np.linspace(0, nz, nz)
    gridToVTK(
        path,
        x,
        y,
        z,
        pointData={
            "Solid": np.ascontiguousarray(result.solid),
            "rho": np.ascontiguousarray(result.rho),
            "velocity": (
                np.ascontiguousarray(result.velocity[..., 0]),
                np.ascontiguousarray(result.velocity[..., 1]),
                np.ascontiguousarray(result.velocity[..., 2]),
            ),
        },
    )

    # Embed simulation metadata as a comment in the XML header so it survives
    # the write/read round-trip.
    metadata = {
        name: getattr(result, name, None)
        for name in (
            "direction",
            "nu",
            "collision_model",
            "rho_in",
            "rho_out",
            "converged",
            "n_iterations",
            "velocity_tol",
            "k_tol",
            "flux_tol",
            "velocity_criterion",
            "k_criterion",
            "flux_criterion",
            "convergence_every",
            "consecutive_passes",
        )
    }
    if any(value is not None for value in metadata.values()):
        attrs = []
        for name, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)):
                value = str(bool(value)).lower()
            attrs.append(f'{name}="{value}"')
        comment = f"<!-- kabs-meta {' '.join(attrs)} -->\n".encode()
        vtr_path = path + ".vtr"
        with open(vtr_path, "rb") as fh:
            data = fh.read()
        first_newline = data.index(b"\n")
        data = data[: first_newline + 1] + comment + data[first_newline + 1 :]
        with open(vtr_path, "wb") as fh:
            fh.write(data)
