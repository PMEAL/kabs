"""
Low-level helpers for reading VTR (VTK Rectilinear Grid) files written by pyevtk.
"""

import re
import struct

import numpy as np


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


def vtr_to_array(filename, array_name="velocity"):
    r"""
    Extracts a named array from a VTR file and returns it as a numpy array.

    Parameters
    ----------
    filename : str
        The VTR file produced by the simulation.
    array_name : str
        Name of the array to extract. Default is ``"velocity"``.

    Returns
    -------
    data : ndarray
        For vector arrays, shape is ``(nx, ny, nz, n_components)``.
        For scalar arrays, shape is ``(nx, ny, nz)``.
    """
    with open(filename, "rb") as fh:
        raw = fh.read()
    marker = raw.index(b'<AppendedData encoding="raw">')
    binary_start = raw.index(b"_", marker) + 1
    xml_header = raw[:marker].decode("utf-8", errors="replace")
    arrays = _parse_xml_arrays(xml_header)
    m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
    x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
    nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    return _read_array(raw, binary_start, arrays, array_name, nx, ny, nz)
