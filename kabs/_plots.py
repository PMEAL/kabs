from kabs._compute_permeability import _parse_xml_arrays, _read_array
import re
import numpy as np


__all__ = [
    'plot_cross_section',
]


def plot_cross_section(filename, direction="x", axis=2):
    r"""
    Generate a 2D image of the velocity field for plotting

    Parameters
    ----------
    filename : str
        The VTR file produced by the simulation
    direction : str
        Specifies which component of the velocity vector to plot.
        The default is "x". "all" will plot the magnitude of the 
        velocity (i.e. the sum of all velocity components)
    axis : int
        The direction where the 2D slice should be taken.
        The default is 2, meaning it views the domain in
        the z-direction, thus shows an 'x-y' plane.
    
    Returns
    -------
    velocity : ndarray
        A 2D array with voxel value corresponding to the velocity.
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
    velocity = _read_array(raw, binary_start, arrays, "velocity", nx, ny, nz)

    if direction in [0, 'x', 'X']:
        v_dir = 0
        vel = velocity[..., v_dir]
    elif direction in [1, 'y', 'Y']:
        v_dir = 1
        vel = velocity[..., v_dir]
    elif direction in [2, 'z', 'Z']:
        v_dir = 2
        vel = velocity[..., v_dir]
    elif direction in ['all', 'All', 'ALL', 'None', None, 'none']:
        vel = np.sum(velocity, axis=-1)
    if axis == 0:
        vx_long = vel[int(vel.shape[0]/2), :, :]
    elif axis == 1:
        vx_long = vel[:, int(vel.shape[1]/2), :].T
    elif axis == 2:
        vx_long = vel[:, :, int(vel.shape[2]/2)].T
    return vx_long
