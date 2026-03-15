from tools.vtr_io import _parse_xml_arrays, _read_array
import re
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'plot_cross_section',
    'vtr_to_array',
    'add_streamlines',
]

def vtr_to_array(filename):
    r"""
    Extracts the velocity values for each voxel from the given VTR file and
    converts to a Numpy array.

    Parameters
    ----------
    filename : str
        The VTR file produced by the simulation

    Returns
    -------
    velocity : ndarray
        An ndarray of size `velocity.ndim + 1`, where the final dimension contains
        the x, y and z velocity components. For e.g. `velocity[..., 0]` returns
        a 3D image with each voxel containing the x component of the velocity.
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
    return velocity


def plot_cross_section(source, direction="x", axis=2, streamlines=None):
    r"""
    Generate a 2D image of the velocity field for plotting

    Parameters
    ----------
    source : FlowResult or str
        Either a ``FlowResult`` returned by ``solve_flow()``, or a path to a
        ``.vtr`` file written by ``SinglePhaseSolver.export_VTK()``.
    direction : str
        Specifies which component of the velocity vector to plot.
        The default is "x". "all" will plot the magnitude of the
        velocity (i.e. the sum of all velocity components)
    axis : int
        The direction where the 2D slice should be taken.
        The default is 2, meaning it views the domain in
        the z-direction, thus shows an 'x-y' plane.
    streamlines : dict or None
        If ``None`` (default), no streamlines are drawn. If a dict,
        ``plt.streamplot`` is called on the current axes using the
        in-plane velocity components at the slice midpoint. Any keys
        in the dict are forwarded as keyword arguments to
        ``plt.streamplot`` (e.g. ``{'color': 'white', 'density': 1.5}``).

    Returns
    -------
    velocity : ndarray
        A 2D array with voxel value corresponding to the velocity.
    """
    from ._solve_flow import FlowResult
    if isinstance(source, FlowResult):
        velocity = source.velocity
    else:
        velocity = vtr_to_array(source)

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


def add_streamlines(source, ax, axis, **kwargs):
    from ._solve_flow import FlowResult
    if isinstance(source, FlowResult):
        velocity = source.velocity
    else:
        velocity = vtr_to_array(source)
    mid = [int(s / 2) for s in velocity.shape[:3]]
    if axis == 0:
        U = velocity[mid[0], :, :, 1]
        V = velocity[mid[0], :, :, 2]
    elif axis == 1:
        U = velocity[:, mid[1], :, 0].T
        V = velocity[:, mid[1], :, 2].T
    elif axis == 2:
        U = velocity[:, :, mid[2], 0].T
        V = velocity[:, :, mid[2], 1].T
    nrows, ncols = U.shape
    X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    ax.streamplot(X, Y, U, V, **kwargs)
    return ax
