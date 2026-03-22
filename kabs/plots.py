import math

import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    "plot_cross_section",
    "add_streamlines",
    "render_flow",
]


def plot_cross_section(soln, direction="x", axis=2, streamlines=None):
    r"""
    Generate a 2D image of the velocity field for plotting

    Parameters
    ----------
    soln : FlowResult
        A ``FlowResult`` returned by ``solve_flow()`` or ``read_flow_vtr()``.
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
    velocity = soln.velocity

    if direction in [0, "x", "X"]:
        v_dir = 0
        vel = velocity[..., v_dir]
    elif direction in [1, "y", "Y"]:
        v_dir = 1
        vel = velocity[..., v_dir]
    elif direction in [2, "z", "Z"]:
        v_dir = 2
        vel = velocity[..., v_dir]
    elif direction in ["all", "All", "ALL", "None", None, "none"]:
        vel = np.sum(velocity, axis=-1)
    if axis == 0:
        vx_long = vel[int(vel.shape[0] / 2), :, :]
    elif axis == 1:
        vx_long = vel[:, int(vel.shape[1] / 2), :].T
    elif axis == 2:
        vx_long = vel[:, :, int(vel.shape[2] / 2)].T
    return vx_long


def add_streamlines(soln, ax, axis=None, **kwargs):
    r"""
    Overlay 2D streamlines on an existing Matplotlib axes.

    Extracts the two in-plane velocity components at the midpoint slice along
    the given axis, then calls ``ax.streamplot`` to draw the streamlines.

    Parameters
    ----------
    soln : FlowResult
        Converged flow result returned by ``solve_flow()`` or
        ``read_flow_vtr()``.
    ax : matplotlib.axes.Axes
        The axes object on which to draw the streamlines.
    axis : int
        The normal axis of the 2D slice to visualize:
        ``0`` → yz-plane (x midpoint), ``1`` → xz-plane (y midpoint),
        ``2`` → xy-plane (z midpoint). If `None`, then an axis
        perpendicular to the direction of flow will be used.
    **kwargs
        Additional keyword arguments forwarded directly to
        ``ax.streamplot`` (e.g. ``color='white'``, ``density=1.5``).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with streamlines added.
    """
    velocity = soln.velocity
    mid = [int(s / 2) for s in velocity.shape[:3]]
    if axis is None:
        options = {"x": 0, "y": 1, "z": 2}
        axis = options[soln.direction]
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


def render_flow(
    result,
    direction=None,
    n_streamlines=200,
    solid_opacity=0.15,
    solid_color="dimgray",
    tube_radius=0.4,
    cmap="Blues",
    save=None,
    show=True,
    off_screen=False,
    window_size=(800, 600),
):
    r"""
    Render a ParaView-style 3D visualisation of the solid structure and flow
    streamlines.

    Requires ``pyvista`` (``pip install pyvista``).

    Parameters
    ----------
    result : FlowResult
        Converged flow result returned by ``solve_flow()`` or
        ``read_flow_vtr()``.
    n_streamlines : int
        Number of seed points arranged as a grid on the inlet face.
        Default 200.
    solid_opacity : float
        Opacity of the solid surface mesh (0 = invisible, 1 = fully opaque).
        Default 0.15.
    solid_color : str
        Colour of the solid surface.  Default ``"dimgray"``.
    tube_radius : float
        Radius of the streamline tubes in voxel units.  Default 0.4.
    cmap : str
        Matplotlib colormap used to colour streamlines by flow speed.
        Default ``"Blues"``.
    save : str or None
        Path at which to save a PNG screenshot.  ``None`` (default) skips saving.
    show : bool
        Open an interactive window after rendering.  Default ``True``.
    off_screen : bool
        Render without opening a display window (useful on headless servers when
        ``save`` is set).  Default ``False``.
    window_size : tuple of int
        Render-window width × height in pixels.  Default ``(800, 600)``.

    Returns
    -------
    plotter : pyvista.Plotter
        The configured plotter (already shown/saved if requested).
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            "pyvista is required for render_flow(). "
            "Install it with:  pip install pyvista"
        )

    solid = result.solid  # (nx, ny, nz), 1 = solid, 0 = pore (internal convention)
    velocity = result.velocity  # (nx, ny, nz, 3)
    direction = result.direction
    nx, ny, nz = solid.shape

    # --- Build a cell-centred ImageData grid ---
    # pyvista/VTK flatten order: x varies fastest, which is Fortran order in numpy.
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    grid.cell_data["solid"] = solid.flatten(order="F").astype(np.float32)

    speed = np.linalg.norm(velocity, axis=-1)  # (nx, ny, nz)
    grid.cell_data["speed"] = speed.flatten(order="F").astype(np.float32)
    grid.cell_data["velocity"] = np.column_stack(
        [
            velocity[..., 0].flatten(order="F"),
            velocity[..., 1].flatten(order="F"),
            velocity[..., 2].flatten(order="F"),
        ]
    ).astype(np.float32)

    # --- Solid surface: threshold to keep solid voxels, then extract faces ---
    solid_surf = grid.threshold(0.5, scalars="solid").extract_surface(
        algorithm="dataset_surface"
    )

    # --- Convert to point data so the streamline integrator can interpolate ---
    grid_pts = grid.cell_data_to_point_data()

    # --- Seed points uniformly covering the inlet face ---
    n_side = max(2, int(math.sqrt(n_streamlines)))
    margin = 0.05  # stay slightly inside the domain boundary
    a = np.linspace(0 + margin, 1 - margin, n_side)
    b = np.linspace(0 + margin, 1 - margin, n_side)
    aa, bb = np.meshgrid(a, b)
    aa, bb = aa.ravel(), bb.ravel()
    if direction == "x":
        pts = np.column_stack([np.full_like(aa, 0.5), aa * ny, bb * nz])
    elif direction == "y":
        pts = np.column_stack([aa * nx, np.full_like(aa, 0.5), bb * nz])
    else:  # z
        pts = np.column_stack([aa * nx, bb * ny, np.full_like(aa, 0.5)])
    seed = pv.PolyData(pts.astype(np.float32))

    # --- Trace streamlines from the inlet ---
    streams = grid_pts.streamlines_from_source(
        seed,
        vectors="velocity",
        # max_time=float(max(nx, ny, nz)) * 3,
        integration_direction="forward",
    )

    # --- Assemble the scene ---
    pl = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
    pl.set_background("white")

    pl.add_mesh(
        solid_surf, color=solid_color, opacity=solid_opacity, smooth_shading=True
    )

    if streams.n_points > 0:
        tubes = streams.tube(radius=tube_radius)
        pl.add_mesh(tubes, scalars="speed", cmap=cmap, show_scalar_bar=False)

    # Camera angle similar to the ParaView default isometric view
    pl.view_isometric()
    pl.camera.azimuth += 15

    if save:
        pl.screenshot(save)
    if show:
        pl.show()

    return pl
