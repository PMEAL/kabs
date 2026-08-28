import numpy as np

from ._solve_flow import solve_flow


__all__ = [
    "check_flow_development",
    "format_hydraulic_conductance_report",
    "solve_hydraulic_conductance",
]


_RHO_IN = 1.00
_RHO_OUT = 0.99


def _select_mid_window(idx, keep_fraction=0.5, min_points=4):
    n = len(idx)
    if n <= min_points:
        return np.asarray(idx, dtype=int)
    keep_n = max(min_points, int(round(n * keep_fraction)))
    keep_n = min(keep_n, n)
    start = (n - keep_n) // 2
    return np.asarray(idx[start : start + keep_n], dtype=int)


def _linear_fit_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coef = np.polyfit(x, y, deg=1)
    yhat = np.polyval(coef, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return coef, r2


def _alpha_for_slice(fluid, u_axis, axis, j):
    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = int(j)
    mm = fluid[tuple(sl)]
    uu = u_axis[tuple(sl)][mm]
    if uu.size < 2:
        return np.nan
    ubar = float(np.mean(uu))
    if np.isclose(ubar, 0.0):
        return np.nan
    return float(np.mean(uu**3) / (ubar**3))


def _slice_polar_moment_and_istar(fluid, axis, valid_idx):
    """Compute per-slice polar moment ``I`` and normalized ``I* = I / A^2``.

    Notes
    -----
    The polar moment is computed from pore-voxel centers relative to the
    pore-area centroid in each flow-normal slice.
    """
    n_slices = fluid.shape[axis]
    i_slice = np.full(n_slices, np.nan, dtype=float)
    istar_slice = np.full(n_slices, np.nan, dtype=float)

    for j in valid_idx:
        sl = [slice(None), slice(None), slice(None)]
        sl[axis] = int(j)
        mask = fluid[tuple(sl)]

        area = int(np.count_nonzero(mask))
        if area <= 0:
            continue

        coords = np.argwhere(mask).astype(float)
        centroid = np.mean(coords, axis=0)
        r2 = np.sum((coords - centroid) ** 2, axis=1)
        i_val = float(np.sum(r2))

        i_slice[j] = i_val
        istar_slice[j] = i_val / float(area**2)

    return i_slice, istar_slice


def check_flow_development(
    fluid,
    u_axis,
    axis,
    valid_idx,
    p_slice,
    q_slice,
    fit_in,
    fit_out,
    alpha_match_tol=0.15,
    alpha_plateau_tol=0.15,
    pressure_r2_min=0.995,
    q_cv_max=0.02,
    min_eval_points=4,
):
    """Check whether lead regions appear sufficiently developed.

    This helper evaluates whether the padded inlet/outlet lead regions behave
    like developed flow zones. It combines profile-shape indicators
    (alpha-matching and alpha-plateau checks), pressure-linearity checks, and
    per-slice flow-rate stability checks into a single pass/fail summary while
    also returning all intermediate diagnostics.

    Returns
    -------
    dict
        Development-check summary and diagnostics.

        +----------------------+-----------------------------------------------+
        | Key                  | Description                                   |
        +======================+===============================================+
        | ``ok``               | Overall boolean pass/fail. ``True`` only      |
        |                      | when every check in ``checks`` passes.        |
        +----------------------+-----------------------------------------------+
        | ``reasons``          | List of check names that failed. Empty list   |
        |                      | means all checks passed.                      |
        +----------------------+-----------------------------------------------+
        | ``checks``           | Mapping from check name to boolean result.    |
        |                      | Useful for programmatic gating and tests.     |
        +----------------------+-----------------------------------------------+
        | ``alpha_slice``      | Per-slice alpha values along the flow axis.   |
        |                      | ``NaN`` where a slice is invalid/undefined.   |
        +----------------------+-----------------------------------------------+
        | ``alpha_in``         | Mean alpha in near-conduit inlet window.      |
        +----------------------+-----------------------------------------------+
        | ``alpha_out``        | Mean alpha in near-conduit outlet window.     |
        +----------------------+-----------------------------------------------+
        | ``alpha_in_mid``     | Mean alpha in middle of inlet lead            |
        |                      | (plateau reference).                          |
        +----------------------+-----------------------------------------------+
        | ``alpha_out_mid``    | Mean alpha in middle of outlet lead           |
        |                      | (plateau reference).                          |
        +----------------------+-----------------------------------------------+
        | ``alpha_match_tol``  | Tolerance for inlet/outlet alpha agreement,   |
        |                      | i.e. ``|alpha_in - alpha_out|``.              |
        +----------------------+-----------------------------------------------+
        | ``alpha_plateau_tol``| Tolerance for near-lead vs mid-lead alpha     |
        |                      | consistency checks.                           |
        +----------------------+-----------------------------------------------+
        | ``coef_in``          | Inlet pressure-fit coefficients               |
        |                      | ``[slope, intercept]`` or ``None``.           |
        +----------------------+-----------------------------------------------+
        | ``coef_out``         | Outlet pressure-fit coefficients              |
        |                      | ``[slope, intercept]`` or ``None``.           |
        +----------------------+-----------------------------------------------+
        | ``r2_in``            | Inlet lead pressure-fit $R^2$ value.          |
        +----------------------+-----------------------------------------------+
        | ``r2_out``           | Outlet lead pressure-fit $R^2$ value.         |
        +----------------------+-----------------------------------------------+
        | ``pressure_r2_min``  | Minimum required $R^2$ threshold for          |
        |                      | pressure-linearity checks.                    |
        +----------------------+-----------------------------------------------+
        | ``q_cv_in``          | Inlet lead flow-rate coefficient of           |
        |                      | variation, $\sigma_Q / |\bar{Q}|$.            |
        +----------------------+-----------------------------------------------+
        | ``q_cv_out``         | Outlet lead flow-rate coefficient of          |
        |                      | variation, $\sigma_Q / |\bar{Q}|$.            |
        +----------------------+-----------------------------------------------+
        | ``q_cv_max``         | Maximum allowed flow-rate coefficient of      |
        |                      | variation threshold.                          |
        +----------------------+-----------------------------------------------+
        | ``in_eval``          | Slice indices used for near-conduit inlet     |
        |                      | alpha evaluation.                             |
        +----------------------+-----------------------------------------------+
        | ``out_eval``         | Slice indices used for near-conduit outlet    |
        |                      | alpha evaluation.                             |
        +----------------------+-----------------------------------------------+
        | ``in_mid``           | Slice indices used for inlet mid-lead         |
        |                      | plateau alpha reference.                      |
        +----------------------+-----------------------------------------------+
        | ``out_mid``          | Slice indices used for outlet mid-lead        |
        |                      | plateau alpha reference.                      |
        +----------------------+-----------------------------------------------+
    """
    n_slices = p_slice.size
    alpha_slice = np.full(n_slices, np.nan, dtype=float)
    for j in valid_idx:
        alpha_slice[j] = _alpha_for_slice(fluid, u_axis, axis, j)

    fit_in = np.asarray(fit_in, dtype=int)
    fit_out = np.asarray(fit_out, dtype=int)

    # Near-conduit evaluation windows.
    n_in = max(min_eval_points, max(1, len(fit_in) // 3))
    n_out = max(min_eval_points, max(1, len(fit_out) // 3))
    in_eval = fit_in[-n_in:] if len(fit_in) else np.array([], dtype=int)
    out_eval = fit_out[:n_out] if len(fit_out) else np.array([], dtype=int)

    # Mid-lead windows for plateau reference.
    in_mid = _select_mid_window(fit_in, keep_fraction=0.5, min_points=min_eval_points)
    out_mid = _select_mid_window(
        fit_out, keep_fraction=0.5, min_points=min_eval_points
    )

    alpha_in = float(np.nanmean(alpha_slice[in_eval])) if in_eval.size else np.nan
    alpha_out = float(np.nanmean(alpha_slice[out_eval])) if out_eval.size else np.nan
    alpha_in_mid = float(np.nanmean(alpha_slice[in_mid])) if in_mid.size else np.nan
    alpha_out_mid = float(np.nanmean(alpha_slice[out_mid])) if out_mid.size else np.nan

    # Pressure linearity.
    coef_in = coef_out = None
    r2_in = r2_out = np.nan
    if len(fit_in) >= 2:
        coef_in, r2_in = _linear_fit_r2(fit_in, p_slice[fit_in])
    if len(fit_out) >= 2:
        coef_out, r2_out = _linear_fit_r2(fit_out, p_slice[fit_out])

    # Flow stability in leads.
    def _cv(a):
        a = np.asarray(a, dtype=float)
        am = float(np.nanmean(a))
        if np.isclose(am, 0.0):
            return np.nan
        return float(np.nanstd(a) / abs(am))

    q_cv_in = _cv(q_slice[fit_in]) if len(fit_in) else np.nan
    q_cv_out = _cv(q_slice[fit_out]) if len(fit_out) else np.nan

    checks = {
        "alpha_in_out_match": bool(
            np.isfinite(alpha_in)
            and np.isfinite(alpha_out)
            and abs(alpha_in - alpha_out) <= alpha_match_tol
        ),
        "alpha_in_plateau": bool(
            np.isfinite(alpha_in)
            and np.isfinite(alpha_in_mid)
            and abs(alpha_in - alpha_in_mid) <= alpha_plateau_tol
        ),
        "alpha_out_plateau": bool(
            np.isfinite(alpha_out)
            and np.isfinite(alpha_out_mid)
            and abs(alpha_out - alpha_out_mid) <= alpha_plateau_tol
        ),
        "pressure_linearity_in": bool(np.isfinite(r2_in) and r2_in >= pressure_r2_min),
        "pressure_linearity_out": bool(
            np.isfinite(r2_out) and r2_out >= pressure_r2_min
        ),
        "q_stability_in": bool(np.isfinite(q_cv_in) and q_cv_in <= q_cv_max),
        "q_stability_out": bool(np.isfinite(q_cv_out) and q_cv_out <= q_cv_max),
    }

    reasons = [k for k, v in checks.items() if not v]
    ok = len(reasons) == 0

    return {
        "ok": ok,
        "reasons": reasons,
        "checks": checks,
        "alpha_slice": alpha_slice,
        "alpha_in": alpha_in,
        "alpha_out": alpha_out,
        "alpha_in_mid": alpha_in_mid,
        "alpha_out_mid": alpha_out_mid,
        "alpha_match_tol": alpha_match_tol,
        "alpha_plateau_tol": alpha_plateau_tol,
        "coef_in": coef_in,
        "coef_out": coef_out,
        "r2_in": r2_in,
        "r2_out": r2_out,
        "pressure_r2_min": pressure_r2_min,
        "q_cv_in": q_cv_in,
        "q_cv_out": q_cv_out,
        "q_cv_max": q_cv_max,
        "in_eval": in_eval,
        "out_eval": out_eval,
        "in_mid": in_mid,
        "out_mid": out_mid,
    }


def format_hydraulic_conductance_report(result):
    """Return a multi-line report string for ``solve_hydraulic_conductance`` output."""
    d = result.get("development", {})
    checks = d.get("checks", {})
    reasons = d.get("reasons", [])
    lines = [
        "Hydraulic conductance report",
        "",
        f"direction                 = {result.get('direction')}",
        f"pad voxels                = {result.get('pad')}",
        f"conduit slice range       = [{result.get('j_start')}, {result.get('j_end')}],",
        f"Q_lu                      = {result.get('Q_lu', np.nan):.6e}",
        f"dP_edge_lu                = {result.get('dP_edge_lu', np.nan):.6e}",
        f"g_lbm_lu                  = {result.get('g_lbm_lu', np.nan):.6e}",
        f"g_model_lu                = {result.get('g_model_lu', np.nan):.6e}",
        f"istar mode                = {result.get('istar_source', 'user')}",
        f"istar mean                = {result.get('istar_mean', np.nan):.6e}",
        f"rho_lu                    = {result.get('rho_lu', np.nan):.6e}",
        f"mu_lu                     = {result.get('mu_lu', np.nan):.6e}",
        f"rho_si                    = {result.get('rho_si', np.nan):.6e}",
        f"mu_si                     = {result.get('mu_si', np.nan):.6e}",
        f"nu_si                     = {result.get('nu_si', np.nan):.6e}",
        f"g_lbm_si                  = {result.get('g_lbm_si', np.nan):.6e}",
        f"g_model_si                = {result.get('g_model_si', np.nan):.6e}",
        "",
        "Resistance components (SI)",
        f"  R_visc_si               = {result.get('R_visc_si', np.nan):.6e}",
        f"  R_acc_si                = {result.get('R_acc_si', np.nan):.6e}",
        f"  R_total_si              = {result.get('R_total_si', np.nan):.6e}",
        "",
        "Development diagnostics",
        f"  overall ok              = {d.get('ok', False)}",
        f"  alpha_in                = {d.get('alpha_in', np.nan):.6f}",
        f"  alpha_out               = {d.get('alpha_out', np.nan):.6f}",
        f"  alpha_in_mid            = {d.get('alpha_in_mid', np.nan):.6f}",
        f"  alpha_out_mid           = {d.get('alpha_out_mid', np.nan):.6f}",
        f"  r2_in                   = {d.get('r2_in', np.nan):.6f}",
        f"  r2_out                  = {d.get('r2_out', np.nan):.6f}",
        f"  q_cv_in                 = {d.get('q_cv_in', np.nan):.6e}",
        f"  q_cv_out                = {d.get('q_cv_out', np.nan):.6e}",
        "",
        "Checks",
    ]
    for key in sorted(checks.keys()):
        lines.append(f"  {key:24s} = {checks[key]}")
    if reasons:
        lines += ["", "Failed checks"]
        for r in reasons:
            lines.append(f"  - {r}")
    return "\n".join(lines)


def solve_hydraulic_conductance(
    im,
    pad=50,
    direction="y",
    voxel_size=1e-6,
    n_steps=20_000,
    tol=1e-3,
    mu=1e-3,
    rho=997,
    Istar=None,
    alpha_1=2.0,
    alpha_2=2.0,
    log_every=500,
    alpha_match_tol=0.15,
    alpha_plateau_tol=0.15,
    pressure_r2_min=0.995,
    q_cv_max=0.02,
):
    """
    Solve flow on a padded conduit image and estimate hydraulic conductance.

    Parameters
    ----------
    im : ndarray
        Binary image in public convention: 1 = pore, 0 = solid.
    pad : int, default=50
        Voxels of lead region added at inlet and outlet (in flow direction).
    direction : {"x", "y", "z"}, default="y"
        Flow direction.
    voxel_size : float, default=1e-6
        Voxel edge length [m].
    n_steps : int, default=20000
        Passed to solve_flow.
    tol : float or None, default=1e-3
        Passed to solve_flow.
    mu : float, default=1e-3
        Dynamic viscosity [Pa.s] used for LU->SI conductance conversion.
    rho_si : float or None, default=None
        Fluid density in SI units [kg/m^3]. If provided, the output also includes
        ``nu_si = mu / rho_si``.

        Notes:
        - This argument is used for SI property reporting only.
        - It does not alter the LBM density field solved internally.
    Istar : float or {None, "auto"}, default=None
        Specific polar moment of inertia corresponding to the cross-sectional shape
        of the conduit.

        - If ``float``, the provided value is used directly.
        - If ``None`` or ``"auto"``, a voxel-based per-slice polar moment is
          computed from the pore cross-sections and converted to
          ``Istar = I / A^2``.
    alpha_1, alpha_2 : float, default=2.0
        Kinetic-energy correction factors for inlet/outlet acceleration term.
    log_every : int, default=500
        Passed to solve_flow.
    alpha_match_tol : float, default=0.15
        Allowed absolute difference between near-conduit inlet/outlet alpha values.
    alpha_plateau_tol : float, default=0.15
        Allowed absolute difference between near-conduit alpha and mid-lead alpha.
    pressure_r2_min : float, default=0.995
        Minimum acceptable R^2 for linear pressure fit in each lead.
    q_cv_max : float, default=0.02
        Maximum acceptable coefficient of variation of per-slice flow in each lead.

    Returns
    -------
    dict
        Full solution bundle containing the solver output, conductance values,
        pressure/flow diagnostics, and development-quality checks.

        +------------------+-----------------------------------------------+
        | Key              | Description                                   |
        +==================+===============================================+
        | ``flow_result``  | ``FlowResult`` returned by ``solve_flow``     |
        |                  | for the padded domain.                        |
        +------------------+-----------------------------------------------+
        | ``padded_image`` | Binary image after adding inlet/outlet lead   |
        |                  | padding in selected flow direction.           |
        +------------------+-----------------------------------------------+
        | ``direction``    | Flow direction used for solve and post-       |
        |                  | processing (``"x"``, ``"y"``, or ``"z"``).   |
        +------------------+-----------------------------------------------+
        | ``pad``          | Number of voxels padded on each side along    |
        |                  | the flow axis.                                |
        +------------------+-----------------------------------------------+
        | ``j_start``      | First slice index of extracted conduit        |
        |                  | region (after inlet lead).                    |
        +------------------+-----------------------------------------------+
        | ``j_end``        | Last slice index of extracted conduit         |
        |                  | region (before outlet lead).                  |
        +------------------+-----------------------------------------------+
        | ``conduit_idx``  | Slice indices spanning conduit region used    |
        |                  | for conductance evaluation.                   |
        +------------------+-----------------------------------------------+
        | ``fit_in``       | Slice indices used for inlet lead pressure    |
        |                  | fitting.                                      |
        +------------------+-----------------------------------------------+
        | ``fit_out``      | Slice indices used for outlet lead pressure   |
        |                  | fitting.                                      |
        +------------------+-----------------------------------------------+
        | ``coef_in``      | Inlet pressure-fit coefficients               |
        |                  | ``[slope, intercept]`` or ``None``.           |
        +------------------+-----------------------------------------------+
        | ``coef_out``     | Outlet pressure-fit coefficients              |
        |                  | ``[slope, intercept]`` or ``None``.           |
        +------------------+-----------------------------------------------+
        | ``dP_fit_lu``    | Pressure drop from inlet/outlet linear fits   |
        |                  | at conduit boundaries (lattice units).        |
        +------------------+-----------------------------------------------+
        | ``dP_edge_lu``   | Pressure drop from edge-window averaging      |
        |                  | near conduit ends (lattice units).            |
        +------------------+-----------------------------------------------+
        | ``Q_lu``         | Mean volumetric flow rate over conduit        |
        |                  | slices (lattice units).                       |
        +------------------+-----------------------------------------------+
        | ``R_lbm_lu``     | Resistance from direct simulation             |
        |                  | observables, $R = \Delta P_{edge}/Q$.         |
        +------------------+-----------------------------------------------+
        | ``g_lbm_lu``     | Conductance from direct simulation            |
        |                  | observables, $g = 1/R_{lbm}$.                 |
        +------------------+-----------------------------------------------+
        | ``R_visc_lu``    | Viscous resistance contribution from          |
        |                  | numerical area-based integration.             |
        +------------------+-----------------------------------------------+
        | ``R_acc_lu``     | Acceleration/inertial resistance from         |
        |                  | inlet/outlet area and alpha terms.            |
        +------------------+-----------------------------------------------+
        | ``R_total_lu``   | Total model resistance,                       |
        |                  | $R_{visc} + R_{acc}$.                         |
        +------------------+-----------------------------------------------+
        | ``g_model_lu``   | Model-derived conductance,                    |
        |                  | $g = 1/R_{total}$.                            |
        +------------------+-----------------------------------------------+
        | ``g_lbm_si``     | Direct-observable conductance converted       |
        |                  | to SI units ($\mathrm{m^3/(Pa\cdot s)}$).     |
        +------------------+-----------------------------------------------+
        | ``g_model_si``   | Model-derived conductance converted           |
        |                  | to SI units ($\mathrm{m^3/(Pa\cdot s)}$).     |
        +------------------+-----------------------------------------------+
        | ``R_visc_si``    | Viscous resistance contribution in SI units   |
        |                  | ($\mathrm{Pa\cdot s/m^3}$).                   |
        +------------------+-----------------------------------------------+
        | ``R_acc_si``     | Acceleration/inertial resistance in SI units  |
        |                  | ($\mathrm{Pa\cdot s/m^3}$).                   |
        +------------------+-----------------------------------------------+
        | ``R_total_si``   | Total model resistance in SI units             |
        |                  | ($\mathrm{Pa\cdot s/m^3}$).                   |
        +------------------+-----------------------------------------------+
        | ``rho_si``       | Input fluid density in SI units               |
        |                  | ($\mathrm{kg/m^3}$), or ``NaN`` if omitted.    |
        +------------------+-----------------------------------------------+
        | ``mu_si``        | Input dynamic viscosity in SI units           |
        |                  | ($\mathrm{Pa\cdot s}$).                       |
        +------------------+-----------------------------------------------+
        | ``nu_si``        | Kinematic viscosity in SI units               |
        |                  | ($\mathrm{m^2/s}$), ``mu_si / rho_si`` when    |
        |                  | ``rho_si`` is supplied.                       |
        +------------------+-----------------------------------------------+
        | ``rho_lu``       | Mean fluid density over pore voxels           |
        |                  | (lattice units).                              |
        +------------------+-----------------------------------------------+
        | ``mu_lu``        | Effective dynamic viscosity in lattice        |
        |                  | units, ``rho_lu * nu``.                       |
        +------------------+-----------------------------------------------+
        | ``areas``        | Per-slice pore area counts normal to flow     |
        |                  | direction (voxel-count units).                |
        +------------------+-----------------------------------------------+
        | ``I_slice``      | Per-slice voxel-based polar moment of pore    |
        |                  | area about slice centroid (voxel units).      |
        +------------------+-----------------------------------------------+
        | ``Istar_slice``  | Per-slice normalized polar moment             |
        |                  | ``I / A^2`` (dimensionless).                  |
        +------------------+-----------------------------------------------+
        | ``istar_source`` | ``"user"`` when ``Istar`` is supplied,        |
        |                  | ``"image"`` when computed from voxels.        |
        +------------------+-----------------------------------------------+
        | ``istar_mean``   | Mean ``Istar`` over conduit slices.           |
        +------------------+-----------------------------------------------+
        | ``p_slice``      | Per-slice mean pressure values over pore      |
        |                  | voxels (lattice units).                       |
        +------------------+-----------------------------------------------+
        | ``q_slice``      | Per-slice volumetric flow values over pore    |
        |                  | voxels (lattice units).                       |
        +------------------+-----------------------------------------------+
        | ``development``  | Output dict from ``check_flow_development``   |
        |                  | with detailed pass/fail diagnostics.          |
        +------------------+-----------------------------------------------+
        | ``development_ok``| Convenience boolean equal to                 |
        |                  | ``development['ok']``.                        |
        +------------------+-----------------------------------------------+
        | ``report_text``  | Preformatted multi-line summary from          |
        |                  | ``format_hydraulic_conductance_report``.      |
        +------------------+-----------------------------------------------+
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    if direction not in axis_map:
        raise ValueError("direction must be one of {'x', 'y', 'z'}")
    if pad < 0:
        raise ValueError("pad must be >= 0")
    try:
        mu = float(mu)
    except (TypeError, ValueError) as exc:
        raise ValueError("mu must be a finite positive float.") from exc
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a finite positive float.")
    if rho is None:
        rho_si_out = np.nan
        nu_si = np.nan
    else:
        try:
            rho_si_out = float(rho)
        except (TypeError, ValueError) as exc:
            raise ValueError("rho_si must be a finite positive float or None.") from exc
        if not np.isfinite(rho_si_out) or rho_si_out <= 0.0:
            raise ValueError("rho_si must be a finite positive float or None.")
        nu_si = mu / rho_si_out

    axis = axis_map[direction]
    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (int(pad), int(pad))
    im_padded = np.pad(im, pad_width=pad_width, mode="edge")

    soln = solve_flow(
        im_padded,
        direction=direction,
        n_steps=n_steps,
        velocity_tol=tol,
        log_every=log_every,
    )

    # Internal convention: 1=solid, 0=pore
    fluid = soln.solid == 0
    u_axis = soln.velocity[..., axis]
    p_lu = soln.rho / 3.0

    transverse_axes = tuple(i for i in range(3) if i != axis)
    areas = fluid.sum(axis=transverse_axes).astype(float)
    valid_idx = np.where(areas > 0)[0]
    if len(valid_idx) < 8:
        raise RuntimeError("Not enough valid slices to estimate conductance.")

    n_slices = fluid.shape[axis]
    p_slice = np.full(n_slices, np.nan, dtype=float)
    q_slice = np.full(n_slices, np.nan, dtype=float)

    for j in valid_idx:
        sl = [slice(None), slice(None), slice(None)]
        sl[axis] = j
        m = fluid[tuple(sl)]
        p_slice[j] = np.mean(p_lu[tuple(sl)][m])
        q_slice[j] = np.sum(u_axis[tuple(sl)][m])

    # Conduit region (excluding added leads)
    j_start = int(valid_idx[0] + pad)
    j_end = int(valid_idx[-1] - pad)
    if j_end <= j_start + 4:
        raise RuntimeError("pad is too large for this domain length.")

    conduit_idx = np.arange(j_start, j_end + 1)

    # Linear-fit pressure extraction (diagnostic)
    fit_in = np.arange(valid_idx[0], j_start)
    fit_out = np.arange(j_end + 1, valid_idx[-1] + 1)
    dP_fit = np.nan
    coef_in = None
    coef_out = None
    if len(fit_in) >= 2 and len(fit_out) >= 2:
        coef_in = np.polyfit(fit_in, p_slice[fit_in], deg=1)
        coef_out = np.polyfit(fit_out, p_slice[fit_out], deg=1)
        dP_fit = np.polyval(coef_in, j_start) - np.polyval(coef_out, j_end)

    # Edge-window pressure extraction (default)
    k_edge = max(3, min(8, len(conduit_idx) // 8))
    pin_idx = conduit_idx[:k_edge]
    pout_idx = conduit_idx[-k_edge:]
    p_in_edge = float(np.nanmean(p_slice[pin_idx]))
    p_out_edge = float(np.nanmean(p_slice[pout_idx]))
    dP_edge = p_in_edge - p_out_edge

    Q_lu = float(np.nanmean(q_slice[conduit_idx]))
    if dP_edge == 0.0:
        raise RuntimeError("Estimated conduit pressure drop is zero.")

    R_lbm_lu = dP_edge / Q_lu
    g_lbm_lu = 1.0 / R_lbm_lu

    # Numerical model resistance over extracted conduit only
    A_conduit = areas[conduit_idx]
    A1 = float(A_conduit[0])
    A2 = float(A_conduit[-1])

    i_slice, istar_slice = _slice_polar_moment_and_istar(fluid, axis, valid_idx)
    auto_istar = Istar is None or (
        isinstance(Istar, str) and Istar.strip().lower() == "auto"
    )

    if auto_istar:
        istar_source = "image"
        istar_conduit = istar_slice[conduit_idx]
        if not np.all(np.isfinite(istar_conduit)):
            raise RuntimeError("Failed to compute valid Istar values in conduit slices.")
    else:
        istar_source = "user"
        try:
            istar_value = float(Istar)
        except (TypeError, ValueError) as exc:
            raise ValueError("Istar must be float, None, or 'auto'.") from exc
        if not np.isfinite(istar_value) or istar_value <= 0.0:
            raise ValueError("Istar must be a finite positive value.")
        istar_conduit = np.full(conduit_idx.shape[0], istar_value, dtype=float)

    istar_mean = float(np.mean(istar_conduit))

    rho_lu = float(np.mean(soln.rho[fluid]))
    mu_lu = rho_lu * float(soln.nu)

    # The viscous model uses sum(I* / A^2) whether I* is user-provided or image-derived.
    R_visc_lu = 16.0 * np.pi**2 * mu_lu * float(
        np.sum(istar_conduit / (A_conduit**2))
    )
    R_acc_lu = 0.5 * rho_lu * Q_lu * ((alpha_2 / A2**2) - (alpha_1 / A1**2))
    R_total_lu = R_visc_lu + R_acc_lu
    g_model_lu = 1.0 / R_total_lu

    # LU -> SI conductance conversion [m^3/(Pa.s)]
    scale = float(soln.nu) * float(voxel_size) ** 3 / mu
    g_lbm_si = g_lbm_lu * scale
    g_model_si = g_model_lu * scale

    # LU -> SI resistance conversion [Pa·s/m^3]
    # Since R = 1/g, the resistance scale is the reciprocal of conductance scale
    r_scale = 1.0 / scale
    R_visc_si = R_visc_lu * r_scale
    R_acc_si = R_acc_lu * r_scale
    R_total_si = R_total_lu * r_scale

    development = check_flow_development(
        fluid=fluid,
        u_axis=u_axis,
        axis=axis,
        valid_idx=valid_idx,
        p_slice=p_slice,
        q_slice=q_slice,
        fit_in=fit_in,
        fit_out=fit_out,
        alpha_match_tol=alpha_match_tol,
        alpha_plateau_tol=alpha_plateau_tol,
        pressure_r2_min=pressure_r2_min,
        q_cv_max=q_cv_max,
    )

    out = {
        "flow_result": soln,
        "padded_image": im_padded,
        "direction": direction,
        "pad": int(pad),
        "j_start": j_start,
        "j_end": j_end,
        "conduit_idx": conduit_idx,
        "fit_in": fit_in,
        "fit_out": fit_out,
        "coef_in": coef_in,
        "coef_out": coef_out,
        "dP_fit_lu": dP_fit,
        "dP_edge_lu": dP_edge,
        "Q_lu": Q_lu,
        "R_lbm_lu": R_lbm_lu,
        "g_lbm_lu": g_lbm_lu,
        "R_visc_lu": R_visc_lu,
        "R_acc_lu": R_acc_lu,
        "R_total_lu": R_total_lu,
        "g_model_lu": g_model_lu,
        "g_lbm_si": g_lbm_si,
        "g_model_si": g_model_si,
        "R_visc_si": R_visc_si,
        "R_acc_si": R_acc_si,
        "R_total_si": R_total_si,
        "rho_si": rho_si_out,
        "mu_si": mu,
        "nu_si": nu_si,
        "rho_lu": rho_lu,
        "mu_lu": mu_lu,
        "areas": areas,
        "I_slice": i_slice,
        "Istar_slice": istar_slice,
        "istar_source": istar_source,
        "istar_mean": istar_mean,
        "p_slice": p_slice,
        "q_slice": q_slice,
        "development": development,
        "development_ok": development["ok"],
    }
    out["report_text"] = format_hydraulic_conductance_report(out)
    return out
