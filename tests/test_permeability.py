"""Tests for compute_permeability against the analytical bundle-of-tubes model.

A "bundle of tubes" consists of N identical, non-overlapping cylindrical pores
aligned along the flow direction.  For this geometry the Darcy permeability is
the sum of the Hagen-Poiseuille contributions divided by the total cross-section:

    k = N * π * R_eff^4 / (8 * A_cross)

where A_cross = ny * nz is the total cross-sectional area and R_eff = R + 0.5 is
the effective radius after the midpoint bounce-back correction (see test_conductance.py).

Test geometry
-------------
4 non-overlapping cylinders of nominal radius R=10, arranged in a 2×2 grid
within a 50×50 cross-section, tube length L=30 along x.

    Centre positions (y, z): (13, 13), (13, 37), (37, 13), (37, 37)
    Nearest-neighbour centre distance: 24 voxels  (> 2R = 20 → 4-voxel gap)

The domain is 30 × 50 × 50 = 75 000 voxels — below the 50³ fast-test limit.
Permeability is independent of L and ν, so using L=30 does not bias the result.

Note on porespy
---------------
porespy is an optional dependency and its random_spheres API places circles via
random sequential addition (RSA) up to a target porosity — it does not accept an
explicit sphere count.  With r=10 in a 50×50 image it achieves at most 2 circles
at maximum packing, making the N=10 bundle model impractical at this domain size.
The image here is therefore constructed directly with NumPy.
"""

import numpy as np
import pytest

from kabs import solve_flow
from kabs._compute_permeability import compute_permeability

# ---------------------------------------------------------------------------
# Bundle geometry parameters
# ---------------------------------------------------------------------------

_N_TUBES = 4
_R = 10
_L = 30  # tube length along x  [voxels]
_NY = _NZ = 50  # cross-section dimensions  [voxels]
_NU = 1.0 / 6.0

# Tube centres in the yz-plane (one centre per tube)
_CENTRES = [(13, 13), (13, 37), (37, 13), (37, 37)]


def _bundle_image():
    """Binary image (1=pore) of 4 parallel cylindrical tubes along x."""
    im = np.zeros((_L, _NY, _NZ), dtype=int)
    y, z = np.mgrid[0:_NY, 0:_NZ]
    for cy, cz in _CENTRES:
        pore = (y - cy) ** 2 + (z - cz) ** 2 <= _R**2
        im[:, pore] = 1
    return im


def _k_analytical():
    """Bundle-of-tubes permeability in lattice units.

    k = N * π * R_eff^4 / (8 * ny * nz)

    Midpoint bounce-back shifts the effective no-slip wall by +0.5 voxels,
    giving R_eff = R + 0.5.
    """
    return _N_TUBES * np.pi * (_R + 0.5) ** 4 / (8 * _NY * _NZ)


# ---------------------------------------------------------------------------
# Shared LBM result
# ---------------------------------------------------------------------------

_SOLVE_KW = dict(
    direction="x",
    nu=_NU,
    n_steps=4000,
    tol=1e-3,
    log_every=200,
    export_vtk=False,
    verbose=False,
)


@pytest.fixture(scope="module")
def bundle_result():
    """Converged LBM solution for the 4-tube bundle (R=10, L=30, 50×50 cross-section)."""
    return solve_flow(_bundle_image(), **_SOLVE_KW)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bundle_permeability_vs_analytical(bundle_result):
    """k_lu must match the bundle-of-tubes analytical formula within 7 %.

    The remaining error after the R_eff = R + 0.5 bounce-back correction is
    due to the discrete voxelisation of the circular boundaries.
    """
    out = compute_permeability(bundle_result, verbose=False)
    assert out["k_lu"] == pytest.approx(_k_analytical(), rel=0.07)


def test_bundle_permeability_positive(bundle_result):
    out = compute_permeability(bundle_result, verbose=False)
    assert out["k_lu"] > 0.0


def test_bundle_darcy_velocity_positive(bundle_result):
    """Pressure gradient must drive flow in the positive x direction."""
    out = compute_permeability(bundle_result, verbose=False)
    assert out["u_darcy"] > 0.0


def test_bundle_no_physical_units_without_dx(bundle_result):
    """k_m2 and k_mD must be None when dx_m is not supplied."""
    out = compute_permeability(bundle_result, verbose=False)
    assert out["k_m2"] is None
    assert out["k_mD"] is None


def test_bundle_physical_units_with_dx(bundle_result):
    """Supplying dx_m must populate k_m2 and k_mD consistently."""
    dx = 1e-6  # 1 µm voxels
    out = compute_permeability(bundle_result, dx_m=dx, verbose=False)
    assert out["k_m2"] == pytest.approx(out["k_lu"] * dx**2, rel=1e-6)
    assert out["k_mD"] == pytest.approx(out["k_m2"] / 9.869233e-16, rel=1e-6)
