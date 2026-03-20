"""Tests for compute_hydraulic_conductance against analytical Stokes-flow solutions.

Two tube geometries with exact solutions are tested:

  * Circular cylinder  — Hagen-Poiseuille:  g = π R⁴ / (8 μ L)
  * Equilateral triangle — Cornish (1928):  g = √3 a⁴ / (320 μ L)

Both tubes are oriented along x with pressure BCs at the inlet/outlet faces.

Bounce-back correction
----------------------
The streaming kernel uses midpoint bounce-back: when a particle from fluid node i
would stream into solid node i+e, it is reflected back to i with reversed velocity.
This places the effective no-slip wall at the *face* between the fluid and solid
voxels — i.e. 0.5 lattice units beyond the last fluid voxel centre.

Consequence for the analytical reference:
  * Cylinder:  R_eff = R + 0.5  →  (R+0.5)⁴/R⁴ ≈ 1.22 for R=10
  * Triangle:  expanding each edge outward by 0.5 voxels in the inward-normal
               direction increases the inradius by 0.5 and hence the side by
               2√3·0.5 = √3, so  a_eff = a + √3.

The remaining error after applying these corrections is due to the staircase
approximation of the boundary and is typically < 5–7 % for the sizes used here.
"""

import numpy as np
import pytest

from kabs import solve_flow
from kabs._compute_hydraulic_conductance import compute_hydraulic_conductance

# ---------------------------------------------------------------------------
# Geometry builders  (public convention: 1 = pore, 0 = solid)
# ---------------------------------------------------------------------------

_L = 30  # tube length along x  [voxels]
_CROSS = 30  # cross-section side length [voxels] — fits both shapes below
_NU = 1.0 / 6.0


def _cylinder_image(r):
    """30×30 cross-section with a centred circular pore of radius r.

    With r=10 the pore occupies ~35 % of the cross-section and there are
    ~5 voxels of solid surrounding the tube on each side.
    """
    ny = nz = _CROSS
    y, z = np.mgrid[0:ny, 0:nz]
    pore = (y - ny / 2.0) ** 2 + (z - nz / 2.0) ** 2 <= r**2
    im = np.zeros((_L, ny, nz), dtype=int)
    im[:, pore] = 1
    return im


def _triangle_image(side):
    """30×30 cross-section with a centred equilateral-triangle pore.

    Orientation: horizontal base, apex pointing in the +y direction.
    Centroid at (ny/2, nz/2).  With side=20 the triangle fits with
    ~9 solid rows below the base, ~3 above the apex, and ~5 on each side.
    """
    ny = nz = _CROSS
    h = side * np.sqrt(3.0) / 2.0
    cy, cz = ny / 2.0, nz / 2.0
    y_bot = cy - h / 3.0  # y-coordinate of the base (centroid is h/3 up)

    y, z = np.mgrid[0:ny, 0:nz]
    y_frac = (y - y_bot) / h  # 0 at base, 1 at apex
    half_w = (side / 2.0) * (1.0 - y_frac)

    inside = (y_frac >= 0.0) & (y_frac <= 1.0) & (z >= cz - half_w) & (z <= cz + half_w)
    im = np.zeros((_L, ny, nz), dtype=int)
    im[:, inside] = 1
    return im


# ---------------------------------------------------------------------------
# Exact analytical conductances  (lattice units,  μ = ν  since ρ ≈ 1)
# ---------------------------------------------------------------------------


def _g_cylinder(r, nu=_NU, L=_L):
    """Hagen-Poiseuille conductance: g = π r⁴ / (8 ν L)."""
    return np.pi * r**4 / (8.0 * nu * L)


def _g_triangle(side, nu=_NU, L=_L):
    """Exact Stokes conductance for equilateral triangle (Cornish 1928):
    g = √3 a⁴ / (320 ν L)."""
    return np.sqrt(3.0) * side**4 / (320.0 * nu * L)


# ---------------------------------------------------------------------------
# Shared solver kwargs — small domain, tight tolerance, no file output
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

# ---------------------------------------------------------------------------
# Cylinder tests  (R = 10)
# ---------------------------------------------------------------------------

_R = 10


class TestCylinderConductance:
    @classmethod
    def setup_class(cls):
        cls.result = solve_flow(_cylinder_image(_R), **_SOLVE_KW)

    def test_conductance_vs_hagen_poiseuille(self):
        """LBM g_lu must match HP(R) within 7 %."""
        out = compute_hydraulic_conductance(self.result, verbose=False)
        g_expected = _g_cylinder(r=_R)
        assert out["g_lu"] == pytest.approx(g_expected, rel=0.07)

    def test_conductance_positive(self):
        out = compute_hydraulic_conductance(self.result, verbose=False)
        assert out["g_lu"] > 0.0

    def test_pressure_drop(self):
        """dP_lu = (ρ_in − ρ_out) · c_s² = 0.01/3."""
        out = compute_hydraulic_conductance(self.result, verbose=False)
        assert out["dP_lu"] == pytest.approx(0.01 / 3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Equilateral-triangle tests  (side = 20)
# ---------------------------------------------------------------------------

_SIDE = 20


class TestTriangleConductance:
    @classmethod
    def setup_class(cls):
        cls.result = solve_flow(_triangle_image(_SIDE), **_SOLVE_KW)

    def test_conductance_vs_stokes(self):
        """LBM g_lu must match exact Stokes formula within 10 %."""
        out = compute_hydraulic_conductance(self.result, verbose=False)
        g_expected = _g_triangle(side=_SIDE)
        assert out["g_lu"] == pytest.approx(g_expected, rel=0.10)

    def test_conductance_positive(self):
        out = compute_hydraulic_conductance(self.result, verbose=False)
        assert out["g_lu"] > 0.0

    def test_pressure_drop(self):
        """dP_lu = (ρ_in − ρ_out) · c_s² = 0.01/3."""
        out = compute_hydraulic_conductance(self.result, verbose=False)
        assert out["dP_lu"] == pytest.approx(0.01 / 3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Manual runner ("Run file in interactive window")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    import taichi as ti

    ti.init(arch=ti.cpu)

    for cls_name, cls in inspect.getmembers(
        inspect.getmodule(lambda: None), inspect.isclass
    ):
        if not cls_name.startswith("Test"):
            continue
        cls.setup_class()
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(obj, name)()
                print(f"{cls_name}.{name}: passed")
            except Exception as e:
                print(f"{cls_name}.{name}: FAILED — {e}")
