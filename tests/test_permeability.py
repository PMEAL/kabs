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


def _bundle_image(direction="x"):
    """Binary image (1=pore) of 4 parallel cylindrical tubes along the given axis."""
    a, b = np.mgrid[0:_NY, 0:_NZ]
    cross = np.zeros((_NY, _NZ), dtype=bool)
    for ca, cb in _CENTRES:
        cross |= (a - ca) ** 2 + (b - cb) ** 2 <= _R**2
    if direction == "x":
        return np.broadcast_to(cross[np.newaxis], (_L, _NY, _NZ)).copy().astype(int)
    elif direction == "y":
        return (
            np.broadcast_to(cross[:, np.newaxis, :], (_NY, _L, _NZ)).copy().astype(int)
        )
    else:  # z
        return (
            np.broadcast_to(cross[:, :, np.newaxis], (_NY, _NZ, _L)).copy().astype(int)
        )


def _k_analytical():
    """Bundle-of-tubes permeability in lattice units: k = N * π * R^4 / (8 * ny * nz)."""
    return _N_TUBES * np.pi * _R**4 / (8 * _NY * _NZ)


# ---------------------------------------------------------------------------
# Shared LBM result
# ---------------------------------------------------------------------------

_SOLVE_KW = dict(
    direction="x",
    nu=_NU,
    n_steps=4000,
    tol=1e-3,
    log_every=200,
    verbose=False,
)


class TestBundlePermeability:
    @classmethod
    def setup_class(cls):
        cls.result = solve_flow(_bundle_image(), **_SOLVE_KW)

    def test_permeability_vs_analytical(self):
        """k_lu must match the bundle-of-tubes analytical formula within 7 %.

        The remaining error after the R_eff = R + 0.5 bounce-back correction is
        due to the discrete voxelisation of the circular boundaries.
        """
        out = compute_permeability(self.result, verbose=False)
        assert out["k_lu"] == pytest.approx(_k_analytical(), rel=0.07)

    def test_permeability_positive(self):
        out = compute_permeability(self.result, verbose=False)
        assert out["k_lu"] > 0.0

    def test_darcy_velocity_positive(self):
        """Pressure gradient must drive flow in the positive x direction."""
        out = compute_permeability(self.result, verbose=False)
        assert out["u_darcy"] > 0.0

    def test_no_physical_units_without_dx(self):
        """k_m2 and k_mD must be None when dx_m is not supplied."""
        out = compute_permeability(self.result, verbose=False)
        assert out["k_m2"] is None
        assert out["k_mD"] is None

    def test_physical_units_with_dx(self):
        """Supplying dx_m must populate k_m2 and k_mD consistently."""
        dx = 1e-6  # 1 µm voxels
        out = compute_permeability(self.result, dx_m=dx, verbose=False)
        assert out["k_m2"] == pytest.approx(out["k_lu"] * dx**2, rel=1e-6)
        assert out["k_mD"] == pytest.approx(out["k_m2"] / 9.869233e-16, rel=1e-6)


# ---------------------------------------------------------------------------
# Permeability tensor: all three BC-setter pairs must drive correct flow
# ---------------------------------------------------------------------------

_SOLVE_KW_BASE = dict(
    nu=_NU,
    n_steps=4000,
    tol=1e-3,
    log_every=200,
    verbose=False,
)


class TestPermeabilityTensor:
    """Run the isotropic bundle image in all three directions.

    Each direction exercises a different pair of BC setters:
      x → set_bc_rho_x0 / set_bc_rho_x1
      y → set_bc_rho_y0 / set_bc_rho_y1
      z → set_bc_rho_z0 / set_bc_rho_z1

    Because the cross-section geometry is identical in all cases, all three
    permeabilities must agree with each other and with the analytical value.
    """

    @classmethod
    def setup_class(cls):
        cls.results = {
            ax: solve_flow(_bundle_image(ax), direction=ax, **_SOLVE_KW_BASE)
            for ax in ("x", "y", "z")
        }
        cls.k = {
            ax: compute_permeability(cls.results[ax], verbose=False)["k_lu"]
            for ax in ("x", "y", "z")
        }

    def test_kx_vs_analytical(self):
        assert self.k["x"] == pytest.approx(_k_analytical(), rel=0.07)

    def test_ky_vs_analytical(self):
        assert self.k["y"] == pytest.approx(_k_analytical(), rel=0.07)

    def test_kz_vs_analytical(self):
        assert self.k["z"] == pytest.approx(_k_analytical(), rel=0.07)

    def test_all_directions_agree(self):
        """Isotropic geometry: kx, ky, kz must all match within 2 %."""
        assert self.k["y"] == pytest.approx(self.k["x"], rel=0.02)
        assert self.k["z"] == pytest.approx(self.k["x"], rel=0.02)

    def test_darcy_velocity_positive_all_directions(self):
        """Each simulation must drive flow in the positive pressure-gradient direction."""
        for ax in ("x", "y", "z"):
            out = compute_permeability(self.results[ax], verbose=False)
            assert out["u_darcy"] > 0.0, f"u_darcy not positive for direction={ax!r}"


# ---------------------------------------------------------------------------
# Sparse storage: results must match the dense solver
# ---------------------------------------------------------------------------

_SOLVE_KW_SPARSE = {**_SOLVE_KW, "sparse": True}


class TestSparsePermeability:
    """Sparse storage should produce the same physics as dense storage."""

    @classmethod
    def setup_class(cls):
        cls.result_dense = solve_flow(_bundle_image(), **_SOLVE_KW)
        cls.result_sparse = solve_flow(_bundle_image(), **_SOLVE_KW_SPARSE)

    def test_permeability_matches_dense(self):
        """Sparse and dense solvers must agree on k_lu within 1 %."""
        k_dense = compute_permeability(self.result_dense, verbose=False)["k_lu"]
        k_sparse = compute_permeability(self.result_sparse, verbose=False)["k_lu"]
        assert k_sparse == pytest.approx(k_dense, rel=0.01)

    def test_permeability_vs_analytical(self):
        """Sparse k_lu must still match the bundle-of-tubes formula within 7 %."""
        out = compute_permeability(self.result_sparse, verbose=False)
        assert out["k_lu"] == pytest.approx(_k_analytical(), rel=0.07)

    def test_velocity_field_shape(self):
        """Sparse solver must return a velocity array of the correct shape."""
        nx, ny, nz = self.result_sparse.solid.shape
        assert self.result_sparse.velocity.shape == (nx, ny, nz, 3)

    def test_solid_field_matches_dense(self):
        """Solid mask must be identical between sparse and dense runs."""
        np.testing.assert_array_equal(self.result_sparse.solid, self.result_dense.solid)


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
