"""Tests for solve_flow_xlb — XLB-based LBM solver.

Two suites:

1. ``TestSolveFlowXlbSmoke`` — fast structural checks (field shapes, types,
   FlowResult attributes) on a tiny domain.

2. ``TestXlbVsTaichi`` — numerical agreement between the XLB (BGK) and Taichi
   (MRT) solvers on the same 4-cylinder bundle-of-tubes geometry used in
   ``test_permeability.py``.  Both solvers should agree with each other and
   with the analytical Hagen-Poiseuille result.

Notes on tolerance
------------------
BGK and MRT are different collision operators but produce identical steady-state
solutions in the Stokes (low-Re) regime.  The permeability results are expected
to agree within 5 % — in practice they are much closer, but a generous bound
accounts for minor floating-point differences due to the differing collision
paths.
"""

import numpy as np
import pytest

pytest.importorskip("xlb", reason="xlb is not installed; skipping XLB tests")

from kabs import solve_flow, solve_flow_xlb, compute_permeability  # noqa: E402


# ---------------------------------------------------------------------------
# Shared geometry helpers (mirrors test_permeability.py)
# ---------------------------------------------------------------------------

_R = 10
_L = 30
_NY = _NZ = 50
_NU = 1.0 / 6.0
_CENTRES = [(13, 13), (13, 37), (37, 13), (37, 37)]
_N_TUBES = len(_CENTRES)


def _bundle_image(direction="x"):
    """Binary image (1=pore) of 4 parallel cylinders along *direction*."""
    a, b = np.mgrid[0:_NY, 0:_NZ]
    cross = np.zeros((_NY, _NZ), dtype=bool)
    for ca, cb in _CENTRES:
        cross |= (a - ca) ** 2 + (b - cb) ** 2 <= _R**2
    if direction == "x":
        return np.broadcast_to(cross[np.newaxis], (_L, _NY, _NZ)).copy().astype(int)
    elif direction == "y":
        return np.broadcast_to(cross[:, np.newaxis, :], (_NY, _L, _NZ)).copy().astype(int)
    else:
        return np.broadcast_to(cross[:, :, np.newaxis], (_NY, _NZ, _L)).copy().astype(int)


def _k_analytical():
    return _N_TUBES * np.pi * _R**4 / (8 * _NY * _NZ)


def _tiny_image():
    """Single cylinder r=4 in a 12×12 cross-section, length 10 — for fast smoke tests."""
    a, b = np.mgrid[0:12, 0:12]
    cross = (a - 6) ** 2 + (b - 6) ** 2 <= 4**2
    return np.broadcast_to(cross[np.newaxis], (10, 12, 12)).copy().astype(int)


def _edge_channel_image(direction, length=12):
    """One-voxel pore channel placed on an edge of each pressure face."""
    im = np.zeros((length, length, length), dtype=np.int8)
    axis = {"x": 0, "y": 1, "z": 2}[direction]
    pore = [0, 0, 0]
    pore[axis] = slice(None)
    im[tuple(pore)] = 1
    return im


# ---------------------------------------------------------------------------
# Suite 1: Structural / smoke tests
# ---------------------------------------------------------------------------

_TINY_KW = dict(
    direction="x",
    nu=_NU,
    n_steps=500,
    velocity_tol=1e-3,
    log_every=100,
    verbose=False,
)


class TestSolveFlowXlbSmoke:
    @classmethod
    def setup_class(cls):
        cls.im = _tiny_image()
        cls.result = solve_flow(cls.im, backend="xlb", **_TINY_KW)

    def test_returns_flow_result(self):
        from kabs._solve_flow import FlowResult

        assert isinstance(self.result, FlowResult)

    def test_velocity_shape(self):
        assert self.result.velocity.shape == (10, 12, 12, 3)

    def test_rho_shape(self):
        assert self.result.rho.shape == (10, 12, 12)

    def test_solid_shape(self):
        assert self.result.solid.shape == (10, 12, 12)

    def test_velocity_dtype(self):
        assert self.result.velocity.dtype == np.float32

    def test_rho_dtype(self):
        assert self.result.rho.dtype == np.float32

    def test_solid_is_flipped(self):
        """Internal convention: 1=solid, 0=pore (opposite of input)."""
        assert self.result.solid.dtype == np.int8
        # Input im has 1=pore; solid should have 0 where im==1
        assert np.all(self.result.solid[self.im == 1] == 0)
        assert np.all(self.result.solid[self.im == 0] == 1)

    def test_direction_stored(self):
        assert self.result.direction == "x"

    def test_nu_stored(self):
        assert self.result.nu == pytest.approx(_NU)

    def test_n_iterations_set(self):
        assert self.result.n_iterations is not None
        assert self.result.n_iterations <= 500

    def test_rho_in_range(self):
        """Density should stay close to 1 everywhere."""
        assert float(self.result.rho.min()) > 0.95
        assert float(self.result.rho.max()) < 1.05

    def test_flow_in_x_direction(self):
        """Pressure gradient along x should produce positive x-velocity."""
        pore = self.result.solid == 0
        assert float(self.result.velocity[..., 0][pore].mean()) > 0.0

    def test_transverse_velocity_small(self):
        """y and z velocities should be negligibly small for axial flow."""
        pore = self.result.solid == 0
        vy_mean = float(np.abs(self.result.velocity[..., 1][pore]).mean())
        vz_mean = float(np.abs(self.result.velocity[..., 2][pore]).mean())
        vx_mean = float(np.abs(self.result.velocity[..., 0][pore]).mean())
        assert vy_mean < 0.05 * vx_mean
        assert vz_mean < 0.05 * vx_mean

    def test_compatible_with_compute_permeability(self):
        """FlowResult must be accepted by compute_permeability without error."""
        out = compute_permeability(self.result, verbose=False)
        assert out["k_lu"] > 0.0

    def test_result_records_srt_collision_model(self):
        assert self.result.collision_model == "srt"

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            solve_flow_xlb(self.im, direction="w")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="compute_backend"):
            solve_flow_xlb(self.im, compute_backend="cuda_nope")

    def test_xlb_rejects_taichi_storage_options(self):
        with pytest.raises(ValueError, match="only supported with backend='taichi'"):
            solve_flow(self.im, backend="xlb", storage="tiled")


@pytest.mark.parametrize(("direction", "axis"), [("x", 0), ("y", 1), ("z", 2)])
def test_flow_reaches_edge_pores_on_all_pressure_faces(direction, axis):
    """Pressure BCs must cover pore voxels at face edges and corners."""
    im = _edge_channel_image(direction)
    result = solve_flow_xlb(
        im,
        direction=direction,
        nu=_NU,
        n_steps=400,
        velocity_tol=None,
        log_every=100,
        verbose=False,
    )
    pore = result.solid == 0
    assert float(result.velocity[..., axis][pore].mean()) > 0.0

    inlet = [0, 0, 0]
    outlet = [0, 0, 0]
    outlet[axis] = -1
    assert result.rho[tuple(inlet)] == pytest.approx(1.00, abs=1e-4)
    assert result.rho[tuple(outlet)] == pytest.approx(0.99, abs=1e-4)


# ---------------------------------------------------------------------------
# Suite 2: XLB vs Taichi numerical agreement
# ---------------------------------------------------------------------------

_SOLVE_KW = dict(
    nu=_NU,
    n_steps=4000,
    velocity_tol=1e-3,
    log_every=200,
    verbose=False,
)


class TestXlbVsTaichi:
    """Compare XLB (BGK) against Taichi (MRT) on the 4-cylinder bundle."""

    @classmethod
    def setup_class(cls):
        im = _bundle_image(direction="x")
        cls.result_ti = solve_flow(im, direction="x", **_SOLVE_KW)
        cls.result_ti_srt = solve_flow(
            im, direction="x", collision_model="srt", **_SOLVE_KW
        )
        cls.result_xlb = solve_flow_xlb(im, direction="x", **_SOLVE_KW)
        cls.k_ana = _k_analytical()

    # --- Both solvers vs analytical ---

    def test_taichi_vs_analytical(self):
        """Taichi result must be within 7 % of Hagen-Poiseuille (reference check)."""
        out = compute_permeability(self.result_ti, verbose=False)
        assert out["k_lu"] == pytest.approx(self.k_ana, rel=0.07)

    def test_xlb_vs_analytical(self):
        """XLB result must be within 7 % of Hagen-Poiseuille."""
        out = compute_permeability(self.result_xlb, verbose=False)
        assert out["k_lu"] == pytest.approx(self.k_ana, rel=0.07)

    def test_taichi_srt_vs_analytical(self):
        """Taichi SRT must remain within 7 % of Hagen-Poiseuille."""
        out = compute_permeability(self.result_ti_srt, verbose=False)
        assert out["k_lu"] == pytest.approx(self.k_ana, rel=0.07)

    # --- Mutual agreement ---

    def test_permeability_agreement(self):
        """XLB and Taichi permeabilities must agree within 5 %."""
        k_ti = compute_permeability(self.result_ti, verbose=False)["k_lu"]
        k_xlb = compute_permeability(self.result_xlb, verbose=False)["k_lu"]
        assert k_xlb == pytest.approx(k_ti, rel=0.05)

    def test_bgk_permeability_agreement(self):
        """Taichi SRT and XLB BGK permeabilities must agree within 5 %."""
        k_ti_srt = compute_permeability(
            self.result_ti_srt, verbose=False
        )["k_lu"]
        k_xlb = compute_permeability(self.result_xlb, verbose=False)["k_lu"]
        assert k_xlb == pytest.approx(k_ti_srt, rel=0.05)

    def test_darcy_velocity_agreement(self):
        """XLB and Taichi Darcy velocities must agree within 5 %."""
        u_ti = compute_permeability(self.result_ti, verbose=False)["u_darcy"]
        u_xlb = compute_permeability(self.result_xlb, verbose=False)["u_darcy"]
        assert u_xlb == pytest.approx(u_ti, rel=0.05)

    def test_mean_rho_agreement(self):
        """Volume-averaged density should be close between solvers."""
        rho_ti = float(self.result_ti.rho.mean())
        rho_xlb = float(self.result_xlb.rho.mean())
        assert rho_xlb == pytest.approx(rho_ti, rel=0.01)


# ---------------------------------------------------------------------------
# Allow direct execution: python tests/test_solve_flow_xlb.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import taichi as ti

    ti.init(arch=ti.cpu)

    obj_smoke = TestSolveFlowXlbSmoke()
    TestSolveFlowXlbSmoke.setup_class()
    obj_smoke = TestSolveFlowXlbSmoke()

    obj_cross = TestXlbVsTaichi()
    TestXlbVsTaichi.setup_class()

    for cls, obj in [(TestSolveFlowXlbSmoke, obj_smoke), (TestXlbVsTaichi, obj_cross)]:
        print(f"\n{'=' * 60}")
        print(f"  {cls.__name__}")
        print(f"{'=' * 60}")
        for name in sorted(dir(obj)):
            if name.startswith("test_"):
                method = getattr(obj, name)
                try:
                    method()
                    print(f"  PASS  {name}")
                except Exception as exc:
                    print(f"  FAIL  {name}: {exc}")
