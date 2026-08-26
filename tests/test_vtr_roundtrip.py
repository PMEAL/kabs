"""Tests for write_flow_vtr / read_flow_vtr round-trip fidelity.

These tests verify that a FlowResult written to disk by write_flow_vtr can be
read back by read_flow_vtr and that the resulting FlowResult's arrays are
numerically identical to the originals.  No Taichi initialisation is required
as the tests construct synthetic numpy arrays directly via FlowResult.from_arrays.
"""

import pathlib
import tempfile

import numpy as np
import pytest

from kabs._solve_flow import FlowResult
from kabs.utils import read_flow_vtr, write_flow_vtr


def _make_result(nx, ny, nz, seed=0):
    """Return a FlowResult built directly from synthetic numpy arrays."""
    rng = np.random.default_rng(seed)
    solid = rng.integers(0, 2, size=(nx, ny, nz), dtype=np.int8)
    rho = rng.random((nx, ny, nz), dtype=np.float32)
    velocity = rng.random((nx, ny, nz, 3), dtype=np.float32)
    return FlowResult.from_arrays(solid, rho, velocity)


class TestVtrRoundtrip:
    def setup_method(self):
        self._tmpdir = pathlib.Path(tempfile.mkdtemp())
        self.result = _make_result(5, 6, 7)

    def _prefix(self, name="test"):
        return str(self._tmpdir / name)

    # -----------------------------------------------------------------------
    # Return type
    # -----------------------------------------------------------------------

    def test_read_returns_flow_result(self):
        prefix = self._prefix()
        write_flow_vtr(prefix, self.result)
        soln = read_flow_vtr(prefix + ".vtr")
        assert isinstance(soln, FlowResult)

    def test_read_direction_and_nu_none_when_absent(self):
        prefix = self._prefix()
        write_flow_vtr(prefix, self.result)
        soln = read_flow_vtr(prefix + ".vtr")
        assert soln.direction is None
        assert soln.nu is None
        assert soln.collision_model is None

    # -----------------------------------------------------------------------
    # Round-trip: write then read
    # -----------------------------------------------------------------------

    def test_roundtrip_shape(self):
        prefix = self._prefix()
        write_flow_vtr(prefix, self.result)
        soln = read_flow_vtr(prefix + ".vtr")
        assert soln.solid.shape == self.result.solid.shape
        assert soln.rho.shape == self.result.rho.shape
        assert soln.velocity.shape == self.result.velocity.shape

    def test_roundtrip_solid_exact(self):
        prefix = self._prefix()
        write_flow_vtr(prefix, self.result)
        soln = read_flow_vtr(prefix + ".vtr")
        np.testing.assert_array_equal(soln.solid, self.result.solid)

    def test_roundtrip_rho_exact(self):
        prefix = self._prefix()
        write_flow_vtr(prefix, self.result)
        soln = read_flow_vtr(prefix + ".vtr")
        np.testing.assert_array_equal(soln.rho, self.result.rho)

    def test_roundtrip_velocity_exact(self):
        prefix = self._prefix()
        write_flow_vtr(prefix, self.result)
        soln = read_flow_vtr(prefix + ".vtr")
        np.testing.assert_array_equal(soln.velocity, self.result.velocity)

    # -----------------------------------------------------------------------
    # dtype checks
    # -----------------------------------------------------------------------

    def test_roundtrip_dtypes(self):
        prefix = self._prefix()
        write_flow_vtr(prefix, self.result)
        soln = read_flow_vtr(prefix + ".vtr")
        assert soln.solid.dtype == np.int8
        assert soln.rho.dtype == np.float32
        assert soln.velocity.dtype == np.float32

    # -----------------------------------------------------------------------
    # direction / nu metadata
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("direction", ["x", "y", "z"])
    def test_roundtrip_direction(self, direction):
        result = FlowResult.from_arrays(
            solid=np.zeros((4, 4, 4), dtype=np.int8),
            rho=np.ones((4, 4, 4), dtype=np.float32),
            velocity=np.zeros((4, 4, 4, 3), dtype=np.float32),
            direction=direction,
        )
        prefix = self._prefix()
        write_flow_vtr(prefix, result)
        soln = read_flow_vtr(prefix + ".vtr")
        assert soln.direction == direction

    def test_roundtrip_nu(self):
        nu = 1.0 / 6.0
        result = FlowResult.from_arrays(
            solid=np.zeros((4, 4, 4), dtype=np.int8),
            rho=np.ones((4, 4, 4), dtype=np.float32),
            velocity=np.zeros((4, 4, 4, 3), dtype=np.float32),
            nu=nu,
        )
        prefix = self._prefix()
        write_flow_vtr(prefix, result)
        soln = read_flow_vtr(prefix + ".vtr")
        assert soln.nu == pytest.approx(nu)

    @pytest.mark.parametrize("collision_model", ["mrt", "srt"])
    def test_roundtrip_collision_model(self, collision_model):
        result = FlowResult.from_arrays(
            solid=np.zeros((4, 4, 4), dtype=np.int8),
            rho=np.ones((4, 4, 4), dtype=np.float32),
            velocity=np.zeros((4, 4, 4, 3), dtype=np.float32),
            collision_model=collision_model,
        )
        prefix = self._prefix()
        write_flow_vtr(prefix, result)
        soln = read_flow_vtr(prefix + ".vtr")
        assert soln.collision_model == collision_model

    # -----------------------------------------------------------------------
    # Velocity component ordering
    # -----------------------------------------------------------------------

    def test_roundtrip_velocity_components(self):
        nx, ny, nz = 4, 5, 6
        solid = np.zeros((nx, ny, nz), dtype=np.int8)
        rho = np.ones((nx, ny, nz), dtype=np.float32)
        velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        velocity[..., 0] = 1.0
        velocity[..., 1] = 2.0
        velocity[..., 2] = 3.0
        result = FlowResult.from_arrays(solid, rho, velocity)
        prefix = self._prefix("components")
        write_flow_vtr(prefix, result)
        soln = read_flow_vtr(prefix + ".vtr")
        np.testing.assert_array_equal(soln.velocity[..., 0], velocity[..., 0])
        np.testing.assert_array_equal(soln.velocity[..., 1], velocity[..., 1])
        np.testing.assert_array_equal(soln.velocity[..., 2], velocity[..., 2])

    # -----------------------------------------------------------------------
    # Non-cubic domains
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("shape", [(3, 4, 5), (10, 10, 10), (2, 8, 4)])
    def test_roundtrip_various_shapes(self, shape):
        result = _make_result(*shape, seed=42)
        prefix = self._prefix("x".join(map(str, shape)))
        write_flow_vtr(prefix, result)
        soln = read_flow_vtr(prefix + ".vtr")
        np.testing.assert_array_equal(soln.solid, result.solid)
        np.testing.assert_array_equal(soln.rho, result.rho)
        np.testing.assert_array_equal(soln.velocity, result.velocity)

    # -----------------------------------------------------------------------
    # All-solid and all-pore edge cases
    # -----------------------------------------------------------------------

    def test_roundtrip_all_pore(self):
        nx, ny, nz = 4, 4, 4
        result = FlowResult.from_arrays(
            solid=np.zeros((nx, ny, nz), dtype=np.int8),
            rho=np.full((nx, ny, nz), 1.0, dtype=np.float32),
            velocity=np.zeros((nx, ny, nz, 3), dtype=np.float32),
        )
        prefix = self._prefix("all_pore")
        write_flow_vtr(prefix, result)
        soln = read_flow_vtr(prefix + ".vtr")
        np.testing.assert_array_equal(soln.solid, result.solid)
        np.testing.assert_array_equal(soln.rho, result.rho)
        np.testing.assert_array_equal(soln.velocity, result.velocity)

    def test_roundtrip_all_solid(self):
        nx, ny, nz = 4, 4, 4
        result = FlowResult.from_arrays(
            solid=np.ones((nx, ny, nz), dtype=np.int8),
            rho=np.ones((nx, ny, nz), dtype=np.float32),
            velocity=np.zeros((nx, ny, nz, 3), dtype=np.float32),
        )
        prefix = self._prefix("all_solid")
        write_flow_vtr(prefix, result)
        soln = read_flow_vtr(prefix + ".vtr")
        np.testing.assert_array_equal(soln.solid, result.solid)


# ---------------------------------------------------------------------------
# Manual runner ("Run file in interactive window")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    obj = TestVtrRoundtrip()
    for name in sorted(dir(obj)):
        if not name.startswith("test_"):
            continue
        fn = getattr(obj.__class__, name)
        marks = getattr(fn, "pytestmark", [])
        parametrize = [m for m in marks if m.name == "parametrize"]
        if parametrize:
            argname = parametrize[0].args[0]
            cases = [(f"{name}[{v}]", v) for v in parametrize[0].args[1]]
        else:
            cases = [(name, None)]
        for label, val in cases:
            obj.setup_method()
            try:
                if val is None:
                    getattr(obj, name)()
                else:
                    getattr(obj, name)(val)
                print(f"{label}: passed")
            except Exception as e:
                print(f"{label}: FAILED — {e}")
