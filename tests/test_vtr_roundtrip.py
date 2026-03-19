"""Tests for write_flow_vtr / read_flow_vtr round-trip fidelity.

These tests verify that a FlowResult written to disk by write_flow_vtr can be
read back by read_flow_vtr and that the resulting FlowResult's arrays are
numerically identical to the originals.  No Taichi initialisation is required
— the tests construct synthetic numpy arrays directly via FlowResult.from_arrays.
"""

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


@pytest.fixture
def small_result():
    return _make_result(5, 6, 7)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_read_returns_flow_result(tmp_path, small_result):
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, small_result)
    soln = read_flow_vtr(prefix + ".vtr")
    assert isinstance(soln, FlowResult)


def test_read_direction_and_nu_none_when_absent(tmp_path, small_result):
    """direction and nu are None when the result had none to begin with."""
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, small_result)
    soln = read_flow_vtr(prefix + ".vtr")
    assert soln.direction is None
    assert soln.nu is None


@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_roundtrip_direction(tmp_path, direction):
    """direction is preserved through write/read when present on the result."""
    result = FlowResult.from_arrays(
        solid=np.zeros((4, 4, 4), dtype=np.int8),
        rho=np.ones((4, 4, 4), dtype=np.float32),
        velocity=np.zeros((4, 4, 4, 3), dtype=np.float32),
        direction=direction,
    )
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, result)
    soln = read_flow_vtr(prefix + ".vtr")
    assert soln.direction == direction


def test_roundtrip_nu(tmp_path):
    """nu is preserved through write/read when present on the result."""
    nu = 1.0 / 6.0
    result = FlowResult.from_arrays(
        solid=np.zeros((4, 4, 4), dtype=np.int8),
        rho=np.ones((4, 4, 4), dtype=np.float32),
        velocity=np.zeros((4, 4, 4, 3), dtype=np.float32),
        nu=nu,
    )
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, result)
    soln = read_flow_vtr(prefix + ".vtr")
    assert soln.nu == pytest.approx(nu)


# ---------------------------------------------------------------------------
# Round-trip: write then read
# ---------------------------------------------------------------------------


def test_roundtrip_shape(tmp_path, small_result):
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, small_result)
    soln = read_flow_vtr(prefix + ".vtr")

    assert soln.solid.shape == small_result.solid.shape
    assert soln.rho.shape == small_result.rho.shape
    assert soln.velocity.shape == small_result.velocity.shape


def test_roundtrip_solid_exact(tmp_path, small_result):
    """Solid (int8) must survive the round-trip without any change."""
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, small_result)
    soln = read_flow_vtr(prefix + ".vtr")

    np.testing.assert_array_equal(soln.solid, small_result.solid)


def test_roundtrip_rho_exact(tmp_path, small_result):
    """Density (float32) must survive the round-trip bit-for-bit."""
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, small_result)
    soln = read_flow_vtr(prefix + ".vtr")

    np.testing.assert_array_equal(soln.rho, small_result.rho)


def test_roundtrip_velocity_exact(tmp_path, small_result):
    """Velocity (float32, 3 components) must survive the round-trip bit-for-bit."""
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, small_result)
    soln = read_flow_vtr(prefix + ".vtr")

    np.testing.assert_array_equal(soln.velocity, small_result.velocity)


# ---------------------------------------------------------------------------
# dtype checks
# ---------------------------------------------------------------------------


def test_roundtrip_dtypes(tmp_path, small_result):
    prefix = str(tmp_path / "test")
    write_flow_vtr(prefix, small_result)
    soln = read_flow_vtr(prefix + ".vtr")

    assert soln.solid.dtype == np.int8
    assert soln.rho.dtype == np.float32
    assert soln.velocity.dtype == np.float32


# ---------------------------------------------------------------------------
# Velocity component ordering
# ---------------------------------------------------------------------------


def test_roundtrip_velocity_components(tmp_path):
    """Each velocity component must map back to the correct axis after round-trip."""
    nx, ny, nz = 4, 5, 6
    solid = np.zeros((nx, ny, nz), dtype=np.int8)
    rho = np.ones((nx, ny, nz), dtype=np.float32)
    # Give each component a distinct constant so mis-ordering is detectable
    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    velocity[..., 0] = 1.0  # vx
    velocity[..., 1] = 2.0  # vy
    velocity[..., 2] = 3.0  # vz

    result = FlowResult.from_arrays(solid, rho, velocity)
    prefix = str(tmp_path / "test_components")
    write_flow_vtr(prefix, result)
    soln = read_flow_vtr(prefix + ".vtr")

    np.testing.assert_array_equal(soln.velocity[..., 0], velocity[..., 0])
    np.testing.assert_array_equal(soln.velocity[..., 1], velocity[..., 1])
    np.testing.assert_array_equal(soln.velocity[..., 2], velocity[..., 2])


# ---------------------------------------------------------------------------
# Non-cubic domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 4, 5), (10, 10, 10), (2, 8, 4)])
def test_roundtrip_various_shapes(tmp_path, shape):
    result = _make_result(*shape, seed=42)
    prefix = str(tmp_path / f"test_{'x'.join(map(str, shape))}")
    write_flow_vtr(prefix, result)
    soln = read_flow_vtr(prefix + ".vtr")

    np.testing.assert_array_equal(soln.solid, result.solid)
    np.testing.assert_array_equal(soln.rho, result.rho)
    np.testing.assert_array_equal(soln.velocity, result.velocity)


# ---------------------------------------------------------------------------
# All-solid and all-pore edge cases
# ---------------------------------------------------------------------------


def test_roundtrip_all_pore(tmp_path):
    nx, ny, nz = 4, 4, 4
    result = FlowResult.from_arrays(
        solid=np.zeros((nx, ny, nz), dtype=np.int8),
        rho=np.full((nx, ny, nz), 1.0, dtype=np.float32),
        velocity=np.zeros((nx, ny, nz, 3), dtype=np.float32),
    )
    prefix = str(tmp_path / "all_pore")
    write_flow_vtr(prefix, result)
    soln = read_flow_vtr(prefix + ".vtr")

    np.testing.assert_array_equal(soln.solid, result.solid)
    np.testing.assert_array_equal(soln.rho, result.rho)
    np.testing.assert_array_equal(soln.velocity, result.velocity)


def test_roundtrip_all_solid(tmp_path):
    nx, ny, nz = 4, 4, 4
    result = FlowResult.from_arrays(
        solid=np.ones((nx, ny, nz), dtype=np.int8),
        rho=np.ones((nx, ny, nz), dtype=np.float32),
        velocity=np.zeros((nx, ny, nz, 3), dtype=np.float32),
    )
    prefix = str(tmp_path / "all_solid")
    write_flow_vtr(prefix, result)
    soln = read_flow_vtr(prefix + ".vtr")

    np.testing.assert_array_equal(soln.solid, result.solid)
