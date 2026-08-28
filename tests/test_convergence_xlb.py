"""Device-side convergence regressions for the XLB solver."""

import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kabs import compute_permeability, solve_flow_xlb
from kabs._convergence import ConvergenceConfig
from kabs._solve_flow_xlb import (
    _convergence_sums_to_host,
    _get_jax_convergence_reducer,
)


def _channel(shape=(6, 6, 6)):
    image = np.zeros(shape, dtype=np.int8)
    image[:, 2:4, 2:4] = 1
    return image


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        ([1.0, -2.0, 3.0], [1.0, -2.0, 3.0]),
        ([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]),
        ([-1.0, 2.0, -3.0], [2.0, -1.0, -4.0]),
        ([1.0, -1.0, 2.0], [0.0, 0.0, 0.0]),
    ],
)
def test_jax_reduction_matches_numpy(previous, current):
    previous_device = jnp.asarray(previous, dtype=jnp.float32)
    current_device = jnp.asarray(current, dtype=jnp.float32)

    v_total, v_change = _convergence_sums_to_host(current_device, previous_device)

    previous_host = np.asarray(previous, dtype=np.float32)
    current_host = np.asarray(current, dtype=np.float32)
    assert v_total == pytest.approx(np.sum(np.abs(current_host)), rel=1e-6)
    assert v_change == pytest.approx(
        np.sum(np.abs(current_host - previous_host)), rel=1e-6
    )


def test_reduction_uses_one_synchronized_device_get(monkeypatch):
    calls = []
    original = jax.device_get

    def tracked_device_get(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(jax, "device_get", tracked_device_get)

    totals = _convergence_sums_to_host(
        jnp.ones((3, 2, 2, 2), dtype=jnp.float32),
        jnp.zeros((3, 2, 2, 2), dtype=jnp.float32),
    )

    assert totals == pytest.approx((24.0, 24.0))
    assert len(calls) == 1
    assert calls[0].shape == (2,)


def _numpy_convergence_sums(current, previous):
    current_host = np.array(current)
    previous_host = np.array(previous)
    return (
        float(np.sum(np.abs(current_host))),
        float(np.sum(np.abs(current_host - previous_host))),
    )


def test_xlb_solver_reports_two_pass_velocity_convergence(capsys):
    kwargs = {
        "n_steps": 4,
        "log_every": 0,
        "convergence_every": 1,
        "velocity_tol": 1e9,
        "verbose": False,
        "compute_backend": "jax",
    }
    image = _channel()

    device = solve_flow_xlb(image, **{**kwargs, "verbose": True})
    output = capsys.readouterr().out
    assert "velocity=" in output
    assert "streak=2" in output
    assert "Converged at step" in output
    assert device.n_iterations == 3
    assert device.converged
    assert device.consecutive_passes == 2
    assert device.velocity_criterion < device.velocity_tol


def test_periodic_checks_do_not_convert_full_velocity_to_numpy(monkeypatch):
    xlb_module = importlib.import_module("kabs._solve_flow_xlb")
    original = np.array
    full_velocity_conversions = []
    device_get_shapes = []
    expected_shape = (3, 6, 6, 6)
    original_device_get = jax.device_get

    def tracked_array(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(value, jax.Array) and result.shape == expected_shape:
            full_velocity_conversions.append(result.shape)
        return result

    def tracked_device_get(value):
        device_get_shapes.append(value.shape)
        return original_device_get(value)

    monkeypatch.setattr(xlb_module.np, "array", tracked_array)
    monkeypatch.setattr(jax, "device_get", tracked_device_get)

    result = solve_flow_xlb(
        _channel(),
        n_steps=3,
        log_every=0,
        convergence_every=1,
        velocity_tol=1e-12,
        verbose=False,
    )

    assert result.convergence_criterion is not None
    assert device_get_shapes == [(2,), (2,)]
    assert full_velocity_conversions == [expected_shape]


def test_single_check_skips_convergence_reduction(monkeypatch):
    xlb_module = importlib.import_module("kabs._solve_flow_xlb")

    def unexpected_reduction(*args):
        raise AssertionError("a single check must not run a convergence reduction")

    monkeypatch.setattr(xlb_module, "_get_jax_convergence_reducer", unexpected_reduction)
    result = solve_flow_xlb(
        _channel(),
        n_steps=1,
        log_every=0,
        convergence_every=1,
        velocity_tol=1e-3,
        verbose=False,
    )

    assert result.n_iterations == 1
    assert not result.converged
    assert result.convergence_criterion is None


@pytest.mark.parametrize(("direction", "axis"), [("x", 0), ("y", 1), ("z", 2)])
def test_jax_combined_reduction_matches_masked_numpy(direction, axis):
    del direction
    rng = np.random.default_rng(17)
    shape = (3, 4, 5)
    current_host = rng.normal(size=(3, *shape)).astype(np.float32)
    previous_host = rng.normal(size=(3, *shape)).astype(np.float32)
    rho_host = rng.uniform(0.9, 1.1, size=(1, *shape)).astype(np.float32)
    pore_host = np.ones(shape, dtype=bool)
    pore_host[1, 2, 3] = False
    values = np.asarray(
        _get_jax_convergence_reducer(True, True, True, axis)(
            jnp.asarray(rho_host),
            jnp.asarray(current_host),
            jnp.asarray(previous_host),
            jnp.asarray(pore_host[None, ...]),
        )
    )
    directional = current_host[axis]
    inlet = [slice(None)] * 3
    outlet = [slice(None)] * 3
    inlet[axis] = 0
    outlet[axis] = -1
    expected = (
        np.sum(np.abs(current_host) * pore_host[None]),
        np.sum(np.abs(current_host - previous_host) * pore_host[None]),
        np.sum(directional * pore_host),
        np.sum((rho_host[0] * directional * pore_host)[tuple(inlet)]),
        np.sum((rho_host[0] * directional * pore_host)[tuple(outlet)]),
    )
    assert values == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_warp_cpu_combined_reduction_matches_masked_numpy(axis):
    import warp as wp

    from kabs._warp_convergence import WarpConvergenceMonitor

    rng = np.random.default_rng(23)
    shape = (3, 4, 5)
    previous = rng.normal(size=(3, *shape)).astype(np.float32)
    current_host = rng.normal(size=(3, *shape)).astype(np.float32)
    rho_host = rng.uniform(0.9, 1.1, size=(1, *shape)).astype(np.float32)
    pore = np.ones(shape, dtype=bool)
    pore[1, 2, 3] = False
    current = wp.array(previous, dtype=wp.float32, device="cpu")
    rho = wp.array(rho_host, dtype=wp.float32, device="cpu")
    config = ConvergenceConfig(1, 1.0, 1.0, 1.0)
    monitor = WarpConvergenceMonitor(current, pore, config, axis)
    monitor.sample(rho, current)
    monitor.snapshot_velocity(current)
    wp.copy(current, wp.array(current_host, dtype=wp.float32, device="cpu"))
    values = monitor.sample(rho, current)

    directional = current_host[axis]
    inlet = [slice(None)] * 3
    outlet = [slice(None)] * 3
    inlet[axis] = 0
    outlet[axis] = -1
    assert values.velocity_total == pytest.approx(
        np.sum(np.abs(current_host) * pore[None]), rel=2e-6
    )
    assert values.velocity_change == pytest.approx(
        np.sum(np.abs(current_host - previous) * pore[None]), rel=2e-6
    )
    assert values.directional_flow == pytest.approx(
        np.sum(directional * pore), rel=2e-6, abs=2e-6
    )
    assert values.inlet_mass_flux == pytest.approx(
        np.sum((rho_host[0] * directional * pore)[tuple(inlet)]),
        rel=2e-6,
        abs=2e-6,
    )
    assert values.outlet_mass_flux == pytest.approx(
        np.sum((rho_host[0] * directional * pore)[tuple(outlet)]),
        rel=2e-6,
        abs=2e-6,
    )


@pytest.mark.skipif(
    os.environ.get("KABS_TEST_WARP_CUDA") != "1",
    reason="set KABS_TEST_WARP_CUDA=1 on a CUDA-enabled Warp test runner",
)
def test_warp_velocity_reduction_matches_numpy():
    import warp as wp

    from kabs._warp_convergence import WarpConvergenceMonitor

    previous = np.arange(3 * 2 * 3 * 4, dtype=np.float32).reshape(3, 2, 3, 4)
    current_host = previous.copy()
    current_host[0] += 1.0
    current_host[1] -= 0.5
    current = wp.array(previous, dtype=wp.float32, device="cuda:0")
    rho = wp.ones((1, 2, 3, 4), dtype=wp.float32, device="cuda:0")
    config = ConvergenceConfig(1, 1e-3, None, None)
    monitor = WarpConvergenceMonitor(
        current, np.ones((2, 3, 4), dtype=bool), config, 0
    )

    assert monitor.sample(rho, current).velocity_change is None
    monitor.snapshot_velocity(current)
    wp.copy(
        current,
        wp.array(current_host, dtype=wp.float32, device="cuda:0"),
    )
    metrics = monitor.sample(rho, current)

    assert metrics.velocity_total == pytest.approx(
        np.sum(np.abs(current_host)), rel=2e-6
    )
    assert metrics.velocity_change == pytest.approx(
        np.sum(np.abs(current_host - previous)), rel=2e-6
    )


@pytest.mark.skipif(
    os.environ.get("KABS_TEST_WARP_CUDA") != "1",
    reason="set KABS_TEST_WARP_CUDA=1 on a CUDA-enabled Warp test runner",
)
def test_warp_backend_uses_device_convergence_path():
    result = solve_flow_xlb(
        _channel(),
        n_steps=3,
        log_every=0,
        convergence_every=1,
        velocity_tol=1e9,
        verbose=False,
        compute_backend="warp",
    )

    assert result.n_iterations == 3
    assert result.converged
    assert result.convergence_criterion is not None
    assert result.rho.shape == (6, 6, 6)
    assert result.velocity.shape == (6, 6, 6, 3)
    assert result.rho.dtype == np.float32
    assert result.velocity.dtype == np.float32
    assert np.isfinite(result.rho).all()
    assert np.isfinite(result.velocity).all()
    assert result.collision_model == "srt"
    assert np.isfinite(compute_permeability(result, verbose=False)["k_lu"])
