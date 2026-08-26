"""Device-side convergence regressions for the XLB solver."""

import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kabs import compute_permeability, solve_flow_xlb
from kabs._solve_flow_xlb import _convergence_sums_to_host


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


def test_xlb_solver_matches_legacy_host_convergence(monkeypatch, capsys):
    xlb_module = importlib.import_module("kabs._solve_flow_xlb")
    kwargs = {
        "n_steps": 20,
        "log_every": 5,
        "tol": 0.2,
        "verbose": False,
        "compute_backend": "jax",
    }
    image = _channel()

    device = solve_flow_xlb(image, **{**kwargs, "verbose": True})
    output = capsys.readouterr().out
    assert "|v|=" in output
    assert "delta|v|=" in output
    assert "Converged at step" in output
    assert "delta|v|/|v|" in output
    assert "residual" not in output.lower()
    with monkeypatch.context() as context:
        context.setattr(
            xlb_module,
            "_convergence_sums_to_host",
            _numpy_convergence_sums,
        )
        reference = solve_flow_xlb(image, **kwargs)

    assert device.n_iterations == reference.n_iterations < kwargs["n_steps"]
    assert device.convergence_criterion == pytest.approx(
        reference.convergence_criterion, rel=2e-5, abs=2e-7
    )
    np.testing.assert_allclose(device.rho, reference.rho, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        device.velocity,
        reference.velocity,
        rtol=1e-6,
        atol=1e-7,
    )
    assert compute_permeability(device, verbose=False)["k_lu"] == pytest.approx(
        compute_permeability(reference, verbose=False)["k_lu"], rel=1e-6
    )


def test_periodic_checks_do_not_convert_full_velocity_to_numpy(monkeypatch):
    xlb_module = importlib.import_module("kabs._solve_flow_xlb")
    original = np.array
    original_reduction = xlb_module._convergence_sums_to_host
    full_velocity_conversions = []
    reduction_calls = 0
    expected_shape = (3, 6, 6, 6)

    def tracked_array(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if isinstance(value, jax.Array) and result.shape == expected_shape:
            full_velocity_conversions.append(result.shape)
        return result

    def tracked_reduction(current, previous):
        nonlocal reduction_calls
        reduction_calls += 1
        return original_reduction(current, previous)

    monkeypatch.setattr(xlb_module.np, "array", tracked_array)
    monkeypatch.setattr(xlb_module, "_convergence_sums_to_host", tracked_reduction)

    result = solve_flow_xlb(
        _channel(),
        n_steps=10,
        log_every=5,
        tol=None,
        verbose=False,
    )

    assert result.convergence_criterion is not None
    assert reduction_calls == 2  # Checks at 5 and 10; step 0 only snapshots.
    assert full_velocity_conversions == [expected_shape]


def test_single_check_skips_convergence_reduction(monkeypatch):
    xlb_module = importlib.import_module("kabs._solve_flow_xlb")

    def unexpected_reduction(*args):
        raise AssertionError("a single check must not run a convergence reduction")

    monkeypatch.setattr(xlb_module, "_convergence_sums_to_host", unexpected_reduction)
    result = solve_flow_xlb(
        _channel(),
        n_steps=0,
        log_every=5,
        tol=None,
        verbose=False,
    )

    assert result.n_iterations == 0
    assert result.convergence_criterion is None


@pytest.mark.skipif(
    os.environ.get("KABS_TEST_WARP_CUDA") != "1",
    reason="set KABS_TEST_WARP_CUDA=1 on a CUDA-enabled Warp test runner",
)
def test_warp_backend_uses_device_convergence_path():
    result = solve_flow_xlb(
        _channel(),
        n_steps=1,
        log_every=1,
        tol=None,
        verbose=False,
        compute_backend="warp",
    )

    assert result.n_iterations == 1
    assert result.convergence_criterion is not None
