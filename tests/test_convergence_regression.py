"""Solver regressions against the former host-side convergence calculation."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from kabs import compute_permeability, solve_flow_taichi
from kabs._single_phase_solver import SinglePhaseSolver
from kabs._solve_flow import FlowResult, _BC_SETTERS, _RHO_IN, _RHO_OUT


# Device reductions can sum float32 values in a different order from NumPy.
# These tolerances cover that ordering difference without masking a changed
# formula. A one-interval stopping difference is accepted only when the legacy
# criterion lies within this same narrow band around the requested tolerance.
_CRITERION_RTOL = 2e-5
_CRITERION_ATOL = 2e-7
_FIELD_RTOL = 2e-6
_FIELD_ATOL = 2e-7
_PERMEABILITY_RTOL = 2e-6


@pytest.fixture(autouse=True)
def _reset_taichi_after_test():
    """Prevent permanent solver fields from accumulating between tests."""
    yield
    ti.reset()
    ti.init(arch=ti.cpu)


def _channel(shape=(6, 6, 6)):
    image = np.zeros(shape, dtype=np.int8)
    image[:, 2:4, 2:4] = 1
    return image


def _new_solver(image, direction, nu, storage, tile_size):
    solid_image = (image == 0).astype(np.int8)
    solver = SinglePhaseSolver(
        solid_image,
        storage=storage,
        tile_size=tile_size,
    )
    set_inlet, set_outlet = _BC_SETTERS[direction]
    getattr(solver, set_inlet)(_RHO_IN)
    getattr(solver, set_outlet)(_RHO_OUT)
    solver.set_viscosity(nu)
    solver.init_simulation()
    return solver


def _solve_with_numpy_reference(
    image,
    *,
    direction="x",
    n_steps,
    nu,
    log_every,
    tol,
    storage,
    tile_size,
):
    """Reproduce the pre-optimization host convergence loop for tests only."""
    solver = _new_solver(image, direction, nu, storage, tile_size)
    previous = None
    trace = []
    final_step = n_steps
    final_criterion = None

    for i in range(n_steps + 1):
        solver.step()
        if i % log_every != 0:
            continue

        current = solver.get_velocity()
        if previous is not None:
            v_total = np.sum(np.abs(current))
            v_change = np.sum(np.abs(current - previous))
            criterion = None
            if v_total > 0:
                criterion = v_change / v_total
                final_criterion = criterion
            trace.append((i, v_total, v_change, criterion))
            if tol is not None and v_total > 0 and criterion < tol:
                final_step = i
                break
        previous = current

    result = FlowResult(
        solver,
        direction,
        nu,
        n_iterations=final_step,
        convergence_criterion=final_criterion,
    )
    return result, trace


def _solve_with_device_trace(monkeypatch, image, **kwargs):
    trace = []
    original = SinglePhaseSolver.get_convergence_sums

    def traced_sums(self):
        totals = original(self)
        trace.append(totals)
        return totals

    with monkeypatch.context() as context:
        context.setattr(SinglePhaseSolver, "get_convergence_sums", traced_sums)
        result = solve_flow_taichi(image, verbose=False, **kwargs)
    return result, trace


def _device_criteria_by_step(trace, log_every):
    return {
        check_number * abs(log_every): (v_change / v_total if v_total > 0 else None)
        for check_number, (v_total, v_change) in enumerate(trace, start=1)
    }


def _assert_stopping_policy(
    device,
    reference,
    log_every,
    tol,
    device_trace,
    reference_trace,
):
    if device.n_iterations == reference.n_iterations:
        return

    assert tol is not None
    assert abs(device.n_iterations - reference.n_iterations) == abs(log_every)
    earlier_step = min(device.n_iterations, reference.n_iterations)
    device_criteria = _device_criteria_by_step(device_trace, log_every)
    reference_criteria = {step: criterion for step, _, _, criterion in reference_trace}
    for criteria in (device_criteria, reference_criteria):
        assert criteria[earlier_step] == pytest.approx(
            tol,
            rel=_CRITERION_RTOL,
            abs=_CRITERION_ATOL,
        )


def _assert_results_equivalent(
    device,
    reference,
    log_every,
    tol,
    device_trace,
    reference_trace,
):
    _assert_stopping_policy(
        device,
        reference,
        log_every,
        tol,
        device_trace,
        reference_trace,
    )
    if device.n_iterations == reference.n_iterations:
        assert device.convergence_criterion == pytest.approx(
            reference.convergence_criterion,
            rel=_CRITERION_RTOL,
            abs=_CRITERION_ATOL,
        )

    device_k = compute_permeability(device, verbose=False)["k_lu"]
    reference_k = compute_permeability(reference, verbose=False)["k_lu"]
    assert device_k == pytest.approx(reference_k, rel=_PERMEABILITY_RTOL)

    if device.n_iterations == reference.n_iterations:
        np.testing.assert_allclose(
            device.rho,
            reference.rho,
            rtol=_FIELD_RTOL,
            atol=_FIELD_ATOL,
        )
        np.testing.assert_allclose(
            device.velocity,
            reference.velocity,
            rtol=_FIELD_RTOL,
            atol=_FIELD_ATOL,
        )


@pytest.mark.parametrize("storage", ["dense", "tiled", "sparse"])
def test_full_run_matches_numpy_reference_for_each_storage(storage, monkeypatch):
    kwargs = {
        "direction": "x",
        "n_steps": 10,
        "nu": 1.0 / 6.0,
        "log_every": 5,
        "tol": None,
        "storage": storage,
        "tile_size": (4, 3, 4),
    }
    image = _channel()

    device, device_trace = _solve_with_device_trace(monkeypatch, image, **kwargs)
    reference, reference_trace = _solve_with_numpy_reference(image, **kwargs)

    assert device.n_iterations == reference.n_iterations == kwargs["n_steps"]
    assert len(device_trace) == len(reference_trace) == 2
    for totals, (_, v_total, v_change, _) in zip(
        device_trace, reference_trace, strict=True
    ):
        assert totals[0] == pytest.approx(
            v_total, rel=_CRITERION_RTOL, abs=_CRITERION_ATOL
        )
        assert totals[1] == pytest.approx(
            v_change, rel=_CRITERION_RTOL, abs=_CRITERION_ATOL
        )
    _assert_results_equivalent(
        device,
        reference,
        kwargs["log_every"],
        None,
        device_trace,
        reference_trace,
    )


def test_early_stopping_matches_numpy_reference(monkeypatch):
    kwargs = {
        "direction": "x",
        "n_steps": 20,
        "nu": 1.0 / 6.0,
        "log_every": 5,
        "tol": 0.5,
        "storage": "dense",
        "tile_size": 4,
    }
    image = _channel()

    device, device_trace = _solve_with_device_trace(monkeypatch, image, **kwargs)
    reference, reference_trace = _solve_with_numpy_reference(image, **kwargs)

    assert device.n_iterations < kwargs["n_steps"]
    _assert_results_equivalent(
        device,
        reference,
        kwargs["log_every"],
        kwargs["tol"],
        device_trace,
        reference_trace,
    )


def test_one_interval_exception_requires_threshold_edge():
    device = SimpleNamespace(n_iterations=5)
    reference = SimpleNamespace(n_iterations=10)
    reference_trace = [(5, 2.0, 1.0, 0.5), (10, 2.0, 0.5, 0.25)]

    _assert_stopping_policy(
        device,
        reference,
        log_every=5,
        tol=0.5,
        device_trace=[(2.0, 1.0)],
        reference_trace=reference_trace,
    )

    with pytest.raises(AssertionError):
        _assert_stopping_policy(
            device,
            reference,
            log_every=5,
            tol=0.5,
            device_trace=[(2.0, 0.8)],
            reference_trace=reference_trace,
        )


def test_progress_output_retains_convergence_vocabulary(capsys):
    result = solve_flow_taichi(
        _channel(),
        n_steps=10,
        log_every=5,
        tol=0.5,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert result.n_iterations == 10
    assert "|v|=" in output
    assert "delta|v|=" in output
    assert "Converged at step 10" in output
    assert "delta|v|/|v|" in output
    assert "residual" not in output.lower()
