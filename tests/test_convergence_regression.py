"""Solver-loop regressions for composable convergence monitoring."""

import numpy as np
import pytest
import taichi as ti

from kabs import solve_flow_taichi
from kabs._single_phase_solver import SinglePhaseSolver


@pytest.fixture(autouse=True)
def _reset_taichi_after_test():
    yield
    ti.reset()
    ti.init(arch=ti.cpu)


def _channel(shape=(6, 6, 6)):
    image = np.zeros(shape, dtype=np.int8)
    image[:, 2:4, 2:4] = 1
    return image


@pytest.mark.parametrize("storage", ["dense", "tiled", "sparse"])
def test_n_steps_is_exact_for_every_storage(storage, monkeypatch):
    calls = 0
    original = SinglePhaseSolver.step

    def counted_step(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(SinglePhaseSolver, "step", counted_step)
    result = solve_flow_taichi(
        _channel(),
        n_steps=4,
        log_every=0,
        velocity_tol=None,
        k_tol=None,
        flux_tol=None,
        storage=storage,
        tile_size=4,
        verbose=False,
    )
    assert calls == 4
    assert result.n_iterations == 4
    assert result.converged is None


def test_zero_steps_executes_no_updates(monkeypatch):
    def unexpected_step(self):
        raise AssertionError("n_steps=0 must not execute a timestep")

    monkeypatch.setattr(SinglePhaseSolver, "step", unexpected_step)
    result = solve_flow_taichi(
        _channel(),
        n_steps=0,
        log_every=0,
        velocity_tol=None,
        verbose=False,
    )
    assert result.n_iterations == 0
    assert result.converged is None


def test_public_default_and_legacy_alias_normalization():
    default = solve_flow_taichi(_channel(), n_steps=0, log_every=0, verbose=False)
    assert default.velocity_tol == pytest.approx(1e-3)
    assert default.converged is False

    with pytest.warns(DeprecationWarning):
        disabled = solve_flow_taichi(
            _channel(), n_steps=0, log_every=0, tol=None, verbose=False
        )
    assert disabled.velocity_tol is None
    assert disabled.converged is None


def test_public_legacy_and_new_velocity_tolerances_conflict():
    with pytest.raises(ValueError, match="cannot both"):
        solve_flow_taichi(
            _channel(),
            n_steps=0,
            tol=1e-3,
            velocity_tol=1e-3,
            verbose=False,
        )


def test_logging_frequency_does_not_change_convergence():
    common = dict(
        n_steps=8,
        convergence_every=2,
        velocity_tol=1e9,
        verbose=False,
    )
    frequent = solve_flow_taichi(_channel(), log_every=1, **common)
    silent = solve_flow_taichi(_channel(), log_every=0, **common)
    assert frequent.n_iterations == silent.n_iterations == 6
    assert frequent.converged and silent.converged
    assert frequent.velocity_criterion == pytest.approx(
        silent.velocity_criterion, rel=2e-5, abs=2e-7
    )


def test_permeability_and_flux_can_run_without_velocity_snapshot():
    result = solve_flow_taichi(
        _channel(),
        n_steps=4,
        log_every=0,
        convergence_every=1,
        velocity_tol=None,
        k_tol=1e9,
        flux_tol=1e9,
        verbose=False,
    )
    assert result._solver.v_previous is None
    assert result.velocity_criterion is None
    assert result.k_criterion is not None
    assert result.flux_criterion is not None


def test_progress_output_lists_only_enabled_criteria(capsys):
    result = solve_flow_taichi(
        _channel(),
        n_steps=3,
        log_every=0,
        convergence_every=1,
        velocity_tol=1e9,
        k_tol=None,
        flux_tol=None,
        verbose=True,
    )
    output = capsys.readouterr().out
    assert "velocity=undefined" in output
    assert "velocity=" in output
    assert "K=" not in output
    assert "flux=" not in output
    assert "streak=2" in output
    assert "Converged at step 3" in output
    assert result.converged
